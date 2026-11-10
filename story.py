"""
The guided story on the Overview page: what the framework does, told with the
real recorded runs. Every number, prompt, tool call and judge reason is read
from the run files (runs/ or demo_data/runs/) and the config files those runs
point at (prompts/, docstrings/, scenarios/; demo_data/config/ in demo mode).
Nothing is typed in. No LLM calls, light imports only (also runs in demo mode).
"""

import json
import os

import yaml

# Plain-English hover text for the terms a first-time visitor meets.
GLOSSARY = {
    "baseline": "The approved version of the system. Every change is tested against it on the same questions.",
    "pass": "An answer passes when its final score (the average of the judges' 0-10 scores) is 7 or more.",
    "noise band": "How much a question's score moves when nothing changed at all (the same setup run twice). "
                  "Moves smaller than this are treated as noise, not as a real change.",
    "▲/▼": "▲ the question scored higher than in the baseline by more than its noise band; "
           "▼ it scored lower by more than its noise band; ═ no move beyond the noise.",
    "McNemar": "A test that counts only the questions that flipped (pass to fail, or fail to pass) and asks "
               "whether the flips lean one way more than chance would.",
    "Wilcoxon": "A test on the score changes of every question, paired with itself: did the scores move "
                "down (or up) more consistently than chance would?",
    "p-value": "How likely a difference this big would be if the change did nothing at all. "
               "Below 0.05 we call it real.",
}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def baseline_run_id(baselines_dir, target="configurable"):
    """The run the target's approved baseline was promoted from."""
    b = _read_json(os.path.join(baselines_dir, f"{target}.json")) or {}
    rd = str(b.get("run_dir") or "").replace("\\", "/").rstrip("/")
    return os.path.basename(rd) or None


def _settings(runs_dir, run_id):
    meta = _read_json(os.path.join(runs_dir, run_id, "run_meta.json")) or {}
    return meta, ((meta.get("resolved_config") or {}).get("settings") or {})


def trace(runs_dir, run_id, scenario_id):
    return _read_json(os.path.join(runs_dir, run_id, "traces", f"{scenario_id}.json")) or {}


def conversation(t):
    """A trace's messages as steps: question, tool call (+ its result), answer. The system prompt is left out."""
    steps, results = [], {m.get("tool_call_id"): m.get("content") for m in t.get("messages") or []
                          if m.get("role") == "tool"}
    for m in t.get("messages") or []:
        role = m.get("role")
        if role == "user":
            steps.append({"kind": "question", "text": m.get("content") or ""})
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"raw": fn.get("arguments")}
                steps.append({"kind": "tool", "name": fn.get("name"), "args": args,
                              "result": results.get(call.get("id"), "")})
            if (m.get("content") or "").strip() and not m.get("tool_calls"):
                steps.append({"kind": "answer", "text": m["content"], "html": answer_html(m["content"])})
    return steps


def answer_html(text):
    """The agent writes Markdown (bold, lists, tables): render it, with any HTML in it escaped first."""
    import markdown
    from markupsafe import Markup, escape
    return Markup(markdown.markdown(str(escape(text)), extensions=["tables", "sane_lists"]))


def system_prompt_of(t):
    return next((m.get("content") or "" for m in t.get("messages") or [] if m.get("role") == "system"), "")


def pick_example(runs_dir, run_id, exclude=()):
    """The baseline's showcase conversation: a passing answer that used the most different tools
    (then the highest score). Chosen from the data, not by hand."""
    from trace_utils import load_latest_traces
    jdir = os.path.join(runs_dir, run_id, "judged")
    judged = load_latest_traces(jdir) if os.path.isdir(jdir) else {}
    best = None
    for sid, j in judged.items():
        score = j.get("final_score")
        if sid in exclude or not isinstance(score, (int, float)) or score < 7:
            continue
        tools = set(trace(runs_dir, run_id, sid).get("tools_used") or [])
        key = (len(tools), score, sid)
        if best is None or key > best[0]:
            best = (key, sid)
    return best[1] if best else None


# ---------------------------------------------------------------- 1. the agent
def agent(runs_dir, baselines_dir, config_dir=".", exclude=("gen_004",)):
    run_id = baseline_run_id(baselines_dir)
    if not run_id:
        return None
    meta, s = _settings(runs_dir, run_id)
    docs = _read_yaml(os.path.join(config_dir, s.get("docstrings") or "")) or {}
    sid = pick_example(runs_dir, run_id, exclude)
    t = trace(runs_dir, run_id, sid) if sid else {}
    from trace_utils import load_latest_traces
    judged = load_latest_traces(os.path.join(runs_dir, run_id, "judged")) if sid else {}
    return {
        "run_id": run_id, "description": meta.get("description"),
        "model": s.get("model"), "temperature": s.get("temperature"),
        "prompt_file": s.get("prompt"), "prompt": system_prompt_of(t),
        "tools": [{"name": k, "description": v} for k, v in docs.items()],
        "example": {"scenario_id": sid, "steps": conversation(t), "tools_used": t.get("tools_used") or [],
                    "tokens": (t.get("usage") or {}).get("total_tokens"), "latency_s": t.get("latency_s"),
                    "final_score": (judged.get(sid) or {}).get("final_score")} if sid else None,
    }


# ---------------------------------------------------------------- 2. the tests
# What each scenario-id prefix means (labels only; the counts are read from scenarios/).
GROUPS = {"normal": "hand-written everyday questions", "edge": "hand-written tricky cases", "rag": "policy questions that need the FAQ search",
          "gen": "written by an LLM (generate.py), so the suite isn't only the builder's own questions"}


def scenarios(runs_dir, run_id, config_dir="."):
    meta, _ = _settings(runs_dir, run_id)
    out = []
    for sid in meta.get("scenario_ids") or []:
        y = _read_yaml(os.path.join(config_dir, "scenarios", f"{sid}.yaml"))
        if y:
            out.append(y)
    return out


def tests(runs_dir, baselines_dir, config_dir="."):
    run_id = baseline_run_id(baselines_dir)
    sc = scenarios(runs_dir, run_id, config_dir) if run_id else []
    if not sc:
        return None
    cats, groups = {}, {}
    for y in sc:
        cats[y.get("category")] = cats.get(y.get("category"), 0) + 1
        g = str(y.get("id")).split("_")[0]
        groups[g] = groups.get(g, 0) + 1
    samples = []
    for g in ("normal", "rag", "gen"):   # the first question of three different groups
        y = next((y for y in sc if str(y.get("id")).startswith(g + "_")), None)
        if y:
            samples.append({"id": y["id"], "question": y.get("question"), "expected": y.get("expected_answer"),
                            "tools": y.get("expected_tools") or []})
    return {"total": len(sc), "with_key": sum(1 for y in sc if y.get("expected_answer")),
            "categories": sorted(cats.items(), key=lambda kv: -kv[1]),
            "groups": [{"prefix": g, "count": n, "label": GROUPS.get(g, "")}
                       for g, n in sorted(groups.items(), key=lambda kv: -kv[1])],
            "samples": samples}


# ---------------------------------------------------------------- 3. how evaluation works
JUDGE_PLAIN = {
    "correctness_gt": "Does the answer say what the answer key says?",
    "completeness": "Did it answer every part of the question (or honestly say it can't)?",
    "groundedness": "Is every claim backed by what the tools returned, with nothing made up?",
    "tool_efficiency": "Did it call the right tools, without missing or wasted calls?",
    "context_relevance": "Did the FAQ search bring back text that fits the question? (only scored when it searched)",
    "tool_match": "Not an LLM: did it call the tools the answer key lists? (shown, not counted in the score)",
}


def evaluation(runs_dir, compared_run, noise_dir):
    s = _read_json(os.path.join(runs_dir, compared_run, "summary.json")) or {}
    c = _read_json(os.path.join(runs_dir, compared_run, "comparison.json")) or {}
    js = s.get("judge_set") or c.get("judge_set") or {}
    noise = _read_json(os.path.join(noise_dir, "configurable_noise.json")) or {}
    bands = [r.get("band") for r in c.get("rows") or [] if isinstance(r.get("band"), (int, float))]
    per = noise.get("scenarios") or {}
    return {
        "judges": [{"name": j, "plain": JUDGE_PLAIN.get(j, "")} for j in js.get("llm_judges") or []],
        "checks": [{"name": j, "plain": JUDGE_PLAIN.get(j, "")} for j in js.get("checks") or []],
        "judge_model": js.get("judge_model"), "judge_set": js.get("name"),
        "threshold": s.get("threshold"), "alpha": c.get("alpha"),
        "noise_runs": [{"run": os.path.basename(str(r.get("run_dir"))), "pass_rate": r.get("pass_rate"),
                        "passed": r.get("passed"), "total": r.get("total")}
                       for r in (noise.get("suite") or {}).get("per_run") or []],
        "band_min": min(bands) if bands else None, "band_max": max(bands) if bands else None,
        "calm": sum(1 for v in per.values() if not v.get("jumpy")), "n_noise": len(per),
        "jumpy_range": noise.get("jumpy_range"),
    }


# ---------------------------------------------------------------- 4. the change tested
def word_diff(a, b):
    """Two token lists [(word, changed?)] for side-by-side highlighting (removed in a, added in b)."""
    import difflib
    wa, wb = a.split(), b.split()
    norm = lambda ws: [w.strip(".,;:!?").lower() for w in ws]  # noqa: E731
    left, right = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, norm(wa), norm(wb), autojunk=False).get_opcodes():
        same = op == "equal" and i2 - i1 >= 2   # a lone shared word ("the") is a coincidence, not kept text
        left += [(w, not same) for w in wa[i1:i2]]
        right += [(w, not same) for w in wb[j1:j2]]
    return left, right


def change(runs_dir, baseline_run, run_id, scenario_id="gen_004"):
    meta_b, sb = _settings(runs_dir, baseline_run)
    meta_r, sr = _settings(runs_dir, run_id)
    pa = system_prompt_of(trace(runs_dir, baseline_run, scenario_id))
    pb = system_prompt_of(trace(runs_dir, run_id, scenario_id))
    left, right = word_diff(pa, pb)
    return {"experiment": meta_r.get("experiment"), "description": meta_r.get("description"),
            "changes": meta_r.get("changes_vs_parent") or [],
            "before": {"file": sb.get("prompt"), "words": left, "n_words": len(pa.split())},
            "after": {"file": sr.get("prompt"), "words": right, "n_words": len(pb.split())}}


# ---------------------------------------------------------------- 5. the result
def odds(p):
    """p as 'about N in 1,000' / 'about N in 100' for a plain sentence."""
    if p is None:
        return None
    return f"about {max(1, round(p * 1000))} in 1,000" if p < 0.01 else f"about {round(p * 100)} in 100"


def result(runs_dir, baseline_run, run_id):
    c = _read_json(os.path.join(runs_dir, run_id, "comparison.json")) or {}
    if not c:
        return None
    sb = _read_json(os.path.join(runs_dir, baseline_run, "summary.json")) or {}
    sr = _read_json(os.path.join(runs_dir, run_id, "summary.json")) or {}
    m, w, f = c.get("mcnemar") or {}, c.get("wilcoxon") or {}, c.get("flips") or {}
    tok = (c.get("resources") or {}).get("total_tokens") or {}
    return {"verdict": c.get("verdict"), "pairs": c.get("pairs"), "alpha": c.get("alpha"),
            "before": {"passed": sb.get("passed"), "total": sb.get("scenarios"), "rate": sb.get("pass_rate"),
                       "mean": sb.get("mean_final_score")},
            "after": {"passed": sr.get("passed"), "total": sr.get("scenarios"), "rate": sr.get("pass_rate"),
                      "mean": sr.get("mean_final_score")},
            "worse": f.get("pass_to_fail"), "better": f.get("fail_to_pass"), "counts": c.get("counts") or {},
            "mcnemar_p": m.get("p"), "mcnemar_odds": odds(m.get("p")),
            "wilcoxon_p": w.get("p"), "wilcoxon_odds": odds(w.get("p")), "wilcoxon_n": w.get("n_nonzero"),
            "tokens": tok, "per_judge": [{"name": k, **v} for k, v in (c.get("per_judge") or {}).items()
                                         if isinstance(v, dict) and v.get("in_final_score")]}


# ---------------------------------------------------------------- 6. one question up close
JUDGE_ORDER = ("correctness_gt", "completeness", "groundedness", "tool_efficiency", "context_relevance", "tool_match")


def up_close(runs_dir, baseline_run, run_id, scenario_id="gen_004", config_dir="."):
    from trace_utils import load_latest_traces
    sides = []
    for label, r in (("Baseline (prompts/v1.txt)", baseline_run), ("Weak prompt", run_id)):
        jdir = os.path.join(runs_dir, r, "judged")
        j = (load_latest_traces(jdir) if os.path.isdir(jdir) else {}).get(scenario_id) or {}
        t = trace(runs_dir, r, scenario_id)
        sides.append({"label": label, "run_id": r, "steps": [s for s in conversation(t) if s["kind"] != "question"],
                      "final": j.get("final_score"), "tokens": (t.get("usage") or {}).get("total_tokens"),
                      "judges": {n: {"score": (j.get(f"{n}_result") or {}).get("score"),
                                     "reason": (j.get(f"{n}_result") or {}).get("reason")}
                                 for n in JUDGE_ORDER if j.get(f"{n}_result")}})
    if not sides[0]["steps"]:
        return None
    y = _read_yaml(os.path.join(config_dir, "scenarios", f"{scenario_id}.yaml")) or {}
    row = next((x for x in (_read_json(os.path.join(runs_dir, run_id, "comparison.json")) or {}).get("rows") or []
                if x.get("scenario_id") == scenario_id), {})
    judges = [n for n in JUDGE_ORDER if any(n in s["judges"] for s in sides)]
    return {"scenario_id": scenario_id, "question": y.get("question") or trace(runs_dir, run_id, scenario_id).get("question"),
            "expected": y.get("expected_answer"), "sides": sides, "judges": judges,
            "band": row.get("band"), "delta": row.get("delta"), "flip": row.get("flip")}


# ---------------------------------------------------------------- 7. diagnosis
def _queries(t):
    return [s["args"].get("query") for s in conversation(t) if s["kind"] == "tool" and s["name"] == "faq_search"]


def stopped_searching(runs_dir, baseline_run, run_id, scenario_id):
    """The honest note for a question labelled RAG whose search worked: both agents ran the same first
    searches, and only the baseline kept going and found the answer. None when the data doesn't show that."""
    tb, tr = trace(runs_dir, baseline_run, scenario_id), trace(runs_dir, run_id, scenario_id)
    qb, qr = _queries(tb), _queries(tr)
    if not qr or len(qb) <= len(qr) or qb[:len(qr)] != qr:
        return None
    outs_b = tb.get("tool_outputs") or []
    found = outs_b[-1] if outs_b else ""
    if found in (tr.get("tool_outputs") or []):
        return None
    return {"scenario_id": scenario_id, "shared": qr, "extra": qb[len(qr):], "found": found,
            "n_base": len(qb), "n_run": len(qr)}


def diagnosis(runs_dir, baseline_run, run_id):
    d = _read_json(os.path.join(runs_dir, run_id, "diagnosis.json")) or {}
    c = _read_json(os.path.join(runs_dir, run_id, "comparison.json")) or {}
    rows = {r["scenario_id"]: r for r in c.get("rows") or []}
    items = d.get("diagnoses") or c.get("diagnosis") or []
    if not items:
        return None
    causes = {}
    for x in items:
        k = str(x.get("cause") or "?").split(" (")[0]
        causes.setdefault(k, []).append(x["scenario_id"])
    notes = [n for n in (stopped_searching(runs_dir, baseline_run, run_id, x["scenario_id"])
                         for x in items if str(x.get("cause")).startswith("RAG")) if n]
    return {"causes": sorted(causes.items(), key=lambda kv: -len(kv[1])),
            "rows": [{**x, "before": (rows.get(x["scenario_id"]) or {}).get("baseline_score"),
                       "after": (rows.get(x["scenario_id"]) or {}).get("run_score"),
                       "flip": (rows.get(x["scenario_id"]) or {}).get("flip")} for x in items],
            "notes": notes}


# ---------------------------------------------------------------- 8. why trust it
def trust(runs_dir, noise_dir, baseline_run):
    import overview
    tt = {t["evidence"]: t for t in overview.truth_tests(runs_dir)}
    aa = tt.get("aa_test") or {}
    # Key findings not already told elsewhere in the story (the weak-prompt result is section 5; judge noise
    # and the judge fix get their own cards here).
    told = ("Judges are noisy", "A weaker prompt", "The judges once rewarded")
    out = {"aa": aa, "judge_fix": tt.get("judge_fix_completeness"),
           "findings": [f for f in overview.key_findings(runs_dir, noise_dir) if not f["title"].startswith(told)]}
    # The judge-noise example, checked here: same answer word for word, two different scores.
    aa_run = aa.get("run")
    if aa_run:
        from trace_utils import load_latest_traces
        ta, tb = trace(runs_dir, baseline_run, "edge_03"), trace(runs_dir, aa_run, "edge_03")
        ja = load_latest_traces(os.path.join(runs_dir, baseline_run, "judged")).get("edge_03") or {}
        jb = load_latest_traces(os.path.join(runs_dir, aa_run, "judged")).get("edge_03") or {}
        sa = (ja.get("tool_efficiency_result") or {}).get("score")
        sb = (jb.get("tool_efficiency_result") or {}).get("score")
        if ta.get("answer") and ta.get("answer") == tb.get("answer") and sa is not None and sb is not None:
            out["noise"] = {"scenario_id": "edge_03", "judge": "tool_efficiency", "a": sa, "b": sb,
                            "answer": ta["answer"], "runs": (baseline_run, aa_run),
                            "reason_a": (ja.get("tool_efficiency_result") or {}).get("reason"),
                            "reason_b": (jb.get("tool_efficiency_result") or {}).get("reason")}
    return out


def known_bad_run():
    import overview
    return next((t.get("run") for t in overview.TRUTH_TESTS if t.get("evidence") == "known_bad_test"), None)


def story(runs_dir="runs", baselines_dir="baselines", config_dir=".", noise_dir="noise"):
    base, run = baseline_run_id(baselines_dir), known_bad_run()
    have = bool(base and run and os.path.isfile(os.path.join(runs_dir, run, "comparison.json"))
                and os.path.isdir(os.path.join(runs_dir, base)))
    out = {"glossary": GLOSSARY, "agent": agent(runs_dir, baselines_dir, config_dir),
           "tests": tests(runs_dir, baselines_dir, config_dir), "baseline_run": base, "run_id": run}
    if have:
        out.update(evaluation=evaluation(runs_dir, run, noise_dir), change=change(runs_dir, base, run),
                   result=result(runs_dir, base, run), up_close=up_close(runs_dir, base, run, config_dir=config_dir),
                   diagnosis=diagnosis(runs_dir, base, run), trust=trust(runs_dir, noise_dir, base))
    return out


def config_files(runs_dir, run_ids):
    """The config files these runs point at (prompt, tool descriptions, scenarios): what
    make_demo_data.py copies into demo_data/config/ so the story matches the recorded runs."""
    files = set()
    for r in run_ids:
        meta, s = _settings(runs_dir, r)
        for k in ("prompt", "docstrings"):
            if s.get(k):
                files.add(str(s[k]).replace("\\", "/"))
        for sid in meta.get("scenario_ids") or []:
            files.add(f"scenarios/{sid}.yaml")
    return sorted(files)
