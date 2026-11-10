"""
Data for the dashboard's home page ("/"): truth tests, latest verdict per
experiment, key findings. Every number is read from the run files on disk
(runs/<run>/summary.json, comparison.json, judged/, noise/), never typed in
here; this file only says WHICH run is the evidence for which truth test.
No LLM calls, light imports only (it also runs in demo mode).
"""

import glob
import json
import os

import yaml

from trace_utils import load_latest_traces

# Which run backs which truth test (see evidence/*.md). "expect" is the verdict
# the framework must give for the test to pass.
TRUTH_TESTS = [
    {"name": "Smoke test", "evidence": "smoke_test", "kind": "smoke",
     "run": "baseline_20261106_101645",
     "question": "Does the pipeline answer and judge 2 questions end to end?",
     "expect": "answered + judged, 0 errors"},
    {"name": "A/A (same config twice)", "evidence": "aa_test", "kind": "verdict",
     "run": "baseline_20261106_111506",
     "question": "Does it stay quiet when nothing changed?",
     "expect": "NO REAL DIFFERENCE"},
    {"name": "Known-bad (weak prompt)", "evidence": "known_bad_test", "kind": "verdict",
     "run": "exp_01_weak_prompt_20261107_164943",
     "question": "Does it catch a change that makes the agent worse?",
     "expect": "DECLINED"},
    {"name": "Judge fix (completeness v2)", "evidence": "judge_fix_completeness", "kind": "judge_fix",
     "question": "Do honest \"I don't have that\" answers pass and invented answers fail?",
     "expect": "honest >= 7, invented <= 3",
     # (run, scenario, honest?) on the completeness judge
     "checks": [("baseline_20261106_101856", "gen_015", True),
                ("baseline_20261106_101856", "rag_06", True),
                ("exp_01_weak_prompt_20261107_164943", "gen_005", False),
                ("exp_01_weak_prompt_20261107_164943", "rag_06", False)]},
    {"name": "Test 4: planted bug (10-day refund window)", "evidence": "test4_test5_plan", "kind": "planned",
     "question": "Does it catch a one-line bug in a tool?", "expect": "DECLINED"},
    {"name": "Test 5: paraphrases (better retrieval)", "evidence": "test4_test5_plan", "kind": "planned",
     "question": "Does embeddings retrieval help when users don't use the FAQ's words?",
     "expect": "more right FAQs found"},
]


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _run_files(runs_dir, run_id):
    p = os.path.join(runs_dir, run_id)
    return (_read_json(os.path.join(p, "run_meta.json")), _read_json(os.path.join(p, "summary.json")),
            _read_json(os.path.join(p, "comparison.json")))


def _judged(runs_dir, run_id):
    d = os.path.join(runs_dir, run_id, "judged")
    return load_latest_traces(d) if os.path.isdir(d) else {}


def _score(t, judge):
    r = (t or {}).get(f"{judge}_result")
    return r.get("score") if isinstance(r, dict) else None


def _fmt_p(p):
    return "-" if p is None else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def _stats_line(c):
    m, w, f = c.get("mcnemar") or {}, c.get("wilcoxon") or {}, c.get("flips") or {}
    return (f"McNemar p={_fmt_p(m.get('p'))} ({f.get('pass_to_fail', 0)} pass->fail, "
            f"{f.get('fail_to_pass', 0)} fail->pass), Wilcoxon p={_fmt_p(w.get('p'))}, "
            f"{c.get('pairs')} paired questions")


def truth_tests(runs_dir="runs"):
    rows = []
    for t in TRUTH_TESTS:
        row = {k: t[k] for k in ("name", "evidence", "question", "expect")}
        row.update(run=t.get("run"), got=None, status="missing", detail="run files not found")
        if t["kind"] == "planned":
            row.update(status="planned", got="not run yet", detail="prepared; needs API tokens")
        elif t["kind"] == "smoke":
            meta, s, _ = _run_files(runs_dir, t["run"])
            if s:
                clean = not s.get("agent_errors") and not s.get("judge_errors") and not s.get("not_judged")
                row.update(status="pass" if clean else "fail",
                           got=f"{s.get('scored')}/{s.get('scenarios')} judged, {s.get('agent_errors', 0)} errors",
                           detail=f"pass {s.get('passed')}/{s.get('scenarios')}, mean score {s.get('mean_final_score')}")
        elif t["kind"] == "verdict":
            _, _, c = _run_files(runs_dir, t["run"])
            if c:
                row.update(status="pass" if c.get("verdict") == t["expect"] else "fail",
                           got=c.get("verdict"), detail=_stats_line(c),
                           p=(c.get("mcnemar") or {}).get("p"))
        elif t["kind"] == "judge_fix":
            parts, good, found = [], True, 0
            for run, sid, honest in t["checks"]:
                sc = _score(_judged(runs_dir, run).get(sid), "completeness")
                if sc is None:
                    good = False
                    parts.append(f"{sid}: no score")
                    continue
                found += 1
                good &= (sc >= 7) if honest else (sc <= 3)
                parts.append(f"{sid} ({'honest' if honest else 'invented'}): {sc:g}")
            if found:
                row.update(status="pass" if good else "fail", got="completeness " + ", ".join(parts),
                           detail="completeness judge v2 sees the tool outputs and the answer key")
        rows.append(row)
    return rows


def latest_verdicts(runs_dir="runs", experiment_dirs=("experiments", "examples/summarizer")):
    """One row per experiment: its latest run and that run's comparison verdict.
    Experiments defined in YAML but never run are listed as 'not run yet'."""
    defined = {}
    for d in experiment_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            try:
                y = yaml.safe_load(open(f, encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(y, dict) and y.get("name") and not y["name"].startswith("tmp_"):
                defined[y["name"]] = y.get("description") or ""
    runs = {}
    for meta_path in glob.glob(os.path.join(runs_dir, "*", "run_meta.json")):
        run_id = os.path.basename(os.path.dirname(meta_path))
        meta, s, c = _run_files(runs_dir, run_id)
        exp = (meta or {}).get("experiment")
        if not exp:
            continue
        rec = {"run_id": run_id, "timestamp": meta.get("timestamp") or "", "summary": s, "comparison": c,
               "target": (meta.get("resolved_config") or {}).get("target")}
        runs.setdefault(exp, []).append(rec)
    out = []
    for exp in sorted(set(defined) | set(runs), key=lambda e: (e != "baseline", e)):
        rs = sorted(runs.get(exp, []), key=lambda r: (r["timestamp"], r["run_id"]))
        row = {"experiment": exp, "description": defined.get(exp, ""), "runs": len(rs), "run_id": None,
               "target": None, "pass_rate": None, "scenarios": None, "verdict": None, "p": None, "pairs": None}
        # The latest compared run that covers the most questions: a 6-question
        # smoke re-run must not stand in for the experiment's full-suite verdict.
        compared = [r for r in rs if r["comparison"]]
        if compared:
            size = lambda r: (r["summary"] or {}).get("scenarios") or r["comparison"].get("pairs") or 0  # noqa: E731
            full = max(size(r) for r in compared)
            compared = [r for r in compared if size(r) == full]
        r = compared[-1] if compared else (rs[-1] if rs else None)
        if r is None:
            row.update(status="not run yet")
        else:
            s, c = r["summary"] or {}, r["comparison"] or {}
            row.update(run_id=r["run_id"], target=r["target"], pass_rate=s.get("pass_rate"),
                       scenarios=s.get("scenarios"), verdict=c.get("verdict"),
                       p=(c.get("mcnemar") or {}).get("p"), pairs=c.get("pairs"),
                       status="compared" if c else ("judged, no baseline to compare" if s else "not judged"))
        out.append(row)
    return out


def key_findings(runs_dir="runs", noise_dir="noise"):
    """Short findings, each with the numbers read from the runs (skipped if the runs are missing)."""
    out = []
    j1, j2 = _judged(runs_dir, "baseline_20261106_101856"), _judged(runs_dir, "baseline_20261106_111506")
    a, b = _score(j1.get("edge_03"), "tool_efficiency"), _score(j2.get("edge_03"), "tool_efficiency")
    if a is not None and b is not None:
        out.append({"title": "Judges are noisy, even at temperature 0",
                    "text": f"edge_03 got the same answer word for word in both A/A runs, yet the tool_efficiency "
                            f"judge scored it {a:g} and then {b:g}. That is why the framework measures noise bands "
                            f"and uses paired significance tests instead of eyeballing per-question scores."})
    _, _, c = _run_files(runs_dir, "exp_01_weak_prompt_20261107_164943")
    if c:
        f = c.get("flips") or {}
        deltas = {k: v.get("delta") for k, v in (c.get("per_judge") or {}).items()
                  if isinstance(v, dict) and v.get("delta") is not None and v.get("in_final_score")}
        worst = sorted(deltas.items(), key=lambda kv: kv[1])[:2]
        worst_s = ", ".join(f"{k} {v:+.2f}" for k, v in worst)
        out.append({"title": "A weaker prompt makes the agent invent answers",
                    "text": f"With the weak prompt, {f.get('pass_to_fail', 0)} questions went from pass to fail and "
                            f"{f.get('fail_to_pass', 0)} the other way (McNemar p={_fmt_p((c.get('mcnemar') or {}).get('p'))})."
                            + (f" The biggest drops: {worst_s}." if worst else "")})
    out.append({"title": "The judges once rewarded making things up",
                "text": "Completeness v1 saw only the question and the answer, so an honest \"I don't have that, "
                        "let me get a human\" scored low and a confident invented answer scored high. Completeness v2 "
                        "also sees the tool outputs and the answer key; honest escalations now pass and invented "
                        "answers score 0."})
    noise = _read_json(os.path.join(noise_dir, "configurable_noise.json")) or {}
    if noise.get("jumpy"):
        out.append({"title": "Some questions flip between runs",
                    "text": f"{len(noise['jumpy'])} questions move more than {noise.get('jumpy_range', 2):g} points "
                            f"between two identical baseline runs: {', '.join(noise['jumpy'])}. A single run's "
                            f"result on these is not trustworthy."})
    return out


def overview(runs_dir="runs", noise_dir="noise"):
    return {"truth_tests": truth_tests(runs_dir), "verdicts": latest_verdicts(runs_dir),
            "findings": key_findings(runs_dir, noise_dir)}


# ---------------------------------------------------------------- /evidence
EVIDENCE_DIR = "evidence"


def evidence_list(evidence_dir=EVIDENCE_DIR):
    """[{name, title, summary}] for evidence/*.md: title = the first '# ' line; summary = the
    "Result/Status/Got" paragraph if there is one, else the first prose paragraph."""
    import re
    out = []
    for f in sorted(glob.glob(os.path.join(evidence_dir, "*.md"))):
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), name)
        paras = [" ".join(ln.strip() for ln in p.splitlines()) for p in re.split(r"\n\s*\n", text)]
        prose = [p for p in paras if p and not p.startswith(("#", "|", "-", "`", "1."))]
        hit = next((m for m in (re.search(r"\*\*(Result|Status)\b", p) for p in prose) if m), None)
        summary = hit.string[hit.start():] if hit else (prose[0] if prose else "")
        summary = summary.replace("**", "").replace("`", "")
        out.append({"name": name, "title": title, "summary": summary[:240] + ("..." if len(summary) > 240 else "")})
    return out


def evidence_html(name, evidence_dir=EVIDENCE_DIR):
    """(title, html) for one evidence note, or None. Only names in evidence_list() are served."""
    import markdown
    match = next((e for e in evidence_list(evidence_dir) if e["name"] == name), None)
    if match is None:
        return None
    with open(os.path.join(evidence_dir, name + ".md"), encoding="utf-8") as f:
        text = f.read()
    return match["title"], markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


# ---------------------------------------------------------------- /systems
# Readable role of a run on the Experiments table, beyond its experiment name.
ROLE_LABELS = {"smoke_test": "smoke test", "aa_test": "A/A re-run"}


def run_role(run_id, is_baseline=False):
    """'approved baseline' / 'A/A re-run' / 'smoke test' / None."""
    if is_baseline:
        return "approved baseline"
    return next((ROLE_LABELS[t["evidence"]] for t in TRUTH_TESTS
                 if t.get("run") == run_id and t.get("evidence") in ROLE_LABELS), None)


TRACK1_TARGETS = ("strong", "weak")   # run_suite.py / pipeline.py: compared with each other, no baseline file


def systems(targets_file="targets.yaml", baselines_dir="baselines", runs_dir="runs"):
    """One row per target in targets.yaml: what it is, how it is judged, its scenarios, its baseline."""
    import judge_sets
    import target_config
    try:
        with open(targets_file, encoding="utf-8") as f:
            targets = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        targets = {}
    run_counts = {}
    for meta_path in glob.glob(os.path.join(runs_dir, "*", "run_meta.json")):
        t = ((_read_json(meta_path) or {}).get("resolved_config") or {}).get("target")
        run_counts[t] = run_counts.get(t, 0) + 1
    out = []
    for name, entry in targets.items():
        if not isinstance(entry, dict):
            continue
        cfg = target_config.get(name, targets)
        track1 = name in TRACK1_TARGETS
        try:
            js = judge_sets.CLASSIC if track1 else judge_sets.for_run([], cfg)
        except (KeyError, ValueError) as e:
            js = {"name": f"invalid: {e}", "llm_judges": [], "checks": []}
        scen_dir = cfg["scenarios"]
        n_scen = len(glob.glob(os.path.join(scen_dir, "*.yaml"))) if os.path.isdir(scen_dir) else 0
        base = None if track1 else _read_json(os.path.join(baselines_dir, f"{name}.json"))
        bsum = (base or {}).get("summary") or {}
        out.append({
            "name": name, "track": "Track 1" if track1 else "Track 2",
            "description": cfg["description"] or ("Track 1 sample support bot (strong prompt)" if name == "strong"
                                                  else "Track 1 sample support bot (weak, degraded prompt)"
                                                  if name == "weak" else ""),
            "function": f"{entry.get('module')}.{entry.get('function')}",
            "accepts_settings": bool(entry.get("accepts_settings")),
            "judge_set": js["name"], "judges": js["llm_judges"], "checks": js.get("checks") or [],
            "not_applicable": js.get("not_applicable") or [],
            "pass_threshold": cfg["pass_threshold"], "scenarios_dir": scen_dir, "scenarios": n_scen,
            "action_tools": [a["name"] for a in cfg["action_tools"]],
            "runs": run_counts.get(name, 0),
            "baseline": None if base is None else {
                "experiment": base.get("experiment"), "run": os.path.basename((base.get("run_dir") or "").replace("\\", "/")),
                "promoted_at": base.get("promoted_at"), "pass_rate": bsum.get("pass_rate"),
                "judge_set": (base.get("judge_set") or {}).get("name")},
        })
    return out
