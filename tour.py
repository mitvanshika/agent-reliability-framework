"""
The guided tour (/tour/<target>): how one system was evaluated, step by step, in plain
words, built from the run files on disk. No LLM calls: every "Explain" panel and every
answer to a "Questions" card is written here, filled with this system's real numbers,
so it is free and safe on the public demo and stays right after new runs.

Nothing here is about one particular system. What a visitor reads about the system comes
from its targets.yaml entry, written by whoever plugged it in (all optional):
    title: a short name                 (default: the target's name)
    about: a plain description          (default: its description)
    tour_question: a question id        (default: the one that wobbled most in the A/A test)
Everything else is read from the run files.

    systems(runs_dir, baselines_dir) -> every plugged-in system, for the /tour list
    tour(target, runs_dir, baselines_dir, noise_dir, config_dir) -> dict for templates/tour.html

Steps: the system -> the test questions -> one answer up close -> the judges ->
the baseline -> noise (the A/A test) -> each experiment -> what it can't tell you.
"""

import glob
import json
import os

import yaml

JUDGE_PLAIN = {
    "correctness_gt": ("Correctness", "Does the answer match the answer key (the facts a correct answer must contain)?"),
    "completeness": ("Completeness", "Does it answer every part of the question? An honest \"the source doesn't say\" counts when the information really isn't there."),
    "groundedness": ("Groundedness", "Is everything it says backed by the evidence it found? Anything not in the evidence counts as made up."),
    "context_relevance": ("Context relevance", "Did the search find the material needed to answer? (Not applicable when nothing was retrieved.)"),
    "tool_efficiency": ("Tool efficiency", "Did it use sensible tools without waste?"),
}

EXPLAIN = {
    "scenarios": ("A test question with an answer key: the facts a correct answer has to contain. The same questions "
                  "are asked of every version of the system, so versions can be compared question by question."),
    "judge": ("Each judge is an LLM given a narrow job and a rubric. It reads the question, the answer, the evidence "
              "and the answer key, and returns a score from 0 to 10, a reason in words and how confident it is. "
              "The judges run at temperature 0 and never change between runs: they are the ruler, and the ruler must "
              "stay the same."),
    "final_score": ("The final score is the average of the judges' scores, weighted by each judge's confidence. "
                    "A question passes when its final score is at least the pass mark."),
    "baseline": ("The baseline is the run a human approved as the reference (\"promoted\" it). Every later version is "
                 "compared with it, question by question. It is never chosen automatically: if the latest run were "
                 "used, a bad run could silently become the new normal."),
    "aa": ("An A/A test runs the SAME system with the SAME settings again. Nothing changed, so every difference between "
           "the two runs is chance. It answers two questions: how much does each question wobble on its own, and does "
           "the comparison correctly say \"no real difference\" when nothing changed (no false alarm)?"),
    "band": ("A question's noise band is how far its score moved between runs where nothing changed. Later, a "
             "question only counts as really up or down if it moves MORE than its own band. A steady question gets a "
             "narrow band (never below 0.5); an unstable one gets a wide band, so its normal wobble isn't mistaken "
             "for a change."),
    "kinds": ("SYSTEM: the two runs found different evidence (or one crashed), so the system itself behaved "
              "differently. JUDGE: the same answer, word for word, got different scores, so the measurement wobbled. "
              "WORDING: same evidence, reworded answer; the judges that moved show where the scores differ. "
              "Borderline: a small move that crossed the pass mark."),
    "mcnemar": ("McNemar's test looks only at the questions that flipped: pass to fail, or fail to pass. If the change "
                "did nothing, flips would go either way like coin tosses. Many flips in ONE direction is unlikely by "
                "chance, so the p-value gets small (below 0.05 = real). Example: 10 flips all one way gives p = 0.002."),
    "wilcoxon": ("The Wilcoxon signed-rank test uses the size of every question's score change, not just the flips. "
                 "If the big moves all go one way, the p-value gets small. Both tests are paired: each question is "
                 "compared with itself, so easy and hard questions don't blur the result."),
    "verdict": ("The verdict comes from the two tests: DECLINED (significantly worse), IMPROVED, IMPROVED WITH "
                "REGRESSIONS (better overall but some questions dropped), NO REAL DIFFERENCE (nothing beyond chance). "
                "When the tests disagree, the more cautious verdict is used."),
    "cost": ("Tokens are the pieces of text sent to and written by the LLM; they are what an API bills for. The "
             "framework counts them for any plugged-in system, per question and per LLM call, so cost is reported "
             "next to quality. Cost has noise too, so a saving must be bigger than what the A/A test moved by chance."),
}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _pct(new, old):
    return round((new - old) / old * 100, 1) if isinstance(new, (int, float)) and old else None


def _clean_diff(lines):
    """Config diff lines without the empty-settings artefact older runs recorded."""
    return [ln for ln in lines or [] if not ln.endswith("{} -> (unset)") and not ln.startswith("settings: {}")]


def _runs(runs_dir, target):
    """[(run_id, meta, comparison or None)] of this target, oldest first."""
    out = []
    for meta_path in sorted(glob.glob(os.path.join(runs_dir, "*", "run_meta.json"))):
        meta = _read_json(meta_path) or {}
        if (meta.get("resolved_config") or {}).get("target") != target:
            continue
        run_dir = os.path.dirname(meta_path)
        out.append((os.path.basename(run_dir), meta, _read_json(os.path.join(run_dir, "comparison.json"))))
    return out


def _scenarios(config_dir, scenarios_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(config_dir, scenarios_dir, "*.yaml"))):
        try:
            with open(f, encoding="utf-8") as fh:
                s = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError):
            continue
        if s.get("id"):
            out[s["id"]] = s
    return out


def _steps(trace):
    """The featured answer's steps in plain words (from its recorded trajectory)."""
    out = []
    for s in trace.get("trajectory") or []:
        if s.get("type") == "tool":
            args = s.get("args") or {}
            what = args.get("query") if isinstance(args, dict) and args.get("query") else json.dumps(args)
            out.append({"kind": "search" if s.get("name") == "retrieve" else "tool",
                        "label": (args.get("strategy") if isinstance(args, dict) else None) or s.get("name"),
                        "text": what, "note": s.get("note") or (s.get("output") or "")[:160]})
        elif s.get("type") == "llm":
            tok = s.get("tokens") or {}
            if tok.get("total") is not None:
                text = f"{tok['total']:,} tokens ({tok.get('input', '?'):,} in, {tok.get('output', '?'):,} out)" \
                    if isinstance(tok.get("input"), int) and isinstance(tok.get("output"), int) else f"{tok['total']:,} tokens"
            elif s.get("error"):
                text = f"failed ({s['error']}): no tokens counted"
            else:
                text = "tokens not reported"
            out.append({"kind": "llm", "label": s.get("label") or s.get("name") or "LLM call",
                        "text": text, "note": "", "failed": bool(s.get("error")),
                        "seconds": s.get("latency_s")})
    return out


def _pricing():
    """The owner's prices (pricing.yaml: model -> USD per 1M input/output tokens), or None. Never guessed."""
    try:
        from run_experiment import load_pricing
        return load_pricing()
    except Exception:  # noqa: BLE001  (a broken pricing file shouldn't break the page)
        return None


def _call_cost(call, pricing):
    price = (pricing or {}).get(call.get("model")) if pricing else None
    if not isinstance(price, dict) or call.get("input_tokens") is None or call.get("output_tokens") is None:
        return None
    if not all(isinstance(price.get(k), (int, float)) for k in ("input", "output")):
        return None
    return (call["input_tokens"] * price["input"] + call["output_tokens"] * price["output"]) / 1_000_000


def _calls(trace, pricing):
    """Every LLM call of one answer, as recorded by the token meter, plus totals. Unknown stays unknown."""
    rows = []
    for c in trace.get("llm_calls") or []:
        rows.append({"label": c.get("label") or "LLM call", "model": c.get("model") or "-",
                     "input": c.get("input_tokens"), "output": c.get("output_tokens"), "total": c.get("total_tokens"),
                     "seconds": c.get("seconds"), "error": c.get("error"), "reason": c.get("reason"),
                     "cost": _call_cost(c, pricing)})
    known = [r for r in rows if r["total"] is not None]
    usage = trace.get("usage") or {}
    costs = [r["cost"] for r in known]
    return {"rows": rows, "failed": sum(1 for r in rows if r["error"]),
            "unmetered": sum(1 for r in rows if r["total"] is None and not r["error"]),
            "input": usage.get("input_tokens"), "output": usage.get("output_tokens"),
            "total": usage.get("total_tokens"),
            "cost": sum(costs) if known and all(c is not None for c in costs) else None}


def _where_tokens_went(runs_dir, run_id, ids, pricing):
    """Across one run: tokens per kind of LLM call, the costliest questions, and what isn't known."""
    by_label, per_q, unknown, failed, calls_n = {}, [], [], 0, 0
    priced, all_priced = 0.0, bool(pricing)
    for sid in ids:
        tr = _judged(runs_dir, run_id, sid) or _read_json(os.path.join(runs_dir, run_id, "traces", f"{sid}.json"))
        if not tr:
            continue
        tot = (tr.get("usage") or {}).get("total_tokens")
        calls = tr.get("llm_calls") or []
        if tot is None:
            unknown.append(sid)
        else:
            per_q.append({"id": sid, "tokens": tot, "calls": len(calls), "score": tr.get("final_score")})
        for c in calls:
            calls_n += 1
            if c.get("error"):
                failed += 1
            lab = by_label.setdefault(c.get("label") or "unlabelled", {"label": c.get("label") or "unlabelled",
                                                                       "calls": 0, "input": 0, "output": 0, "total": 0})
            lab["calls"] += 1
            for k, src in (("input", "input_tokens"), ("output", "output_tokens"), ("total", "total_tokens")):
                lab[k] += c.get(src) or 0
            cost = _call_cost(c, pricing)
            if c.get("total_tokens") is not None:
                if cost is None:
                    all_priced = False
                else:
                    priced += cost
    grand = sum(x["total"] for x in by_label.values()) or 0
    labels = sorted(by_label.values(), key=lambda x: -x["total"])
    for x in labels:
        x["share"] = round(100 * x["total"] / grand, 1) if grand else None
    per_q.sort(key=lambda x: -x["tokens"])
    return {"by_label": labels, "costliest": per_q[:5], "unknown": unknown, "failed_calls": failed,
            "calls": calls_n, "avg_per_question": round(sum(q["tokens"] for q in per_q) / len(per_q)) if per_q else None,
            "cost": round(priced, 4) if pricing and all_priced and calls_n else None}


def _judged(runs_dir, run_id, sid):
    return _read_json(os.path.join(runs_dir, run_id, "judged", f"{sid}.json"))


def _judges_of(trace):
    out = []
    for name, (title, what) in JUDGE_PLAIN.items():
        r = trace.get(f"{name}_result")
        if isinstance(r, dict) and r.get("score") is not None and not r.get("na"):
            out.append({"name": name, "title": title, "what": what, "score": r["score"],
                        "reason": r.get("reason") or "", "confidence": r.get("confidence")})
    return out


def _how_plugged_in(meta, trace):
    """One plain sentence on how the system is connected, from what the run saved (no system-specific text)."""
    fn = str(meta.get("target_function") or "")
    parts = []
    if fn.startswith("adapters."):
        parts.append(f"It is plugged in through an adapter ({fn.rsplit('.', 1)[0].replace('.', '/')}.py), "
                     "so the system's own code is untouched")
    elif fn:
        parts.append(f"The framework calls it as {fn}()")
    if meta.get("accepts_settings"):
        parts.append("Experiments change its settings, one change at a time")
    src = (trace or {}).get("usage_source")
    if src == "meter":
        parts.append("It doesn't report its own token use, so the framework counted every LLM call it made")
    elif src == "target":
        parts.append("Token use is reported with each answer")
    if (trace or {}).get("trajectory"):
        parts.append("Each step it took was recorded")
    return " ".join(p + "." for p in parts) if parts else None


def systems(runs_dir="runs", baselines_dir="baselines", targets=None):
    """Every plugged-in system (targets.yaml), for the /tour list: its title, what it does, and how far it got."""
    import target_config
    targets = targets if targets is not None else target_config._targets()
    out = []
    for name in targets:
        cfg = target_config.get(name, targets)
        base = _read_json(os.path.join(baselines_dir, f"{name}.json")) or {}
        recs = base.get("scenarios") or {}
        out.append({"target": name, "title": cfg["title"] or name, "about": cfg["about"] or cfg["description"],
                    "runs": len(_runs(runs_dir, name)), "has_baseline": bool(recs),
                    "passed": sum(1 for r in recs.values() if r.get("passed")), "total": len(recs)})
    return sorted(out, key=lambda x: (not x["has_baseline"], -x["runs"], x["target"]))


def _pick_question(cfg, recs, noise, runs_dir, base_run):
    """The question shown up close: the owner's choice, else the one that wobbled most between identical runs
    (the most to learn from), else the first. Only questions with a saved judged answer qualify."""
    have = [sid for sid in sorted(recs) if os.path.isfile(os.path.join(runs_dir, base_run, "judged", f"{sid}.json"))]
    if cfg.get("tour_question") in have:
        return cfg["tour_question"]
    ranges = {sid: st.get("range") for sid, st in ((noise or {}).get("scenarios") or {}).items()
              if st.get("range") is not None and sid in have}
    if ranges:
        return max(ranges, key=lambda sid: ranges[sid])
    return have[0] if have else None


def tour(target, runs_dir="runs", baselines_dir="baselines", noise_dir="noise", config_dir=".",
         targets=None):
    import target_config
    cfg = target_config.get(target, targets)
    threshold = cfg["pass_threshold"]
    scen = _scenarios(config_dir, cfg["scenarios"])
    base = _read_json(os.path.join(baselines_dir, f"{target}.json"))
    noise = _read_json(os.path.join(noise_dir, f"{target}_noise.json")) or {}
    runs = _runs(runs_dir, target)
    base_run = os.path.basename(str((base or {}).get("run_dir") or "").replace("\\", "/").rstrip("/"))
    t = {"target": target, "title": cfg["title"] or target, "plain": cfg["about"] or cfg["description"],
         "threshold": threshold, "explain": EXPLAIN, "has_baseline": bool(base),
         "judges_used": [JUDGE_PLAIN.get(j, (j,))[0] for j in (cfg["judges"] or JUDGE_PLAIN)]}

    # 2. the test questions
    cats = {}
    for s in scen.values():
        cats.setdefault(s.get("category") or "other", []).append(s["id"])
    t["questions"] = {"count": len(scen), "categories": sorted(((c, len(v)) for c, v in cats.items()),
                                                                key=lambda x: -x[1])}
    if not base:
        return t

    # 5. the baseline
    recs = base.get("scenarios") or {}
    summ = base.get("summary") or {}
    passed = [sid for sid, r in recs.items() if r.get("passed")]
    failed = sorted((sid for sid in recs if sid not in passed), key=lambda s: recs[s].get("final_score") or 0)
    by_cat = []
    for c, ids in sorted(cats.items()):
        got = [recs[i] for i in ids if i in recs]
        if got:
            by_cat.append({"category": c, "n": len(got), "passed": sum(1 for r in got if r.get("passed")),
                           "mean": round(sum(r.get("final_score") or 0 for r in got) / len(got), 2)})
    judge_means = summ.get("per_judge_mean") or {}
    pricing = _pricing()
    tok = summ.get("tokens") or {}
    t["cost_note"] = ("Cost in money uses the prices in pricing.yaml, for the system's own calls only (not the judges)."
                      if pricing else "Tokens only: no pricing.yaml, so no money cost is shown. The system's owner "
                      "can add one (model -> USD per 1M input and output tokens); the framework never guesses prices.")
    t["baseline"] = {"tokens_in": tok.get("input_tokens"), "tokens_out": tok.get("output_tokens"),
                     "spend": _where_tokens_went(runs_dir, base_run, sorted(recs), pricing),
                     "run": base_run, "promoted": (base.get("promoted_at") or "")[:10], "note": base.get("note"),
                     "passed": len(passed), "total": len(recs), "mean": summ.get("mean_final_score"),
                     "tokens": (summ.get("tokens") or {}).get("total_tokens"),
                     "latency": summ.get("latency_mean_s"), "by_category": by_cat,
                     "judge_means": [{"title": JUDGE_PLAIN.get(n, (n,))[0], "mean": m}
                                     for n, m in judge_means.items() if m is not None],
                     "failed": [{"id": sid, "score": recs[sid].get("final_score"),
                                 "lowest": min(((n, s) for n, s in (recs[sid].get("judges") or {}).items()),
                                               key=lambda x: x[1], default=(None, None)),
                                 "question": (scen.get(sid) or {}).get("question")} for sid in failed]}

    # 3 + 4. one answer up close, with its judges
    featured = _pick_question(cfg, recs, noise, runs_dir, base_run)
    tr = _judged(runs_dir, base_run, featured) if featured else None
    base_meta = next((m for rid, m, c in runs if rid == base_run), {})
    t["plugged_in"] = _how_plugged_in(base_meta, tr)
    if tr:
        s = scen.get(featured) or {}
        t["featured"] = {"id": featured, "question": tr.get("question"), "category": tr.get("category"),
                         "answer_key": s.get("expected_answer"), "answer": tr.get("answer"),
                         "steps": _steps(tr), "judges": _judges_of(tr), "final": tr.get("final_score"),
                         "passed": (tr.get("final_score") or 0) >= threshold,
                         "tokens": (tr.get("usage") or {}).get("total_tokens"), "seconds": tr.get("latency_s"),
                         "calls": len(tr.get("llm_calls") or []), "call_table": _calls(tr, pricing),
                         "why_this_one": "chosen by the system's owner" if featured == cfg.get("tour_question")
                         else "picked because it wobbled the most between identical runs"
                         if featured in ((noise or {}).get("scenarios") or {}) else "the first question"}

    # 6. noise: the A/A run(s) and the bands
    aa = [(rid, m, c) for rid, m, c in runs if c and not _clean_diff(c.get("config_diff"))]
    if aa:
        rid, meta, c = aa[-1]
        res = c.get("resources") or {}
        tok = res.get("total_tokens") or {}
        t["aa"] = {"run": rid, "verdict": c.get("verdict"), "pairs": c.get("pairs"),
                   "b": (c.get("mcnemar") or {}).get("b_pass_to_fail"),
                   "c": (c.get("mcnemar") or {}).get("c_fail_to_pass"),
                   "p_mcnemar": (c.get("mcnemar") or {}).get("p"), "p_wilcoxon": (c.get("wilcoxon") or {}).get("p"),
                   "tokens_pct": tok.get("pct"),
                   "moved": sorted(r["scenario_id"] for r in c.get("rows") or [] if abs(r.get("delta") or 0) >= 0.5)}
    sc = noise.get("scenarios") or {}
    if sc:
        ranges = {sid: st.get("range") for sid, st in sc.items() if st.get("range") is not None}
        expl = noise.get("jumpy_explained") or {}
        jumpy = []
        for sid in noise.get("jumpy") or []:
            e = expl.get(sid) or {}
            jumpy.append({"id": sid, "range": ranges.get(sid), "kind": e.get("kind"), "why": e.get("why"),
                          "borderline": e.get("borderline"), "detail": e.get("detail") or [],
                          "scores": e.get("scores") or [], "judges_moved": e.get("judges_moved") or {}})
        suite = noise.get("suite") or {}
        t["noise"] = {"runs": noise.get("runs"), "steady": sum(1 for r in ranges.values() if r < 0.5),
                      "total": len(ranges), "jumpy": jumpy,
                      "widest": max(ranges.items(), key=lambda x: x[1]) if ranges else None,
                      "mean_range": suite.get("mean_score_range"), "pass_rate_range": suite.get("pass_rate_range")}

    # 7. experiments (each experiment's latest compared run)
    latest = {}
    for rid, meta, c in runs:
        if c and _clean_diff(c.get("config_diff")):
            latest[meta.get("experiment")] = (rid, meta, c)
    exps = []
    for name, (rid, meta, c) in latest.items():
        res = c.get("resources") or {}
        tok, lat = res.get("total_tokens") or {}, res.get("latency_mean_s") or {}
        rows = {r["scenario_id"]: r for r in c.get("rows") or []}
        moved = [{"id": sid, "dir": rows[sid]["direction"], "base": rows[sid]["baseline_score"],
                  "run": rows[sid]["run_score"], "band": rows[sid]["band"], "flip": rows[sid]["flip"],
                  "judges": rows[sid].get("judges_moved") or [],
                  "also_in_aa": sid in set((t.get("aa") or {}).get("moved") or [])}
                 for sid in (c.get("down") or []) + (c.get("up") or []) if sid in rows]
        qc = c.get("question_costs") or {}
        per = {x["scenario_id"]: x for x in qc.get("per_question") or []}
        run_summ = _read_json(os.path.join(runs_dir, rid, "summary.json")) or {}
        exps.append({"name": name, "run": rid, "description": meta.get("description"),
                     "changes": _clean_diff(c.get("config_diff")), "verdict": c.get("verdict"),
                     "flips": c.get("flips") or {}, "counts": c.get("counts") or {},
                     "p_mcnemar": (c.get("mcnemar") or {}).get("p"), "p_wilcoxon": (c.get("wilcoxon") or {}).get("p"),
                     "passed": run_summ.get("passed"), "total": run_summ.get("scenarios"),
                     "mean": run_summ.get("mean_final_score"),
                     "tokens": tok, "latency": lat, "moved": moved,
                     "cost_rows": [per[s] for s in qc.get("biggest_changes") or [] if s in per][:5]})
    t["experiments"] = exps
    t["qa"] = questions(t)
    return t


def _f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def questions(t):
    """The pre-answered questions: written here from this system's own numbers (no LLM)."""
    qa = []
    b, f, n, aa = t.get("baseline"), t.get("featured"), t.get("noise"), t.get("aa")
    if b:
        if b["failed"]:
            lines = "; ".join(f"{x['id']} scored {_f(x['score'], 2)}"
                              + (f" (lowest judge: {x['lowest'][0]} {x['lowest'][1]})" if x["lowest"][0] else "")
                              for x in b["failed"])
            qa.append(("Why didn't every question pass?",
                       f"{b['passed']} of {b['total']} questions scored at least {t['threshold']:g}. The rest: {lines}. "
                       "A failing question isn't necessarily a bug: it means the answer missed facts the answer key "
                       "requires. A test set where everything passes can't show improvements, so a few hard questions "
                       "are useful."))
        qa.append(("Who decides what \"correct\" is?",
                   "Each question has an answer key written in advance. The judges used for this system are "
                   f"{', '.join(t['judges_used'])}; each checks one thing and gives a score with a written reason. "
                   f"A question passes at {t['threshold']:g} or more."))
    qa.append(("Can the judges be wrong?",
               "Yes, and the framework is built around that. The same answer can get different scores on different "
               "runs (see the noise step), so single scores are never trusted; changes are judged over all questions "
               "with paired statistics. A judge can also misunderstand its instructions, which is why every score "
               "comes with a written reason to read. A fix becomes a new judge version, and old runs are re-graded "
               "with it (regrade.py) so they stay comparable."))
    if n and n["jumpy"]:
        sys_q = next((j for j in n["jumpy"] if j["kind"] == "SYSTEM"), None)
        if sys_q:
            extra = "; ".join(ln.strip().rstrip(":") for ln in sys_q["detail"] if ln.strip())
            extra = f"In detail: {extra}." if extra else ""
            qa.append((f"Why did {sys_q['id']} change so much when nothing changed?",
                       f"{sys_q['id']} scored {' and '.join(_f(s) for s in sys_q['scores'])} in two runs of the same "
                       f"system. The framework compared what each run saved: {sys_q['why']}. {extra} So the "
                       "system itself did something different between the runs; the judges weren't the cause."))
        judge_q = next((j for j in n["jumpy"] if j["kind"] in ("JUDGE", "WORDING")), None)
        if judge_q:
            moved = ", ".join(f"{k} {' -> '.join(str(v) for v in vals)}" for k, vals in judge_q["judges_moved"].items())
            qa.append((f"And {judge_q['id']}?",
                       f"{judge_q['id']}: {judge_q['why']}" + (f" (judges that moved: {moved})." if moved else ".")
                       + (" It sits right at the pass mark, so a small wobble flips it." if judge_q["borderline"] else "")))
    if n:
        w = n.get("widest")
        qa.append(("What is noise, and why measure it?",
                   f"Running the same system twice gives slightly different scores: the system's LLM and the judges "
                   f"both vary. From {n['runs']} identical runs, {n['steady']} of {n['total']} questions moved less than "
                   "0.5 points" + (f"; {w[0]} moved {_f(w[1], 2)}" if w else "") + ". Without knowing this wobble, a "
                   "random move could be mistaken for the effect of a change."))
    if aa:
        qa.append(("How do I know the framework doesn't raise false alarms?",
                   f"The A/A test: the same system run twice, nothing changed. The verdict was {aa['verdict']} "
                   f"({aa['b']} pass->fail vs {aa['c']} fail->pass, McNemar p = {_f(aa['p_mcnemar'], 3)}; Wilcoxon "
                   f"p = {_f(aa['p_wilcoxon'], 3)}). If it had said DECLINED with nothing changed, its verdicts couldn't "
                   "be trusted."))
    for e in t.get("experiments") or []:
        tok = e["tokens"]
        noise_pct = abs(aa["tokens_pct"]) if aa and aa.get("tokens_pct") is not None else None
        real = (" That is well beyond the "
                + f"{noise_pct:.0f}% the token count moved by chance in the A/A test, so the saving is real."
                if noise_pct is not None and tok.get("pct") is not None and abs(tok["pct"]) > 2 * noise_pct else "")
        qa.append((f"What did {e['name']} show?",
                   f"One change: {'; '.join(e['changes'])}. Verdict: {e['verdict']} "
                   f"({e['flips'].get('pass_to_fail', 0)} pass->fail, {e['flips'].get('fail_to_pass', 0)} fail->pass; "
                   f"McNemar p = {_f(e['p_mcnemar'], 3)}, Wilcoxon p = {_f(e['p_wilcoxon'], 3)}). Tokens "
                   f"{tok.get('baseline') or 0:,} -> {tok.get('run') or 0:,} ({_f(tok.get('pct'))}%)." + real))
    sp = (b or {}).get("spend") or {}
    if sp.get("by_label"):
        top = sp["by_label"][0]
        heavy_in = [x["label"] for x in sp["by_label"] if x["total"] and x["label"] != "unlabelled" and x["input"] > 2 * x["output"]]
        heavy_out = [x["label"] for x in sp["by_label"] if x["total"] and x["output"] > x["input"]]
        a = (f"Most of them went to {top['label']}: {top['share']}% of all tokens over {top['calls']} calls. ")
        if heavy_in:
            a += (f"{', '.join(heavy_in)} {'is' if len(heavy_in) == 1 else 'are'} mostly input: the model is sent a lot of text (prompts, passages) and "
                  "writes little back, so sending less text is the lever there. ")
        if heavy_out:
            a += (f"{', '.join(heavy_out)} writes more than it reads, so its output (including any hidden "
                  "reasoning the model does) is the lever there. ")
        if sp.get("costliest"):
            q = sp["costliest"][0]
            a += f"The most expensive question was {q['id']} ({q['tokens']:,} tokens, {q['calls']} calls)."
        qa.append(("Where do the tokens go?", a.strip()))
    qa.append(("What can't these results tell you?",
               f"Each version was run once or twice on {t['questions']['count']} questions, so only fairly large quality "
               "changes are detectable; noise bands from two runs are rough; the judges are LLMs (validated by reading "
               "their reasons, not yet against human labels); and the results hold for these questions, not every "
               "possible question."))
    return [{"q": q, "a": a} for q, a in qa]
