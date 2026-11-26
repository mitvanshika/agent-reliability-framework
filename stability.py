"""
Why is a question unstable? Explains the jumpy questions of runs where nothing changed
(the baseline and its A/A re-run), from what the runs already saved. No API calls.

    python stability.py runs/<run_a> runs/<run_b> [--scenarios id ...]
    (calibrate_noise.py --from-runs prints the same explanations for its jumpy questions)

The evidence a system found (its tool_outputs: retrieved chunks, tool results) is the
reliable signal. The answer text is not: a generating system rewords every run, so even
an answer with the same content can look only 20% similar as text.

    SYSTEM   the runs found different evidence (or one crashed): the system itself
             behaved differently. Look at the system (its search, its tools).
    JUDGE    the same answer, word for word, got different scores: measurement noise.
             Don't trust single scores of this question.
    WORDING  the same evidence, the answer reworded, the scores differ: the system
             wrote it differently and/or the judges wobbled; the judges that moved say which.
    + borderline   a small move (under the jumpy range) that crossed the pass mark.

Systems that report no evidence get SYSTEM / JUDGE / WORDING from the answer alone.
"""

import argparse
import difflib
import os
import re
import sys

EVIDENCE_SAME = 0.5    # Jaccard overlap of the evidence sets; below this the runs found different material
ANSWER_SAME = 0.95     # text similarity of the answers; at or above this they are the same answer
JUMPY_RANGE = 2.0      # as in calibrate_noise.py
JUDGE_MOVED = 2.0      # a judge counts as "moved" when its scores span at least this


def _norm(text):
    return re.sub(r"\[[^\]]*\]", "", (text or "").lower()).strip()   # citation markers [1] [S2] vary freely


def answer_similarity(a, b):
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def evidence_overlap(a, b):
    """Jaccard overlap of two evidence lists, or None if neither run reported evidence."""
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def _ordered(text):
    return list(dict.fromkeys(re.findall(r"[a-z0-9][a-z0-9\-]*", (text or "").lower())))


def _words(text):
    return set(_ordered(text))


def _retrieval_steps(trace):
    return [s for s in (trace.get("trajectory") or []) if s.get("type") == "tool"]


def _judge_scores(trace):
    out = {}
    for key, val in trace.items():
        if key.endswith("_result") and isinstance(val, dict) and isinstance(val.get("score"), (int, float)) \
                and not val.get("na"):
            out[key[:-len("_result")]] = val["score"]
    return out


def explain_one(traces, threshold=7.0):
    """traces: the judged trace of ONE question from each run (2 or more). Returns a dict with
    kind (SYSTEM / JUDGE / WORDING), borderline, and the evidence for the explanation."""
    first = traces[0]
    scores = [t.get("final_score") for t in traces]
    scored = [s for s in scores if isinstance(s, (int, float))]
    spread = max(scored) - min(scored) if len(scored) >= 2 else None
    crossed = len({s >= threshold for s in scored}) > 1
    overlaps = [evidence_overlap(first.get("tool_outputs"), t.get("tool_outputs")) for t in traces[1:]]
    known = [o for o in overlaps if o is not None]
    evidence = min(known) if known else None
    answer = min(answer_similarity(first.get("answer"), t.get("answer")) for t in traces[1:])
    crashed = [i + 1 for i, t in enumerate(traces) if t.get("error")]

    per_judge = [_judge_scores(t) for t in traces]
    names = sorted(set().union(*per_judge))
    moved = {n: [p.get(n) for p in per_judge] for n in names
             if len([p[n] for p in per_judge if n in p]) >= 2
             and max(p[n] for p in per_judge if n in p) - min(p[n] for p in per_judge if n in p) >= JUDGE_MOVED}

    if crashed:
        kind, why = "SYSTEM", f"run {', '.join(map(str, crashed))} crashed"
    elif evidence is not None and evidence < EVIDENCE_SAME:
        kind, why = "SYSTEM", f"the runs found different evidence (overlap {evidence:.0%})"
    elif answer >= ANSWER_SAME:
        kind, why = "JUDGE", "the same answer got different scores"
    elif evidence is not None:
        kind, why = "WORDING", f"same evidence (overlap {evidence:.0%}), answer reworded"
    else:
        kind, why = "SYSTEM", f"different answers ({answer:.0%} similar), no evidence reported to compare"

    detail = []
    if kind == "SYSTEM" and not crashed:
        steps = [_retrieval_steps(t) for t in traces]
        for i, (a, b) in enumerate(zip(steps[0], steps[-1])):
            if a.get("args") != b.get("args") or a.get("output") != b.get("output"):
                qa, qb = (str((s.get("args") or {}).get("query") or s.get("args")) for s in (a, b))
                wa, wb = _words(qa), _words(qb)
                detail.append(f"step {i + 1} ({a.get('name', 'tool')}) searched differently:")
                if wa != wb:
                    detail.append(f"  only in run 1: {' '.join(w for w in _ordered(qa) if w in wa - wb) or '-'}")
                    detail.append(f"  only in run {len(traces)}: {' '.join(w for w in _ordered(qb) if w in wb - wa) or '-'}")
                for run, s in ((1, a), (len(traces), b)):
                    if s.get("note"):
                        detail.append(f"  run {run} found: {s['note'].split(';')[0]}")
                break
        if not detail and len(steps[0]) != len(steps[-1]):
            detail.append(f"{len(steps[0])} search steps in run 1, {len(steps[-1])} in run {len(traces)}")
    return {"kind": kind, "why": why, "borderline": bool(crossed and spread is not None and spread < JUMPY_RANGE),
            "scores": scores, "judges_moved": moved, "evidence_overlap": evidence,
            "answer_similarity": round(answer, 2), "detail": detail}


def explain(run_traces, ids, threshold=7.0):
    """run_traces: [{scenario_id: judged trace}] per run. Explains each id found in 2+ runs."""
    out = {}
    for sid in ids:
        traces = [rt[sid] for rt in run_traces if sid in rt]
        if len(traces) >= 2:
            out[sid] = explain_one(traces, threshold)
    return out


def print_explanations(expl):
    if not expl:
        return
    print("\nWhy these questions are unstable (from the saved runs, no API calls):")
    for sid, e in expl.items():
        scores = " / ".join(f"{s:.1f}" if isinstance(s, (int, float)) else "-" for s in e["scores"])
        tag = e["kind"] + (" + borderline" if e["borderline"] else "")
        print(f"  {sid:12} {tag:20} scores {scores}: {e['why']}")
        for line in e["detail"]:
            print(f"  {'':12}   {line}")
        if e["judges_moved"]:
            moved = ", ".join(f"{n} {' -> '.join('-' if v is None else f'{v:g}' for v in vals)}"
                              for n, vals in e["judges_moved"].items())
            print(f"  {'':12}   judges that moved: {moved}")
        if e["borderline"]:
            print(f"  {'':12}   a small move that crossed the pass mark")
    print("  SYSTEM = the system behaved differently (look at the system). JUDGE = measurement noise.\n"
          "  WORDING = same evidence, reworded answer: the judges that moved show where the scores differ.")


def load_run(run_dir):
    from trace_utils import load_latest_traces
    return load_latest_traces(os.path.join(run_dir, "judged"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Explain why questions are unstable across runs where nothing changed.")
    p.add_argument("runs", nargs="+", help="2 or more judged runs of the same config")
    p.add_argument("--scenarios", nargs="+", help="only these ids (default: those whose scores differ)")
    args = p.parse_args(argv)
    if len(args.runs) < 2:
        sys.exit("needs at least 2 judged runs")
    run_traces = [load_run(d.rstrip("/\\")) for d in args.runs]
    ids = args.scenarios or sorted(
        sid for sid in set().union(*run_traces)
        if len({round(rt[sid].get("final_score") or 0, 2) for rt in run_traces if sid in rt}) > 1)
    expl = explain(run_traces, ids)
    print_explanations(expl)
    return expl


if __name__ == "__main__":
    main()
