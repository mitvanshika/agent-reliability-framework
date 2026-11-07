"""
Judges one experiment run folder with the experiment judge set
(judge_experiment.py: 5 LLM judges incl. answer-key correctness and
context relevance, plus the tool_match code check), then prints and
saves a summary. The judge set used is saved in summary.json.

    python judge_run.py runs/<experiment>_<timestamp> [--yes] [--summary-only]

Judging goes through run_judges.judge_all_traces() (runs/<run>/traces ->
runs/<run>/judged): same crashed-trace handling, same latest-trace-per-
scenario logic, with judge_experiment's evaluator.

--summary-only recomputes summary.json from an already-judged run with no
API calls (e.g. after you add pricing.yaml).

Summary: what changed, pass rate (final_score >= 7, same rule as
detect_regressions.py: crashed / unjudged scenarios count as fails),
mean final_score, per-judge means, tokens, latency, errors by type and
estimated cost. Tokens and cost cover the system under test only, not
the judges.
"""

import argparse
import json
import os
import statistics
import sys

import judge_sets
from calibrate_noise import JUDGE_SLEEP_SECONDS, confirm
from detect_regressions import passed
from experiment_config import format_changes
from run_experiment import compute_totals, estimate_cost, load_pricing
from trace_utils import latest_trace_files, load_latest_traces

THRESHOLD = 7   # passed() default in detect_regressions.py


def load_meta(run_dir):
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found: is {run_dir} a folder created by run_experiment.py?")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _what_changed(meta):
    return format_changes({"parent": meta.get("parent"), "changes_vs_parent": meta.get("changes_vs_parent") or []})


def summarize(run_dir, meta, pricing):
    ids = meta.get("scenario_ids") or []
    run_traces = load_latest_traces(os.path.join(run_dir, "traces"))
    judged_dir = os.path.join(run_dir, "judged")
    judged = load_latest_traces(judged_dir) if os.path.isdir(judged_dir) else {}
    ids = ids or sorted(run_traces)

    scores = [judged[sid]["final_score"] for sid in ids
              if sid in judged and judged[sid].get("final_score") is not None]
    n_pass = sum(1 for sid in ids if sid in judged and passed(judged[sid], THRESHOLD))

    js = judge_sets.of_judged_traces(judged[sid] for sid in ids if sid in judged) if judged else None
    if js is None or js["name"] == "mixed":
        names = [n for n in judge_sets.ALL_NAMES if n not in judge_sets.EXPERIMENT["checks"]]
    else:
        names = js["llm_judges"]
    per_judge, per_judge_na = {}, {}
    for name in names:
        results = [judged[sid].get(f"{name}_result") for sid in ids if sid in judged]
        vals = [r["score"] for r in results if isinstance(r, dict) and not r.get("na") and "score" in r]
        nas = sum(1 for r in results if isinstance(r, dict) and r.get("na"))
        if js is not None and js["name"] == "mixed" and not vals and not nas:
            continue
        per_judge[name] = round(statistics.fmean(vals), 3) if vals else None
        if nas:
            per_judge_na[name] = nas
    tm = [judged[sid]["tool_match_result"]["score"] for sid in ids if sid in judged
          and isinstance(judged[sid].get("tool_match_result"), dict) and "score" in judged[sid]["tool_match_result"]]
    fallbacks = sorted(sid for sid in ids if sid in judged and isinstance(judged[sid].get("correctness_gt_result"), dict)
                       and judged[sid]["correctness_gt_result"].get("mode") == "fallback_no_answer_key")

    totals = compute_totals([run_traces[sid] for sid in ids if sid in run_traces])
    settings = (meta.get("resolved_config") or {}).get("settings") or {}
    cost, cost_note = estimate_cost(totals, settings, pricing)

    return {
        "experiment": meta.get("experiment"),
        "run_dir": run_dir,
        "parent": meta.get("parent"),
        "changes_vs_parent": meta.get("changes_vs_parent") or [],
        "what_changed": _what_changed(meta),
        "scenarios": len(ids),
        "threshold": THRESHOLD,
        "passed": n_pass,
        "pass_rate": round(n_pass / len(ids), 4) if ids else None,
        "scored": len(scores),
        "mean_final_score": round(statistics.fmean(scores), 3) if scores else None,
        "judge_set": js,
        "per_judge_mean": per_judge,
        "per_judge_na": per_judge_na,
        "tool_match_mean": round(statistics.fmean(tm), 3) if tm else None,
        "correctness_fallbacks": fallbacks,
        "not_judged": sorted(sid for sid in ids if sid not in judged),
        "judge_errors": sorted(sid for sid in ids if sid in judged and judged[sid].get("judge_error")),
        "agent_errors": totals["errors"],
        "errors_by_type": totals["errors_by_type"],
        "tokens": {k: totals[k] for k in ("input_tokens", "output_tokens", "total_tokens")},
        "scenarios_with_usage": totals["scenarios_with_usage"],
        "latency_mean_s": totals["latency_mean_s"],
        "latency_p95_s": totals["latency_p95_s"],
        "estimated_cost_usd": cost,
        "cost_note": cost_note,
    }


def _num(x, fmt="{:.2f}"):
    return fmt.format(x) if x is not None else "-"


def print_summary(s):
    print(f"\nExperiment:   {s['experiment']}   ({s['run_dir']})")
    print(f"What changed: {s['what_changed']}")
    rate = f"{s['pass_rate'] * 100:.0f}%" if s["pass_rate"] is not None else "-"
    print(f"Pass rate (final_score >= {s['threshold']}): {s['passed']}/{s['scenarios']} ({rate})"
          f"      Mean final_score: {_num(s['mean_final_score'])} ({s['scored']} scored)")
    if s.get("judge_set"):
        print(f"Judge set:    {judge_sets.describe(s['judge_set'])}")
    na = s.get("per_judge_na") or {}
    print("Per-judge mean: " + " | ".join(
        f"{k} {_num(v, '{:.1f}')}" + (f" (N/A {na[k]})" if k in na else "") for k, v in s["per_judge_mean"].items()))
    if s.get("tool_match_mean") is not None:
        print(f"tool_match (code check vs expected_tools, not in final_score): {s['tool_match_mean']:.1f}")
    if s.get("correctness_fallbacks"):
        print(f"No answer key, classic correctness used for: {' '.join(s['correctness_fallbacks'])}")
    errs = ", ".join(f"{k} {v}" for k, v in s["errors_by_type"].items())
    print(f"Agent errors: {s['agent_errors']} of {s['scenarios']}" + (f" ({errs})" if errs else ""))
    if s["judge_errors"]:
        print(f"Judge errors: {len(s['judge_errors'])} ({' '.join(s['judge_errors'])}): re-run judge_run.py to retry")
    if s["not_judged"]:
        print(f"Not judged:   {len(s['not_judged'])} ({' '.join(s['not_judged'])})")
    t = s["tokens"]
    if t["total_tokens"] is not None:
        print(f"Tokens (system under test, not judges): {t['input_tokens']:,} in / {t['output_tokens']:,} out / "
              f"{t['total_tokens']:,} total ({s['scenarios_with_usage']}/{s['scenarios']} scenarios reported usage)")
    else:
        print("Tokens:       not reported by this target")
    print(f"Latency:      mean {_num(s['latency_mean_s'], '{:.1f}s')}, p95 {_num(s['latency_p95_s'], '{:.1f}s')}")
    if s["estimated_cost_usd"] is not None:
        print(f"Est. cost:    ${s['estimated_cost_usd']:.4f} ({s['cost_note']})")
    else:
        print(f"Est. cost:    - ({s['cost_note']})")


def next_step_hint(run_dir, meta):
    import compare   # imported here because compare.py imports this module
    target = compare.run_target(meta)
    if target and os.path.isfile(compare.baseline_path(target, suite=meta.get("suite") or "main")):
        return f"Compare with: python compare.py {run_dir}"
    return f"No baseline yet: python promote_baseline.py {run_dir}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Judge one run_experiment.py run and summarize it.")
    p.add_argument("run_dir")
    p.add_argument("--yes", action="store_true", help="skip the API-call confirmation")
    p.add_argument("--summary-only", action="store_true", help="no judging, just recompute summary.json")
    args = p.parse_args(argv)

    run_dir = args.run_dir.rstrip("/\\")
    meta = load_meta(run_dir)
    traces_dir = os.path.join(run_dir, "traces")
    judged_dir = os.path.join(run_dir, "judged")

    if not args.summary_only:
        latest = latest_trace_files(traces_dir)
        judgeable = sum(1 for _, t in latest.values() if t.get("error") is None)
        print(f"Experiment: {meta.get('experiment')}")
        print(f"What changed: {_what_changed(meta)}")
        print(f"Traces: {len(latest)} ({len(latest) - judgeable} crashed: recorded as fails, no judge calls)")
        from run_experiment import load_scenarios
        scenario_dir = meta.get("scenario_dir")
        ids = set(meta.get("scenario_ids") or [])
        js = judge_sets.for_scenarios([s for s in load_scenarios(scenario_dir) if s.get("id") in ids])
        per_trace = len(js["llm_judges"])
        print(f"Judge set: {judge_sets.describe(js)}" + (f"  (suite: {meta.get('suite')})" if meta.get("suite") else ""))
        print(f"Estimated judge API calls: up to {judgeable * per_trace} ({per_trace} per answered trace; "
              f"fewer when a judge is N/A), plus retries. "
              f"Takes at least ~{round(len(latest) * JUDGE_SLEEP_SECONDS / 60, 1)} min.")
        if not confirm("Proceed?", args.yes):
            print("Aborted. No API calls were made.")
            return None
        import judge_experiment   # imported late: loads the judges + Groq client
        import compare
        import run_suite
        description = (run_suite.load_targets().get(compare.run_target(meta)) or {}).get("description")
        judge_experiment.judge_run_folder(meta.get("experiment") or "run", traces_dir, judged_dir,
                                          scenario_dir, description)

    summary = summarize(run_dir, meta, load_pricing())
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_summary(summary)
    print(f"\nSaved {os.path.join(run_dir, 'summary.json')}")
    print(next_step_hint(run_dir, meta))
    return summary


if __name__ == "__main__":
    main()
