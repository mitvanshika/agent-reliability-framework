"""
Re-grade an already-judged run with a newer judge set, re-running ONLY the
judges that changed (e.g. experiment_v1 -> experiment_v2 re-runs completeness
only). The other judges' scores are copied, final_score is recomputed.
No agent calls.

    python regrade.py runs/<run> --to experiment_v2 [--judge-errors full] [--yes]

- judged/ is first backed up to judged_<old set>/ (the old scores stay on disk)
- answers the old set judged OK: only the changed judges are called
- answers that had a judge error have no old scores to copy:
    --judge-errors keep  (default) they stay errors, marked with the new set
    --judge-errors full  judge them fully with the new set (all its judges)
- crashed answers (agent error) are copied, nothing to judge
- saves after every trace and skips traces already re-graded, so after a
  daily-limit stop the same command continues where it stopped
- recomputes summary.json and prints old vs new per scenario
"""

import argparse
import json
import os
import shutil
import sys

import judge_sets
import rate_limit
from calibrate_noise import EST_TOKENS_PER_JUDGE_CALL, confirm
from detect_regressions import passed
from trace_utils import latest_trace_files


def _load_meta(run_dir):
    with open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _write(path, trace):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


def plan(source, new_base, current_dir, errors_mode):
    """[(filename, trace, action)] with action in skip / copy / rejudge / full / mark."""
    old_js = judge_sets.of_judged_traces(t for _, t in source.values())
    changed = judge_sets.changed_judges(old_js, new_base)
    rows = []
    for sid, (fn, t) in sorted(source.items()):
        cur_path = os.path.join(current_dir, fn)
        if os.path.exists(cur_path):
            with open(cur_path, encoding="utf-8") as f:
                cur = json.load(f)
            if judge_sets.same(cur.get("judge_set"), {**new_base, "judge_model": (cur.get("judge_set") or {}).get(
                    "judge_model")}) and (cur.get("judge_error") is None or errors_mode == "keep"):
                rows.append((fn, t, "skip"))
                continue
        if t.get("error") is not None:
            rows.append((fn, t, "copy"))
        elif t.get("judge_error") or t.get("final_score") is None:
            rows.append((fn, t, "full" if errors_mode == "full" else "mark"))
        else:
            rows.append((fn, t, "rejudge"))
    return old_js, changed, rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-grade a judged run, re-running only the judges that changed.")
    p.add_argument("run_dir")
    p.add_argument("--to", required=True, help=f"judge set name: {', '.join(judge_sets.BY_NAME)}")
    p.add_argument("--judge-errors", choices=["keep", "full"], default="keep")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    run_dir = args.run_dir.rstrip("/\\")
    if args.to not in judge_sets.BY_NAME:
        sys.exit(f"Unknown judge set '{args.to}'. Known: {', '.join(judge_sets.BY_NAME)}")
    new_base = judge_sets.BY_NAME[args.to]
    judged = os.path.join(run_dir, "judged")
    if not os.path.isdir(judged):
        sys.exit(f"{run_dir} has not been judged yet.")

    first = latest_trace_files(judged)
    old_js_now = judge_sets.of_judged_traces(t for _, t in first.values())
    backup = None
    for name in judge_sets.BY_NAME:
        if name != args.to and os.path.isdir(os.path.join(run_dir, f"judged_{name}")):
            backup = os.path.join(run_dir, f"judged_{name}")
    if backup is None:
        if old_js_now["name"] == args.to:
            sys.exit(f"{run_dir} is already judged with {args.to}.")
        if old_js_now["name"] == "mixed":
            sys.exit(f"{run_dir}/judged mixes judge sets and has no backup to re-grade from.")
        backup = os.path.join(run_dir, f"judged_{old_js_now['name']}")
        shutil.copytree(judged, backup)
        print(f"Backed up {judged} -> {backup}")
    source = latest_trace_files(backup)
    old_js, changed, rows = plan(source, new_base, judged, args.judge_errors)
    if not changed:
        sys.exit(f"{judge_sets.describe(old_js)} and {args.to} run the same judges: nothing to re-grade.")

    calls = sum(len(changed) for *_, a in rows if a == "rejudge") + \
        sum(len(new_base["llm_judges"]) for *_, a in rows if a == "full")
    counts = {a: sum(1 for *_, x in rows if x == a) for a in ("rejudge", "full", "mark", "copy", "skip")}
    print(f"Run: {run_dir}\nFrom {judge_sets.describe(old_js)}\nTo   {args.to}: re-running only {', '.join(changed)}")
    print(f"Traces: {counts['rejudge']} re-grade, {counts['full']} judge fully (had judge errors), "
          f"{counts['mark']} judge errors kept, {counts['copy']} crashed (copied), {counts['skip']} already done")
    print(f"Estimated judge API calls: {calls} (~{calls * EST_TOKENS_PER_JUDGE_CALL:,} tokens), no agent calls. "
          f"Takes at least ~{rate_limit.min_minutes(calls * EST_TOKENS_PER_JUDGE_CALL):.0f} min.")
    if calls and not confirm("Proceed?", args.yes):
        print("Aborted. No API calls were made.")
        return None

    import judge_experiment   # imported late: loads the judges + Groq client
    import run_experiment
    import run_suite
    import compare
    meta = _load_meta(run_dir)
    import target_config
    global THRESHOLD
    THRESHOLD = target_config.for_meta(meta)["pass_threshold"]
    scenarios = {s["id"]: s for s in run_experiment.load_scenarios(meta.get("scenario_dir"))}
    description = (run_suite.load_targets().get(compare.run_target(meta)) or {}).get("description")
    new_js = judge_experiment.judge_set(new_base)
    stopped = None
    for fn, t, action in rows:
        if action == "skip":
            continue
        out = {**t, "judge_set": new_js}
        scenario = scenarios.get(t["scenario_id"], {})
        try:
            if action == "rejudge":
                print(f"Re-grading {fn} ({', '.join(changed)})...")
                out.update(judge_experiment.rejudge(t, scenario, new_base, changed, description))
                out.update(regraded_from=old_js["name"], regraded_judges=changed)
            elif action == "full":
                print(f"Judging {fn} fully (it had a judge error)...")
                out.update(judge_experiment.evaluate(t, scenario, new_base, description))
                out.update(judge_error=None, regraded_from=old_js["name"], regraded_judges="all")
        except Exception as e:
            if rate_limit.is_daily_limit_error(e):
                stopped = rate_limit.STATS.daily_limit_message or str(e)
                print(f"  DAILY API LIMIT reached: stopping; {fn} and later traces are not re-graded yet.")
                break
            print(f"  Judging failed: {e}")
            out.update(final_score=None, disagreement=None, judge_error=str(e))
        _write(os.path.join(judged, fn), out)

    print(rate_limit.summary())
    if stopped:
        sys.exit(f"Stopped at the daily API limit. Continue later with the same command:\n"
                 f"  python regrade.py {run_dir} --to {args.to} --judge-errors {args.judge_errors} --yes\n"
                 f"Provider said: {stopped}")

    import judge_run
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        summary = judge_run.main([run_dir, "--summary-only"])
    print_table(source, latest_trace_files(judged), changed, old_js["name"], args.to)
    print(f"\nPass rate {old_js['name']} -> {args.to}: "
          f"{_pass_rate(source)} -> {summary['passed']}/{summary['scenarios']}")
    return summary


THRESHOLD = 7   # replaced per run from the target's pass_threshold in main()


def _pass_rate(files):
    ok = sum(1 for _, t in files.values() if passed(t, THRESHOLD))
    return f"{ok}/{len(files)}"


def print_table(before, after, changed, old_name, new_name):
    def fmt(x):
        return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))

    def judge(t, n):
        r = t.get(f"{n}_result") or {}
        return r.get("score") if not r.get("na") else "N/A"

    cols = [f"{n} old->new" for n in changed]
    print(f"\n{'scenario':12} " + " ".join(f"{c:>22}" for c in cols) + f" {'final old->new':>16} {'pass old->new':>14}")
    for sid in sorted(after):
        b, a = before.get(sid, (None, {}))[1], after[sid][1]
        js = " ".join(f"{fmt(judge(b, n)) + ' -> ' + fmt(judge(a, n)):>22}" for n in changed)
        pb = passed(b, THRESHOLD) if b.get("final_score") is not None else None
        pa = passed(a, THRESHOLD) if a.get("final_score") is not None else None
        pf = {True: "pass", False: "FAIL", None: "-"}
        mark = "  <-- flip" if pb is not None and pa is not None and pb != pa else ""
        print(f"{sid:12} {js} {fmt(b.get('final_score')) + ' -> ' + fmt(a.get('final_score')):>16} "
              f"{pf[pb] + ' -> ' + pf[pa]:>14}{mark}")


if __name__ == "__main__":
    main()
