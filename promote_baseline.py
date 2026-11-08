"""
Promotes a judged experiment run to be the baseline for its target.

    python promote_baseline.py runs/<run> [--note "why"] [--yes] [--force]

Writes baselines/<target>.json (one per target; commit it to git so the
whole team compares against the same thing). It holds the run's config,
what changed vs its parent, git commit, date, your note, the run summary
and per-scenario final_score / passed / judge scores / error /
error_type / latency / tokens.

Always shows the run summary and how it compares with the current
baseline, then asks for confirmation. Never promotes automatically
(--yes is you confirming up front, e.g. in a script).

Refuses a run that is not fully judged, or has judge errors (a missing
score would silently become a hole in every later comparison), unless
--force. The previous baseline is backed up to baselines/history/.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import compare
import judge_run
import judge_sets
from calibrate_noise import confirm
from run_experiment import load_pricing


def history_dir():
    return os.path.join(compare.BASELINES_DIR, "history")


def build_baseline(run_dir, meta, records, summary, note):
    return {
        "target": compare.run_target(meta),
        "experiment": meta.get("experiment"),
        "description": meta.get("description"),
        "run_dir": run_dir.replace("\\", "/"),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "note": note or "",
        "run_timestamp": meta.get("timestamp"),
        "resolved_config": meta.get("resolved_config"),
        "parent": meta.get("parent"),
        "changes_vs_parent": meta.get("changes_vs_parent") or [],
        "git": meta.get("git"),
        "suite": meta.get("suite") or "main",
        "scenario_dir": meta.get("scenario_dir"),
        "judge_set": summary.get("judge_set"),
        "summary": summary,
        "scenarios": records,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Promote a judged run to the baseline for its target.")
    p.add_argument("run_dir")
    p.add_argument("--note", default="", help="why this is the new baseline")
    p.add_argument("--yes", action="store_true", help="confirm without the prompt")
    p.add_argument("--force", action="store_true", help="promote even with judge errors / unjudged scenarios")
    args = p.parse_args(argv)
    run_dir = args.run_dir.rstrip("/\\")

    meta, records = compare.load_run_records(run_dir)
    target = compare.run_target(meta)
    not_judged = sorted(s for s, r in records.items() if r["status"] == "not_judged")
    judge_errors = sorted(s for s, r in records.items() if r["status"] == "judge_error")
    if (not_judged or judge_errors) and not args.force:
        problems = []
        if not_judged:
            problems.append(f"{len(not_judged)} not judged ({' '.join(not_judged)})")
        if judge_errors:
            problems.append(f"{len(judge_errors)} judge errors ({' '.join(judge_errors)})")
        print(f"Refused: {run_dir} has {'; '.join(problems)}. Judge it first "
              f"(python judge_run.py {run_dir}) or pass --force.", file=sys.stderr)
        sys.exit(2)

    summary = judge_run.summarize(run_dir, meta, load_pricing())
    if summary.get("judge_set") and summary["judge_set"]["name"] == "mixed":
        print(f"Refused: {run_dir} was judged with a mix of judge sets. Re-judge it first "
              f"(python judge_run.py {run_dir}).", file=sys.stderr)
        sys.exit(2)
    print(f"Promote {meta.get('experiment')} ({run_dir}) as the baseline for target '{target}'")
    judge_run.print_summary(summary)
    if args.force and (not_judged or judge_errors):
        print(f"\n--force: {len(not_judged) + len(judge_errors)} scenarios without a score are stored as such.")

    suite = meta.get("suite") or "main"
    dest = compare.baseline_path(target, suite=suite)
    current = None
    if os.path.isfile(dest):
        with open(dest, encoding="utf-8") as f:
            current = json.load(f)
        print(f"\nCompared with the current baseline ({current.get('experiment')}, "
              f"promoted {current.get('promoted_at', '?')[:10]} from {current.get('run_dir')}):")
        cur_js = current.get("judge_set") or judge_sets.CLASSIC
        # only scored answers carry an answer_key_hash (judge errors / crashes are excluded anyway)
        changed_key = sorted(s for s in set(current["scenarios"]) & set(records)
                             if current["scenarios"][s].get("status") == "ok" and records[s]["status"] == "ok"
                             and current["scenarios"][s].get("answer_key_hash") != records[s].get("answer_key_hash"))
        if not judge_sets.same(cur_js, summary.get("judge_set")):
            print(f"  Not comparable: the current baseline was judged with {judge_sets.describe(cur_js)}, "
                  f"this run with {judge_sets.describe(summary.get('judge_set'))}.")
            print("  (Promoting a run judged with the new set is how you switch judge sets.)")
        elif changed_key:
            print(f"  Not comparable: the answer key changed for {' '.join(changed_key)}.")
        else:
            result = compare.compare_records(current["scenarios"], records,
                                             compare.load_bands(target, baseline=current, suite=suite)[0])
            compare.print_comparison(result, show_rows=False)
            print(f"Verdict vs current baseline: {result['verdict']}")
    else:
        print(f"\nNo current baseline for '{target}': this will be the first.")

    if not confirm(f"Promote to {dest}?", args.yes):
        print("Not promoted.")
        return None

    os.makedirs(compare.BASELINES_DIR, exist_ok=True)
    if current is not None:
        os.makedirs(history_dir(), exist_ok=True)
        stamp = (current.get("promoted_at") or "unknown").replace(":", "").replace("-", "")
        backup = os.path.join(history_dir(), f"{compare.suite_key(target, suite)}_{stamp}.json")
        n = 2
        while os.path.exists(backup):
            backup = os.path.join(history_dir(), f"{compare.suite_key(target, suite)}_{stamp}_{n}.json")
            n += 1
        shutil.copy2(dest, backup)
        print(f"Old baseline backed up to {backup}")

    baseline = build_baseline(run_dir, meta, records, summary, args.note)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    print(f"Promoted: {dest}  (commit it so everyone compares against the same baseline)")
    return baseline


if __name__ == "__main__":
    main()
