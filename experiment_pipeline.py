"""
One command for a whole experiment:
    python experiment_pipeline.py experiments/<name>.yaml [--scenarios id ...] [--yes] [--alert]

    1. run_experiment   run the scenarios          (asks once, unless --yes)
    2. judge_run        judge with the experiment judge set
    3. compare          vs baselines/<target>.json (skipped, with the promote command, if none yet)
    4. diagnose         causes for the down questions

Calls each tool's own main(), so behaviour is exactly the same as running them
by hand. Stops at the first failed step and prints how to resume from it.
Never promotes a baseline. Saves runs/<run>/pipeline.json.

Exit code: 0 done (or no baseline yet), 1 DECLINED, 2 a step failed / refused / not confirmed.

--summary-md FILE appends a Markdown report (CI: $GITHUB_STEP_SUMMARY).
"""

import argparse
import json
import os
import sys
import traceback

import compare
import diagnose
import judge_run
import run_experiment

STEPS = ["run", "judge", "compare", "diagnose"]


class StepFailed(Exception):
    def __init__(self, step, message, resume=None):
        super().__init__(message)
        self.step, self.message, self.resume = step, message, resume


def _call(step, fn, argv, resume=None):
    """Run one tool's main(); turn sys.exit(...) into a result or StepFailed."""
    try:
        return fn(argv), 0
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
        if step == "compare" and code == 1:          # DECLINED is a verdict, not a failure
            return None, 1
        msg = e.code if isinstance(e.code, str) else f"exited with code {code}"
        raise StepFailed(step, msg, resume) from None
    except Exception as e:
        traceback.print_exc()
        raise StepFailed(step, f"{type(e).__name__}: {e}", resume) from None


def run_pipeline(argv=None) -> int:
    p = argparse.ArgumentParser(description="run -> judge -> compare -> diagnose, in one command")
    p.add_argument("experiment")
    p.add_argument("--scenarios", nargs="+", metavar="ID")
    p.add_argument("--scenario-dir", metavar="DIR", help="passed to run_experiment.py (e.g. an adversarial suite)")
    p.add_argument("--yes", action="store_true", help="no confirmation prompt (needed in CI)")
    p.add_argument("--alert", action="store_true", help="passed to compare.py (Slack if DECLINED / anything down)")
    p.add_argument("--summary-md", metavar="FILE", help="append a Markdown report (CI: $GITHUB_STEP_SUMMARY)")
    args = p.parse_args(argv)
    record = {"experiment_file": args.experiment, "steps": {}}
    run_dir = None
    try:
        print(f"=== Step 1/4: run_experiment {args.experiment} ===")
        run_args = [args.experiment] + (["--scenarios", *args.scenarios] if args.scenarios else []) \
            + (["--scenario-dir", args.scenario_dir] if args.scenario_dir else []) + (["--yes"] if args.yes else [])
        run_dir, _ = _call("run", run_experiment.main, run_args)
        if run_dir is None:
            raise StepFailed("run", "not confirmed: nothing was run")
        record["run_dir"] = run_dir
        record["steps"]["run"] = "ok"

        print(f"\n=== Step 2/4: judge_run {run_dir} ===")
        resume = f"python judge_run.py {run_dir} --yes"
        summary, _ = _call("judge", judge_run.main, [run_dir, "--yes"], resume)
        answered = summary["scenarios"] - summary["agent_errors"]
        if answered and len(summary["judge_errors"]) >= answered:
            raise StepFailed("judge", "every answered scenario had a judge error (check the key / judge model)", resume)
        if summary["judge_errors"]:
            print(f"Warning: {len(summary['judge_errors'])} judge errors ({' '.join(summary['judge_errors'])}); "
                  "compare.py leaves those scenarios out.")
        record["steps"]["judge"] = "ok"
        record.update(pass_rate=summary["pass_rate"], mean_final_score=summary["mean_final_score"])

        run_meta = judge_run.load_meta(run_dir)
        target = compare.run_target(run_meta)
        if not os.path.isfile(compare.baseline_path(target, suite=run_meta.get("suite") or "main")):
            record["steps"]["compare"] = "skipped: no baseline"
            record["verdict"] = None
            print(f"\nNo baseline yet for target '{target}': stopping after judging (nothing is promoted automatically).")
            print(f"Promote this run if it should be the baseline: python promote_baseline.py {run_dir}")
            return _finish(record, run_dir, 0, summary_md=args.summary_md)

        print(f"\n=== Step 3/4: compare {run_dir} ===")
        out, code = _call("compare", compare.main, [run_dir] + (["--alert"] if args.alert else []),
                          f"python compare.py {run_dir}")
        if code == 1:   # DECLINED: compare exited before returning, so read what it saved
            with open(os.path.join(run_dir, "comparison.json"), encoding="utf-8") as f:
                out = json.load(f)
        record["steps"]["compare"] = "ok"
        record.update(verdict=out["verdict"], down=out["down"], up=out["up"])

        print(f"\n=== Step 4/4: diagnose {run_dir} ===")
        diag, _ = _call("diagnose", diagnose.main, [run_dir], f"python diagnose.py {run_dir}")
        record["steps"]["diagnose"] = "ok"
        record["causes"] = {d["scenario_id"]: d["cause"] for d in diag["diagnoses"]}
        return _finish(record, run_dir, 1 if out["verdict"] == "DECLINED" else 0, summary_md=args.summary_md)

    except StepFailed as e:
        n = STEPS.index(e.step) + 1
        record["steps"][e.step] = f"failed: {e.message}"
        record["resume"] = e.resume
        print(f"\nPipeline stopped at step {n}/4 ({e.step}): {e.message}", file=sys.stderr)
        if run_dir:
            print("The run itself is saved." + (f" Resume with: {e.resume}" if e.resume else ""), file=sys.stderr)
        return _finish(record, run_dir, 2, quiet=True, summary_md=args.summary_md)


def _fmt(x, fmt="{:.2f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


def write_summary_md(path, record, run_dir):
    """Markdown report: verdict, pass rate, down/up table, diagnosis, resource deltas."""
    cmp = None
    if run_dir and os.path.isfile(os.path.join(run_dir, "comparison.json")):
        with open(os.path.join(run_dir, "comparison.json"), encoding="utf-8") as f:
            cmp = json.load(f)
    verdict = record.get("verdict")
    icon = {"DECLINED": "❌", "IMPROVED": "✅", "IMPROVED WITH REGRESSIONS": "⚠️",
            "NO REAL DIFFERENCE": "✅", "NOT ENOUGH DATA": "⚠️"}.get(verdict, "ℹ️")
    lines = [f"## Experiment check: {icon} {verdict or 'not compared'}", "",
             f"- Experiment: `{record['experiment_file']}`" + (f", run `{run_dir}`" if run_dir else "")]
    failed = {s: v for s, v in record["steps"].items() if str(v).startswith("failed")}
    if failed:
        step, msg = next(iter(failed.items()))
        lines += [f"- **Pipeline stopped at step `{step}`**: {msg[len('failed: '):]}"]
        if record.get("resume"):
            lines += [f"- Resume with: `{record['resume']}`"]
    if record.get("pass_rate") is not None:
        lines += [f"- Pass rate: {record['pass_rate'] * 100:.0f}%, mean final_score {_fmt(record.get('mean_final_score'))}"]
    if record["steps"].get("compare", "").startswith("skipped"):
        lines += ["- No committed baseline for this target: promote a run and commit `baselines/<target>.json`."]
    if cmp:
        lines += [f"- Baseline: `{cmp.get('baseline_experiment')}` (`{cmp.get('baseline_run_dir')}`)",
                  f"- Config diff vs baseline: {'; '.join(cmp.get('config_diff') or []) or 'identical config'}",
                  f"- Noise bands: {cmp.get('noise_bands')}",
                  f"- {cmp['counts']['up']} up, {cmp['counts']['down']} down, {cmp['counts']['same']} same of "
                  f"{cmp['pairs']} paired; McNemar p={cmp['mcnemar']['p']:.4f}, Wilcoxon p={cmp['wilcoxon']['p']:.4f}"]
        if cmp.get("missing_in_run"):
            lines += [f"- Not run (call budget / subset): {len(cmp['missing_in_run'])} scenario(s)"]
        moved = [r for r in cmp["rows"] if r["direction"] != "same"]
        if moved:
            causes = {d["scenario_id"]: d["cause"] for d in cmp.get("diagnosis") or []}
            lines += ["", "| | scenario | baseline | run | delta | judges moved | likely cause |", "|---|---|---|---|---|---|---|"]
            for r in sorted(moved, key=lambda r: r["delta"]):
                arrow = "▼" if r["direction"] == "down" else "▲"
                lines.append(f"| {arrow} | {r['scenario_id']} | {r['baseline_score']:.1f} | {r['run_score']:.1f} | "
                             f"{r['delta']:+.1f} | {', '.join(r['judges_moved']) or '-'} | {causes.get(r['scenario_id'], '')} |")
        res = cmp.get("resources") or {}
        parts = [f"{label} {_fmt(res[k]['baseline'], fmt)} -> {_fmt(res[k]['run'], fmt)}"
                 for k, label, fmt in [("total_tokens", "tokens", "{:,.0f}"), ("latency_mean_s", "latency", "{:.2f}s"),
                                       ("agent_errors", "errors", "{:.0f}")] if k in res]
        if parts:
            lines += ["", "Resources: " + "; ".join(parts)]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def _finish(record, run_dir, code, quiet=False, summary_md=None):
    record["exit_code"] = code
    if run_dir:
        with open(os.path.join(run_dir, "pipeline.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    if summary_md:
        write_summary_md(summary_md, record, run_dir)
    if not quiet:
        print("\n" + "=" * 60)
        print(f"Experiment: {record['experiment_file']}   run: {run_dir}")
        if record.get("pass_rate") is not None:
            print(f"Pass rate: {record['pass_rate'] * 100:.0f}%   mean final_score: {record['mean_final_score']}")
        print(f"VERDICT: {record.get('verdict') or 'no baseline yet (not compared)'}")
        if record.get("down"):
            print(f"Down: {' '.join(record['down'])}")
            for sid, cause in record.get("causes", {}).items():
                print(f"  {sid:15} {cause}")
        print(f"Exit code {code} (0 ok / no baseline, 1 DECLINED, 2 step failed)")
    return code


if __name__ == "__main__":
    sys.exit(run_pipeline())
