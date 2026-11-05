"""
Noise calibration: how much do scores wobble when NOTHING changes?

Runs the same variant several times with no code change and measures
the spread of final_score per scenario. Later, a score drop only
counts as real if it is bigger than this natural wobble.

Two modes:
  full (default)  agent + judges re-run every time. Measures agent
                  noise AND judge noise together, which is what two
                  real pipeline runs differ by.
  --judge-only    ONE fixed set of traces re-judged N times. The
                  answers never change, so any spread is purely the
                  judges' own noise.

Works for any variant in targets.yaml (Track 1: run_suite + the classic
judges), or for any experiment file with --experiment (run_experiment +
the experiment judge set, i.e. the same judges compare.py compares).
Everything goes into noise_runs/<name>/<mode>_<timestamp>/run_k/{traces,judged},
so traces/, judged_traces/ and runs/ are never touched.

Saved to noise/<variant>_noise.json (detect_regressions.py), or for an
experiment noise/<target>_noise.json (compare.py), together with the
experiment's config_hash and judge_set: compare.py ignores the bands as
stale when they don't match the baseline it compares against.

Usage:
    python calibrate_noise.py <variant> [--runs 3] [--judge-only] ...
    python calibrate_noise.py --experiment experiments/baseline.yaml [--runs 3] [--judge-only] ...
        [--scenarios id1 id2 ...] [--from-traces DIR] [--yes]

    --from-traces  (judge-only) where to take the fixed traces from. Defaults
                   to the variant's traces folder, or for an experiment the
                   newest runs/<experiment>_*/traces.
    --yes          skip the "this will make N API calls" confirmation.
"""

import argparse
import json
import os
import shutil
import statistics
import sys
from datetime import datetime

import run_suite
from trace_utils import latest_trace_files, load_latest_traces

THRESHOLD = 7             # same pass rule as detect_regressions.passed()
JUMPY_RANGE = 2.0         # a scenario whose score spans >= this is "jumpy"
JUDGES_PER_TRACE = 4
EST_AGENT_CALLS = 3       # typical LLM calls per agent run (question -> tool -> answer)
JUDGE_SLEEP_SECONDS = 5   # run_judges.py pauses this long after each judged trace
NOISE_RUNS_DIR = "noise_runs"
NOISE_DIR = "noise"


# ---------------------------------------------------------------- statistics

def passed(score, threshold=THRESHOLD):
    # None (agent crashed / judge failed) counts as a fail, like detect_regressions.
    return score is not None and score >= threshold


def count_flips(pass_fail):
    """Number of times the pass/fail result changed between consecutive runs."""
    return sum(1 for a, b in zip(pass_fail, pass_fail[1:]) if a != b)


def scenario_stats(scores, threshold=THRESHOLD):
    """Stats for one scenario's final_score across runs (None = not scored)."""
    scored = [s for s in scores if s is not None]
    pf = [passed(s, threshold) for s in scores]
    st = {
        "scores": scores,
        "n_runs": len(scores),
        "n_scored": len(scored),
        "pass_fail": "".join("P" if p else "F" for p in pf),
        "flips": count_flips(pf),
        "mean": None, "std": None, "min": None, "max": None, "range": None,
    }
    if scored:
        st["mean"] = round(statistics.fmean(scored), 3)
        # Sample std (n-1): we are estimating the wobble from a few samples.
        st["std"] = round(statistics.stdev(scored), 3) if len(scored) > 1 else 0.0
        st["min"] = min(scored)
        st["max"] = max(scored)
        st["range"] = round(max(scored) - min(scored), 3)
    st["jumpy"] = st["flips"] > 0 or (st["range"] is not None and st["range"] >= JUMPY_RANGE)
    return st


def compute_noise(judged_dirs, threshold=THRESHOLD):
    """Per-scenario and suite-level noise from one judged folder per run."""
    runs = [load_latest_traces(d) for d in judged_dirs]
    ids = sorted(set().union(*runs))

    scenarios = {
        sid: scenario_stats([r.get(sid, {}).get("final_score") for r in runs], threshold)
        for sid in ids
    }

    per_run, raw_means, raw_rates = [], [], []   # ranges use unrounded values
    for i, r in enumerate(runs, 1):
        scored = [r[sid]["final_score"] for sid in ids
                  if sid in r and r[sid].get("final_score") is not None]
        n_pass = sum(1 for sid in ids if passed(r.get(sid, {}).get("final_score"), threshold))
        if scored:
            raw_means.append(statistics.fmean(scored))
        if ids:
            raw_rates.append(n_pass / len(ids))
        per_run.append({
            "run": i,
            "mean_score": round(statistics.fmean(scored), 3) if scored else None,
            "passed": n_pass,
            "total": len(ids),
            "pass_rate": round(n_pass / len(ids), 4) if ids else None,
        })

    return {
        "threshold": threshold,
        "jumpy_range": JUMPY_RANGE,
        "scenarios": scenarios,
        "suite": {
            "per_run": per_run,
            "mean_score_range": round(max(raw_means) - min(raw_means), 3) if raw_means else None,
            "pass_rate_range": round(max(raw_rates) - min(raw_rates), 4) if raw_rates else None,
        },
        "jumpy": [sid for sid, st in scenarios.items() if st["jumpy"]],
    }


# ---------------------------------------------------------------- output

def _fmt(x, width=6):
    return f"{x:{width}.2f}" if x is not None else f"{'-':>{width}}"


def print_report(report):
    mode = report.get("mode", "?")
    what = "agent + judge noise" if mode == "full" else "judge noise only"
    print(f"\nNoise calibration: {report.get('variant', '?')} ({mode} mode: {what}), "
          f"{report.get('runs', '?')} runs, pass threshold {report['threshold']}")
    print("-" * 78)
    print(f"{'scenario':15} {'mean':>6} {'std':>6} {'min':>6} {'max':>6} {'range':>6} "
          f"{'flips':>5}  P/F")
    for sid, st in report["scenarios"].items():
        mark = "  JUMPY" if st["jumpy"] else ""
        print(f"{sid:15} {_fmt(st['mean'])} {_fmt(st['std'])} {_fmt(st['min'])} {_fmt(st['max'])} "
              f"{_fmt(st['range'])} {st['flips']:>5}  {st['pass_fail']}{mark}")
    print("-" * 78)
    suite = report["suite"]
    for p in suite["per_run"]:
        rate = f"{p['pass_rate'] * 100:.0f}%" if p["pass_rate"] is not None else "-"
        print(f"run {p['run']}: mean score {_fmt(p['mean_score'], 5).strip()}, "
              f"passed {p['passed']}/{p['total']} ({rate})")
    if suite["mean_score_range"] is not None:
        print(f"suite mean-score range across runs: {suite['mean_score_range']:.2f}")
    if suite["pass_rate_range"] is not None:
        print(f"suite pass-rate range across runs:  {suite['pass_rate_range'] * 100:.1f} percentage points")
    if report["jumpy"]:
        print(f"\nJumpy scenarios (range >= {report['jumpy_range']} or any pass/fail flip): "
              f"{' '.join(report['jumpy'])}")
    else:
        print("\nNo jumpy scenarios.")


def save_report(report, variant, noise_dir=None):
    noise_dir = noise_dir or NOISE_DIR   # looked up at call time, not definition time
    os.makedirs(noise_dir, exist_ok=True)
    path = os.path.join(noise_dir, f"{variant}_noise.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


# ---------------------------------------------------------------- running

def estimate_calls(n_scenarios, runs, judge_only, n_judgeable=None, judges_per_trace=JUDGES_PER_TRACE):
    """Upper-bound estimate of API calls (retries come on top)."""
    if judge_only:
        agent = 0
        judge = (n_judgeable if n_judgeable is not None else n_scenarios) * runs * judges_per_trace
        traces_judged = (n_judgeable if n_judgeable is not None else n_scenarios) * runs
    else:
        agent = n_scenarios * runs * EST_AGENT_CALLS
        judge = n_scenarios * runs * judges_per_trace
        traces_judged = n_scenarios * runs
    return {"agent": agent, "judge": judge, "total": agent + judge,
            "min_minutes": round(traces_judged * JUDGE_SLEEP_SECONDS / 60, 1)}


def confirm(message, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Not running interactively; re-run with --yes to confirm.")
        return False
    try:
        return input(f"{message} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:   # no input available (e.g. stdin from NUL, which Windows reports as a tty)
        print()
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Measure score noise when nothing changes.")
    p.add_argument("variant", nargs="?", help="a variant name from targets.yaml (Track 1)")
    p.add_argument("--experiment", metavar="FILE", help="calibrate an experiment file instead (e.g. the baseline)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--judge-only", action="store_true",
                   help="re-judge one fixed set of traces N times (judge noise only)")
    p.add_argument("--scenarios", nargs="+", metavar="ID", help="only these scenario ids")
    p.add_argument("--from-traces", metavar="DIR",
                   help="judge-only: traces folder to re-judge (default: variant's normal traces folder)")
    p.add_argument("--yes", action="store_true", help="skip the API-call confirmation")
    return p.parse_args(argv)


def _select(all_ids, wanted):
    if not wanted:
        return list(all_ids)
    unknown = sorted(set(wanted) - set(all_ids))
    if unknown:
        sys.exit(f"Unknown scenario ids: {' '.join(unknown)}")
    return [sid for sid in all_ids if sid in set(wanted)]


def newest_run_traces(experiment_name, runs_dir="runs"):
    if not os.path.isdir(runs_dir):
        return None
    runs = sorted(d for d in os.listdir(runs_dir)
                  if d.startswith(f"{experiment_name}_") and os.path.isdir(os.path.join(runs_dir, d, "traces")))
    return os.path.join(runs_dir, runs[-1], "traces") if runs else None


def main_experiment(args):
    """--experiment: same idea as the variant mode, but through run_experiment + judge_experiment."""
    import run_experiment
    from experiment_config import ExperimentConfigError, config_hash, load_experiment
    import judge_sets

    try:
        exp = load_experiment(args.experiment)
    except ExperimentConfigError as e:
        sys.exit(f"Experiment error: {e}")
    fn, accepts, validate = run_experiment.resolve_target(exp["target"])
    if accepts and validate:
        try:
            validate(exp["settings"])
        except Exception as e:
            sys.exit(f"Settings error in {args.experiment}:\n{e}")

    scenarios = run_suite.load_scenarios()
    selected = _select([s["id"] for s in scenarios], args.scenarios)
    mode = "judge_only" if args.judge_only else "full"
    session_dir = os.path.join(NOISE_RUNS_DIR, exp["name"], f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    per_trace = len(judge_sets.EXPERIMENT["llm_judges"])

    if args.judge_only:
        src_dir = args.from_traces or newest_run_traces(exp["name"])
        if not src_dir or not os.path.isdir(src_dir):
            sys.exit(f"No traces to re-judge for '{exp['name']}'. Run run_experiment.py first or pass --from-traces DIR.")
        latest = latest_trace_files(src_dir)
        missing = [sid for sid in selected if sid not in latest]
        if missing:
            print(f"Warning: no trace in {src_dir} for: {' '.join(missing)} (skipped)")
        selected = [sid for sid in selected if sid in latest]
        if not selected:
            sys.exit("Nothing to judge.")
        n_judgeable = sum(1 for sid in selected if latest[sid][1].get("error") is None)
        est = estimate_calls(len(selected), args.runs, True, n_judgeable, per_trace)
        print(f"Judge-only: re-judging {len(selected)} fixed traces of '{exp['name']}' from {src_dir} {args.runs} times.")
    else:
        est = estimate_calls(len(selected), args.runs, False, judges_per_trace=per_trace)
        print(f"Full mode: running experiment '{exp['name']}' (agent + {per_trace} judges) for "
              f"{len(selected)} scenarios, {args.runs} times.")
    print(f"Estimated API calls: ~{est['total']} (agent ~{est['agent']}, judges up to {est['judge']}), "
          f"plus retries. Takes at least ~{est['min_minutes']} min.")
    print(f"Output: {session_dir}/ and {NOISE_DIR}/{exp['target']}_noise.json")
    if not confirm("Proceed?", args.yes):
        print("Aborted. No API calls were made.")
        return None

    import judge_experiment   # imported late: loads the judges + Groq client

    judged_dirs, used_js = [], None
    by_id = {s["id"]: s for s in scenarios}
    if args.judge_only:
        fixed = os.path.join(session_dir, "source_traces")
        os.makedirs(fixed, exist_ok=True)
        for sid in selected:
            shutil.copy2(os.path.join(src_dir, latest[sid][0]), fixed)
    for k in range(1, args.runs + 1):
        print(f"\n=== {mode} run {k}/{args.runs} ===")
        run_dir = os.path.join(session_dir, f"run_{k}")
        judged = os.path.join(run_dir, "judged")
        if args.judge_only:
            traces = fixed
        else:
            run_experiment.execute(exp, fn, accepts, [by_id[sid] for sid in selected], run_dir, args.experiment)
            traces = os.path.join(run_dir, "traces")
        used_js = judge_experiment.judge_run_folder(exp["name"], traces, judged)
        judged_dirs.append(judged)

    report = {"variant": exp["name"], "experiment": exp["name"], "experiment_file": args.experiment,
              "target": exp["target"], "config_hash": config_hash(exp),
              "resolved_config": {"target": exp["target"], "settings": exp["settings"]},
              "judge_set": used_js, "mode": mode, "runs": args.runs,
              "created": datetime.now().isoformat(timespec="seconds"),
              "session_dir": session_dir, "scenario_ids": selected}
    report.update(compute_noise(judged_dirs))
    path = save_report(report, exp["target"])
    print_report(report)
    print(f"\nSaved {path}")
    return report


def main(argv=None):
    args = parse_args(argv)
    if args.runs < 2:
        sys.exit("--runs must be at least 2 to measure any wobble.")
    if bool(args.variant) == bool(args.experiment):
        sys.exit("Give either a variant name (Track 1) or --experiment FILE, not both / neither.")
    if args.experiment:
        return main_experiment(args)

    targets = run_suite.load_targets()
    if args.variant not in targets:
        sys.exit(f"'{args.variant}' is not in {run_suite.TARGETS_FILE}. Found: {list(targets)}")

    all_ids = [s["id"] for s in run_suite.load_scenarios()]
    if args.scenarios:
        unknown = sorted(set(args.scenarios) - set(all_ids))
        if unknown:
            sys.exit(f"Unknown scenario ids: {' '.join(unknown)}")
        selected = [sid for sid in all_ids if sid in set(args.scenarios)]
    else:
        selected = all_ids

    mode = "judge_only" if args.judge_only else "full"
    session_dir = os.path.join(NOISE_RUNS_DIR, args.variant,
                               f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    source = None
    if args.judge_only:
        src_dir = args.from_traces or run_suite.TRACES_DIRS.get(args.variant)
        if not src_dir or not os.path.isdir(src_dir):
            sys.exit(f"No traces folder for '{args.variant}'"
                     f"{f' ({src_dir} not found)' if src_dir else ''}. "
                     "Run run_suite.py first or pass --from-traces DIR.")
        latest = latest_trace_files(src_dir)
        missing = [sid for sid in selected if sid not in latest]
        if missing:
            print(f"Warning: no trace in {src_dir} for: {' '.join(missing)} (skipped)")
        selected = [sid for sid in selected if sid in latest]
        if not selected:
            sys.exit("Nothing to judge.")
        source = (src_dir, {sid: latest[sid][0] for sid in selected})
        n_judgeable = sum(1 for sid in selected if latest[sid][1].get("error") is None)
        est = estimate_calls(len(selected), args.runs, True, n_judgeable)
        print(f"Judge-only: re-judging {len(selected)} fixed traces from {src_dir} "
              f"{args.runs} times ({len(selected) - n_judgeable} crashed traces cost nothing).")
    else:
        est = estimate_calls(len(selected), args.runs, False)
        print(f"Full mode: running agent + judges for {len(selected)} scenarios, {args.runs} times.")

    print(f"Estimated API calls: ~{est['total']} "
          f"(agent ~{est['agent']} at ~{EST_AGENT_CALLS}/scenario, judges {est['judge']}), "
          f"plus retries. Takes at least ~{est['min_minutes']} min.")
    print(f"Output: {session_dir}/ and {NOISE_DIR}/{args.variant}_noise.json")
    if not confirm("Proceed?", args.yes):
        print("Aborted. No API calls were made.")
        return None

    from run_judges import judge_all_traces   # imported late: loads the judges + Groq client

    judged_dirs = []
    if args.judge_only:
        src_dir, files = source
        fixed = os.path.join(session_dir, "source_traces")
        os.makedirs(fixed, exist_ok=True)
        for filename in files.values():
            shutil.copy2(os.path.join(src_dir, filename), fixed)
        for k in range(1, args.runs + 1):
            print(f"\n=== judge-only run {k}/{args.runs} ===")
            judged = os.path.join(session_dir, f"run_{k}", "judged")
            judge_all_traces(args.variant, traces_dir=fixed, judged_dir=judged)
            judged_dirs.append(judged)
    else:
        for k in range(1, args.runs + 1):
            print(f"\n=== full run {k}/{args.runs} ===")
            traces = os.path.join(session_dir, f"run_{k}", "traces")
            judged = os.path.join(session_dir, f"run_{k}", "judged")
            run_suite.run_suite(args.variant, traces_dir=traces, scenario_ids=selected)
            judge_all_traces(args.variant, traces_dir=traces, judged_dir=judged)
            judged_dirs.append(judged)

    report = {"variant": args.variant, "mode": mode, "runs": args.runs,
              "created": datetime.now().isoformat(timespec="seconds"),
              "session_dir": session_dir, "scenario_ids": selected}
    report.update(compute_noise(judged_dirs))
    path = save_report(report, args.variant)
    print_report(report)
    print(f"\nSaved {path}")
    return report


if __name__ == "__main__":
    main()
