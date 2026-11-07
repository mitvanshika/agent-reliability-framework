"""
Runs one experiment (experiments/<name>.yaml) over the scenario suite.

    python run_experiment.py experiments/<name>.yaml [--scenarios id1 id2 ...] [--yes]

Works for any target in targets.yaml. If the target has
accepts_settings: true it is called as fn(question, settings=...),
otherwise fn(question) exactly like run_suite.py. If the target's module
defines validate_settings(settings), it is called before anything else,
so a typo in the experiment file fails before any API call.

Per scenario it records the answer, tools, error/error_type, wall-clock
latency (works for any target, includes retry waits) and token usage if
the target returns one ("usage": {input_tokens, output_tokens,
total_tokens}); otherwise usage is null.

Output (judge it afterwards with judge_run.py):
    runs/<experiment>_<timestamp>/
        run_meta.json          what ran, git commit, totals
        traces/<scenario_id>.json

Optional pricing.yaml (you fill it in; prices change, nothing is
hard-coded) adds an estimated cost:
    openai/gpt-oss-120b: {input: 0.15, output: 0.60}   # USD per 1M tokens
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

import yaml

import run_suite
import judge_sets
import prompt_addons
import rate_limit
from calibrate_noise import EST_AGENT_CALLS, confirm
from experiment_config import ExperimentConfigError, format_changes, load_experiment

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = "runs"
PRICING_FILE = "pricing.yaml"
USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


# ---------------------------------------------------------------- target

def resolve_target(name: str):
    """(function, accepts_settings, validate_settings hook or None)."""
    targets = run_suite.load_targets()
    if name not in targets:
        raise ValueError(f"target '{name}' is not in {run_suite.TARGETS_FILE}. Found: {list(targets)}")
    fn = run_suite.resolve_target_function(name)
    accepts = targets[name].get("accepts_settings") is True
    hook = getattr(sys.modules[fn.__module__], "validate_settings", None)
    return fn, accepts, hook if callable(hook) else None


def _clean_usage(usage):
    if not isinstance(usage, dict):
        return None
    if not all(isinstance(usage.get(k), int) for k in ("input_tokens", "output_tokens")):
        return None
    total = usage.get("total_tokens")
    return {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "total_tokens": total if isinstance(total, int) else usage["input_tokens"] + usage["output_tokens"]}


def run_scenario(fn, accepts_settings, settings, scenario, timestamp, experiment):
    start = time.perf_counter()
    usage = messages = None
    try:
        raw = fn(scenario["question"], settings=settings) if accepts_settings else fn(scenario["question"])
        normalized = run_suite.normalize_result(raw)
        if isinstance(raw, dict):
            usage = _clean_usage(raw.get("usage"))
            messages = raw.get("messages") if isinstance(raw.get("messages"), list) else None
    except Exception as e:
        normalized = {"answer": None, "tools_used": [], "tool_outputs": [], "retries_needed": 0,
                      "error": str(e), "error_type": "crash"}
    latency = round(time.perf_counter() - start, 3)
    return {
        "scenario_id": scenario["id"],
        "question": scenario["question"],
        "category": scenario.get("category"),
        "answer": normalized["answer"],
        "tools_used": normalized["tools_used"],
        "tool_outputs": normalized["tool_outputs"],
        "retries_needed": normalized["retries_needed"],
        "timestamp": timestamp,
        "error": normalized["error"],
        "error_type": normalized["error_type"],
        "experiment": experiment,
        "latency_s": latency,
        "usage": usage,
        "messages": messages,   # optional full conversation (None if the target doesn't return one)
    }


# ---------------------------------------------------------------- totals / cost

def p95(values):
    """Nearest-rank 95th percentile (with few values this is close to the max)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def compute_totals(traces):
    with_usage = [t["usage"] for t in traces if t.get("usage")]
    latencies = [t["latency_s"] for t in traces if t.get("latency_s") is not None]
    errors = {}
    for t in traces:
        if t.get("error") is not None:
            kind = t.get("error_type") or "unknown"
            errors[kind] = errors.get(kind, 0) + 1
    return {
        "scenarios": len(traces),
        "scenarios_with_usage": len(with_usage),
        **{k: sum(u[k] for u in with_usage) if with_usage else None for k in USAGE_KEYS},
        "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else None,
        "latency_p95_s": p95(latencies),
        "errors": sum(errors.values()),
        "errors_by_type": dict(sorted(errors.items())),
    }


def load_pricing(path=None):
    path = path or PRICING_FILE
    full = path if os.path.isabs(path) else os.path.join(REPO_DIR, path)
    if not os.path.isfile(full):
        return None
    with open(full, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must map model names to {{input: <USD per 1M>, output: <USD per 1M>}}")
    return data


def estimate_cost(totals, settings, pricing):
    """(USD or None, note). Covers the system under test only, not the judges."""
    if pricing is None:
        return None, f"no {os.path.basename(PRICING_FILE)}: tokens only"
    model = settings.get("model") if isinstance(settings, dict) else None
    if not model:
        return None, "settings has no 'model' to look up in pricing.yaml"
    price = pricing.get(model)
    if not isinstance(price, dict) or not all(isinstance(price.get(k), (int, float)) for k in ("input", "output")):
        return None, f"no input/output price for '{model}' in {os.path.basename(PRICING_FILE)}"
    if totals.get("input_tokens") is None:
        return None, "the target reported no token usage"
    cost = (totals["input_tokens"] * price["input"] + totals["output_tokens"] * price["output"]) / 1_000_000
    note = "system under test only (not judges)"
    if totals["scenarios_with_usage"] < totals["scenarios"]:
        note += f"; {totals['scenarios'] - totals['scenarios_with_usage']} scenarios had no usage"
    return round(cost, 6), note


# ---------------------------------------------------------------- provenance

def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    try:
        changed = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(changed), "changed_files": len(changed)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_files": None}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def new_run_dir(name, timestamp):
    base = os.path.join(RUNS_DIR, f"{name}_{timestamp}")
    path, n = base, 2
    while os.path.exists(path):
        path, n = f"{base}_{n}", n + 1
    os.makedirs(os.path.join(path, "traces"))
    return path


def load_scenarios(scenario_dir=None):
    """Scenarios from scenario_dir (default: run_suite's scenarios/, exactly as before)."""
    if not scenario_dir or os.path.normpath(scenario_dir) == os.path.normpath(run_suite.SCENARIOS_DIR):
        return run_suite.load_scenarios()
    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"scenario folder not found: {scenario_dir}")
    out = []
    for filename in sorted(os.listdir(scenario_dir)):
        if filename.endswith(".yaml"):
            with open(os.path.join(scenario_dir, filename), encoding="utf-8") as f:
                out.append(yaml.safe_load(f))
    return out


def suite_name(scenario_dir=None, target=None) -> str:
    """'main' for the target's own scenario folder (targets.yaml `scenarios:`, default scenarios/),
    'adversarial' for scenarios_adversarial/<target>, else the folder name.
    Baselines and noise files are kept per suite, so suites never mix."""
    if not scenario_dir or os.path.normpath(scenario_dir) == os.path.normpath(run_suite.SCENARIOS_DIR):
        return "main"
    if target:
        import target_config
        if target_config.is_main_suite(target, scenario_dir):
            return "main"
    parts = os.path.normpath(scenario_dir).replace("\\", "/").split("/")
    if "scenarios_adversarial" in parts:
        return "adversarial"
    return parts[-1]


def execute(exp, fn, accepts, scenarios, run_dir, experiment_file, timestamp=None, scenario_dir=None):
    """Run every scenario into run_dir (traces/ + run_meta.json). Used by main()
    and by calibrate_noise.py --experiment, so there is one experiment runner."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(run_dir, "traces"), exist_ok=True)
    n = len(scenarios)
    meta = {
        "experiment": exp["name"],
        "description": exp["description"],
        "experiment_file": experiment_file,
        "parent": exp["parent"],
        "chain": exp["chain"],
        "changes_vs_parent": exp["changes_vs_parent"],
        "resolved_config": {"target": exp["target"], "settings": exp["settings"]},
        "target_function": f"{fn.__module__}.{fn.__name__}",
        "accepts_settings": accepts,
        "git": git_info(),
        "timestamp": timestamp,
        "python": platform.python_version(),
        "scenario_ids": [s["id"] for s in scenarios],
        "scenario_dir": (scenario_dir or run_suite.SCENARIOS_DIR).replace("\\", "/"),
        "suite": suite_name(scenario_dir, exp["target"]),
        "status": "running",
    }
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)

    traces = []
    stopped = None
    for i, scenario in enumerate(scenarios, 1):
        trace = run_scenario(fn, accepts, exp["settings"], scenario, timestamp, exp["name"])
        if trace.get("error_type") == "rate_limit_daily" or rate_limit.STATS.daily_limit:
            # Not saved: a quota stop is not an answer and must not be judged as a fail.
            stopped = {"after": i - 1, "not_run": [s["id"] for s in scenarios[i - 1:]],
                       "message": rate_limit.STATS.daily_limit_message or trace.get("error")}
            print(f"[{i}/{n}] {scenario['id']:15} DAILY API LIMIT reached: stopping. "
                  f"{i - 1} of {n} scenarios saved.")
            break
        _write_json(os.path.join(run_dir, "traces", f"{scenario['id']}.json"), trace)
        traces.append(trace)
        status = f"ERROR {trace['error_type']}" if trace["error"] is not None else "ok"
        tokens = f", {trace['usage']['total_tokens']} tokens" if trace["usage"] else ""
        print(f"[{i}/{n}] {scenario['id']:15} {trace['latency_s']:6.2f}s{tokens}  {status}")

    totals = compute_totals(traces)
    cost, cost_note = estimate_cost(totals, exp["settings"], load_pricing())
    totals["estimated_cost_usd"] = cost
    totals["cost_note"] = cost_note
    if stopped:
        meta.update(totals=totals, status="stopped_daily_limit", stopped=stopped)
    else:
        meta.update(totals=totals, status="complete")
    _write_json(os.path.join(run_dir, "run_meta.json"), meta)
    totals["stopped"] = stopped
    return totals


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run one experiment over the scenario suite.")
    p.add_argument("experiment", help="path to experiments/<name>.yaml")
    p.add_argument("--scenarios", nargs="+", metavar="ID", help="only these scenario ids")
    p.add_argument("--scenario-dir", metavar="DIR",
                   help="run another scenario folder, e.g. scenarios_adversarial/<target> (default: scenarios/)")
    p.add_argument("--yes", action="store_true", help="skip the API-call confirmation")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        exp = load_experiment(args.experiment)
    except ExperimentConfigError as e:
        sys.exit(f"Experiment error: {e}")
    try:
        fn, accepts, validate = resolve_target(exp["target"])
    except (ValueError, ImportError, AttributeError) as e:
        sys.exit(f"Target error: {e}")
    if accepts and validate:
        try:
            validate(exp["settings"])
        except Exception as e:
            sys.exit(f"Settings error in {args.experiment}:\n{e}")
    if accepts and exp["settings"].get("prompt_addons"):
        # framework feature, checked for every target before any API call
        try:
            prompt_addons.validate(exp["settings"]["prompt_addons"],
                                   (run_suite.load_targets().get(exp["target"]) or {}).get("description"))
        except ValueError as e:
            sys.exit(f"Settings error in {args.experiment}:\nsettings.prompt_addons: {e}")

    if not args.scenario_dir:
        import target_config
        args.scenario_dir = target_config.get(exp["target"])["scenarios"]   # default: scenarios/
    try:
        scenarios = load_scenarios(args.scenario_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not scenarios:
        sys.exit(f"No scenarios found in {args.scenario_dir or run_suite.SCENARIOS_DIR}")
    if args.scenarios:
        known = {s["id"] for s in scenarios}
        unknown = sorted(set(args.scenarios) - known)
        if unknown:
            sys.exit(f"Unknown scenario ids: {' '.join(unknown)}")
        scenarios = [s for s in scenarios if s["id"] in set(args.scenarios)]

    how = "settings passed" if accepts else "called with the question only"
    print(f"Experiment: {exp['name']}" + (f'  ("{exp["description"]}")' if exp["description"] else ""))
    print(f"Target:     {exp['target']} ({fn.__module__}.{fn.__name__}, {how})")
    if not accepts and exp["settings"]:
        print("Warning:    this target does not have accepts_settings: true, so settings are IGNORED.")
    if exp["parent"]:
        print(f"Changes vs {exp['parent']}:")
        for change in exp["changes_vs_parent"] or ["(none)"]:
            print(f"  {change}")
    else:
        print("Changes:    (baseline: no parent)")
    n = len(scenarios)
    print(f"Scenarios:  {n}")
    print(f"Estimated API calls: ~{n * EST_AGENT_CALLS} for the system under test "
          f"(~{EST_AGENT_CALLS} per scenario for the demo agent; yours may differ), plus retries.")
    print(f"            Judging afterwards (judge_run.py) adds up to {n * len(judge_sets.EXPERIMENT['llm_judges'])}.")
    if not confirm("Proceed?", args.yes):
        print("Aborted. No API calls were made.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = new_run_dir(exp["name"], timestamp)
    totals = execute(exp, fn, accepts, scenarios, run_dir, args.experiment, timestamp, args.scenario_dir)
    cost, cost_note = totals["estimated_cost_usd"], totals["cost_note"]

    tok = (f"{totals['total_tokens']:,} tokens ({totals['input_tokens']:,} in / {totals['output_tokens']:,} out)"
           if totals["total_tokens"] is not None else "no token usage reported")
    print(f"\nDone: {n} scenarios, {totals['errors']} errors {totals['errors_by_type'] or ''}".rstrip())
    print(f"Latency: mean {totals['latency_mean_s']}s, p95 {totals['latency_p95_s']}s | {tok}")
    if cost is not None:
        print(f"Estimated cost: ${cost:.6f} ({cost_note})")
    print(f"Saved {run_dir}")
    print(rate_limit.summary())
    if totals.get("stopped"):
        st = totals["stopped"]
        sys.exit(f"Stopped at the daily API limit after {st['after']} of {n} scenarios ({run_dir}).\n"
                 f"Not run: {' '.join(st['not_run'])}\nProvider said: {st['message']}")
    print(f"Next: python judge_run.py {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
