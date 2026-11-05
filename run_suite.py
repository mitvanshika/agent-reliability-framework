"""
Runs the fixed scenario suite against either the "strong" or "weak"
target defined in targets.yaml — which can point at ANY Python
function, not just the built-in support-bot agents. See targets.yaml
for how to point this at your own agent/feature.

Usage:
    python run_suite.py strong
    python run_suite.py weak
"""

import sys
import importlib
import yaml
import json
import os
from datetime import datetime

SCENARIOS_DIR = "scenarios"
TARGETS_FILE = "targets.yaml"

TRACES_DIRS = {
    "strong": "traces",
    "weak": "traces_weak",
}


def load_targets() -> dict:
    with open(TARGETS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scenarios():
    scenarios = []
    for filename in sorted(os.listdir(SCENARIOS_DIR)):
        if filename.endswith(".yaml"):
            with open(os.path.join(SCENARIOS_DIR, filename), encoding="utf-8") as f:
                scenarios.append(yaml.safe_load(f))
    return scenarios


def resolve_target_function(variant: str):
    targets = load_targets()
    if variant not in targets:
        raise ValueError(f"'{variant}' is not defined in {TARGETS_FILE}. Found: {list(targets.keys())}")
    config = targets[variant]
    module = importlib.import_module(config["module"])
    return getattr(module, config["function"])


def normalize_result(raw_result) -> dict:
    """
    Accepts whatever shape the target function returned and fills in
    defaults for anything missing, so run_suite.py works whether the
    target is a full LangGraph agent or a one-line function that just
    returns a string.
    """
    if isinstance(raw_result, str):
        return {
            "answer": raw_result,
            "tools_used": [],
            "tool_outputs": [],
            "error": None,
            "error_type": None,
            "retries_needed": 0,
        }
    if isinstance(raw_result, dict):
        return {
            "answer": raw_result.get("answer"),
            "tools_used": raw_result.get("tools_used", []),
            "tool_outputs": raw_result.get("tool_outputs", []),
            "error": raw_result.get("error"),
            "error_type": raw_result.get("error_type"),
            "retries_needed": raw_result.get("retries_needed", 0),
        }
    raise TypeError(
        f"Target function returned {type(raw_result)}, expected a string or a dict. "
        f"See targets.yaml for supported return shapes."
    )


def run_suite(variant: str, traces_dir: str | None = None, scenario_ids: list[str] | None = None):
    # traces_dir / scenario_ids are optional (used by calibrate_noise.py).
    # Left as None, behaviour is exactly as before.
    if traces_dir is None:
        if variant not in TRACES_DIRS:
            raise ValueError(f"Unknown variant '{variant}'. Use 'strong' or 'weak'.")
        traces_dir = TRACES_DIRS[variant]
    run_target = resolve_target_function(variant)

    os.makedirs(traces_dir, exist_ok=True)
    scenarios = load_scenarios()
    if scenario_ids is not None:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s["id"] in wanted]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scenario in scenarios:
        print(f"[{variant}] Running {scenario['id']}...")
        try:
            raw_result = run_target(scenario["question"])
            normalized = normalize_result(raw_result)
            result = {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "category": scenario["category"],
                "answer": normalized["answer"],
                "tools_used": normalized["tools_used"],
                "tool_outputs": normalized["tool_outputs"],
                "retries_needed": normalized["retries_needed"],
                "timestamp": timestamp,
                "error": normalized["error"],
                "error_type": normalized["error_type"],
            }
        except Exception as e:
            result = {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "category": scenario["category"],
                "answer": None,
                "tools_used": [],
                "tool_outputs": [],
                "retries_needed": 0,
                "timestamp": timestamp,
                "error": str(e),
                "error_type": "crash",
            }

        output_path = os.path.join(traces_dir, f"{scenario['id']}_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "strong"
    run_suite(variant)