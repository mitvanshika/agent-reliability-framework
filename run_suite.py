"""
Runs the fixed scenario suite against either the strong or weak
agent. Replaces the old separate run_suite.py / run_suite_weak.py
files with one script that takes a variant argument.

Also fixes a latent bug from the old run_suite_weak.py: its except
block referenced result_data even in the case where run_agent()
itself raised before result_data was ever assigned, which would
have crashed with a NameError. This version's except block doesn't
depend on anything from the try block.

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

VARIANTS = {
    "strong": {"agent_module": "test_agent", "traces_dir": "traces"},
    "weak": {"agent_module": "agent_v2_weak", "traces_dir": "traces_weak"},
}


def load_scenarios():
    scenarios = []
    for filename in sorted(os.listdir(SCENARIOS_DIR)):
        if filename.endswith(".yaml"):
            with open(os.path.join(SCENARIOS_DIR, filename)) as f:
                scenarios.append(yaml.safe_load(f))
    return scenarios


def run_suite(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Use 'strong' or 'weak'.")
    config = VARIANTS[variant]
    agent_module = importlib.import_module(config["agent_module"])
    run_agent = agent_module.run_agent
    traces_dir = config["traces_dir"]

    os.makedirs(traces_dir, exist_ok=True)
    scenarios = load_scenarios()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scenario in scenarios:
        print(f"[{variant}] Running {scenario['id']}...")
        try:
            result_data = run_agent(scenario["question"])
            result = {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "category": scenario["category"],
                "answer": result_data["answer"],
                "tools_used": result_data["tools_used"],
                "tool_outputs": result_data.get("tool_outputs", []),
                "retries_needed": result_data.get("retries_needed", 0),
                "timestamp": timestamp,
                "error": result_data.get("error"),
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
            }

        output_path = os.path.join(traces_dir, f"{scenario['id']}_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "strong"
    run_suite(variant)