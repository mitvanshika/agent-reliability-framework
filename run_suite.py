import yaml
import json
import os
from datetime import datetime
from test_agent import run_agent

SCENARIOS_DIR = "scenarios"
TRACES_DIR = "traces"

def load_scenarios():
    scenarios = []
    for filename in sorted(os.listdir(SCENARIOS_DIR)):
        if filename.endswith(".yaml"):
            with open(os.path.join(SCENARIOS_DIR, filename)) as f:
                scenarios.append(yaml.safe_load(f))
    return scenarios

def run_suite():
    os.makedirs(TRACES_DIR, exist_ok=True)
    scenarios = load_scenarios()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scenario in scenarios:
        print(f"Running {scenario['id']}...")
        try:
            answer = run_agent(scenario["question"])
            result = {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "category": scenario["category"],
                "answer": answer,
                "timestamp": timestamp,
                "error": None
            }
        except Exception as e:
            result = {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "category": scenario["category"],
                "answer": None,
                "timestamp": timestamp,
                "error": str(e)
            }

        output_path = os.path.join(TRACES_DIR, f"{scenario['id']}_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {output_path}")

if __name__ == "__main__":
    run_suite()