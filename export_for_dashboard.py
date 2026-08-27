"""
Bundles judged_traces/ and judged_traces_weak/ into a single JSON
snapshot file.

NOTE: this is no longer required to use the dashboard — app.py now
serves live data directly from /api/data, reading straight off disk.
This script is kept around only if you want to save a point-in-time
snapshot of results (e.g. to compare "results as of last Tuesday"
against current results, or to archive a run for a report).

Usage:
    python export_for_dashboard.py

Produces: dashboard_data.json
"""

import json
import os

STRONG_FOLDER = "judged_traces"
WEAK_FOLDER = "judged_traces_weak"
OUTPUT_FILE = "dashboard_data.json"


def load_judged_traces(folder: str) -> dict:
    results = {}
    if not os.path.isdir(folder):
        return results
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            trace = json.load(f)
        scenario_id = trace["scenario_id"]
        results[scenario_id] = trace
    return results


def main():
    strong = load_judged_traces(STRONG_FOLDER)
    weak = load_judged_traces(WEAK_FOLDER)

    bundle = {
        "strong": strong,
        "weak": weak,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Exported {len(strong)} strong traces and {len(weak)} weak traces to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()