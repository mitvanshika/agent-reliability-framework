"""
Re-judge specific scenario IDs without re-running the full batch —
useful when a judge flaked on a couple of traces and you don't want
to burn time/rate-limit re-judging everything. Replaces the old
rejudge_specific.py + rejudge_specific_strong.py.

Usage:
    python rejudge_specific.py weak edge_05 normal_02 normal_05
    python rejudge_specific.py strong normal_05
"""

import sys
import json
import os
import time

from run_judges import evaluate_trace, VARIANTS
from trace_utils import latest_trace_files


def find_trace_file(traces_dir: str, scenario_id: str):
    # Newest run for this scenario (see trace_utils.py), matched on the
    # trace's scenario_id field rather than a filename prefix.
    match = latest_trace_files(traces_dir).get(scenario_id)
    if match is None:
        return None
    return os.path.join(traces_dir, match[0])


def rejudge_scenario(variant: str, scenario_id: str):
    config = VARIANTS[variant]
    trace_path = find_trace_file(config["traces_dir"], scenario_id)
    if trace_path is None:
        print(f"[{scenario_id}] No trace file found in {config['traces_dir']}/ — skipping.")
        return

    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)

    if trace.get("error") is not None:
        print(f"[{scenario_id}] Agent itself errored — nothing to judge, skipping.")
        return

    print(f"[{scenario_id}] Re-judging...")
    try:
        judge_result = evaluate_trace(
            trace["question"], trace["answer"],
            trace.get("tools_used", []), trace.get("tool_outputs", [])
        )
        trace["final_score"] = judge_result["final_score"]
        trace["disagreement"] = judge_result["disagreement"]
        trace["low_confidence_judges"] = judge_result["low_confidence_judges"]
        trace["correctness_result"] = judge_result["correctness_result"]
        trace["tool_efficiency_result"] = judge_result["tool_efficiency_result"]
        trace["completeness_result"] = judge_result["completeness_result"]
        trace["groundedness_result"] = judge_result["groundedness_result"]
        trace["judge_error"] = None
    except Exception as e:
        print(f"[{scenario_id}] Judging failed again: {e}")
        trace["final_score"] = None
        trace["disagreement"] = None
        trace["judge_error"] = str(e)

    final_score = trace.get("final_score")
    if final_score is None:
        print(f"[{scenario_id}] Still ERROR after re-judge.")
    else:
        print(f"[{scenario_id}] SUCCESS — final_score: {final_score:.1f}")

    out_path = os.path.join(config["judged_dir"], os.path.basename(trace_path))
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"[{scenario_id}] Saved to {out_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python rejudge_specific.py <strong|weak> <scenario_id> [scenario_id ...]")
        return
    variant = sys.argv[1]
    if variant not in VARIANTS:
        print(f"Unknown variant '{variant}'. Use 'strong' or 'weak'.")
        return
    scenario_ids = sys.argv[2:]
    for i, scenario_id in enumerate(scenario_ids):
        rejudge_scenario(variant, scenario_id)
        if i < len(scenario_ids) - 1:
            time.sleep(5)


if __name__ == "__main__":
    main()