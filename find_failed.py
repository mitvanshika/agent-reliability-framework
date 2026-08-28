"""
Scans judged_traces/ and judged_traces_weak/ and prints exactly which
scenario_ids still have a judge_error (failed judging, likely from a
rate-limit hit) — so you can re-judge only those specific ones with
rejudge_specific.py instead of burning tokens re-running everything.

Usage:
    python find_failed.py
"""

import json
import os

VARIANTS = {
    "strong": "judged_traces",
    "weak": "judged_traces_weak",
}


def find_failed(folder: str) -> list[str]:
    failed = []
    if not os.path.isdir(folder):
        return failed
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            trace = json.load(f)
        if trace.get("judge_error") is not None or trace.get("final_score") is None:
            failed.append(trace["scenario_id"])
    return sorted(set(failed))


def main():
    for variant, folder in VARIANTS.items():
        failed = find_failed(folder)
        print(f"\n{variant} ({len(failed)} failed):")
        if failed:
            print("  " + " ".join(failed))
            print(f"\n  Re-judge with:\n  python rejudge_specific.py {variant} {' '.join(failed)}")
        else:
            print("  none — all judged successfully")


if __name__ == "__main__":
    main()