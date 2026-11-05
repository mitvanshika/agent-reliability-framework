"""
Shared helper for picking the latest trace per scenario.

run_suite.py writes traces/<scenario_id>_<timestamp>.json and never
deletes old runs, so a folder can hold several files for the same
scenario. Every script that reads traces or judged traces goes
through here, so they all agree on which run counts.
"""

import json
import os


def latest_trace_files(folder: str) -> dict[str, tuple[str, dict]]:
    """Return {scenario_id: (filename, trace)}, keeping only the newest
    file per scenario_id.

    "Newest" is decided by the trace's own "timestamp" field, not file
    mtime (copies and git checkouts reset mtime). The format is
    %Y%m%d_%H%M%S, so plain string comparison is chronological. A
    missing timestamp counts as oldest, and ties fall back to the
    filename, so the result never depends on os.listdir() order.

    A missing folder raises, same as os.listdir(), so callers can't
    mistake "no results" for "everything passed".
    """
    latest = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            trace = json.load(f)
        scenario_id = trace["scenario_id"]
        key = (trace.get("timestamp") or "", filename)
        current = latest.get(scenario_id)
        if current is None or key > (current[1].get("timestamp") or "", current[0]):
            latest[scenario_id] = (filename, trace)
    return latest


def load_latest_traces(folder: str) -> dict[str, dict]:
    """Same as latest_trace_files() without filenames: {scenario_id: trace}."""
    return {sid: trace for sid, (_, trace) in latest_trace_files(folder).items()}
