"""
Builds fake run_experiment.py run folders (run_meta.json, traces/,
judged/) with known numbers, for offline tests of judge_run / compare /
promote_baseline. No API calls.
"""

import json
import os

JUDGES = ["correctness", "tool_efficiency", "completeness", "groundedness"]
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def _dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def make_run(root, name, rows, target="configurable", settings=None, judged=True, timestamp="20261105_120000"):
    """rows: {scenario_id: score | dict}. A dict may set: score, judges (4 ints),
    error + error_type (agent crash), judge_error, not_judged, latency, usage,
    tools_used, tool_outputs."""
    run_dir = os.path.join(root, f"{name}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "traces"))
    if judged:
        os.makedirs(os.path.join(run_dir, "judged"))
    _dump(os.path.join(run_dir, "run_meta.json"), {
        "experiment": name, "description": f"fake {name}", "parent": None if name == "baseline" else "baseline",
        "changes_vs_parent": [] if name == "baseline" else [f"settings.prompt: prompts/v1.txt -> prompts/{name}.txt"],
        "resolved_config": {"target": target, "settings": settings or {"model": "m1", "prompt": f"prompts/{name}.txt"}},
        "git": {"commit": "f" * 40, "dirty": False, "changed_files": 0},
        "timestamp": timestamp, "scenario_ids": sorted(rows)})
    for sid, spec in rows.items():
        spec = spec if isinstance(spec, dict) else {"score": spec}
        crashed = spec.get("error") is not None
        trace = {"scenario_id": sid, "question": f"{sid}?", "answer": None if crashed else "answer",
                 "tools_used": spec.get("tools_used", []), "tool_outputs": spec.get("tool_outputs", []),
                 "retries_needed": 0, "timestamp": timestamp,
                 "error": spec.get("error"), "error_type": spec.get("error_type"), "experiment": name,
                 "latency_s": spec.get("latency", 1.0), "usage": None if crashed else spec.get("usage", USAGE)}
        _dump(os.path.join(run_dir, "traces", f"{sid}.json"), trace)
        if not judged or spec.get("not_judged"):
            continue
        j = dict(trace)
        if crashed:
            j.update(final_score=None, disagreement=None, low_confidence_judges=[], judge_error=None,
                     not_judged_reason="agent_error")
        elif spec.get("judge_error"):
            j.update(final_score=None, disagreement=None, judge_error=spec["judge_error"])
        else:
            score = spec["score"]
            judges = spec.get("judges") or [round(score)] * 4
            j.update(final_score=score, disagreement=False, low_confidence_judges=[], judge_error=None)
            for name_, s in zip(JUDGES, judges):
                j[f"{name_}_result"] = {"score": s, "confidence": 0.9, "reason": "x"}
        _dump(os.path.join(run_dir, "judged", f"{sid}.json"), j)
    return run_dir
