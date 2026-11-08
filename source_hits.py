"""
Did the system FIND the right source? A plain-code retrieval check, no LLM.

    python source_hits.py runs/<run> [runs/<run2> ...]

A scenario may set `expected_source`: a phrase (or a list of phrases) that
appears only in the right document / FAQ / record, e.g. "5-7 business days".
For each answer this checks whether any tool output of that answer contained
it (case-insensitive), and prints the hit rate per run. Scenarios with
`expected_source: none` must NOT have retrieved any of the other scenarios'
phrases (a wrong document was fetched).

Judges can disagree about "context relevance"; this cannot.
"""

import json
import os
import sys

import run_experiment
import trajectory
from trace_utils import load_latest_traces


def _phrases(v):
    if v is None or v == "none":
        return []
    return [v] if isinstance(v, str) else list(v)


def check_run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
    scenarios = {s["id"]: s for s in run_experiment.load_scenarios(meta.get("scenario_dir"))
                 if "expected_source" in s}
    all_phrases = {p.lower() for s in scenarios.values() for p in _phrases(s["expected_source"])}
    traces = load_latest_traces(os.path.join(run_dir, "traces"))
    rows = []
    for sid, s in sorted(scenarios.items()):
        t = traces.get(sid)
        if t is None:
            continue
        steps, _ = trajectory.of_trace(t)
        outputs = " \n".join(str(x.get("output") or "") for x in steps if x.get("type") == "tool").lower()
        outputs = outputs or " \n".join(map(str, t.get("tool_outputs") or [])).lower()
        want = [p.lower() for p in _phrases(s["expected_source"])]
        if want:
            hit = all(p in outputs for p in want)
            kind = "hit" if hit else "miss"
        else:
            wrong = sorted(p for p in all_phrases if p in outputs)
            hit, kind = not wrong, ("correctly nothing" if not wrong else f"WRONG source: {wrong[0]!r}")
        rows.append({"scenario_id": sid, "expected": s["expected_source"], "hit": hit, "result": kind,
                     "searches": sum(1 for x in steps if x.get("type") == "tool")})
    n_hit = sum(r["hit"] for r in rows)
    report = {"run_dir": run_dir.replace("\\", "/"), "experiment": meta.get("experiment"),
              "settings": (meta.get("resolved_config") or {}).get("settings"), "scenarios": len(rows),
              "hits": n_hit, "hit_rate": round(n_hit / len(rows), 3) if rows else None, "rows": rows}
    with open(os.path.join(run_dir, "source_hits.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def main(argv=None):
    reports = []
    for d in (argv if argv is not None else sys.argv[1:]):
        d = d.rstrip("/\\")
        r = check_run(d)
        reports.append(r)
        method = ((r["settings"] or {}).get("retrieval") or {}).get("method", "?")
        print(f"\n{r['run_dir']}  ({r['experiment']}, retrieval {method}): "
              f"{r['hits']}/{r['scenarios']} found the right source")
        for x in r["rows"]:
            print(f"  {'OK  ' if x['hit'] else 'MISS'}  {x['scenario_id']:12} {x['result']:<22} "
                  f"({x['searches']} tool calls)")
    return reports


if __name__ == "__main__":
    main()
