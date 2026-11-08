"""
Plain-code checks on the steps of every answer in a run. No LLM, no API calls.

    python step_checks.py runs/<run> [runs/<run2> ...] [--max-steps N] [--slow-step S]

Uses trace["trajectory"] (recorded by trajectory.py) when present, else the
steps rebuilt from the saved conversation (trace["messages"]), so older runs
work too (without per-step timing).

Checks, per answer:
  repeated_call      the same tool with the same (normalized) arguments twice or more
  same_result_repeat the same tool returned exactly the same output as an earlier call
                     with DIFFERENT arguments: the extra calls added nothing (an agent
                     that rephrases a search and keeps getting the same page)
  empty_then_repeat  a tool returned nothing useful ("not found", "no match", ...) and
                     was then called again with the same arguments: retrying can't help
  too_many_steps     more than max_steps LLM turns or tool calls (targets.yaml
                     `max_steps`, default 8)
  tool_error         a step that raised
  slow_step          a step slower than slow_step_s (default 10 s); timing needed
Not flagged, only counted: "rephrased after an empty result" (the same tool
called again with other arguments after it found nothing), because that is
often the right thing to do. Per run it also lists the slowest step (recorded timing) or, without timing, the
slowest answers with an ESTIMATED time per LLM turn (latency / turns).

Saves runs/<run>/step_checks.json.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import trajectory
from trace_utils import load_latest_traces

DEFAULT_MAX_STEPS = 8
DEFAULT_SLOW_STEP_S = 10.0
EMPTY = re.compile(r"\b(no matching|not found|no results?|nothing found|no order found|no data|unknown)\b", re.I)


def _norm_args(args):
    import idempotent
    return json.dumps(idempotent.normalize(args or {}), sort_keys=True, default=str)


def check_steps(steps, max_steps=DEFAULT_MAX_STEPS, slow_step_s=DEFAULT_SLOW_STEP_S):
    """[finding] for one answer's steps. A finding: {"check", "detail", ...}."""
    findings = []
    tools = [s for s in steps if s.get("type") == "tool"]
    llms = [s for s in steps if s.get("type") == "llm"]
    calls = Counter((s.get("name"), _norm_args(s.get("args"))) for s in tools)
    for (name, args), n in sorted(calls.items()):
        if n >= 2:
            findings.append({"check": "repeated_call", "tool": name, "args": json.loads(args), "count": n,
                             "detail": f"{name}({args}) called {n}x with the same arguments"})
    outputs = {}
    for s in tools:
        outputs.setdefault((s.get("name"), str(s.get("output") or "")), []).append(s)
    for (name, out), group in outputs.items():
        distinct = {_norm_args(s.get("args")) for s in group}
        if len(group) >= 2 and len(distinct) >= 2:
            findings.append({"check": "same_result_repeat", "tool": name, "count": len(group),
                             "args": [s.get("args") for s in group],
                             "detail": f"{name} returned the same output for {len(distinct)} different arguments "
                                       f"({', '.join(json.dumps(s.get('args')) for s in group)}): "
                                       f"{len(group) - 1} call(s) added nothing"})
    seen_empty = {}
    for s in tools:
        key = (s.get("name"), _norm_args(s.get("args")))
        if key in seen_empty:
            findings.append({"check": "empty_then_repeat", "tool": s.get("name"), "step": s.get("index"),
                             "detail": f"{s.get('name')} returned nothing useful at step {seen_empty[key]} and was "
                                       f"called again with the same arguments at step {s.get('index')}"})
            del seen_empty[key]
        elif EMPTY.search(str(s.get("output") or "")):
            seen_empty[key] = s.get("index")
    if len(llms) > max_steps or len(tools) > max_steps:
        findings.append({"check": "too_many_steps", "llm_turns": len(llms), "tool_calls": len(tools),
                         "detail": f"{len(llms)} LLM turns, {len(tools)} tool calls (limit {max_steps})"})
    for s in steps:
        if s.get("error"):
            findings.append({"check": "tool_error" if s.get("type") == "tool" else "llm_error", "step": s.get("index"),
                             "detail": f"step {s.get('index')} ({s.get('type')} {s.get('name')}): {s['error'][:120]}"})
    timed = [s for s in steps if isinstance(s.get("latency_s"), (int, float))]
    for s in timed:
        if s["latency_s"] > slow_step_s:
            findings.append({"check": "slow_step", "step": s.get("index"), "latency_s": s["latency_s"],
                             "detail": f"step {s.get('index')} ({s.get('type')} {s.get('name')}) took "
                                       f"{s['latency_s']:.1f}s (limit {slow_step_s:g}s)"})
    return findings


def slowest(steps):
    timed = [s for s in steps if isinstance(s.get("latency_s"), (int, float))]
    if not timed:
        return None
    s = max(timed, key=lambda x: x["latency_s"])
    total = sum(x["latency_s"] for x in timed) or 1
    return {"step": s.get("index"), "type": s.get("type"), "name": s.get("name"), "latency_s": s["latency_s"],
            "share": round(s["latency_s"] / total, 3)}


def check_run(run_dir, max_steps=None, slow_step_s=DEFAULT_SLOW_STEP_S):
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.isfile(meta_path) else {}
    if max_steps is None:
        target = (meta.get("resolved_config") or {}).get("target")
        try:
            import run_suite
            max_steps = int(((run_suite.load_targets() or {}).get(target) or {}).get("max_steps", DEFAULT_MAX_STEPS))
        except Exception:
            max_steps = DEFAULT_MAX_STEPS
    traces = load_latest_traces(os.path.join(run_dir, "traces"))
    rows, sources = [], Counter()
    for sid in sorted(traces):
        t = traces[sid]
        steps, source = trajectory.of_trace(t)
        sources[source] += 1
        llm_turns = sum(1 for s in steps if s.get("type") == "llm")
        tool_steps = [s for s in steps if s.get("type") == "tool"]
        rephrased = any(EMPTY.search(str(s.get("output") or "")) and any(
            x.get("name") == s.get("name") and _norm_args(x.get("args")) != _norm_args(s.get("args"))
            for x in tool_steps[i + 1:]) for i, s in enumerate(tool_steps))
        rows.append({
            "scenario_id": sid, "source": source, "steps": len(steps), "llm_turns": llm_turns,
            "tool_calls": sum(1 for s in steps if s.get("type") == "tool"),
            "tools": [s.get("name") for s in steps if s.get("type") == "tool"],
            "latency_s": t.get("latency_s"),
            "est_s_per_llm_turn": round(t["latency_s"] / llm_turns, 2)
            if isinstance(t.get("latency_s"), (int, float)) and llm_turns else None,
            "slowest_step": slowest(steps),
            "rephrased_after_empty": rephrased,
            "findings": check_steps(steps, max_steps, slow_step_s) if steps else [],
        })
    counts = Counter(f["check"] for r in rows for f in r["findings"])
    report = {"run_dir": run_dir.replace("\\", "/"), "experiment": meta.get("experiment"), "max_steps": max_steps,
              "slow_step_s": slow_step_s, "scenarios": len(rows), "sources": dict(sources),
              "counts": dict(counts), "scenarios_with_findings": sum(1 for r in rows if r["findings"]),
              "mean_steps": round(sum(r["steps"] for r in rows) / len(rows), 2) if rows else None,
              "max_steps_seen": max((r["steps"] for r in rows), default=0),
              "rephrased_after_empty": sum(1 for r in rows if r["rephrased_after_empty"]), "rows": rows}
    with open(os.path.join(run_dir, "step_checks.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def print_report(rep):
    print(f"\n{rep['run_dir']}  ({rep['experiment']})")
    if rep["sources"].get("none") == rep["scenarios"]:
        print(f"  {rep['scenarios']} answers, but no steps saved (no trajectory, no messages): nothing to check")
        return
    src = ", ".join(f"{n} {k}" for k, n in rep["sources"].items())
    print(f"  {rep['scenarios']} answers (steps from: {src}); steps per answer: mean {rep['mean_steps']}, "
          f"max {rep['max_steps_seen']}; limit {rep['max_steps']}")
    if rep["counts"]:
        print("  findings: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["counts"].items()))
              + f"  ({rep['scenarios_with_findings']} answers)")
        for r in rep["rows"]:
            for f in r["findings"]:
                print(f"    {r['scenario_id']:<12} {f['check']:<18} {f['detail']}"[:160])
    else:
        print("  findings: none")
    if rep["rephrased_after_empty"]:
        print(f"  (not a finding) {rep['rephrased_after_empty']} answers rephrased a call after it found nothing")
    timed = [r for r in rep["rows"] if r["slowest_step"]]
    if timed:
        s = max(timed, key=lambda r: r["slowest_step"]["latency_s"])
        st = s["slowest_step"]
        print(f"  slowest step: {s['scenario_id']} step {st['step']} ({st['type']} {st['name']}) {st['latency_s']:.2f}s, "
              f"{st['share'] * 100:.0f}% of that answer's step time")
    else:
        top = sorted((r for r in rep["rows"] if isinstance(r["latency_s"], (int, float))),
                     key=lambda r: -r["latency_s"])[:3]
        if top:
            print("  no per-step timing in this run (recorded before trajectory.py). Slowest ANSWERS, with an "
                  "ESTIMATED time per LLM turn:")
            for r in top:
                print(f"    {r['scenario_id']:<12} {r['latency_s']:.1f}s, {r['llm_turns']} LLM turns, "
                      f"{r['tool_calls']} tool calls (~{r['est_s_per_llm_turn']}s per turn)")


def main(argv=None):
    p = argparse.ArgumentParser(description="Plain-code checks on every answer's steps (no LLM).")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--max-steps", type=int, help=f"LLM turns / tool calls limit (default: targets.yaml max_steps, "
                                                 f"else {DEFAULT_MAX_STEPS})")
    p.add_argument("--slow-step", type=float, default=DEFAULT_SLOW_STEP_S, help="seconds (default 10)")
    args = p.parse_args(argv)
    reports = []
    for d in args.run_dirs:
        d = d.rstrip("/\\")
        if not os.path.isdir(os.path.join(d, "traces")):
            print(f"{d}: no traces/ folder, skipped", file=sys.stderr)
            continue
        rep = check_run(d, args.max_steps, args.slow_step)
        print_report(rep)
        reports.append(rep)
    return reports


if __name__ == "__main__":
    main()
