"""
Step-level confidence: judge every TOOL step of every answer in a run with the
step judge (judges/step.py, judge set steps_v1). Opt-in and separate: results
go to runs/<run>/judged_steps/, never into judged/ or final_score.

    python judge_steps.py runs/<run> [--yes]

Per answer: a score per tool step, the answer's step score (min and mean) and
its weakest step. Answers without tool steps are N/A. Steps come from the
recorded trajectory, else from the saved conversation.

One LLM call per tool step (the estimate is shown and confirmed first);
already-judged answers are skipped, and a daily-limit stop keeps what is done,
so the same command continues later.
"""

import argparse
import json
import os
import sys

import judge_sets
import rate_limit
import trajectory
from calibrate_noise import EST_TOKENS_PER_JUDGE_CALL, confirm
from trace_utils import latest_trace_files


def tool_steps(steps):
    return [(i, s) for i, s in enumerate(steps) if s.get("type") == "tool"]


def judge_trace_steps(trace, description=None, judge=None):
    """{"steps": [...], "step_score_min", "step_score_mean", "weakest_step", "na"}."""
    if judge is None:
        from judges.step import judge_step as judge
    steps, source = trajectory.of_trace(trace)
    tools = tool_steps(steps)
    if not tools:
        return {"na": True, "reason": "no tool steps", "steps": [], "source": source}
    scored = []
    for i, s in tools:
        v = judge(trace.get("question") or "", steps[:i], s, description)
        scored.append({"index": s.get("index", i), "name": s.get("name"), "args": s.get("args"),
                       "score": v.score, "confidence": v.confidence, "reason": v.reason})
    scores = [x["score"] for x in scored]
    weakest = min(scored, key=lambda x: x["score"])
    return {"na": False, "source": source, "steps": scored, "step_score_min": min(scores),
            "step_score_mean": round(sum(scores) / len(scores), 2),
            "weakest_step": {k: weakest[k] for k in ("index", "name", "args", "score", "reason")}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Judge every tool step of a run (steps_v1, opt-in).")
    p.add_argument("run_dir")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)
    run_dir = args.run_dir.rstrip("/\\")
    files = latest_trace_files(os.path.join(run_dir, "traces"))
    out_dir = os.path.join(run_dir, "judged_steps")
    os.makedirs(out_dir, exist_ok=True)
    js = {**judge_sets.STEPS_V1, "judge_model": _judge_model()}
    todo, skipped, calls = [], 0, 0
    for sid, (fn, t) in sorted(files.items()):
        path = os.path.join(out_dir, fn)
        if os.path.exists(path):
            old = json.load(open(path, encoding="utf-8"))
            if judge_sets.same(old.get("judge_set"), js) and not old.get("judge_error"):
                skipped += 1
                continue
        if t.get("error") is not None:
            continue
        n = len(tool_steps(trajectory.of_trace(t)[0]))
        calls += n
        todo.append((sid, fn, t))
    print(f"Run: {run_dir}\nJudge set: {judge_sets.describe(js)} (one call per tool step; not in final_score)")
    print(f"Answers to judge: {len(todo)} ({skipped} already done). Estimated judge API calls: {calls} "
          f"(~{calls * EST_TOKENS_PER_JUDGE_CALL:,} tokens), at least "
          f"~{rate_limit.min_minutes(calls * EST_TOKENS_PER_JUDGE_CALL):.0f} min.")
    if calls and not confirm("Proceed?", args.yes):
        print("Aborted. No API calls were made.")
        return None
    import run_suite
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
    target = (meta.get("resolved_config") or {}).get("target")
    description = ((run_suite.load_targets() or {}).get(target) or {}).get("description")
    for sid, fn, t in todo:
        print(f"Judging the steps of {sid}...")
        try:
            result = judge_trace_steps(t, description)
            record = {"scenario_id": sid, "judge_set": js, "judge_error": None, **result}
        except Exception as e:  # noqa: BLE001
            if rate_limit.is_daily_limit_error(e):
                print(rate_limit.summary())
                sys.exit(f"Stopped at the daily API limit; done so far is saved in {out_dir}.\n"
                         f"Continue later: python judge_steps.py {run_dir} --yes")
            record = {"scenario_id": sid, "judge_set": js, "judge_error": str(e)}
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    summary = summarize(out_dir)
    with open(os.path.join(run_dir, "steps_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(rate_limit.summary())
    print(f"Mean step score {summary['mean_step_score']}; weakest steps: " +
          ", ".join(f"{w['scenario_id']} step {w['index']} {w['name']} ({w['score']})" for w in summary["weakest"][:5]))
    return summary


def summarize(out_dir):
    rows = []
    for fn in sorted(os.listdir(out_dir)):
        r = json.load(open(os.path.join(out_dir, fn), encoding="utf-8"))
        if not r.get("na") and not r.get("judge_error") and r.get("steps"):
            rows.append(r)
    weakest = sorted(({"scenario_id": r["scenario_id"], **r["weakest_step"]} for r in rows), key=lambda w: w["score"])
    means = [r["step_score_mean"] for r in rows]
    return {"judge_set": judge_sets.STEPS_V1["name"], "answers": len(rows),
            "mean_step_score": round(sum(means) / len(means), 2) if means else None,
            "low_steps": sum(1 for r in rows for s in r["steps"] if s["score"] <= 4), "weakest": weakest}


def _judge_model():
    try:
        import judges.base
        return judges.base.llm.model_name
    except Exception:
        return None


if __name__ == "__main__":
    main()
