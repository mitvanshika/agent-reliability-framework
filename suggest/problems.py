"""
Collect the problems a run already found, so the fix suggester can look at each one.
No LLM calls: everything here is read from the run's saved files.

    problems(run_dir) -> (problems, skipped)

A problem is one thing worth fixing, with the evidence the framework already has:
  crash       the system raised an error on a question (trace["error"]); grouped by error type
  llm_error   one LLM call inside the system failed (a step with "error"); grouped by call + error
  tool_error  one tool call failed; grouped by tool + error
  drop        a question got worse than the baseline and diagnose.py named a fixable cause
              (PROMPT, RAG, TOOL DESCRIPTIONS, TOOLS/PROMPT, SAFETY, AGENT CRASH)

Skipped, with the reason (shown to the user, never sent to a model):
  drops diagnosed INFRA (rate limit/timeout: re-run), CONFIG (key/model), MEASUREMENT UNSURE
  and UNCLEAR (no single cause to fix), and anything inside the noise band.
A drop on a question that also moved in an A/A test (same config, run again) is kept but
marked: it may be noise, so the suggester is told to say so.
"""

import json
import os
from collections import OrderedDict

from trace_utils import load_latest_traces

FIXABLE_CAUSES = ("AGENT CRASH", "TOOLS/PROMPT", "RAG", "RAG (the change being tested)", "MODEL (the change being tested)",
                  "PROMPT (used info badly)", "PROMPT (missed part)", "TOOL DESCRIPTIONS", "SAFETY")
SKIP_REASONS = {
    "INFRA": "a rate limit or timeout, not the system's code: re-run those questions",
    "CONFIG": "a key or model setting: check .env and settings.model",
    "MEASUREMENT UNSURE": "the judges disagreed or weren't confident: read the answer by hand first",
    "UNCLEAR": "no single cause stands out: read the run and baseline answers side by side first",
}
AA_MOVED = 0.5   # same rule as the guided tour: a question "moved" in an A/A test if |delta| >= this


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _traces(run_dir, sub="traces"):
    folder = os.path.join(run_dir, sub)
    return load_latest_traces(folder) if os.path.isdir(folder) else {}


def _short(text, n=300):
    text = str(text or "").strip()
    return text if len(text) <= n else text[:n] + "..."


def aa_moved(runs_dir, target):
    """Questions that moved >= AA_MOVED in an A/A test of this target (a run compared with its
    baseline with nothing changed). Changes on these may be chance."""
    import overview
    moved = set()
    if not os.path.isdir(runs_dir):
        return moved
    for rid in os.listdir(runs_dir):
        comp = _read_json(os.path.join(runs_dir, rid, "comparison.json"))
        if not comp or comp.get("target") != target or "config_diff" not in comp:
            continue
        if overview.real_changes(comp.get("config_diff")):
            continue
        moved |= {r["scenario_id"] for r in comp.get("rows") or [] if abs(r.get("delta") or 0) >= AA_MOVED}
    return moved


def _steps(trace):
    import trajectory_read
    return trajectory_read.of_trace(trace)[0]


def _judge_reasons(judged):
    out = {}
    for k, v in (judged or {}).items():
        if k.endswith("_result") and isinstance(v, dict) and v.get("score") is not None and not v.get("na"):
            out[k[:-7]] = {"score": v["score"], "reason": _short(v.get("reason"), 300)}
    return out


def problems(run_dir, runs_dir=None):
    """(problems, skipped) for one run. Each problem has a stable id, so a re-run skips done ones."""
    runs_dir = runs_dir or os.path.dirname(os.path.normpath(run_dir)) or "runs"
    meta = _read_json(os.path.join(run_dir, "run_meta.json")) or {}
    comp = _read_json(os.path.join(run_dir, "comparison.json")) or {}
    target = (meta.get("resolved_config") or {}).get("target") or comp.get("target")
    traces = _traces(run_dir)
    judged = _traces(run_dir, "judged")
    found, skipped = OrderedDict(), []

    def add(pid, **p):
        if pid in found:
            found[pid]["scenario_ids"].append(p["scenario_ids"][0])
        else:
            found[pid] = {"id": pid, "target": target, "run": os.path.basename(os.path.normpath(run_dir)), **p}

    # 1. crashes: the whole question failed
    for sid, t in sorted(traces.items()):
        if t.get("error"):
            etype = t.get("error_type") or "crash"
            if etype in ("rate_limit", "timeout", "auth"):
                skipped.append({"scenario_id": sid, "kind": "crash", "why": f"{etype}: " + (
                    SKIP_REASONS["CONFIG"] if etype == "auth" else SKIP_REASONS["INFRA"])})
                continue
            add(f"crash:{etype}", kind="crash", scenario_ids=[sid], title=f"The system crashed ({etype})",
                evidence={"error_type": etype, "error": _short(t.get("error"), 1500),
                          "question": t.get("question"), "last_steps": _step_lines(_steps(t))[-6:]})

    # 2. failed steps inside an answer (the system recovered or not, but a call failed)
    for sid, t in sorted(traces.items()):
        for s in _steps(t):
            if not s.get("error"):
                continue
            kind = "tool_error" if s.get("type") == "tool" else "llm_error"
            name = s.get("label") or s.get("name") or "?"
            err = str(s["error"]).split(":")[0][:60]
            add(f"{kind}:{name}:{err}", kind=kind, scenario_ids=[sid],
                title=f"{'Tool' if kind == 'tool_error' else 'LLM call'} '{name}' failed with {err}",
                evidence={"step": s.get("index"), "call": name, "model": s.get("name"), "error": _short(s["error"], 800),
                          "question": t.get("question"), "steps_around": _around(_steps(t), s.get("index"))})

    # 3. drops diagnose.py explained
    moved_in_aa = aa_moved(runs_dir, target) if target else set()
    rows = {r["scenario_id"]: r for r in comp.get("rows") or []}
    base_dir = comp.get("baseline_run_dir")
    base_dir = os.path.join(runs_dir, os.path.basename(base_dir.replace("\\", "/").rstrip("/"))) if base_dir else None
    base_judged = _traces(base_dir, "judged") if base_dir else {}
    for d in _diagnosis(comp):
        sid, cause = d.get("scenario_id"), d.get("cause") or ""
        if cause not in FIXABLE_CAUSES:
            skipped.append({"scenario_id": sid, "kind": "drop", "cause": cause,
                            "why": SKIP_REASONS.get(cause.split(" (")[0], "not a cause code can fix")})
            continue
        r, run_t, base_t = rows.get(sid) or {}, judged.get(sid) or traces.get(sid) or {}, base_judged.get(sid) or {}
        only_base, only_run = _diff_items(base_t.get("tool_outputs"), run_t.get("tool_outputs"))
        ids_base, ids_run = _diff_items(_found_ids(base_t), _found_ids(run_t))
        add(f"drop:{sid}", kind="drop", scenario_ids=[sid], cause=cause,
            title=f"{sid} got worse: {cause}", maybe_noise=sid in moved_in_aa,
            evidence={"question": run_t.get("question") or base_t.get("question"),
                      "answer_key": _answer_key(target, sid),
                      "found_only_in_baseline": {"ids": ids_base[:10], "passages": [_short(x, 400) for x in only_base[:3]]},
                      "found_only_in_run": {"ids": ids_run[:10], "passages": [_short(x, 400) for x in only_run[:2]]},
                      "what_changed": comp.get("config_diff") or [],
                      "score": {"baseline": r.get("baseline_score"), "run": r.get("run_score"), "band": r.get("band")},
                      "diagnosis": d.get("evidence"), "framework_hint": d.get("suggestion"),
                      "baseline_answer": _short(base_t.get("answer"), 800), "run_answer": _short(run_t.get("answer"), 800),
                      "baseline_judges": _judge_reasons(base_t), "run_judges": _judge_reasons(run_t),
                      "baseline_steps": _step_lines(_steps(base_t))[:8], "run_steps": _step_lines(_steps(run_t))[:8]})
    return list(found.values()), skipped


def _diagnosis(comp):
    """The causes as diagnose.py gives them NOW (it reads what the experiment changed), so older runs get
    the current rules without a re-run; the saved diagnosis if that isn't possible."""
    if any("judge_deltas" in r for r in comp.get("rows") or []):
        try:
            import diagnose
            return diagnose.diagnose(comp)
        except Exception:  # noqa: BLE001
            pass
    return comp.get("diagnosis") or []   # an older comparison.json without the data diagnose needs


def _answer_key(target, sid):
    """The question's answer key, from its scenario file (the judges' reference, never shown to the system)."""
    try:
        import target_config
        import tour
        return (tour._scenarios(".", target_config.get(target)["scenarios"]).get(sid) or {}).get("expected_answer")
    except Exception:  # noqa: BLE001
        return None


def _found_ids(trace):
    """Ids of what each search found, from step outputs like 'chunks: A-0171, A-0223' (any system's own ids)."""
    out = []
    for s in _steps(trace):
        text = str(s.get("output") or "") if s.get("type") == "tool" else ""
        if ":" in text.split(",")[0]:
            text = text.split(":", 1)[1]
        out += [x.strip() for x in text.replace("\n", ",").split(",") if x.strip() and len(x.strip()) < 80]
    return out


def _diff_items(base, run):
    """(only in base, only in run), order kept: what one side found that the other didn't."""
    base, run = [str(x) for x in base or []], [str(x) for x in run or []]
    return [x for x in base if x not in run], [x for x in run if x not in base]


def _step_lines(steps):
    out = []
    for s in steps or []:
        if s.get("type") == "tool":
            line = f"#{s.get('index')} tool {s.get('name')}({_short(json.dumps(s.get('args'), ensure_ascii=False), 160)})"
            if s.get("note"):
                line += f" -> {_short(s['note'], 160)}"
            elif s.get("output"):
                line += f" -> {_short(s['output'], 160)}"
        else:
            tok = (s.get("tokens") or {}).get("total")
            line = f"#{s.get('index')} llm {s.get('label') or s.get('name') or ''}" + (f" ({tok} tokens)" if tok else "")
        if s.get("error"):
            line += f" FAILED: {_short(s['error'], 200)}"
        out.append(line)
    return out


def _around(steps, index, n=2):
    lines = _step_lines(steps)
    pos = next((i for i, s in enumerate(steps) if s.get("index") == index), 0)
    return lines[max(0, pos - n):pos + n + 1]
