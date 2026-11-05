"""
Diagnoses WHY questions got worse in a comparison.

    python diagnose.py runs/<run> [--draft]

Reads runs/<run>/comparison.json (written by compare.py). For every ▼
question it picks the most likely cause, checking in this order:

  1. run error rate_limit / timeout -> INFRA            not agent quality: re-run
  2. run error auth                 -> CONFIG           check the key / model
  3. run error loop_limit           -> TOOLS/PROMPT     agent got stuck calling tools
     any other run crash            -> AGENT CRASH      the target raised an exception
  (several judges dropped by about the same amount -> UNCLEAR: no single judge
   stands out, so naming one cause would be a guess)
  4. context_relevance dropped      -> RAG              retrieval fetched the wrong / no info
  5. context fine, but correctness_gt / groundedness dropped
                                    -> PROMPT           had the info, used it badly
  6. tool_efficiency / tool_match dropped -> TOOL DESCRIPTIONS
  7. completeness dropped           -> PROMPT           missed part of the question
  8. judges disagreed / low confidence -> MEASUREMENT UNSURE   check by hand
  9. anything else                  -> UNCLEAR          lists every judge that went down

Each diagnosis carries its evidence (the judge deltas) and a suggested next
experiment. Errors are checked first on purpose: a rate-limited answer has
terrible judge scores, but the cause is the infrastructure, not the agent.

Also scans runs/ for the same target: a scenario that FAILED in 3+ judged
runs with DIFFERENT prompt settings is a fine-tuning candidate (prompting
isn't fixing it).

--draft writes experiments/draft_<cause>_<timestamp>.yaml extending the run's
experiment, with a TODO on the setting to change. It is never run for you.
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------- thresholds
DROP = 2.0                  # a judge "dropped" if its score fell by at least this much (0-10 scale)
EQUAL_MIN_JUDGES = 3        # "several judges dropped equally": at least this many dropped...
EQUAL_MARGIN = 1.0          # ...and their drops are within this of each other
FINE_TUNE_MIN_PROMPTS = 3   # failed under at least this many DIFFERENT prompt settings
PASS_THRESHOLD = 7          # same pass rule as detect_regressions.passed()

RUNS_DIR = "runs"
EXPERIMENTS_DIR = "experiments"

CORRECTNESS_JUDGES = ("correctness_gt", "correctness")   # classic judge set has plain "correctness"
TOOL_JUDGES = ("tool_efficiency", "tool_match")

# cause -> (setting an experiment would change, suggested next experiment)
CAUSES = {
    "INFRA": (None, "not an agent-quality signal: re-run the same experiment for these scenarios "
                    "(python run_experiment.py <experiment> --scenarios ...)"),
    "CONFIG": (None, "check GROQ_API_KEY and settings.model, then re-run"),
    "TOOLS/PROMPT": ("prompt", "try a prompt change: 'call each tool at most once; if a tool returns no data, "
                               "answer from what you have or escalate'"),
    "AGENT CRASH": (None, "read the error in the trace; fix the target's code, then re-run"),
    "RAG": ("retrieval", "try a retrieval change: retrieval.method embeddings, or top_k: 2"),
    "PROMPT (used info badly)": ("prompt", "try a prompt change: 'answer only from the most relevant FAQ / tool "
                                           "output, and repeat exact numbers, dates and statuses'"),
    "TOOL DESCRIPTIONS": ("docstrings", "try a docstrings change: clearer tool descriptions "
                                        "(start from docstrings/detailed.yaml)"),
    "PROMPT (missed part)": ("prompt", "try a prompt change: 'list every part of the question first, then answer each'"),
    "MEASUREMENT UNSURE": (None, "read the answer by hand; re-judge (judge_run.py) or calibrate noise before trusting it"),
    "UNCLEAR": (None, "open the run and baseline traces side by side and read them"),
}
ACTIONABLE = [c for c, (setting, _) in CAUSES.items() if setting]


# ---------------------------------------------------------------- rules

def _dropped(deltas, names):
    return {n: deltas[n] for n in names if n in deltas and deltas[n] <= -DROP}


def diagnose_row(row) -> dict:
    """{scenario_id, cause, evidence, suggestion, setting} for one comparison row."""
    sid = row["scenario_id"]
    deltas = row.get("judge_deltas") or {}
    deltas_text = ", ".join(f"{n} {d:+g}" for n, d in deltas.items()) or "no judge scores on both sides"
    score_text = f"final_score {row['baseline_score']:g} -> {row['run_score']:g} ({row['delta']:+g})"

    def result(cause, why):
        setting, suggestion = CAUSES[cause]
        return {"scenario_id": sid, "cause": cause, "evidence": f"{why}. {score_text}; judge deltas: {deltas_text}",
                "suggestion": suggestion, "setting": setting}

    err = row.get("run_error_type")
    if err in ("rate_limit", "timeout"):
        return result("INFRA", f"the run hit {err}: the answer never happened")
    if err == "auth":
        return result("CONFIG", "the run failed authentication")
    if err == "loop_limit":
        return result("TOOLS/PROMPT", "the run hit the recursion limit (stuck calling tools)")
    if err is not None or row.get("run_error"):
        return result("AGENT CRASH", f"the run crashed ({err or 'unknown'}): {(row.get('run_error') or '')[:120]}")

    all_drops = _dropped(deltas, deltas)
    if len(all_drops) >= EQUAL_MIN_JUDGES and max(all_drops.values()) - min(all_drops.values()) <= EQUAL_MARGIN:
        return result("UNCLEAR", f"{len(all_drops)} judges dropped about equally "
                                 f"({', '.join(all_drops)}): no single judge stands out")

    if "context_relevance" in deltas and deltas["context_relevance"] <= -DROP:
        return result("RAG", f"context_relevance dropped {deltas['context_relevance']:+g}: "
                             "retrieval did not bring back what was needed")
    base_ctx = "context_relevance" in (row.get("baseline_judges") or {})
    run_ctx = "context_relevance" in (row.get("run_judges") or {})
    context_fine = ("context_relevance" in deltas) or (not base_ctx and not run_ctx)
    bad = _dropped(deltas, CORRECTNESS_JUDGES + ("groundedness",))
    if context_fine and bad:
        why = "context was fine" if "context_relevance" in deltas else "no retrieval on either side"
        return result("PROMPT (used info badly)", f"{why}, but {', '.join(f'{n} {d:+g}' for n, d in bad.items())}")
    tools = _dropped(deltas, TOOL_JUDGES)
    if tools:
        return result("TOOL DESCRIPTIONS", f"{', '.join(f'{n} {d:+g}' for n, d in tools.items())}")
    if "completeness" in deltas and deltas["completeness"] <= -DROP:
        return result("PROMPT (missed part)", f"completeness dropped {deltas['completeness']:+g}")
    if row.get("run_disagreement") or row.get("run_low_confidence"):
        flags = (["judges disagreed (spread >= 4)"] if row.get("run_disagreement") else []) + \
                ([f"low confidence: {', '.join(row['run_low_confidence'])}"] if row.get("run_low_confidence") else [])
        return result("MEASUREMENT UNSURE", "; ".join(flags))
    down = [f"{n} {d:+g}" for n, d in deltas.items() if d < 0]
    return result("UNCLEAR", f"judges that went down: {', '.join(down) if down else 'none clearly'}")


def diagnose(comparison) -> list[dict]:
    rows = [r for r in comparison.get("rows", []) if r.get("direction") == "down"]
    if rows and "judge_deltas" not in rows[0]:
        raise ValueError("this comparison.json is from an older compare.py without diagnosis data: re-run compare.py")
    return [diagnose_row(r) for r in rows]


# ---------------------------------------------------------------- fine-tuning signal

def _failed(record):
    if record["status"] == "agent_error":
        return True
    return record["status"] == "ok" and not (record["final_score"] is not None and record["final_score"] >= PASS_THRESHOLD)


def fine_tune_candidates(target, runs_dir=None):
    """{scenario_id: [evidence runs]} for scenarios that failed under >= FINE_TUNE_MIN_PROMPTS
    different prompt settings, across every judged run of this target in runs/."""
    import compare
    runs_dir = runs_dir or RUNS_DIR
    failures, no_prompt_setting = {}, False
    if not os.path.isdir(runs_dir):
        return {}, False
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not (os.path.isfile(os.path.join(run_dir, "run_meta.json")) and os.path.isdir(os.path.join(run_dir, "judged"))):
            continue
        try:
            meta, records = compare.load_run_records(run_dir)
        except SystemExit:
            continue
        if compare.run_target(meta) != target:
            continue
        prompt = ((meta.get("resolved_config") or {}).get("settings") or {}).get("prompt")
        if prompt is None:
            no_prompt_setting = True
            continue
        for sid, rec in records.items():
            if _failed(rec):
                failures.setdefault(sid, []).append({
                    "run_dir": run_dir.replace("\\", "/"), "experiment": meta.get("experiment"), "prompt": prompt,
                    "final_score": rec["final_score"], "error_type": rec["error_type"] if rec["status"] == "agent_error" else None})
    candidates = {sid: runs for sid, runs in failures.items()
                  if len({r["prompt"] for r in runs}) >= FINE_TUNE_MIN_PROMPTS}
    return dict(sorted(candidates.items())), no_prompt_setting


# ---------------------------------------------------------------- draft

def _slug(cause):
    return "".join(ch if ch.isalnum() else "_" for ch in cause.lower()).strip("_").replace("__", "_")


def draft_experiment(diagnoses, comparison, experiments_dir=None, now=None):
    """Write experiments/draft_<cause>_<timestamp>.yaml for the most common actionable cause.
    Returns the path, or None if no ▼ question has an actionable cause."""
    counts = {}
    for d in diagnoses:
        if d["setting"]:
            counts.setdefault(d["cause"], []).append(d["scenario_id"])
    if not counts:
        return None
    cause = max(counts, key=lambda c: (len(counts[c]), -ACTIONABLE.index(c)))
    setting, suggestion = CAUSES[cause]
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    name = f"draft_{_slug(cause)}_{stamp}"
    parent = comparison.get("experiment")
    examples = {
        "prompt": "  # TODO: copy the current prompt file to a new one, edit it, and point this at it\n"
                  "  prompt: prompts/TODO_new_prompt.txt\n",
        "docstrings": "  # TODO: copy docstrings/detailed.yaml to a new file, rewrite the descriptions of the tools\n"
                      "  # that went wrong, and point this at it\n"
                      "  docstrings: docstrings/TODO_new_docstrings.yaml\n",
        "retrieval": "  # TODO: pick one change, e.g. {method: embeddings} or {method: bm25, top_k: 2}\n"
                     "  retrieval: {method: TODO}\n",
    }
    body = examples[setting] if comparison.get("target") == "configurable" else \
        f"  # TODO: change the setting that controls the {setting} of your system\n"
    text = (
        f"# DRAFT written by diagnose.py on {(now or datetime.now()).isoformat(timespec='seconds')}\n"
        f"# from {comparison.get('run_dir')} (compared with baseline '{comparison.get('baseline_experiment')}').\n"
        f"# Most common cause among the down questions: {cause} "
        f"({len(counts[cause])} of {len(diagnoses)}: {' '.join(counts[cause])})\n"
        f"# Suggested: {suggestion}\n"
        f"# NOT run automatically. Fill in the TODOs, then:\n"
        f"#   python run_experiment.py {experiments_dir or EXPERIMENTS_DIR}/{name}.yaml\n"
        f"name: {name}\n"
        f'description: "TODO: describe the change ({cause.lower()})"\n'
        f"extends: {parent}\n"
        f"settings:\n{body}"
    )
    path = os.path.join(experiments_dir or EXPERIMENTS_DIR, f"{name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\r\n") as f:   # "x": never overwrite
        f.write(text)
    return path


# ---------------------------------------------------------------- output

def print_diagnoses(diagnoses, header=True):
    try:
        import compare
        down = compare._symbols()["down"]
    except Exception:
        down = "v"
    if header:
        print(f"\nDiagnosis of {len(diagnoses)} {down} question(s):")
    for d in diagnoses:
        print(f"  {d['scenario_id']:15} {d['cause']}")
        print(f"  {'':15}   evidence: {d['evidence']}")
        print(f"  {'':15}   next:     {d['suggestion']}")
    if diagnoses:
        counts = {}
        for d in diagnoses:
            counts[d["cause"]] = counts.get(d["cause"], 0) + 1
        print("  by cause: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])))


def print_fine_tune(candidates, no_prompt_setting):
    print(f"\nFine-tuning signal (failed under >= {FINE_TUNE_MIN_PROMPTS} different prompt settings):")
    if not candidates:
        print("  none" + (" (some runs of this target have no 'prompt' setting, so they can't count)"
                          if no_prompt_setting else ""))
        return
    for sid, runs in candidates.items():
        prompts = sorted({r["prompt"] for r in runs})
        print(f"  {sid}: failed in {len(runs)} runs across {len(prompts)} prompts -> "
              "prompting isn't fixing this, fine-tuning candidate")
        for r in runs:
            score = "crash " + (r["error_type"] or "") if r["error_type"] else f"{r['final_score']:.1f}"
            print(f"      {r['run_dir']}  ({r['experiment']}, prompt {r['prompt']}): {score}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose why questions went down in a comparison.")
    p.add_argument("run_dir")
    p.add_argument("--draft", action="store_true", help="write a draft next experiment (never run automatically)")
    args = p.parse_args(argv)
    run_dir = args.run_dir.rstrip("/\\")
    path = os.path.join(run_dir, "comparison.json")
    if not os.path.isfile(path):
        sys.exit(f"{path} not found: run python compare.py {run_dir} first.")
    with open(path, encoding="utf-8") as f:
        comparison = json.load(f)
    try:
        diagnoses = diagnose(comparison)
    except ValueError as e:
        sys.exit(str(e))

    print(f"Diagnosis for {comparison.get('experiment')} vs baseline '{comparison.get('baseline_experiment')}' "
          f"(verdict {comparison.get('verdict')})")
    if diagnoses:
        print_diagnoses(diagnoses)
    else:
        print("No question went down beyond its noise band: nothing to diagnose.")
    candidates, no_prompt = fine_tune_candidates(comparison.get("target"))
    print_fine_tune(candidates, no_prompt)

    out = {"run_dir": run_dir, "diagnoses": diagnoses, "fine_tune_candidates": candidates}
    if args.draft:
        draft = draft_experiment(diagnoses, comparison)
        out["draft"] = draft
        print(f"\nDraft written (not run): {draft}" if draft else
              "\nNo draft: no down question has a cause an experiment setting could address.")
    with open(os.path.join(run_dir, "diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {os.path.join(run_dir, 'diagnosis.json')}")
    return out


if __name__ == "__main__":
    main()
