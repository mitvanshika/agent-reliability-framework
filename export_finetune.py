"""
Exports fine-tuning data from judged experiment runs.

    python export_finetune.py [--target configurable] [--scenarios id ...] [--dry-run] [--yes]

Only TRAIN-split scenarios (split: train in the scenario file; missing =
test) are ever exported, and only answers that were clearly good:
    final_score >= 9, no agent error, no judge error, judges did not
    disagree, no low-confidence judge, judged against the CURRENT answer
    key, and the trace has the full conversation ("messages").
Identical conversations are kept once (best score, then newest run).

Leakage guard: a test-split scenario is never exported. Asking for one is
refused, and every example is checked again right before writing. Test
scenarios are the exam that compare.py grades: training on them lets the
model memorise the answers, and the exam stops measuring anything.

Always writes finetune/review_<ts>.md first (every example + why the rest
were rejected). The JSONL (finetune/train_<ts>.jsonl, OpenAI chat format
with tool schemas) is written only after you confirm (--yes skips the
question, --dry-run never writes it).
"""

import argparse
import json
import os
import sys
from datetime import datetime

import judge_sets
import run_suite
from calibrate_noise import confirm
from trace_utils import load_latest_traces

FINETUNE_DIR = "finetune"
RUNS_DIR = "runs"
MIN_SCORE = 9.0
LEAKAGE_EXPLANATION = ("it is split: test, part of the exam compare.py grades. Training on it lets the model "
                       "memorise the answer, and the exam would stop measuring anything")


class LeakageError(RuntimeError):
    pass


def split_of(scenario) -> str:
    return (scenario or {}).get("split") or "test"


def tool_schemas(meta):
    """The 4 demo tools as OpenAI tool schemas, with the descriptions that run used. None for other targets."""
    settings = (meta.get("resolved_config") or {}).get("settings") or {}
    if (meta.get("resolved_config") or {}).get("target") != "configurable" or "docstrings" not in settings:
        return None
    from langchain_core.utils.function_calling import convert_to_openai_tool

    import agent_configurable
    import test_agent
    descriptions = agent_configurable._load_docstrings(settings["docstrings"])
    return [convert_to_openai_tool(t.model_copy(update={"description": descriptions[t.name]}))
            for t in test_agent.tools]


def _conversation_key(messages):
    """Same conversation regardless of tool-call ids."""
    out = []
    for m in messages:
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
        out.append((m["role"], " ".join((m.get("content") or "").split()), calls))
    return json.dumps(out, ensure_ascii=False)


def collect(target, runs_dir=None, scenarios=None, only=None):
    """(examples, rejected {reason: count}) from every judged run of target."""
    scenarios = scenarios if scenarios is not None else {s["id"]: s for s in run_suite.load_scenarios()}
    runs_dir = runs_dir or RUNS_DIR
    candidates, rejected = [], {}

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for name in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        run_dir = os.path.join(runs_dir, name)
        meta_path, judged_dir = os.path.join(run_dir, "run_meta.json"), os.path.join(run_dir, "judged")
        if not (os.path.isfile(meta_path) and os.path.isdir(judged_dir)):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("resolved_config") or {}).get("target") != target:
            continue
        tools = None
        for sid, t in sorted(load_latest_traces(judged_dir).items()):
            if only and sid not in only:
                continue
            sc = scenarios.get(sid)
            if sc is None:
                reject("scenario no longer exists")
            elif split_of(sc) != "train":
                reject("split: test (the exam: never exported)" if split_of(sc) == "test"
                       else f"unknown split '{split_of(sc)}' (treated as test)")
            elif t.get("error") is not None:
                reject("agent error")
            elif t.get("judge_error"):
                reject("judge error")
            elif t.get("final_score") is None or t["final_score"] < MIN_SCORE:
                reject(f"final_score < {MIN_SCORE:g}")
            elif t.get("disagreement"):
                reject("judges disagreed")
            elif t.get("low_confidence_judges"):
                reject("low-confidence judge")
            elif t.get("answer_key_hash") != judge_sets.answer_key_hash(sc):
                reject("judged against an older answer key")
            elif not isinstance(t.get("messages"), list) or not t["messages"]:
                reject("trace has no full conversation (older run)")
            else:
                if tools is None:
                    tools = tool_schemas(meta) or []
                judges = {n: (t.get(f"{n}_result") or {}).get("score")
                          for n in judge_sets.ALL_NAMES if isinstance(t.get(f"{n}_result"), dict)
                          and "score" in t[f"{n}_result"]}
                candidates.append({"scenario_id": sid, "run_dir": run_dir.replace("\\", "/"),
                                   "experiment": meta.get("experiment"), "timestamp": meta.get("timestamp") or "",
                                   "final_score": t["final_score"], "judges": judges,
                                   "messages": t["messages"], "tools": tools})
    best = {}
    for ex in candidates:
        key = (ex["scenario_id"], _conversation_key(ex["messages"]))
        cur = best.get(key)
        if cur is None or (ex["final_score"], ex["timestamp"]) > (cur["final_score"], cur["timestamp"]):
            if cur is not None:
                reject("duplicate conversation")
            best[key] = ex
        else:
            reject("duplicate conversation")
    examples = sorted(best.values(), key=lambda e: (e["scenario_id"], e["run_dir"]))
    return examples, rejected


def guard(examples, scenarios):
    leaked = sorted({e["scenario_id"] for e in examples if split_of(scenarios.get(e["scenario_id"])) != "train"})
    if leaked:
        raise LeakageError(f"refusing to export {' '.join(leaked)}: {LEAKAGE_EXPLANATION}.")


def write_review(path, examples, rejected, target):
    lines = [f"# Fine-tuning export review ({datetime.now().isoformat(timespec='seconds')})", "",
             f"Target: `{target}`. Only `split: train` scenarios; final_score >= {MIN_SCORE:g}; no agent/judge "
             "errors; no judge disagreement or low confidence; current answer key; full conversation.", "",
             f"**{len(examples)} example(s)** would be exported.", "", "## Rejected", ""]
    lines += [f"- {reason}: {n}" for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1])] or ["- none"]
    for i, ex in enumerate(examples, 1):
        judges = ", ".join(f"{k} {v}" for k, v in ex["judges"].items())
        lines += ["", f"## {i}. {ex['scenario_id']} ({ex['final_score']:.2f})",
                  f"run `{ex['run_dir']}` ({ex['experiment']}); judges: {judges}", ""]
        for m in ex["messages"]:
            text = (m.get("content") or "").strip()
            calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls") or [])
            shown = text if m["role"] != "system" else text[:200] + ("..." if len(text) > 200 else "")
            lines.append(f"- **{m['role']}**: {shown}" + (f" -> calls {calls}" if calls else ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            line = {"messages": ex["messages"]}
            if ex["tools"]:
                line["tools"] = ex["tools"]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Export train-split, high-quality answers as fine-tuning data.")
    p.add_argument("--target", default="configurable")
    p.add_argument("--scenarios", nargs="+", metavar="ID", help="only these (must be split: train)")
    p.add_argument("--dry-run", action="store_true", help="write review.md only, never the JSONL")
    p.add_argument("--yes", action="store_true", help="write the JSONL without asking")
    args = p.parse_args(argv)

    scenarios = {s["id"]: s for s in run_suite.load_scenarios()}
    if args.scenarios:
        unknown = sorted(set(args.scenarios) - set(scenarios))
        if unknown:
            sys.exit(f"Unknown scenario ids: {' '.join(unknown)}")
        test_ids = sorted(s for s in args.scenarios if split_of(scenarios[s]) != "train")
        if test_ids:
            print(f"Refused: {' '.join(test_ids)}: {LEAKAGE_EXPLANATION}.", file=sys.stderr)
            sys.exit(2)

    examples, rejected = collect(args.target, scenarios=scenarios, only=set(args.scenarios or []) or None)
    guard(examples, scenarios)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review = os.path.join(FINETUNE_DIR, f"review_{stamp}.md")
    write_review(review, examples, rejected, args.target)
    print(f"{len(examples)} example(s) pass every filter. Review: {review}")
    for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {n:4}  {reason}")
    if args.dry_run:
        print("Dry run: JSONL not written.")
        return {"examples": examples, "rejected": rejected, "review": review, "jsonl": None}
    if not examples:
        print("Nothing to export.")
        return {"examples": [], "rejected": rejected, "review": review, "jsonl": None}
    if not confirm(f"Read {review}. Write {len(examples)} example(s) to {FINETUNE_DIR}/train_{stamp}.jsonl?", args.yes):
        print("Not written.")
        return {"examples": examples, "rejected": rejected, "review": review, "jsonl": None}
    guard(examples, scenarios)   # once more, right before anything leaves the building
    jsonl = os.path.join(FINETUNE_DIR, f"train_{stamp}.jsonl")
    write_jsonl(jsonl, examples)
    print(f"Wrote {jsonl}")
    return {"examples": examples, "rejected": rejected, "review": review, "jsonl": jsonl}


if __name__ == "__main__":
    main()
