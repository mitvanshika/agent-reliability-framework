"""
Runs the 4-judge parallel evaluation pipeline against either the
strong or weak agent's traces. Replaces the old separate
run_judges.py / run_judges_weak.py files with one script that takes
a variant argument, so there's only one place to fix bugs.

Scoring change from the original version: final_score is now a
confidence-weighted average instead of a flat average. A judge that
says "score 9, confidence 0.98" counts more than one that says
"score 9, confidence 0.2" (which is basically guessing). Falls back
to a flat average only if every judge somehow reports 0 confidence.

Also now tracks which judges (if any) were under 0.5 confidence on
a given trace, so you can see where the evaluation itself was shaky
— not just where the agent's answer was.

Usage:
    python run_judges.py strong
    python run_judges.py weak
"""

import sys
import json
import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from judges.correctness import judge_correctness
from judges.tool_efficiency import judge_tool_efficiency
from judges.completeness import judge_completeness
from judges.groundedness import judge_groundedness

VARIANTS = {
    "strong": {"traces_dir": "traces", "judged_dir": "judged_traces"},
    "weak": {"traces_dir": "traces_weak", "judged_dir": "judged_traces_weak"},
}


class JudgeState(TypedDict):
    question: str
    answer: str
    tools_used: list[str]
    tool_outputs: list[str]
    correctness_result: dict
    tool_efficiency_result: dict
    completeness_result: dict
    groundedness_result: dict
    disagreement: bool
    low_confidence_judges: list[str]
    final_score: float


def start(state: JudgeState):
    return {}


def run_correctness(state: JudgeState):
    verdict = judge_correctness(state["question"], state["answer"])
    return {"correctness_result": verdict.model_dump()}


def run_tool_efficiency(state: JudgeState):
    verdict = judge_tool_efficiency(state["question"], state["answer"], state["tools_used"])
    return {"tool_efficiency_result": verdict.model_dump()}


def run_completeness(state: JudgeState):
    verdict = judge_completeness(state["question"], state["answer"])
    return {"completeness_result": verdict.model_dump()}


def run_groundedness(state: JudgeState):
    verdict = judge_groundedness(state["question"], state["answer"], state["tool_outputs"])
    return {"groundedness_result": verdict.model_dump()}


def combine_verdicts(state: JudgeState):
    verdicts = {
        "correctness": state["correctness_result"],
        "tool_efficiency": state["tool_efficiency_result"],
        "completeness": state["completeness_result"],
        "groundedness": state["groundedness_result"],
    }
    scores = [v["score"] for v in verdicts.values()]
    confidences = [v["confidence"] for v in verdicts.values()]

    total_confidence = sum(confidences)
    if total_confidence > 0:
        final_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
    else:
        final_score = sum(scores) / len(scores)

    spread = max(scores) - min(scores)
    disagreement = spread >= 4
    low_confidence_judges = [name for name, v in verdicts.items() if v["confidence"] < 0.5]

    return {
        "final_score": final_score,
        "disagreement": disagreement,
        "low_confidence_judges": low_confidence_judges,
    }


graph = StateGraph(JudgeState)
graph.add_node("start", start)
graph.add_node("correctness", run_correctness)
graph.add_node("tool_efficiency", run_tool_efficiency)
graph.add_node("completeness", run_completeness)
graph.add_node("groundedness", run_groundedness)
graph.add_node("combine", combine_verdicts)

graph.set_entry_point("start")
graph.add_edge("start", "correctness")
graph.add_edge("start", "tool_efficiency")
graph.add_edge("start", "completeness")
graph.add_edge("start", "groundedness")
graph.add_edge("correctness", "combine")
graph.add_edge("tool_efficiency", "combine")
graph.add_edge("completeness", "combine")
graph.add_edge("groundedness", "combine")
graph.add_edge("combine", END)

app = graph.compile()


def evaluate_trace(question: str, answer: str, tools_used: list[str], tool_outputs: list[str]):
    return app.invoke({
        "question": question,
        "answer": answer,
        "tools_used": tools_used,
        "tool_outputs": tool_outputs,
        "correctness_result": {},
        "tool_efficiency_result": {},
        "completeness_result": {},
        "groundedness_result": {},
        "disagreement": False,
        "low_confidence_judges": [],
        "final_score": 0.0,
    })


def judge_all_traces(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Use 'strong' or 'weak'.")
    config = VARIANTS[variant]
    os.makedirs(config["judged_dir"], exist_ok=True)
    trace_files = [f for f in os.listdir(config["traces_dir"]) if f.endswith(".json")]

    for filename in trace_files:
        with open(os.path.join(config["traces_dir"], filename)) as f:
            trace = json.load(f)

        if trace["error"] is not None:
            print(f"Skipping {filename} (agent errored, nothing to judge)")
            continue

        print(f"[{variant}] Judging {filename}...")
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
            print(f"  Judging failed after retries: {e}")
            trace["final_score"] = None
            trace["disagreement"] = None
            trace["judge_error"] = str(e)

        with open(os.path.join(config["judged_dir"], filename), "w") as f:
            json.dump(trace, f, indent=2)
        print(f"  Saved to {config['judged_dir']}/{filename}")
        time.sleep(5)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "strong"
    judge_all_traces(variant)