from langgraph.graph import StateGraph, END
from typing import TypedDict
from judges.correctness import judge_correctness
from judges.tool_efficiency import judge_tool_efficiency
from judges.completeness import judge_completeness
import json
import os
import time

class JudgeState(TypedDict):
    question: str
    answer: str
    tools_used: list[str]
    correctness_result: dict
    tool_efficiency_result: dict
    completeness_result: dict
    disagreement: bool
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

def combine_verdicts(state: JudgeState):
    scores = [
        state["correctness_result"]["score"],
        state["tool_efficiency_result"]["score"],
        state["completeness_result"]["score"]
    ]
    final_score = sum(scores) / len(scores)
    spread = max(scores) - min(scores)
    disagreement = spread >= 4
    return {"final_score": final_score, "disagreement": disagreement}

graph = StateGraph(JudgeState)
graph.add_node("start", start)
graph.add_node("correctness", run_correctness)
graph.add_node("tool_efficiency", run_tool_efficiency)
graph.add_node("completeness", run_completeness)
graph.add_node("combine", combine_verdicts)

graph.set_entry_point("start")
graph.add_edge("start", "correctness")
graph.add_edge("start", "tool_efficiency")
graph.add_edge("start", "completeness")
graph.add_edge("correctness", "combine")
graph.add_edge("tool_efficiency", "combine")
graph.add_edge("completeness", "combine")
graph.add_edge("combine", END)

app = graph.compile()

def evaluate_trace(question: str, answer: str, tools_used: list[str]):
    result = app.invoke({
        "question": question,
        "answer": answer,
        "tools_used": tools_used,
        "correctness_result": {},
        "tool_efficiency_result": {},
        "completeness_result": {},
        "disagreement": False,
        "final_score": 0.0
    })
    return result

def judge_all_traces():
    os.makedirs("judged_traces_weak", exist_ok=True)
    trace_files = [f for f in os.listdir("traces_weak") if f.endswith(".json")]

    for filename in trace_files:
        with open(os.path.join("traces_weak", filename)) as f:
            trace = json.load(f)

        if trace["error"] is not None:
            print(f"Skipping {filename} (agent errored, nothing to judge)")
            continue

        print(f"Judging {filename}...")
        judge_result = evaluate_trace(trace["question"], trace["answer"], trace.get("tools_used", []))

        trace["final_score"] = judge_result["final_score"]
        trace["disagreement"] = judge_result["disagreement"]
        trace["correctness_result"] = judge_result["correctness_result"]
        trace["tool_efficiency_result"] = judge_result["tool_efficiency_result"]
        trace["completeness_result"] = judge_result["completeness_result"]

        with open(os.path.join("judged_traces_weak", filename), "w") as f:
            json.dump(trace, f, indent=2)
        print(f"  Saved to judged_traces_weak/{filename}")
        time.sleep(5)

if __name__ == "__main__":
    judge_all_traces()