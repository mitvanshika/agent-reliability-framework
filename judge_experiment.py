"""
The experiment judge set (judge_sets.EXPERIMENT): 5 LLM judges plus a
code check, in the same fan-out / fan-in LangGraph shape as run_judges.py.

    correctness_gt     answer vs the scenario's expected_answer (falls back to
                       the classic correctness judge when there is none)
    completeness       did it answer every part of the question
    groundedness       is it supported by what the tools returned
    tool_efficiency    sensible tool use
    context_relevance  did the retrieved tool outputs contain what was needed
                       (N/A when no retrieval happened)
    tool_match         code check vs expected_tools (display only)

final_score = confidence-weighted average of the APPLICABLE LLM judges:
N/A judges are left out, not counted as 0. tool_match is never in it.

The answer key is read from scenarios/ at judge time (so runs made
before the key existed can be re-judged). Each judged trace records
judge_set and answer_key_hash; compare.py refuses to compare scores made
with a different judge set or a different answer key.

judge_run.py uses this for experiment runs; Track 1 (run_judges.py /
pipeline.py) keeps the classic 4 judges.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

import judge_sets
import run_judges
from judge_sets import answer_key_hash  # noqa: F401  (re-exported; lives in the light module)
import run_suite
from judges import base as judges_base
from judges.completeness import judge_completeness, judge_completeness_v2
from judges.context_relevance import judge_context_relevance
from judges.correctness_gt import judge_correctness_gt
from judges.groundedness import judge_groundedness
from judges.safety import judge_safety
from judges.tool_efficiency import judge_tool_efficiency
from judges.tool_match import tool_match

LLM_JUDGES = judge_sets.EXPERIMENT["llm_judges"]


def judge_set(base=None):
    return {**(base or judge_sets.EXPERIMENT), "judge_model": judges_base.llm.model_name}


class ExperimentJudgeState(TypedDict, total=False):
    question: str
    answer: str
    tools_used: list
    tool_outputs: list
    expected_answer: str
    expected_tools: list
    attack: str
    description: str
    non_retrieval_tools: list
    safety_result: dict
    correctness_gt_result: dict
    completeness_result: dict
    groundedness_result: dict
    tool_efficiency_result: dict
    context_relevance_result: dict
    tool_match_result: dict
    final_score: float
    disagreement: bool
    low_confidence_judges: list
    na_judges: list


def _na(reason):
    return {"na": True, "reason": reason}


def run_correctness_gt(state):
    verdict, mode = judge_correctness_gt(state["question"], state["answer"], state.get("expected_answer"))
    return {"correctness_gt_result": {**verdict.model_dump(), "mode": mode}}


def run_completeness(state):
    return {"completeness_result": judge_completeness(state["question"], state["answer"]).model_dump()}


def run_completeness_v2(state):
    return {"completeness_result": judge_completeness_v2(state["question"], state["answer"],
                                                         state.get("tool_outputs", []),
                                                         state.get("expected_answer")).model_dump()}


def run_groundedness(state):
    return {"groundedness_result": judge_groundedness(state["question"], state["answer"],
                                                      state.get("tool_outputs", [])).model_dump()}


def run_tool_efficiency(state):
    return {"tool_efficiency_result": judge_tool_efficiency(state["question"], state["answer"],
                                                            state.get("tools_used", [])).model_dump()}


def run_context_relevance(state):
    verdict = judge_context_relevance(state["question"], state.get("tools_used", []),
                                      state.get("tool_outputs", []), state.get("expected_answer"),
                                      non_retrieval=state.get("non_retrieval_tools"))
    if verdict is None:
        return {"context_relevance_result": _na("no retrieval happened (no tool outputs from retrieval tools)")}
    return {"context_relevance_result": verdict.model_dump()}


def run_safety(state):
    return {"safety_result": judge_safety(state["question"], state["answer"], state.get("tool_outputs", []),
                                          state.get("attack"), state.get("expected_answer"),
                                          state.get("description")).model_dump()}


def run_tool_match(state):
    return {"tool_match_result": tool_match(state.get("tools_used", []), state.get("expected_tools"))}


def combine(verdicts: dict) -> dict:
    """Confidence-weighted final_score over the applicable judges ({name: result dict})."""
    applicable = {n: v for n, v in verdicts.items() if v and not v.get("na") and "score" in v}
    na = [n for n in verdicts if n not in applicable]
    if not applicable:
        return {"final_score": None, "disagreement": None, "low_confidence_judges": [], "na_judges": na}
    scores = [v["score"] for v in applicable.values()]
    confidences = [v["confidence"] for v in applicable.values()]
    total = sum(confidences)
    final = (sum(s * c for s, c in zip(scores, confidences)) / total) if total > 0 else sum(scores) / len(scores)
    return {
        "final_score": final,
        "disagreement": max(scores) - min(scores) >= 4,
        "low_confidence_judges": [n for n, v in applicable.items() if v["confidence"] < 0.5],
        "na_judges": na,
    }


_NODES = {"correctness_gt": run_correctness_gt, "completeness": run_completeness,
          "groundedness": run_groundedness, "tool_efficiency": run_tool_efficiency,
          "context_relevance": run_context_relevance, "tool_match": run_tool_match,
          "safety": run_safety}
# (judge, version) -> node, for judge sets with "versions"; the result key stays <judge>_result
_VERSIONED = {("completeness", "v2"): run_completeness_v2}
_APPS = {}


def _node(js, name):
    """The node function for one judge of a judge set (its version, if the set pins one)."""
    version = (js.get("versions") or {}).get(name)
    return _VERSIONED.get((name, version), _NODES[name])


def app_for(js):
    """The fan-out / fan-in graph for a judge set (built once per set)."""
    cache_key = judge_sets.key(js)
    if cache_key not in _APPS:
        judges, checks = list(js["llm_judges"]), list(js.get("checks") or [])
        graph = StateGraph(ExperimentJudgeState)
        graph.add_node("start", lambda state: {})
        graph.add_node("combine", lambda state: combine({n: state.get(f"{n}_result") for n in judges}))
        graph.set_entry_point("start")
        for node_name in judges + checks:
            graph.add_node(node_name, _node(js, node_name))
            graph.add_edge("start", node_name)
            graph.add_edge(node_name, "combine")
        graph.add_edge("combine", END)
        _APPS[cache_key] = graph.compile()
    return _APPS[cache_key]


app = app_for(judge_sets.EXPERIMENT)


def evaluate(trace: dict, scenario: dict, js=None, description=None, non_retrieval_tools=None) -> dict:
    """All judged fields for one answered trace (experiment judge set by default)."""
    js = js or judge_sets.EXPERIMENT
    result = app_for(js).invoke(_state(trace, scenario, description, non_retrieval_tools))
    fields = {k: result.get(k) for k in ("final_score", "disagreement", "low_confidence_judges", "na_judges")}
    for name in list(js["llm_judges"]) + list(js.get("checks") or []):
        fields[f"{name}_result"] = result.get(f"{name}_result")
    for name in js.get("not_applicable") or []:      # targets.yaml judges: not called, not in final_score
        fields[f"{name}_result"] = _na("not applicable to this target (targets.yaml judges)")
        fields["na_judges"] = list(fields.get("na_judges") or []) + [name]
    fields["answer_key_hash"] = answer_key_hash(scenario)
    if fields["final_score"] is None:
        raise RuntimeError("every judge was N/A, nothing to score")
    return fields


def _state(trace, scenario, description=None, non_retrieval_tools=None):
    return {
        "non_retrieval_tools": non_retrieval_tools,
        "question": trace["question"], "answer": trace["answer"],
        "tools_used": trace.get("tools_used", []), "tool_outputs": trace.get("tool_outputs", []),
        "expected_answer": scenario.get("expected_answer"), "expected_tools": scenario.get("expected_tools"),
        "attack": scenario.get("attack"), "description": description,
    }


def rejudge(judged_trace: dict, scenario: dict, js, only, description=None) -> dict:
    """Re-run only the judges in `only` on an already-judged trace; every other judge's
    result is kept as it is, and final_score etc. are recomputed with the same combine()."""
    state = _state(judged_trace, scenario, description)
    fresh = {}
    for name in only:
        fresh.update(_node(js, name)(state))
    verdicts = {n: fresh.get(f"{n}_result", judged_trace.get(f"{n}_result")) for n in js["llm_judges"]}
    fields = combine(verdicts)
    fields.update(fresh)
    if fields["final_score"] is None:
        raise RuntimeError("every judge was N/A, nothing to score")
    return fields


def judge_run_folder(name: str, traces_dir: str, judged_dir: str, scenario_dir=None, description=None,
                     target=None):
    """Judge one run folder, reusing run_judges.judge_all_traces (latest trace per scenario,
    crashed-trace handling, judge-error handling, pacing). The answer key comes from the
    run's own scenario folder; an adversarial suite gets the adversarial judge set."""
    import run_experiment
    scenario_list = run_experiment.load_scenarios(scenario_dir)
    scenarios = {s["id"]: s for s in scenario_list}
    import target_config
    cfg = target_config.get(target)
    base = judge_sets.for_target(judge_sets.for_scenarios(scenario_list), cfg["judges"])
    js = judge_set(base)
    description = description if description is not None else cfg["description"]
    non_retrieval = cfg["non_retrieval_tools"] if target else None

    def evaluator(trace):
        return evaluate(trace, scenarios.get(trace["scenario_id"], {}), base, description, non_retrieval)

    run_judges.judge_all_traces(name, traces_dir=traces_dir, judged_dir=judged_dir,
                                evaluator=evaluator, extra_fields={"judge_set": js}, skip_judged=True)
    return js
