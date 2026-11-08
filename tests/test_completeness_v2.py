"""
Tests for completeness v2 and the experiment_v2 judge set (LLM faked, no API calls).
    python tests/test_completeness_v2.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import judge_experiment  # noqa: E402
import judge_sets  # noqa: E402
import run_suite  # noqa: E402
from judges import completeness  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


SC = {s["id"]: s for s in run_suite.load_scenarios()}
RAG06 = SC["rag_06"]
TRACE = {"scenario_id": "rag_06", "question": RAG06["question"],
         "answer": "I don't have information about gift cards, so I've passed this to a human agent.",
         "tools_used": ["faq_search", "escalate_to_human"],
         "tool_outputs": ["No matching FAQ found for this query", "Escalated to human support. Reason: gift cards"]}

print("v2 prompt:")
fake_judges.install()
completeness.judge_completeness_v2(TRACE["question"], TRACE["answer"], TRACE["tool_outputs"], RAG06["expected_answer"])
p = fake_judges.called("completeness")[0]
ok("shows the tool outputs", "- No matching FAQ found for this query" in p and "- Escalated to human support" in p)
ok("shows the expected answer", RAG06["expected_answer"] in p)
ok("honest 'not available' / escalation counts as complete", "information needed was NOT in the tool outputs" in p
   and "escalates" in p)
ok("escalating when the info WAS available is not complete", "although the needed information WAS in" in p)
ok("an invented answer is never complete", "An invented answer is never complete" in p)
template = open(os.path.join("judges", "completeness.py"), encoding="utf-8").read().split("def judge_completeness_v2")[1]
ok("generic: no support-bot words in the v2 template", all(w not in template.lower()
   for w in ("order", "refund", "faq", "shipping", "gift")))
fake_judges.install()
completeness.judge_completeness_v2("q?", "a", [], None)
p = fake_judges.called("completeness")[0]
ok("no tool outputs / no key -> said so", "(no tool outputs)" in p and "(not given)" in p)

print("v1 untouched:")
src = open(os.path.join("judges", "completeness.py"), encoding="utf-8").read()
v1_src = src.split("def judge_completeness_v2")[0]
ok("v1 function text unchanged", "def judge_completeness(question: str, answer: str, max_retries: int = 2):" in v1_src
   and "Consider: if the question had multiple parts" in v1_src)
fake_judges.install()
completeness.judge_completeness("q?", "a")
ok("v1 prompt has no tool outputs / answer key", "tools returned" not in fake_judges.called("completeness")[0]
   and "Reference answer" not in fake_judges.called("completeness")[0])
ok("experiment_v1 unchanged (name, judges, checks, no versions)", judge_sets.EXPERIMENT == {
   "name": "experiment_v1",
   "llm_judges": ["correctness_gt", "completeness", "groundedness", "tool_efficiency", "context_relevance"],
   "checks": ["tool_match"]})

print("experiment_v2 judge set:")
V2 = judge_sets.EXPERIMENT_V2
ok("same judges + checks as v1, name experiment_v2", V2["llm_judges"] == judge_sets.EXPERIMENT["llm_judges"]
   and V2["checks"] == judge_sets.EXPERIMENT["checks"] and V2["name"] == "experiment_v2")
ok("only completeness is versioned", V2["versions"] == {"completeness": "v2"})
ok("key() differs from v1", judge_sets.key(V2) != judge_sets.key(judge_sets.EXPERIMENT))
ok("same() says v1 != v2 (with and without judge_model)", not judge_sets.same(V2, judge_sets.EXPERIMENT)
   and not judge_sets.same({**V2, "judge_model": "m"}, {**judge_sets.EXPERIMENT, "judge_model": "m"}))
ok("default for main scenarios is now v2",
   judge_sets.for_scenarios(list(SC.values()))["name"] == "experiment_v2")


def prompts_for(js):
    fake_judges.install()
    fields = judge_experiment.evaluate(TRACE, RAG06, js)
    return {n: pr for n, pr in fake_judges.CALLS}, fields


v1_prompts, v1_fields = prompts_for(judge_sets.EXPERIMENT)
v2_prompts, v2_fields = prompts_for(V2)
print("judging one trace with v1 vs v2:")
ok("v2 calls the v2 completeness prompt", "Reference answer" in v2_prompts["completeness"])
ok("v1 still calls the v1 completeness prompt", "Reference answer" not in v1_prompts["completeness"])
ok("the other 4 judges get byte-identical prompts", all(v1_prompts[n] == v2_prompts[n]
   for n in ("correctness_gt", "groundedness", "tool_efficiency", "context_relevance")))
ok("result still stored as completeness_result (same fields as v1)", set(v1_fields) == set(v2_fields)
   and "completeness_result" in v2_fields)

print("compare refuses to mix v1 and v2 (compare.main raises 'Judge sets differ' when same() is False):")
ok("v1-judged run vs v2 baseline -> same() is False",
   not judge_sets.same({**judge_sets.EXPERIMENT, "judge_model": "m"}, {**V2, "judge_model": "m"}))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
