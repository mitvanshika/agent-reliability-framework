"""
Tests for the domain-neutral judge set generic_v1 (judges/generic.py). LLM faked, no API calls.
    python tests/test_generic_judges.py
"""

import os
import re
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
import target_config  # noqa: E402
from judges import generic  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


G = judge_sets.GENERIC_V1
DESC = "summarises a news article in two sentences"
ARTICLE = "Summarise: The council voted 7-2 on Tuesday to build a new library on Elm Street, opening in 2027."
# "order" alone is fine ("in a sensible order"); purchase senses are caught by "orders" / "order id|status|..."
SUPPORT_WORDS = (r"\b(store|refunds?|faq|escalat\w*|customers?|shipping|orders|order (?:id|status|number|lookup)"
                 r"|gift|warranty|human agent|purchase)\b")


def kind(prompt):
    first = prompt.splitlines()[0].lower()
    for k in ("reference answer written", "is correct", "fully handles", "supported by", "tools sensibly", "retrieval"):
        if k in first:
            return {"reference answer written": "correctness_gt", "is correct": "correctness_gt", "fully handles": "completeness", "supported by": "groundedness",
                    "tools sensibly": "tool_efficiency", "retrieval": "context_relevance"}[k]
    return "?"


def prompts(trace, scenario, js, description=DESC, non_retrieval=None):
    fake_judges.install(lambda name, prompt: (8, 0.9))
    fields = judge_experiment.evaluate(trace, scenario, js, description, non_retrieval)
    return {kind(p): p for n, p in fake_judges.CALLS if n == "generic"}, fields, [n for n, _ in fake_judges.CALLS]


TOOL_TRACE = {"scenario_id": "x", "question": "What's the capital of France?", "answer": "Paris.",
              "tools_used": ["search"], "tool_outputs": ["Paris is the capital of France."]}
P, fields, called = prompts(TOOL_TRACE, {"expected_answer": "Paris"}, G)

print("the generic set:")
ok("same 5 judges + tool_match, every judge versioned 'generic'", G["llm_judges"] == judge_sets.EXPERIMENT["llm_judges"]
   and G["checks"] == ["tool_match"] and set(G["versions"].values()) == {"generic"})
ok("all 5 calls went to judges/generic.py (none to the old files)", called == ["generic"] * 5, called)
ok("one prompt per judge", sorted(P) == sorted(G["llm_judges"]), sorted(P))
ok("not the default: main scenarios still get experiment_v2",
   judge_sets.for_scenarios(run_suite.load_scenarios())["name"] == "experiment_v2")
ok("differs from experiment_v2 (compare won't mix)", not judge_sets.same(G, judge_sets.EXPERIMENT_V2))

print("wording is domain-neutral:")
for name, p in sorted(P.items()):
    hits = sorted(set(m.group(0).lower() for m in re.finditer(SUPPORT_WORDS, p, re.I)))
    ok(f"{name}: no support-bot words", not hits, hits)
src = open(os.path.join("judges", "generic.py"), encoding="utf-8").read().split('"""', 2)[2]   # code, not the docstring
ok("judges/generic.py code has no support-bot words",
   not re.findall(SUPPORT_WORDS, src, re.I), re.findall(SUPPORT_WORDS, src, re.I))

print("every prompt carries the target's description, the input and the source:")
for name in ("correctness_gt", "completeness", "groundedness", "context_relevance"):
    ok(f"{name}: description + input + source", DESC in P[name] and TOOL_TRACE["question"] in P[name]
       and "Paris is the capital of France." in P[name])
ok("tool_efficiency: description + the tools called", DESC in P["tool_efficiency"] and "['search']" in P["tool_efficiency"])
ok("honest-refusal rule in groundedness + completeness, without 'escalate'",
   all("doesn't know, says it can't do it, or passes the request on" in P[n] for n in ("groundedness", "completeness")))
ok("completeness v2.1: small extras = small deduction, invented core = not complete",
   "a small deduction, not a zero" in P["completeness"] and "core of the answer is made up" in P["completeness"])

print("a system with no tools (summarizer):")
SUM = {"scenario_id": "s", "question": ARTICLE, "answer": "The council approved a new Elm Street library, opening 2027.",
       "tools_used": [], "tool_outputs": []}
js_sum = judge_sets.for_target(G, ["correctness_gt", "completeness", "groundedness"])
Ps, fs, called = prompts(SUM, {"expected_answer": "Council voted 7-2 for a new library on Elm Street, opening 2027."},
                         js_sum)
ok("only the 3 applicable judges called", sorted(Ps) == ["completeness", "correctness_gt", "groundedness"], sorted(Ps))
ok("groundedness is told the source is the INPUT itself", "(none: only the input above)" in Ps["groundedness"]
   and "supported by the INPUT or the SOURCE MATERIAL" in Ps["groundedness"] and ARTICLE in Ps["groundedness"])
ok("tool_efficiency / context_relevance / tool_match N/A, score from the 3", fs["final_score"] == 8 and
   {"tool_efficiency", "context_relevance", "tool_match"} <= set(fs["na_judges"]))
_, _, _ = prompts({**SUM, "question": ""}, {}, js_sum)
ok("empty input shown as such", any("(an empty input)" in p for n, p in fake_judges.CALLS))
fake_judges.install()
_, mode = generic.judge_correctness("q?", "a", None, DESC)
ok("no reference answer -> generic fallback, says so", mode == "fallback_no_answer_key"
   and "No reference answer is available" in fake_judges.CALLS[0][1])

print("opting in through targets.yaml:")
REAL = run_suite.load_targets()
cfg = target_config.get("x", {"x": {"module": "m", "function": "f", "judge_set": "generic_v1",
                                    "judges": ["correctness_gt", "groundedness"]}})
js = judge_sets.for_run(run_suite.load_scenarios(), cfg)
ok("judge_set: generic_v1 + judges -> generic_v1 restricted", js["name"] == "generic_v1"
   and js["llm_judges"] == ["correctness_gt", "groundedness"])
ok("an attack suite still gets the attack judges", judge_sets.for_run(
   [{"id": "a", "attack": "injection"}], cfg)["name"] == "adversarial_v1")
ok("unknown judge_set rejected", target_config.validate("x", {"judge_set": "generic_v9"}, judge_sets.ALL_NAMES) != [])
ok("judge_set must be a main set", target_config.validate("x", {"judge_set": "adversarial_v1"}, []) != [])

print("support bot untouched:")
c = target_config.get("configurable")
ok("configurable: no judge_set -> experiment_v2, full", c["judge_set"] is None
   and judge_sets.for_run(run_suite.load_scenarios(), c) is judge_sets.EXPERIMENT_V2)
Pv2, _, called = prompts(TOOL_TRACE, {"expected_answer": "Paris"}, judge_sets.EXPERIMENT_V2)
ok("experiment_v2 still uses the original judge files (no generic call)", "generic" not in called, called)

print("generic_v2: the completeness wording fix (fdd_q10):")
G2 = judge_sets.GENERIC_V2
ok("generic_v2 differs from generic_v1 in completeness ONLY (regrade re-runs just that judge)",
   judge_sets.changed_judges(G, G2) == ["completeness"], judge_sets.changed_judges(G, G2))
ok("a target may pick it", "generic_v2" in judge_sets.MAIN_SETS and judge_sets.BY_NAME["generic_v2"] is G2)
ok("its completeness node is the v2 one; the other judges stay generic v1",
   judge_experiment._node(G2, "completeness") is judge_experiment._gen_completeness_v2
   and judge_experiment._node(G2, "groundedness") is judge_experiment._node(G, "groundedness"))
ok("v1 and v2 runs are never compared as if the same", not judge_sets.same({**G, "judge_model": "m"},
                                                                         {**G2, "judge_model": "m"}))
seen = {}
real_call = generic.call_judge
generic.call_judge = lambda prompt, max_retries=2: seen.setdefault("p", prompt) and None
try:
    generic.judge_completeness_v2("What cybersecurity risks does Apple describe?", "Viruses and ransomware [1].",
                                  ["chunk: ransomware ..."], "Hacking, ransomware, high-profile target.", "a 10-K RAG")
finally:
    generic.call_judge = real_call
p2 = seen["p"]
ok("v2 prompt names the INPUT as the thing to split, not 'the request'",
   "Split the INPUT the system received" in p2 and "Split the request" not in p2)
ok("v2 prompt says: don't grade against these instructions, the system never saw them",
   "do not grade the output against these" in p2 and "never shown them" in p2)
ok("same rules as v1 (honest refusal complete, made-up core not, small extras a small deduction)",
   all(x in p2 for x in ("COMPLETE if the information needed is in neither", "core of the answer is made up",
                          "a small deduction, not a zero")))
ok("the question, output, sources and reference all reach the judge",
   all(x in p2 for x in ("cybersecurity risks", "Viruses and ransomware", "chunk: ransomware", "high-profile target")))
ok("financial_dd uses generic_v2; the summarizer keeps generic_v1 (its baseline was judged with it)",
   target_config.get("financial_dd")["judge_set"] == "generic_v2"
   and target_config.get("summarizer")["judge_set"] == "generic_v1")

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
