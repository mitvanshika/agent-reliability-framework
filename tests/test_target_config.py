"""
Tests for per-target settings in targets.yaml (target_config.py). LLM faked, no API calls.
    python tests/test_target_config.py
"""

import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import compare  # noqa: E402
import judge_experiment  # noqa: E402
import judge_sets  # noqa: E402
import run_experiment  # noqa: E402
import run_suite  # noqa: E402
import target_config  # noqa: E402
from judges.context_relevance import retrieval_outputs  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


REAL = run_suite.load_targets()

print("defaults = old behaviour:")
c = target_config.get("configurable")
ok("configurable: threshold 7, scenarios/, all judges, its description",
   c["pass_threshold"] == 7 and c["scenarios"] == run_suite.SCENARIOS_DIR and c["judges"] is None
   and c["description"] == REAL["configurable"]["description"])
ok("configurable: escalate_to_human is non-retrieval (moved from judge code to targets.yaml)",
   c["non_retrieval_tools"] == ["escalate_to_human"])
for name in ("strong", "weak", "not-a-target", None):
    d = target_config.get(name)
    ok(f"{name}: all defaults", d == {"description": None, "judge_set": None, "judges": None, "pass_threshold": 7.0,
                                      "scenarios": run_suite.SCENARIOS_DIR, "non_retrieval_tools": []}, d)
ok("every real targets.yaml entry validates", all(
   not target_config.validate(n, e, judge_sets.ALL_NAMES) for n, e in REAL.items()),
   [target_config.validate(n, e, judge_sets.ALL_NAMES) for n, e in REAL.items()])

print("validation:")
bad = target_config.validate("x", {"module": "m", "function": "f", "judges": ["completness"], "pass_threshold": 11,
                                    "scenarios": "", "non_retrieval_tools": "send", "descripton": "typo"},
                             judge_sets.ALL_NAMES)
text = "\n".join(bad)
ok("typo'd key, unknown judge, threshold > 10, empty folder, bad tool list all reported", all(
   w in text for w in ("unknown keys ['descripton']", "unknown ['completness']", "0 to 10", "x.scenarios",
                       "x.non_retrieval_tools")), text)
ok("threshold True rejected (bool is not a number here)",
   target_config.validate("x", {"pass_threshold": True}, []) != [])

print("judges per target:")
V2 = judge_sets.EXPERIMENT_V2
ok("judges None -> the set unchanged (same object)", judge_sets.for_target(V2, None) is V2)
r = judge_sets.for_target(V2, ["correctness_gt", "completeness", "groundedness"])
ok("restricted: 3 LLM judges, no checks, the rest listed not_applicable",
   r["llm_judges"] == ["correctness_gt", "completeness", "groundedness"] and r["checks"] == []
   and r["not_applicable"] == ["tool_efficiency", "context_relevance", "tool_match"], r)
ok("versions kept (still completeness v2)", r["versions"] == {"completeness": "v2"} and r["name"] == "experiment_v2")
ok("restricted set != full set (compare won't mix them)", not judge_sets.same(r, V2))
ok("attack suite keeps its own judges", judge_sets.for_target(judge_sets.ADVERSARIAL, ["correctness_gt"])
   is judge_sets.ADVERSARIAL)
try:
    judge_sets.for_target(V2, ["safety"])
    ok("no applicable judge -> error", False)
except ValueError as e:
    ok("no applicable judge -> error", "none of the target's judges" in str(e))

TRACE = {"scenario_id": "x", "question": "Summarise: The cat sat.", "answer": "A cat sat.", "tools_used": [],
         "tool_outputs": []}
fake_judges.install(lambda name, prompt: (8, 0.9))
fields = judge_experiment.evaluate(TRACE, {"expected_answer": "A cat sat."}, r)
ok("only the 3 applicable judges were called", sorted(n for n, _ in fake_judges.CALLS)
   == ["completeness", "correctness_gt", "groundedness"], fake_judges.CALLS and [n for n, _ in fake_judges.CALLS])
ok("the others recorded as N/A with the reason, listed in na_judges",
   all(fields[f"{n}_result"].get("na") and "targets.yaml" in fields[f"{n}_result"]["reason"]
       for n in ("tool_efficiency", "context_relevance", "tool_match"))
   and {"tool_efficiency", "context_relevance", "tool_match"} <= set(fields["na_judges"]))
ok("final_score from the applicable judges only (all 8 -> 8)", fields["final_score"] == 8)
fake_judges.install(lambda name, prompt: (8, 0.9))
full = judge_experiment.evaluate({**TRACE, "tools_used": ["faq_search"], "tool_outputs": ["text"]}, {}, V2)
ok("the full set still calls all 5 (graph cache keyed by judges, not just name)",
   len({n for n, _ in fake_judges.CALLS}) == 5, sorted({n for n, _ in fake_judges.CALLS}))

print("non_retrieval_tools per target:")
uses, outs = ["faq_search", "escalate_to_human"], ["faq text", "Escalated to human support. Reason: x"]
ok("None (no target) -> legacy behaviour, escalate ignored", retrieval_outputs(uses, outs) == ["faq text"])
ok("configurable's list -> same result", retrieval_outputs(uses, outs, ["escalate_to_human"]) == ["faq text"])
ok("a target with [] -> every tool's output counts", retrieval_outputs(uses, outs, []) == outs)
ok("a target with its own non-retrieval tool", retrieval_outputs(["search", "send_email"], ["hit", "sent"],
   ["send_email"]) == ["hit"])

print("pass_threshold per target (compare / promote / diagnose use this):")
root = tempfile.mkdtemp()
d = os.path.join(root, "run")
os.makedirs(os.path.join(d, "traces"))
os.makedirs(os.path.join(d, "judged"))
json.dump({"experiment": "e", "scenario_ids": ["a", "b"], "resolved_config": {"target": "thr_target", "settings": {}}},
          open(os.path.join(d, "run_meta.json"), "w"))
for sid, score in (("a", 8.0), ("b", 9.5)):
    t = {"scenario_id": sid, "question": "q", "answer": "x", "error": None, "timestamp": "t"}
    json.dump(t, open(os.path.join(d, "traces", f"{sid}.json"), "w"))
    json.dump({**t, "final_score": score, "judge_error": None, "judge_set": V2},
              open(os.path.join(d, "judged", f"{sid}.json"), "w"))
run_suite.load_targets = lambda: {**REAL, "thr_target": {"module": "m", "function": "f", "pass_threshold": 9}}
_, recs = compare.load_run_records(d)
ok("threshold 9: 8.0 fails, 9.5 passes", (recs["a"]["passed"], recs["b"]["passed"]) == (False, True))
run_suite.load_targets = lambda: {**REAL, "thr_target": {"module": "m", "function": "f"}}
_, recs = compare.load_run_records(d)
ok("default 7: both pass (unchanged behaviour)", recs["a"]["passed"] and recs["b"]["passed"])
import diagnose  # noqa: E402
ok("diagnose's fine-tune signal uses the same pass flag", diagnose._failed({"status": "ok", "passed": False,
   "final_score": 8.0}) and not diagnose._failed({"status": "ok", "passed": True, "final_score": 6.0}))

print("scenarios folder per target:")
run_suite.load_targets = lambda: {**REAL, "sumtarget": {"module": "m", "function": "f",
                                                         "scenarios": "examples/summarizer/scenarios"}}
ok("its own folder is its MAIN suite (baselines/<target>.json)",
   run_experiment.suite_name("examples/summarizer/scenarios", "sumtarget") == "main")
ok("scenarios/ is still main for the support bot",
   run_experiment.suite_name(None, "configurable") == "main" == run_experiment.suite_name("scenarios", "configurable"))
ok("the attack suite is still 'adversarial'",
   run_experiment.suite_name("scenarios_adversarial/configurable", "configurable") == "adversarial")
run_suite.load_targets = lambda: REAL

print("support bot untouched:")
from experiment_config import config_hash, load_experiment  # noqa: E402
base = load_experiment(os.path.join("experiments", "baseline.yaml"))
ok("baseline settings / config_hash unchanged (targets.yaml is not part of the experiment)",
   config_hash(base) == config_hash({"target": "configurable", "settings": base["settings"]}))
ok("configurable judged with the full experiment_v2 (no not_applicable)",
   judge_sets.for_target(judge_sets.for_scenarios(run_suite.load_scenarios()), c["judges"]) is V2)

shutil.rmtree(root)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
