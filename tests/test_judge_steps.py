"""
Tests for the step-level judge (judges/step.py, judge_steps.py, judge set steps_v1). LLM faked, no API calls.
    python tests/test_judge_steps.py
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import judge_sets  # noqa: E402
import judge_steps  # noqa: E402
import rate_limit  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def msg_trace(sid, calls, question="Where is A1001 and how much does shipping cost?"):
    """A saved conversation: each (tool, args, output) is one assistant turn + its tool result."""
    msgs = [{"role": "user", "content": question}]
    for i, (name, args, out) in enumerate(calls):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": str(i), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": out})
    msgs.append({"role": "assistant", "content": "final answer"})
    return {"scenario_id": sid, "question": question, "answer": "final answer", "error": None, "timestamp": "t",
            "messages": msgs}


T = msg_trace("gen_012", [("order_lookup", {"order_id": "A1001"}, "Order A1001: delivered"),
                          ("faq_search", {"query": "shipping cost"}, "Standard shipping takes 5-7 days"),
                          ("faq_search", {"query": "shipping cost amount"}, "Standard shipping takes 5-7 days")])


def score(name, prompt):
    """Stand-in judge: the near-repeat (2nd faq_search) is avoidable -> 5, the rest 10."""
    this = prompt.split("THIS step:", 1)[1]
    return (5, 0.8) if '"shipping cost amount"' in this else (10, 0.9)


print("the step judge prompt:")
fake_judges.install(score)
res = judge_steps.judge_trace_steps(T, "a store's support assistant")
prompts = fake_judges.called("step")
ok("one call per TOOL step (3), none for model turns", len(prompts) == 3, len(prompts))
third = prompts[2]
ok("it sees the request, the description and every earlier step with its output",
   "Where is A1001" in third and "a store's support assistant" in third
   and '1. model turn: asked for order_lookup' in third and 'order_lookup({"order_id": "A1001"}) -> Order A1001' in third
   and 'faq_search({"query": "shipping cost"}) -> Standard shipping' in third)
ok("and THIS step's call + output", 'called faq_search with arguments {"query": "shipping cost amount"}' in third
   and "What it returned: Standard shipping takes 5-7 days" in third)
ok("asks 'necessary and correct'; a step that finds nothing is still correct",
   "Was this step necessary and correct?" in third and "correctly finds that the information does not exist" in third)
ok("first tool step: its history is just the model turn that asked for it",
   prompts[0].split("Steps before this one:\n", 1)[1].split("\nTHIS step", 1)[0] == "1. model turn: asked for order_lookup")
from judges.step import judge_step  # noqa: E402
fake_judges.install()
judge_step("q?", [], {"type": "tool", "name": "t", "args": {}, "output": "x"})
ok("no history at all -> '(none: this is the first step)'", "(none: this is the first step)" in fake_judges.called("step")[0])
fake_judges.install(score)
src = open(os.path.join("judges", "step.py"), encoding="utf-8").read()
words = r"\b(store|refunds?|faq|escalat\w*|customers?|shipping|orders|gift|warranty)\b"
ok("the template is domain-neutral (no support-bot words)", not re.findall(words, src, re.I), re.findall(words, src, re.I))

print("scores per answer:")
ok("scores per step", [s["score"] for s in res["steps"]] == [10, 10, 5] and [s["name"] for s in res["steps"]]
   == ["order_lookup", "faq_search", "faq_search"])
ok("step score: min 5, mean 8.33, weakest = the near-repeat (index 5 in the step list)",
   res["step_score_min"] == 5 and res["step_score_mean"] == 8.33 and res["weakest_step"]["args"]
   == {"query": "shipping cost amount"} and res["weakest_step"]["index"] == 5, res["weakest_step"])
na = judge_steps.judge_trace_steps(msg_trace("x", []))
ok("no tool steps -> N/A, no call", na["na"] and na["reason"] == "no tool steps")

print("the judge set:")
ok("steps_v1 is separate: not a main set, not selectable as a target judge_set", judge_sets.STEPS_V1["name"]
   == "steps_v1" and "steps_v1" not in judge_sets.MAIN_SETS and "steps_v1" not in judge_sets.BY_NAME)
ok("the default is untouched", judge_sets.for_scenarios(run_suite.load_scenarios())["name"] == "experiment_v2")

print("judge_steps.py on a run (separate folder, resumable):")
root = tempfile.mkdtemp()
d = os.path.join(root, "run")
os.makedirs(os.path.join(d, "traces"))
os.makedirs(os.path.join(d, "judged"))
open(os.path.join(d, "judged", "keep.json"), "w").write('{"untouched": true}')
json.dump({"experiment": "e", "resolved_config": {"target": "configurable"}}, open(os.path.join(d, "run_meta.json"), "w"))
for t in (T, msg_trace("normal_01", [("faq_search", {"query": "return policy"}, "30 days")]), msg_trace("none_01", [])):
    json.dump(t, open(os.path.join(d, "traces", f"{t['scenario_id']}.json"), "w"))
crashed = {**msg_trace("crash", []), "error": "boom"}
json.dump(crashed, open(os.path.join(d, "traces", "crash.json"), "w"))
fake_judges.install(score)
with contextlib.redirect_stdout(io.StringIO()) as out:
    s = judge_steps.main([d, "--yes"])
ok("estimate = tool steps (3 + 1 = 4), crashed answers skipped", "Estimated judge API calls: 4" in out.getvalue()
   and len(fake_judges.CALLS) == 4, out.getvalue()[:300])
ok("results in judged_steps/, judged/ untouched", sorted(os.listdir(os.path.join(d, "judged_steps")))
   == ["gen_012.json", "none_01.json", "normal_01.json"] and os.listdir(os.path.join(d, "judged")) == ["keep.json"])
ok("steps_summary.json: mean of answer means, weakest step first", s["answers"] == 2 and s["mean_step_score"]
   == round((8.33 + 10) / 2, 2) and s["weakest"][0]["scenario_id"] == "gen_012" and s["low_steps"] == 0, s)
rec = json.load(open(os.path.join(d, "judged_steps", "gen_012.json")))
ok("each file records the judge set", rec["judge_set"]["name"] == "steps_v1")
fake_judges.install(score)
with contextlib.redirect_stdout(io.StringIO()):
    judge_steps.main([d, "--yes"])
ok("re-run: everything already judged -> 0 calls", fake_judges.CALLS == [])

print("daily limit -> stop, keep what's done:")
shutil.rmtree(os.path.join(d, "judged_steps"))
n = {"k": 0}


def quota(name, prompt):
    n["k"] += 1
    if n["k"] == 2:
        rate_limit.STATS.daily_limit = True
        raise RuntimeError("Rate limit reached on tokens per day (TPD)")
    return (9, 0.9)


rate_limit.STATS.reset()
fake_judges.install(quota)
code = None
with contextlib.redirect_stdout(io.StringIO()):
    try:
        judge_steps.main([d, "--yes"])
    except SystemExit as e:
        code = e.code
ok("stops with a resume command", isinstance(code, str) and "Continue later: python judge_steps.py" in code, code)
rate_limit.STATS.reset()
shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
