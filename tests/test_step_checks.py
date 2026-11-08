"""
Tests for step_checks.py (no LLM, no API calls).
    python tests/test_step_checks.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import step_checks as sc  # noqa: E402
import target_config  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def llm(i, calls=(), lat=None, err=None):
    return {"index": i, "type": "llm", "name": "m", "tool_calls": [{"name": n, "args": a} for n, a in calls],
            "latency_s": lat, "error": err}


def tool(i, name, args, out, lat=None, err=None):
    return {"index": i, "type": "tool", "name": name, "args": args, "output": out, "latency_s": lat, "error": err}


def checks(steps, **kw):
    return sorted(f["check"] for f in sc.check_steps(steps, **kw))


print("each check fires on its pattern and only there:")
clean = [llm(0), tool(1, "order_lookup", {"order_id": "A1"}, "Order A1: delivered"), llm(2)]
ok("a clean answer -> no findings", checks(clean) == [])
rep = [llm(0), tool(1, "order_lookup", {"order_id": "A1"}, "Order A1"), llm(2),
       tool(3, "order_lookup", {"order_id": " A1 "}, "Order A1"), llm(4)]
f = sc.check_steps(rep)
ok("repeated_call: same tool + same normalized args (' A1 ' == 'A1')", "repeated_call" in checks(rep)
   and next(x for x in f if x["check"] == "repeated_call")["count"] == 2)
same = [tool(0, "search", {"q": "shipping cost"}, "Shipping takes 5 days"),
        tool(1, "search", {"q": "shipping cost amount"}, "Shipping takes 5 days"),
        tool(2, "search", {"q": "standard shipping cost"}, "Shipping takes 5 days")]
f = sc.check_steps(same)
ok("same_result_repeat: 3 different queries, the same output -> 2 calls added nothing",
   checks(same) == ["same_result_repeat"] and "2 call(s) added nothing" in f[0]["detail"])
differ = [tool(0, "search", {"q": "a"}, "No matching FAQ"), tool(1, "search", {"q": "customs"}, "Customs fees: ...")]
ok("rephrasing that FOUND something is not a finding", checks(differ) == [])
empty = [tool(0, "search", {"q": "gift"}, "No matching FAQ found"), tool(1, "search", {"q": "gift"}, "No matching FAQ found")]
ok("empty_then_repeat: nothing found, then the identical call again", "empty_then_repeat" in checks(empty))
many = [llm(i) for i in range(5)] + [tool(5 + i, "t", {"i": i}, f"r{i}") for i in range(9)]
ok("too_many_steps: 9 tool calls > 8", "too_many_steps" in checks(many))
ok("...limit per call: 9 calls under max_steps=10 is fine", "too_many_steps" not in checks(many, max_steps=10))
ok("tool_error and llm_error", checks([llm(0, err="RateLimitError: x"), tool(1, "t", {}, "", err="KeyError: db")])
   == ["llm_error", "tool_error"])
slow = [llm(0, lat=2.0), tool(1, "t", {}, "x", lat=12.5), llm(2, lat=1.0)]
ok("slow_step: 12.5 s > 10 s", checks(slow) == ["slow_step"])
s = sc.slowest(slow)
ok("slowest step + its share of the answer's step time", s["step"] == 1 and s["latency_s"] == 12.5
   and s["share"] == round(12.5 / 15.5, 3), s)
ok("no timing -> no slowest step (never invented)", sc.slowest(clean) is None)

print("on a run folder (recorded trajectory + rebuilt messages):")
root = tempfile.mkdtemp()
d = os.path.join(root, "run")
os.makedirs(os.path.join(d, "traces"))
json.dump({"experiment": "e", "resolved_config": {"target": "configurable"}}, open(os.path.join(d, "run_meta.json"), "w"))
json.dump({"scenario_id": "a", "timestamp": "t", "latency_s": 3.0, "trajectory": slow}, open(os.path.join(d, "traces", "a.json"), "w"))
msgs = [{"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "search", "arguments": json.dumps({"q": "x"})}}]},
        {"role": "tool", "tool_call_id": "1", "content": "same page"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "type": "function", "function": {
            "name": "search", "arguments": json.dumps({"q": "x y"})}}]},
        {"role": "tool", "tool_call_id": "2", "content": "same page"},
        {"role": "assistant", "content": "answer"}]
json.dump({"scenario_id": "b", "timestamp": "t", "latency_s": 9.0, "messages": msgs}, open(os.path.join(d, "traces", "b.json"), "w"))
json.dump({"scenario_id": "c", "timestamp": "t", "latency_s": 1.0}, open(os.path.join(d, "traces", "c.json"), "w"))
r = sc.check_run(d)
by = {x["scenario_id"]: x for x in r["rows"]}
ok("sources: recorded / messages / none", (by["a"]["source"], by["b"]["source"], by["c"]["source"])
   == ("recorded", "messages", "none"))
ok("findings per answer", [f["check"] for f in by["a"]["findings"]] == ["slow_step"]
   and [f["check"] for f in by["b"]["findings"]] == ["same_result_repeat"] and by["c"]["findings"] == [])
ok("estimated time per LLM turn only where turns exist (9 s / 3 turns)", by["b"]["est_s_per_llm_turn"] == 3.0
   and by["c"]["est_s_per_llm_turn"] is None)
ok("max_steps from targets.yaml default (8)", r["max_steps"] == 8)
ok("step_checks.json saved", json.load(open(os.path.join(d, "step_checks.json")))["counts"] == {
   "slow_step": 1, "same_result_repeat": 1})
out = io.StringIO()
with contextlib.redirect_stdout(out):
    sc.main([d])
ok("report: findings listed, slowest recorded step shown", "same_result_repeat" in out.getvalue()
   and "slowest step: a step 1" in out.getvalue(), out.getvalue()[:400])
shutil.rmtree(root, ignore_errors=True)

print("targets.yaml max_steps:")
ok("valid / invalid values", target_config.validate("x", {"max_steps": 12}, []) == []
   and target_config.validate("x", {"max_steps": 0}, []) != [] and target_config.validate("x", {"max_steps": True}, []) != [])

print("the real saved baseline (if present): the known finding is still found:")
real = os.path.join("runs", "baseline_20261106_101856")
if os.path.isdir(os.path.join(real, "traces")):
    tmp = tempfile.mkdtemp()
    shutil.copytree(os.path.join(real, "traces"), os.path.join(tmp, "traces"))
    shutil.copy(os.path.join(real, "run_meta.json"), tmp)
    r = sc.check_run(tmp)
    g = next(x for x in r["rows"] if x["scenario_id"] == "gen_012")
    ok("gen_012: faq_search 3x, same output -> same_result_repeat", any(
       f["check"] == "same_result_repeat" and f["count"] == 3 for f in g["findings"]), g["findings"])
    ok("31 answers, all rebuilt from messages, no timing invented", r["sources"] == {"messages": 31}
       and all(x["slowest_step"] is None for x in r["rows"]))
    shutil.rmtree(tmp, ignore_errors=True)
else:
    print("  (skipped: runs/ not present)")

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
