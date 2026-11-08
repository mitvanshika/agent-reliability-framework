"""
Tests for trajectory.py (the step recorder) and its use in the support bot and
run_experiment. A scripted LangChain chat model stands in for the LLM; no API calls.
    python tests/test_trajectory.py
"""

import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

import agent_configurable as ac  # noqa: E402
import run_experiment  # noqa: E402
import trajectory  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


class ScriptedChat(BaseChatModel):
    """A real LangChain chat model (so callbacks fire) that replies from a script."""
    script: list

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def script_lookup():
    return [AIMessage(content="", tool_calls=[{"name": "order_lookup", "args": {"order_id": "A1001"}, "id": "c1"}],
                      usage_metadata=usage(900, 25)),
            AIMessage(content="Your order A1001 was delivered.", usage_metadata=usage(1000, 12))]


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))


def run(script, settings=None):
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: ScriptedChat(script=script)
    return ac.run_agent("Where is my order A1001?", settings or BASE["settings"])


print("recording the support bot (scripted LLM):")
out = run(script_lookup())
steps = out["trajectory"]
ok("answer unchanged by the recorder", out["answer"] == "Your order A1001 was delivered." and out["error"] is None)
ok("3 steps in order: llm -> tool -> llm", [(s["type"], s["name"]) for s in steps] ==
   [("llm", "ScriptedChat"), ("tool", "order_lookup"), ("llm", "ScriptedChat")], [(s["type"], s["name"]) for s in steps])
ok("llm step 0 asked for order_lookup(A1001), tokens 900/25", steps[0]["tool_calls"] == [
   {"name": "order_lookup", "args": {"order_id": "A1001"}}] and steps[0]["tokens"] == {"input": 900, "output": 25,
                                                                                       "total": 925}, steps[0])
ok("tool step: args + real output from the tool", steps[1]["args"] == {"order_id": "A1001"}
   and steps[1]["output"].startswith("Order A1001: status=delivered"), steps[1])
ok("final llm step: its text + tokens", steps[2]["output"] == "Your order A1001 was delivered."
   and steps[2]["tokens"]["total"] == 1012)
ok("every step has latency >= 0, a start offset, index, attempt 0, no error", all(
   s["latency_s"] >= 0 and s["started_at"] >= 0 and s["index"] == i and s["attempt"] == 0 and s["error"] is None
   for i, s in enumerate(steps)))
ok("per-step tokens add up to the answer's usage", sum(s["tokens"]["total"] for s in steps if s["type"] == "llm")
   == out["usage"]["total_tokens"], out["usage"])
ok("the trajectory is JSON-serializable (it goes into the trace file)", json.loads(json.dumps(steps)) == steps)

print("rebuilt from the saved conversation (for old runs):")
rebuilt = trajectory.from_messages(out["messages"])
ok("same steps (type, tool, args) as recorded", [(s["type"], s.get("args"), [c["name"] for c in s.get("tool_calls") or []])
   for s in rebuilt] == [(s["type"], s.get("args"), [c["name"] for c in s.get("tool_calls") or []]) for s in steps],
   rebuilt)
ok("rebuilt steps say they have no timing", all(s["latency_s"] is None and s["source"] == "messages" for s in rebuilt))
ok("of_trace prefers the recorded one", trajectory.of_trace({"trajectory": steps, "messages": out["messages"]})[1]
   == "recorded" and trajectory.of_trace({"messages": out["messages"]})[1] == "messages"
   and trajectory.of_trace({}) == ([], "none"))

print("errors are recorded:")
boom = ScriptedChat(script=[])          # pop from empty -> the LLM call raises
ac._graph_cache.clear()
ac._make_llm = lambda model, temperature: boom
ac.time.sleep = lambda s: None
out = ac.run_agent("q", BASE["settings"], max_retries=1)
ok("a failing LLM call -> llm steps with the error, one per attempt", out["error"] is not None
   and [s["attempt"] for s in out["trajectory"]] == [0, 1] and all("IndexError" in s["error"] for s in out["trajectory"]),
   out["trajectory"])

print("the recorder can never break the agent:")


class Broken(trajectory.TrajectoryRecorder):
    def on_tool_end(self, output, **kw):
        raise RuntimeError("recorder bug")


orig = trajectory.TrajectoryRecorder
trajectory.TrajectoryRecorder = Broken
try:
    out = run(script_lookup())
finally:
    trajectory.TrajectoryRecorder = orig
ok("a recorder that raises: the agent still answers", out["answer"] == "Your order A1001 was delivered."
   and out["error"] is None)

print("plain function: the recorder works with any LangChain runnable:")
rec = trajectory.TrajectoryRecorder()
ScriptedChat(script=[AIMessage(content="hi", usage_metadata=usage(3, 1))]).invoke("hello", config={"callbacks": [rec]})
ok("one llm step from a bare chat model call", [(s["type"], s["output"], s["tokens"]["total"]) for s in rec.steps()]
   == [("llm", "hi", 4)])

print("run_experiment keeps it in the trace (additive):")
root = tempfile.mkdtemp()
calls = []


def target(question):
    calls.append(question)
    return {"answer": "x", "trajectory": [{"index": 0, "type": "llm", "name": "m", "latency_s": 0.1}]}


exp = {"name": "t", "description": "", "parent": None, "chain": ["t"], "changes_vs_parent": [], "target": "x",
       "settings": {}}
import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    run_experiment.execute(exp, target, False, [{"id": "s1", "question": "q?"}], root, "t.yaml")
tr = json.load(open(os.path.join(root, "traces", "s1.json")))
ok("trace has the trajectory", tr["trajectory"] == [{"index": 0, "type": "llm", "name": "m", "latency_s": 0.1}])
with contextlib.redirect_stdout(io.StringIO()):
    run_experiment.execute(exp, lambda q: "plain", False, [{"id": "s2", "question": "q?"}], root, "t.yaml")
tr = json.load(open(os.path.join(root, "traces", "s2.json")))
ok("a target that returns a plain string: trajectory None, everything else as before", tr["trajectory"] is None
   and tr["answer"] == "plain")
shutil.rmtree(root, ignore_errors=True)

print("baseline untouched:")
ok("config_hash unchanged (the recorder is not a setting)", config_hash(BASE) == config_hash(
   {"target": "configurable", "settings": BASE["settings"]}))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
