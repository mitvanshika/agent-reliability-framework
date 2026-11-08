"""
Tests for agent_configurable.py with the LLM mocked (no API calls).
    python tests/test_agent_configurable.py
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import httpx  # noqa: E402
import groq  # noqa: E402
from langchain_core.messages import AIMessage, SystemMessage  # noqa: E402

import agent_configurable as ac  # noqa: E402
import test_agent  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


BASE = {"model": "openai/gpt-oss-120b", "temperature": 0, "prompt": "prompts/v1.txt",
        "docstrings": "docstrings/detailed.yaml", "retrieval": {"method": "substring"}, "recursion_limit": 25}


class FakeLLM:
    """Records what it was built with; replays scripted AIMessages."""
    instances = []

    def __init__(self, model, temperature, script):
        self.model, self.temperature, self.script = model, temperature, list(script)
        self.bound_tools, self.seen = None, []
        FakeLLM.instances.append(self)

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="final")
        if isinstance(step, Exception):
            raise step
        return step


def install(script):
    FakeLLM.instances.clear()
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: FakeLLM(model, temperature, script)


def usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


TOOL_CALL = AIMessage(content="", tool_calls=[{"name": "order_lookup", "args": {"order_id": "A1002"}, "id": "c1"}],
                      usage_metadata=usage(100, 10))
FINAL = AIMessage(content="Your order A1002 was delivered.", usage_metadata=usage(150, 25))

print("prompt, docstrings, model, temperature:")
install([TOOL_CALL, FINAL])
r = ac.run_agent("Where is A1002?", BASE)
llm = FakeLLM.instances[0]
first_call = llm.seen[0]
ok("system message is prompts/v1.txt text", isinstance(first_call[0], SystemMessage)
   and first_call[0].content == test_agent.SYSTEM_PROMPT)
ok("model + temperature passed through", (llm.model, llm.temperature) == ("openai/gpt-oss-120b", 0.0),
   (llm.model, llm.temperature))
ok("tool descriptions from docstrings/detailed.yaml",
   [t.description for t in llm.bound_tools] == [t.description for t in test_agent.tools])
ok("tools reuse test_agent's functions (no re-implementation)",
   all(c.func is o.func for c, o in zip(llm.bound_tools, test_agent.tools)))
ok("the real tool ran (store.db)", any("A1002" in o for o in r["tool_outputs"]), r["tool_outputs"])
ok("usage summed over 2 LLM calls", r["usage"] == usage(250, 35), r["usage"])
ok("same dict shape as test_agent.run_agent + usage + messages + trajectory (additive)",
   set(r) == {"answer", "tools_used", "tool_outputs", "error", "error_type", "retries_needed", "usage", "messages",
              "trajectory"}, sorted(r))
roles = [m["role"] for m in r["messages"]]
call = r["messages"][2]["tool_calls"][0]
ok("messages: full chat history incl. tool call + tool result", roles == ["system", "user", "assistant", "tool", "assistant"]
   and call["function"] == {"name": "order_lookup", "arguments": '{"order_id": "A1002"}'}
   and r["messages"][3]["tool_call_id"] == call["id"] and r["messages"][-1]["content"] == FINAL.content, roles)
ok("answer/tools/error", r["answer"] == FINAL.content and r["tools_used"] == ["order_lookup"]
   and r["error"] is None and r["error_type"] is None)

print("vague docstrings + weak prompt:")
install([AIMessage(content="ok")])
ac.run_agent("q", {**BASE, "prompt": "prompts/weak.txt", "docstrings": "docstrings/vague.yaml"})
import agent_v2_weak  # noqa: E402
llm = FakeLLM.instances[0]
ok("weak prompt is the system message", llm.seen[0][0].content == agent_v2_weak.SYSTEM_PROMPT)
ok("vague descriptions bound", [t.description for t in llm.bound_tools] == [t.description for t in agent_v2_weak.tools])
ok("original test_agent tools untouched", test_agent.order_lookup.description.startswith("Looks up an order"))

print("usage unavailable:")
install([AIMessage(content="no usage reported")])
ok("usage None when no AIMessage reports it", ac.run_agent("q", BASE)["usage"] is None)

print("recursion_limit passed through:")
loop = [AIMessage(content="", tool_calls=[{"name": "faq_search", "args": {"query": "shipping"}, "id": f"c{i}"}])
        for i in range(50)]
install(loop)
r = ac.run_agent("q", {**BASE, "recursion_limit": 5})
calls = len(FakeLLM.instances[0].seen)
ok("loop stopped by recursion_limit=5", r["error_type"] == "loop_limit" and calls <= 3, f"LLM calls={calls}")
install(loop)
r = ac.run_agent("q", {**BASE, "recursion_limit": 11})
calls11 = len(FakeLLM.instances[0].seen)
ok("higher limit -> more steps", calls11 > calls, f"LLM calls={calls11}")

print("errors use agent_retry classification:")
req = httpx.Request("POST", "https://api.groq.com/x")
install([groq.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)])
r = ac.run_agent("q", BASE)
ok("auth -> no retry", r["error_type"] == "auth" and r["retries_needed"] == 0 and r["usage"] is None, r["error_type"])
ac.time.sleep = lambda s: None
install([groq.RateLimitError("429", response=httpx.Response(429, request=req), body=None), FINAL])
r = ac.run_agent("q", BASE)
ok("rate limit -> retried, then success", r["error"] is None and r["retries_needed"] == 1)

print("settings validation:")


def expect_error(label, settings, must_contain):
    try:
        ac.validate_settings(settings)
        ok(label, False, "no error")
    except ac.SettingsError as e:
        ok(label, all(s in str(e) for s in must_contain), str(e).replace("\n", " | "))


expect_error("unknown key with suggestion", {**BASE, "temperatur": 0},
             ["settings.temperatur: unknown key", "did you mean 'temperature'"])
missing = dict(BASE)
del missing["recursion_limit"]
expect_error("missing key", missing, ["settings.recursion_limit: required but missing"])
expect_error("unknown retrieval method", {**BASE, "retrieval": {"method": "vector_db"}},
             ["settings.retrieval.method", "'substring', 'bm25' or 'embeddings'"])
expect_error("unknown nested key", {**BASE, "retrieval": {"method": "substring", "topk": 3}},
             ["settings.retrieval.topk: unknown key"])
expect_error("temperature out of range", {**BASE, "temperature": 5}, ["settings.temperature"])
expect_error("prompt file missing", {**BASE, "prompt": "prompts/nope.txt"}, ["file not found: prompts/nope.txt"])
bad = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
bad.write('order_lookup: "x"\nrefund_calculator: "x"\nfaq_search: "x"\nescalate: "x"\n')
bad.close()
expect_error("docstrings must cover exactly the 4 tools", {**BASE, "docstrings": bad.name},
             ["missing ['escalate_to_human']", "unknown ['escalate']"])
try:
    ac.run_agent("q")
    ok("run_agent without settings -> clear error", False)
except ac.SettingsError as e:
    ok("run_agent without settings -> clear error", "run_experiment.py" in str(e), str(e))

print("graph cache:")
install([AIMessage(content="a"), AIMessage(content="b")])
ac.run_agent("q1", BASE)
ac.run_agent("q2", BASE)
ok("built once for the same settings", len(FakeLLM.instances) == 1, f"builds={len(FakeLLM.instances)}")

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
