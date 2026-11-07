"""
Deterministic tests for validated_tool.py and tool_validation (no LLM, no API calls).
    python tests/test_validated_tool.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

from langchain_core.messages import AIMessage  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agent_configurable as ac  # noqa: E402
import test_agent  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402
from validated_tool import validated_function, validated_tool  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))["settings"]


class Fake:
    def __init__(self):
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="ok")


def tools_for(settings):
    fake = Fake()
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: fake
    ac.run_agent("q", settings)
    return {t.name: t for t in fake.bound}


OFF = tools_for(BASE)
STRICT = tools_for({**BASE, "tool_validation": "strict"})


def call(tool, args):
    try:
        return tool.invoke(args)
    except Exception as e:
        return f"RAISED {type(e).__name__}"


print("valid inputs: identical output with validation off and strict:")
valid = [("order_lookup", {"order_id": "A1001"}), ("order_lookup", {"order_id": "Z9999"}),
         ("order_lookup", {"order_id": "a1002"}),
         ("refund_calculator", {"order_total": 89.99, "days_since_purchase": 12}),
         ("refund_calculator", {"order_total": 45.5, "days_since_purchase": 45}),
         ("refund_calculator", {"order_total": 0, "days_since_purchase": 0}),
         ("faq_search", {"query": "return policy"}), ("faq_search", {"query": "weather"}),
         ("escalate_to_human", {"reason": "customer wants a manager"})]
for name, args in valid:
    a, b = call(OFF[name], args), call(STRICT[name], args)
    ok(f"{name}{args}", a == b and not a.startswith("Error"), b[:60])

print("each invalid input -> the right clean error, never raises:")
invalid = [
    ("order_lookup", {"order_id": "order A1001"}, "order_id", "like A1234"),
    ("order_lookup", {"order_id": ""}, "order_id", "like A1234"),
    ("order_lookup", {}, "order_id", "required but missing"),
    ("refund_calculator", {"order_total": -50, "days_since_purchase": 12}, "order_total", "greater than or equal to 0"),
    ("refund_calculator", {"order_total": 50, "days_since_purchase": -3}, "days_since_purchase", "greater than or equal to 0"),
    ("refund_calculator", {"order_total": "abc", "days_since_purchase": 3}, "order_total", "valid number"),
    ("faq_search", {"query": ""}, "query", "at least 1 character"),
    ("faq_search", {"query": "   "}, "query", "at least 1 character"),
    ("faq_search", {"query": "x" * 600}, "query", "at most 500 characters"),
    ("escalate_to_human", {"reason": ""}, "reason", "at least 1 character"),
]
for name, args, field, words in invalid:
    out = call(STRICT[name], args)
    shown = {k: (v if len(str(v)) < 20 else f"{str(v)[:5]}...({len(v)})") for k, v in args.items()}
    ok(f"{name}{shown}", out.startswith(f"Error: invalid input for {name}: {field}") and words in out
       and out.endswith("Fix the input and try again."), out[:120])
ok("off: the old nonsense is still what 'off' does (refund of $-50.0)",
   call(OFF["refund_calculator"], {"order_total": -50, "days_since_purchase": 3}) == "Eligible for full refund of $-50.0")

print("a tool that raises inside -> clean error, run not crashed:")


def boom(order_id):
    raise KeyError("database down")


boom_tool = validated_tool(test_agent.order_lookup.model_copy(update={"func": boom}), ac.TOOL_CONSTRAINTS["order_lookup"])
out = call(boom_tool, {"order_id": "A1001"})
ok("exception -> 'Error: order_lookup failed (KeyError: ...)'", out.startswith("Error: order_lookup failed (KeyError"), out)

print("the model sees exactly the same tools:")
for name in OFF:
    ok(f"{name}: same name, description, argument schema", (STRICT[name].name, STRICT[name].description,
       STRICT[name].args) == (OFF[name].name, OFF[name].description, OFF[name].args))

print("default off = old behaviour:")
ok("tool_validation defaults to off", ac.AgentSettings(**BASE).tool_validation == "off")
ok("baseline tools run test_agent's own functions (no wrapper)", all(OFF[t.name].func is t.func for t in test_agent.tools))
ok("strict tools are wrapped", all(getattr(STRICT[n].func, "__wrapped__", None) is OFF[n].func for n in OFF))
ok("baseline settings + config_hash untouched", "tool_validation" not in BASE
   and config_hash({"target": "configurable", "settings": BASE}) == config_hash(load_experiment(
       os.path.join("experiments", "baseline.yaml"))))
try:
    ac.validate_settings({**BASE, "tool_validation": "on"})
    ok("unknown value rejected", False)
except ac.SettingsError as e:
    ok("unknown value rejected", "settings.tool_validation" in str(e) and "'off' or 'strict'" in str(e))
strict_bm25 = tools_for({**BASE, "tool_validation": "strict", "retrieval": {"method": "bm25"}})
ok("combines with bm25 retrieval: validated, and still finds 'import fees'",
   call(strict_bm25["faq_search"], {"query": ""}).startswith("Error: invalid input")
   and "customs" in call(strict_bm25["faq_search"], {"query": "import fees"}))

print("generic: works for plain functions too:")


class AddArgs(BaseModel):
    a: int = Field(ge=0)
    b: int = Field(ge=0)


safe_add = validated_function(lambda a, b: a + b, AddArgs, name="add")
ok("valid -> same result", safe_add(a=2, b=3) == 5)
ok("invalid -> clean error", safe_add(a=-1, b=3).startswith("Error: invalid input for add: a:"))
ok("missing -> clean error", "b: required but missing" in safe_add(a=1))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
