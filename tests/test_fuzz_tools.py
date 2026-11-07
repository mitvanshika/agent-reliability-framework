"""
Deterministic tests for fuzz_tools.py (no LLM, no API calls).
    python tests/test_fuzz_tools.py
"""

import contextlib
import io
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agent_configurable as ac  # noqa: E402
import fuzz_tools  # noqa: E402
import test_agent  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402
from validated_tool import validated_function, validated_tool  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        code = fn(*a)
    return code, out.getvalue(), err.getvalue()


@tool
def repeat(text: str, times: int) -> str:
    """Repeat text."""
    return text * times


class RepeatArgs(BaseModel):
    text: str = Field(min_length=1)
    times: int = Field(ge=0, le=10)


print("inputs come from the argument types (nothing tool-specific):")
labels = [label for label, _ in fuzz_tools.cases(repeat)]
ok("string arg gets the string set", all(f"text={x}" in labels for x in ("empty", "10k chars", "None", "int 123")))
ok("integer arg gets the integer set", all(f"times={x}" in labels for x in ("-1", "1e12", "1.5", "'abc'")))
ok("missing each required arg, all missing, extra arg",
   {"missing text", "missing times", "all args missing", "unknown extra arg"} <= set(labels))
bad_text = dict(fuzz_tools.cases(repeat))["text=empty"]
ok("other args get neutral values", bad_text == {"text": "", "times": 1}, bad_text)
ok("plain functions work too", [l for l, _ in fuzz_tools.cases(lambda q: q)][:2] == ["q=empty", "q=whitespace"])

print("classification:")
res = fuzz_tools.fuzz([repeat])["repeat"]
by = {label: status for label, status, _ in res}
ok("wrong type -> CRASH", by["times='abc'"] == "CRASH")
ok("missing arg -> CRASH", by["missing times"] == "CRASH")
ok("bad but typed value -> ACCEPTED", by["times=-1"] == "ACCEPTED")
safe = fuzz_tools.fuzz([validated_tool(repeat, RepeatArgs)])["repeat"]
ok("validated_tool -> no CRASH at all", all(s != "CRASH" for _, s, _ in safe))
ok("validated_tool -> -1 and 1e12 now CLEAN", {l: s for l, s, _ in safe}["times=-1"] == "CLEAN"
   and {l: s for l, s, _ in safe}["times=1e12"] == "CLEAN")


def boom(order_id: str):
    raise KeyError("db down")


ok("an exception inside a plain function -> CRASH",
   fuzz_tools.fuzz([boom])["boom"][0][1] == "CRASH")
ok("...and CLEAN once wrapped", all(s == "CLEAN" for _, s, _ in fuzz_tools.fuzz(
    [validated_function(boom, ac.TOOL_CONSTRAINTS["order_lookup"], name="boom")])["validated_boom"]))

print("on the support bot:")
code, out, _ = quiet(fuzz_tools.main, ["configurable"])
ok("baseline (validation off): crashes -> exit 1", code == 1 and "FAIL" in out and "TOTAL: 17 crash" in out,
   out.strip().splitlines()[-3])
ok("baseline: accepts a negative refund (the warning column)", "Eligible for full refund of $-1.0" in out)
tmp = tempfile.mkdtemp()
strict_yaml = os.path.join(tmp, "tmp_strict.yaml")
open(strict_yaml, "w").write(open(os.path.join("experiments", "baseline.yaml")).read()
                             .replace("name: baseline", "name: tmp_strict") + "  tool_validation: strict\n")
code, out, _ = quiet(fuzz_tools.main, ["configurable", "--experiment", strict_yaml])
ok("strict: 0 crashes -> exit 0", code == 0 and "TOTAL: 0 crash" in out and "PASS" in out)

print("opt-in and errors:")
code, _, err = quiet(fuzz_tools.main, ["strong"])
ok("target without tools: -> refused, exit 2", code == 2 and "hasn't opted in" in err, err.strip()[:90])
code, _, err = quiet(fuzz_tools.main, ["nope"])
ok("unknown target -> exit 2", code == 2 and "Unknown target" in err)
bad_yaml = os.path.join(tmp, "wrong_name.yaml")
open(bad_yaml, "w").write("name: other\ntarget: configurable\nsettings: {}\n")
code, _, err = quiet(fuzz_tools.main, ["configurable", "--experiment", bad_yaml])
ok("bad experiment file -> clean message, exit 2", code == 2 and "must equal the file name" in err, err.strip()[:90])
try:
    fuzz_tools.load_tools("configurable", targets={"configurable": {"tools": "test_agent:tools_getter_missing"}})
    ok("missing getter raises", False)
except AttributeError:
    ok("missing getter raises", True)

print("build_tools == what the agent binds; baseline untouched:")


class Fake:
    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="ok")


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))["settings"]
for settings in (BASE, {**BASE, "tool_validation": "strict"}, {**BASE, "retrieval": {"method": "bm25"}}):
    fake = Fake()
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: fake
    ac.run_agent("q", settings)
    built = ac.build_tools(settings)
    # bm25 builds a fresh search closure each time, so compare what the model sees + the output
    same = [(t.name, t.description, t.args) for t in built] == [(t.name, t.description, t.args) for t in fake.bound]
    same_out = [t.invoke({"query": "import fees"}) for t in built if t.name == "faq_search"] == \
               [t.invoke({"query": "import fees"}) for t in fake.bound if t.name == "faq_search"]
    ok(f"build_tools matches bound tools ({settings.get('tool_validation', 'off')}, {settings['retrieval']['method']})",
       same and same_out)
ok("baseline tools are still test_agent's functions", all(a.func is b.func for a, b in zip(ac.build_tools(BASE), test_agent.tools)))
ok("config_hash of baseline unchanged by the tools: key (it lives in targets.yaml, not settings)",
   "tools" not in BASE and config_hash(load_experiment(os.path.join("experiments", "baseline.yaml")))
   == config_hash({"target": "configurable", "settings": BASE}))
try:
    ac.build_tools({**BASE, "tool_validation": "on"})
    ok("build_tools validates settings", False)
except ac.SettingsError:
    ok("build_tools validates settings", True)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
