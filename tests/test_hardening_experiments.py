"""
Offline checks for exp_07_hardened_prompt and exp_08_tool_validation (no LLM, no API calls).
    python tests/test_hardening_experiments.py
"""

import contextlib
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

from langchain_core.messages import AIMessage, SystemMessage  # noqa: E402

import agent_configurable as ac  # noqa: E402
import fuzz_tools  # noqa: E402
import run_suite  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def path(name):
    return os.path.join("experiments", f"{name}.yaml")


BASE = load_experiment(path("baseline"))
E7 = load_experiment(path("exp_07_hardened_prompt"))
E8 = load_experiment(path("exp_08_tool_validation"))


class Fake:
    def __init__(self):
        self.system = None

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.system = next(m.content for m in messages if isinstance(m, SystemMessage))
        return AIMessage(content="ok")


def system_prompt(settings):
    fake = Fake()
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: fake
    ac.run_agent("q", settings)
    return fake.system


print("exp_07_hardened_prompt:")
ok("exactly one change vs baseline", len(E7["changes_vs_parent"]) == 1
   and E7["changes_vs_parent"][0].startswith("settings.prompt_addons:"), E7["changes_vs_parent"])
ok("everything else = baseline", {k: v for k, v in E7["settings"].items() if k != "prompt_addons"} == BASE["settings"])
ok("config_hash differs from baseline", config_hash(E7) != config_hash(BASE))
ok("settings validate (addons + description known)", isinstance(ac.validate_settings(E7["settings"]), ac.AgentSettings))
base_prompt, hard_prompt = system_prompt(BASE["settings"]), system_prompt(E7["settings"])
desc = run_suite.load_targets()["configurable"]["description"]
ok("prompt = baseline prompt + 'Additional rules:' block", hard_prompt.startswith(base_prompt + "\n\nAdditional rules:\n- "))
ok("three rules in the block", hard_prompt.split("Additional rules:\n", 1)[1].count("\n- ") == 2)
ok("scope rule has the description filled in", f"Your job: {desc}." in hard_prompt and "{description}" not in hard_prompt)
ok("baseline prompt has no addon block", "Additional rules:" not in base_prompt)

print("exp_08_tool_validation:")
ok("exactly one change vs baseline", len(E8["changes_vs_parent"]) == 1 and "tool_validation" in E8["changes_vs_parent"][0],
   E8["changes_vs_parent"])
ok("everything else = baseline", {k: v for k, v in E8["settings"].items() if k != "tool_validation"} == BASE["settings"])
ok("config_hash differs from baseline and exp_07", len({config_hash(BASE), config_hash(E7), config_hash(E8)}) == 3)
ok("settings validate", ac.validate_settings(E8["settings"]).tool_validation == "strict")
ok("prompt unchanged vs baseline (only the tools change)", system_prompt(E8["settings"]) == base_prompt)
with contextlib.redirect_stdout(io.StringIO()) as out:
    code = fuzz_tools.main(["configurable", "--experiment", path("exp_08_tool_validation")])
ok("fuzz_tools on exp_08 -> 0 crashes, exit 0", code == 0 and "TOTAL: 0 crash" in out.getvalue())
with contextlib.redirect_stdout(io.StringIO()):
    code = fuzz_tools.main(["configurable"])
ok("fuzz_tools on baseline -> exit 1 (the before picture)", code == 1)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
