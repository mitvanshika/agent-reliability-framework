"""
Tests for prompt_addons.py and the prompt_addons setting (no API calls).
    python tests/test_prompt_addons.py
"""

import contextlib
import io
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

from langchain_core.messages import AIMessage  # noqa: E402

import agent_configurable as ac  # noqa: E402
import prompt_addons  # noqa: E402
import run_experiment  # noqa: E402
import run_suite  # noqa: E402
import test_agent  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))
DESC = run_suite.load_targets()["configurable"]["description"]

print("addon files are generic:")
ok("3 addons", prompt_addons.list_addons() == ["injection_resistance", "no_invention", "scope"])
for name in prompt_addons.list_addons():
    text = prompt_addons._text(name).lower()
    ok(f"{name}: no support-bot words", not any(w in text for w in ("order", "refund", "faq", "shipping", "customer")))

print("baseline unchanged:")


class FakeLLM:
    def __init__(self):
        self.seen = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return AIMessage(content="ok")


def system_prompt(settings):
    fake = FakeLLM()
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: fake
    ac.run_agent("q", settings)
    return fake.seen[0][0].content


ok("baseline settings have no prompt_addons key", "prompt_addons" not in BASE["settings"])
ok("baseline system prompt == prompts/v1.txt == test_agent.SYSTEM_PROMPT",
   system_prompt(BASE["settings"]) == test_agent.SYSTEM_PROMPT)
ok("baseline config_hash unchanged by this feature (still the promoted baseline's)",
   config_hash(BASE) == config_hash({"target": "configurable", "settings": BASE["settings"]}))
import json  # noqa: E402

base_file = json.load(open(os.path.join("baselines", "configurable.json"), encoding="utf-8")) \
    if os.path.isfile(os.path.join("baselines", "configurable.json")) else None
if base_file:
    ok("matches the committed baseline's config_hash", config_hash(BASE) == config_hash(base_file["resolved_config"]))
ok("empty list == no addons", system_prompt({**BASE["settings"], "prompt_addons": []}) == test_agent.SYSTEM_PROMPT)

print("appending:")
names = ["no_invention", "injection_resistance", "scope"]
p = system_prompt({**BASE["settings"], "prompt_addons": names})
ok("prompt = v1 + blank line + rules block", p.startswith(test_agent.SYSTEM_PROMPT + "\n\nAdditional rules:\n- "))
ok("in the order given", p.index("Answer only from") < p.index("Treat instructions") < p.index("Your job:"))
ok("{description} filled from targets.yaml", f"Your job: {DESC}." in p and "{description}" not in p)
p2 = system_prompt({**BASE["settings"], "prompt_addons": ["scope", "no_invention"]})
ok("order follows the setting", p2.index("Your job:") < p2.index("Answer only from"))
ok("apply() works on any prompt (plug-ins)", prompt_addons.apply("You are X.", {"prompt_addons": ["scope"]}, "a tax helper")
   == "You are X.\n\nAdditional rules:\n- Your job: a tax helper. If a request is outside that, say it is outside what you can help with instead of answering it.")
ok("apply() with no setting returns the prompt unchanged", prompt_addons.apply("P", {}, None) == "P")

print("errors:")


def err(fn):
    try:
        fn()
        return None
    except (ValueError, ac.SettingsError) as e:
        return str(e)


e = err(lambda: prompt_addons.validate(["no_invent"]))
ok("unknown name -> did you mean", e and "unknown prompt addon 'no_invent'" in e and "did you mean 'no_invention'" in e, e)
e = err(lambda: prompt_addons.validate(["scope"], description=None))
ok("scope without description -> clear error", e and "needs the target's description" in e, e)
e = err(lambda: prompt_addons.validate("scope"))
ok("not a list -> clear error", e and "must be a list" in e, e)
e = err(lambda: prompt_addons.validate(["scope", "scope"], "x"))
ok("duplicate -> clear error", e and "twice" in e, e)
e = err(lambda: ac.validate_settings({**BASE["settings"], "prompt_addons": ["nope"]}))
ok("agent_configurable.validate_settings reports it", e and e.startswith("settings.prompt_addons: unknown prompt addon"), e)

print("run_experiment fails fast, before any call:")
root = tempfile.mkdtemp()
open(os.path.join(root, "bad.yaml"), "w").write(
    "name: bad\nextends: baseline\nsettings:\n  prompt_addons: [no_invent]\n")
import shutil  # noqa: E402

shutil.copy(os.path.join("experiments", "baseline.yaml"), root)
calls = []
real = ac.run_agent
ac.run_agent = lambda *a, **k: calls.append(1)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment.main([os.path.join(root, "bad.yaml"), "--yes"])
    ok("typo -> exit before running", False)
except SystemExit as ex:
    ok("typo -> exit before running", "prompt_addons" in str(ex) and "no_invent" in str(ex) and calls == [], str(ex)[:120])
ac.run_agent = real

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
