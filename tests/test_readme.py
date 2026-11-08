"""
Checks that README.md only mentions things that exist (no API calls):
every .py file, folder, --flag and YAML example in it.
    python tests/test_readme.py
"""

import os
import re
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


text = open("README.md", encoding="utf-8").read()
ok("title appears once", text.count("# Agent Reliability") == 1)
ok("no reference to the non-existent generate_scenarios.py", "generate_scenarios.py" not in text)

print(".py files mentioned exist:")
example_only = {"entry.py"}   # my_rag/entry.py is the plug-in example
py = sorted(set(re.findall(r"([A-Za-z_][\w/]*\.py)\b", text)))
missing = [p for p in py if os.path.basename(p) not in example_only and not os.path.isfile(p)
           and not os.path.isfile(os.path.join("tests", os.path.basename(p)))
           and not os.path.isfile(os.path.join("judges", os.path.basename(p)))]
ok(f"{len(py)} .py names", not missing, missing)

print("folders / files mentioned exist:")
for path in ["scenarios", "experiments", "prompts", "docstrings", "judges", "templates", "tests", ".github/workflows",
             "targets.yaml", "requirements.txt", "requirements-rag.txt", ".github/workflows/regression-check.yml",
             ".github/workflows/experiment-check.yml", "templates/experiments.html", "experiments/baseline.yaml",
             "experiments/exp_01_weak_prompt.yaml", "experiments/exp_02_vague_docstrings.yaml",
             "experiments/exp_04_bm25.yaml", "experiments/exp_05_embeddings.yaml", "prompts/weak.txt"]:
    ok(path, os.path.exists(path))

print("--flags mentioned exist in the script they're used with:")
for line in text.splitlines():
    m = re.search(r"python ([\w/]+\.py)(.*)", line)
    if not m or not os.path.isfile(m.group(1)):
        continue
    script, rest = m.group(1), m.group(2).split("#")[0]
    flags = re.findall(r"(--[a-z][a-z-]*)", rest)
    if not flags:
        continue
    helptext = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True,
                              env={**os.environ, "GROQ_API_KEY": "offline-test"}).stdout
    for flag in flags:
        ok(f"{script} {flag}", flag in helptext)

print("YAML examples parse:")
blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
for b in blocks:
    try:
        d = yaml.safe_load(b)
        ok(f"yaml block ({b.splitlines()[0][:40]})", isinstance(d, dict))
    except yaml.YAMLError as e:
        ok("yaml block parses", False, e)
targets_example = next(yaml.safe_load(b) for b in blocks if "accepts_settings" in b)
ok("targets.yaml example has module/function/accepts_settings",
   targets_example["financial_dd"] == {"module": "my_rag.entry", "function": "answer", "accepts_settings": True})

print("plug-in section matches the real example:")
four = text.split("\n## Plug in your system in 4 steps\n", 1)
ok("'Plug in your system in 4 steps' comes before the architecture section",
   len(four) == 2 and four[1].find("\n## Architecture: two tracks\n") > 0)
real_targets = yaml.safe_load(open("targets.yaml", encoding="utf-8"))
shown = next(yaml.safe_load(b) for b in blocks if "summarizer:" in b)
ok("its targets.yaml block = the real summarizer entry (can't go stale)", shown["summarizer"] == real_targets["summarizer"],
   shown)
ok("the example experiment it runs exists", os.path.isfile("examples/summarizer/summarizer_baseline.yaml"))
sys.path.insert(0, REPO)
import target_config  # noqa: E402
key_table = text.split("**Per-target settings in `targets.yaml`**", 1)[-1].split("\n\n", 2)[1]
table_keys = re.findall(r"^\| `(\w+)` \|", key_table, re.M)
ok("every per-target key in the README table is a real key", table_keys and all(
   k in target_config.KEYS for k in table_keys), table_keys)
ok("...and every real key is in the table", set(target_config.KEYS) - {"module", "function", "accepts_settings"}
   <= set(table_keys), set(target_config.KEYS) - set(table_keys))

print("actions section matches the code:")
act = text.split("\n## Making any agent's actions safe\n", 1)[-1].split("\n## ", 1)[0]
import approval  # noqa: E402
names = re.findall(r"^\| `(\w+)` (?:\(default\) )?\|", act, re.M)
ok("every approver in the table exists", names and all(n in approval.APPROVERS for n in names), names)
ok("the actions yaml example is a valid target entry", not target_config.validate(
   "my_target", {"module": "m", "function": "f", **next(yaml.safe_load(b) for b in blocks
                                                       if "action_tools" in b)["my_target"]}, []))
ok("the support-bot experiment it names exists", os.path.isfile("experiments/exp_09_actions.yaml"))

print("DeepEval section matches the code:")
ok("requirements-deepeval.txt exists and pins deepeval", "deepeval==" in open("requirements-deepeval.txt").read())
ok("the judge names it gives exist in deepeval_v1", all(n in text for n in __import__("judge_sets").DEEPEVAL_V1[
   "llm_judges"]))

print("sections:")
for h in ["Architecture: two tracks", "How it works (Track 1)", "The experiment workflow (Track 2)", "Experiments",
          "Judges and the answer key", "Baselines and comparison", "Diagnosis", "Plug in your own system",
          "Fine-tuning data", "CI", "Dashboards", "Key features", "Tech stack", "Project structure", "Setup",
          "Results", "Known limitations / future work", "The headline finding", "Plug in your system in 4 steps",
          "Making any agent's actions safe", "Using DeepEval metrics inside this framework"]:
    ok(f"## {h}", f"\n## {h}\n" in text)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
