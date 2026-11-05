"""
Proves the experiment files are exact copies of the demo agents:
  prompts/v1.txt + docstrings/detailed.yaml  == test_agent.py    (strong)
  prompts/weak.txt + docstrings/vague.yaml   == agent_v2_weak.py (weak)

So the "baseline" experiment is provably the same agent as `strong`, and
exp_01/exp_02 change exactly one thing each. If someone edits a prompt in
either place, this fails and shows where the texts start to differ.

Run from the repo root (no API calls):
    python tests/test_prompt_equivalence.py
"""

import os
import sys
import tempfile

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")   # importing the agents builds a client, no call

import agent_v2_weak  # noqa: E402
import test_agent  # noqa: E402

failures = []


def read_text(path):
    # Same way agent_configurable.py reads it: text mode, so \r\n -> \n on every OS.
    with open(path, encoding="utf-8") as f:
        return f.read()


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at char {i}: {a[max(0, i - 15):i + 15]!r} vs {b[max(0, i - 15):i + 15]!r}"
    return f"one is a prefix of the other (lengths {len(a)} vs {len(b)})"


def check(label, actual, expected):
    if actual == expected:
        print(f"  OK    {label}")
    else:
        print(f"  FAIL  {label}: {first_diff(actual, expected)}")
        failures.append(label)


def check_prompt(rel, module):
    check(f"{rel} == {module.__name__}.SYSTEM_PROMPT", read_text(os.path.join(REPO, rel)), module.SYSTEM_PROMPT)


def check_docstrings(rel, module):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = [t.name for t in module.tools]
    if sorted(data) != sorted(names):
        print(f"  FAIL  {rel} keys {sorted(data)} != {module.__name__} tools {sorted(names)}")
        failures.append(rel)
        return
    for t in module.tools:
        # tool.description is exactly what LangChain sends to the model
        check(f"{rel}[{t.name}] == {module.__name__}.{t.name}.description", data[t.name], t.description)


def check_line_endings(rel):
    raw = open(os.path.join(REPO, rel), "rb").read()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    tmp = tempfile.mkdtemp()
    texts = []
    for name, data in (("lf", lf), ("crlf", crlf)):
        p = os.path.join(tmp, name)
        open(p, "wb").write(data)
        texts.append(read_text(p))
    check(f"{rel} reads the same with LF and CRLF line endings", texts[0], texts[1])


print("strong agent (test_agent.py):")
check_prompt("prompts/v1.txt", test_agent)
check_docstrings("docstrings/detailed.yaml", test_agent)
print("weak agent (agent_v2_weak.py):")
check_prompt("prompts/weak.txt", agent_v2_weak)
check_docstrings("docstrings/vague.yaml", agent_v2_weak)
print("line endings (Windows checkout = CRLF, Linux CI = LF):")
for rel in ("prompts/v1.txt", "prompts/weak.txt"):
    check_line_endings(rel)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
