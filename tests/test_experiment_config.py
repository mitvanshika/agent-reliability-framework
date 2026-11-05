"""
Tests for experiment_config.py (no API calls).
    python tests/test_experiment_config.py
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiment_config import ExperimentConfigError, format_changes, load_experiment  # noqa: E402

failures = []


def make(files: dict) -> str:
    d = tempfile.mkdtemp()
    for name, text in files.items():
        with open(os.path.join(d, f"{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(text)
    return d


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def raises(label, d, name, must_contain):
    try:
        load_experiment(os.path.join(d, f"{name}.yaml"))
        ok(label, False, "no error raised")
    except ExperimentConfigError as e:
        ok(label, all(s in str(e) for s in must_contain), str(e))


BASE = """name: baseline
description: "the baseline"
target: configurable
settings:
  model: openai/gpt-oss-120b
  temperature: 0
  prompt: prompts/v1.txt
  retrieval: {method: substring, top_k: 3}
  recursion_limit: 25
"""

print("simple extends:")
d = make({"baseline": BASE, "exp_a": "name: exp_a\nextends: baseline\nsettings:\n  prompt: prompts/weak.txt\n"})
r = load_experiment(os.path.join(d, "exp_a.yaml"))
ok("prompt overridden", r["settings"]["prompt"] == "prompts/weak.txt")
ok("other settings inherited", r["settings"]["model"] == "openai/gpt-oss-120b" and r["settings"]["recursion_limit"] == 25)
ok("target inherited", r["target"] == "configurable")
ok("description NOT inherited", r["description"] == "")
ok("parent + chain", r["parent"] == "baseline" and r["chain"] == ["baseline", "exp_a"])
ok("changes vs parent", r["changes_vs_parent"] == ["settings.prompt: prompts/v1.txt -> prompts/weak.txt"],
   r["changes_vs_parent"])
ok("format_changes", format_changes(r) == "settings.prompt: prompts/v1.txt -> prompts/weak.txt   (vs baseline)",
   format_changes(r))
b = load_experiment(os.path.join(d, "baseline.yaml"))
ok("baseline has no parent and no changes", b["parent"] is None and b["changes_vs_parent"] == [], format_changes(b))

print("nested merge:")
d = make({"baseline": BASE, "exp_n": "name: exp_n\nextends: baseline\nsettings:\n  retrieval: {method: embeddings}\n"})
r = load_experiment(os.path.join(d, "exp_n.yaml"))
ok("retrieval.method overridden, top_k kept", r["settings"]["retrieval"] == {"method": "embeddings", "top_k": 3},
   r["settings"]["retrieval"])
ok("only the nested key reported", r["changes_vs_parent"] == ["settings.retrieval.method: substring -> embeddings"],
   r["changes_vs_parent"])
ok("parent's dict not mutated", load_experiment(os.path.join(d, "baseline.yaml"))["settings"]["retrieval"]["method"] == "substring")

print("3-level chain:")
d = make({"baseline": BASE,
          "mid": "name: mid\nextends: baseline\nsettings:\n  temperature: 0.5\n  new_key: [1, 2]\n",
          "leaf": "name: leaf\nextends: mid\ntarget: other_target\nsettings:\n  prompt: prompts/weak.txt\n"})
r = load_experiment(os.path.join(d, "leaf.yaml"))
ok("chain", r["chain"] == ["baseline", "mid", "leaf"], r["chain"])
ok("values from every level", r["settings"]["temperature"] == 0.5 and r["settings"]["prompt"] == "prompts/weak.txt"
   and r["settings"]["model"] == "openai/gpt-oss-120b" and r["target"] == "other_target")
ok("changes only vs direct parent (mid)",
   r["changes_vs_parent"] == ["settings.prompt: prompts/v1.txt -> prompts/weak.txt", "target: configurable -> other_target"],
   r["changes_vs_parent"])
m = load_experiment(os.path.join(d, "mid.yaml"))
ok("added key shows (unset)", "settings.new_key: (unset) -> [1, 2]" in m["changes_vs_parent"]
   and "settings.temperature: 0 -> 0.5" in m["changes_vs_parent"], m["changes_vs_parent"])

print("errors:")
d = make({"a": "name: a\nextends: b\ntarget: t\n", "b": "name: b\nextends: c\n", "c": "name: c\nextends: a\n"})
raises("3-file cycle", d, "a", ["Cycle", "a -> b -> c -> a"])
d = make({"selfref": "name: selfref\nextends: selfref\ntarget: t\n"})
raises("self cycle", d, "selfref", ["Cycle", "selfref -> selfref"])
d = make({"x": "name: x\nextends: baselin\ntarget: t\n"})
raises("missing parent", d, "x", ["extends 'baselin'", "does not exist"])
d = make({"x": "name: x\ntarget: t\nsetings:\n  prompt: p\n"})
raises("unknown key with suggestion", d, "x", ["Unknown key 'setings'", "did you mean 'settings'"])
d = make({"x": "name: y\ntarget: t\n"})
raises("name must equal file name", d, "x", ["must equal the file name 'x'"])
d = make({"x": "name: x\nsettings: {a: 1}\n"})
raises("no target anywhere", d, "x", ["no 'target'"])
d = make({"x": "name: x\ntarget: t\nsettings: [1, 2]\n"})
raises("settings must be a mapping", d, "x", ["'settings' must be a mapping"])
d = make({"x": "- just\n- a list\n"})
raises("file must be a mapping", d, "x", ["must be a mapping"])
d = make({"x": "name: x\ntarget: t\nsettings: {a: 1\n"})
raises("invalid YAML", d, "x", ["not valid YAML"])
raises("missing file", d, "nope", ["not found"])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
