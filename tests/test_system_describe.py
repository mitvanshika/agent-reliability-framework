"""
Tests for the optional target hook describe(): which version of the system ran
(its own git commit and config), saved in run_meta.json as "system", carried into
the baseline by promote_baseline.py, and compared by compare.py. No API calls.
    python tests/test_system_describe.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import compare  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def fake_module(name, describe=None):
    """A target module: run(question, settings) plus an optional describe(settings)."""
    mod = types.ModuleType(name)
    exec("def run(question, settings=None):\n    return 'answer'\n", mod.__dict__)
    if describe is not None:
        mod.describe = describe
    sys.modules[name] = mod
    return mod


EXP = {"name": "t", "description": "", "parent": None, "chain": ["t"], "changes_vs_parent": [], "target": "x",
       "settings": {"self_healing": {"max_attempts": 1}}}


def meta_after_run(mod, accepts=True):
    root = tempfile.mkdtemp()
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment.execute(EXP, mod.run, accepts, [{"id": "s1", "question": "q?"}], root, "t.yaml")
    meta = json.load(open(os.path.join(root, "run_meta.json"), encoding="utf-8"))
    shutil.rmtree(root, ignore_errors=True)
    return meta


print("run_meta.json:")
seen = []
info = {"system": "fake", "git": {"commit": "a" * 40, "dirty": False}, "config_hash": "c1"}
meta = meta_after_run(fake_module("sys_fake_1", lambda settings=None: seen.append(settings) or info))
ok("describe() result saved as run_meta['system']", meta.get("system") == info)
ok("describe() gets the experiment's settings", seen == [EXP["settings"]])
meta = meta_after_run(fake_module("sys_fake_2"))
ok("no describe() hook: no 'system' key (old targets unchanged)", "system" not in meta)


def broken(settings=None):
    raise RuntimeError("worker gone")


meta = meta_after_run(fake_module("sys_fake_3", broken))
ok("a failing describe() is recorded, the run still happens",
   meta.get("system") == {"error": "RuntimeError: worker gone"} and meta.get("status") != "running")

print("promote_baseline:")
base = promote_baseline.build_baseline("runs/x", {"system": info}, {}, {}, "note")
ok("baseline keeps the system record", base.get("system") == info)

print("compare.system_differences:")
same = compare.system_differences(info, dict(info))
ok("same commit + config: no changes", same == [])
moved = compare.system_differences(info, {**info, "git": {"commit": "b" * 40, "dirty": False}})
ok("different commit reported", moved == ["commit aaaaaaa -> bbbbbbb"], moved)
dirty = compare.system_differences(info, {**info, "git": {"commit": "a" * 40, "dirty": True,
                                                           "changed_files": ["config.yaml"], "untracked_files": 7}})
ok("uncommitted edits in the run reported, with the files",
   dirty == ["run had uncommitted edits in the system's repo: config.yaml"], dirty)
leftovers = compare.system_differences(info, {**info, "git": {"commit": "a" * 40, "dirty": False,
                                                               "changed_files": [], "untracked_files": 7}})
ok("only untracked leftovers (eval outputs): not a difference", leftovers == [], leftovers)
cfg = compare.system_differences(info, {**info, "config_hash": "c2"})
ok("different config file reported", cfg == ["its config file c1 -> c2"], cfg)
ok("either side without a record: nothing to compare",
   compare.system_differences(None, info) == [] and compare.system_differences(info, None) == [])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
