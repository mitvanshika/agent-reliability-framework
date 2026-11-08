"""
Tests for generate_adversarial.py and run_experiment --scenario-dir (no API calls).
    python tests/test_adversarial.py
"""

import contextlib
import glob
import io
import json
import os
import sys
import tempfile
import textwrap

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import generate_adversarial as ga  # noqa: E402
import run_experiment  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


SUPPORT = run_suite.load_targets()["configurable"]["description"]
main_before = {p: open(p, "rb").read() for p in glob.glob(os.path.join("scenarios", "*.yaml"))}

print("templates are generic:")
raw = open(ga.TEMPLATES_FILE, encoding="utf-8").read().lower()
ok("no support-bot words in the templates", not any(w in raw for w in ("order", "refund", "faq", "shipping", "customer")))
ok("7 attack kinds", [t["kind"] for t in ga.load_templates()]
   == ["injection", "fake_authority", "unanswerable", "out_of_scope", "empty", "gibberish", "very_long"])

print("generated support-bot set:")
root = tempfile.mkdtemp()
with contextlib.redirect_stdout(io.StringIO()):
    out_dir = ga.main(["configurable", "--out-root", root])
files = sorted(os.listdir(out_dir))
ok("7 files adv_<kind>.yaml", files == sorted(f"adv_{t['kind']}.yaml" for t in ga.load_templates()), files)
docs = {f: yaml.safe_load(open(os.path.join(out_dir, f), encoding="utf-8")) for f in files}
ok("valid YAML with id, question, attack, expected_answer, category adversarial",
   all(set(d) >= {"id", "question", "attack", "expected_answer"} and d["category"] == "adversarial" for d in docs.values()))
ok("unique ids", len({d["id"] for d in docs.values()}) == 7)
ok("no expected_tools (tool_match N/A for attacks)", all("expected_tools" not in d for d in docs.values()))
ok("{description} filled in", SUPPORT in docs["adv_injection.yaml"]["expected_answer"]
   and all("{description}" not in d["expected_answer"] for d in docs.values()))
ok("out of scope for the support bot = the Python question",
   docs["adv_out_of_scope.yaml"]["question"].startswith("Can you write me a Python function"))
ok("empty question is an empty string", docs["adv_empty.yaml"]["question"] == "")
long_q = docs["adv_very_long.yaml"]["question"]
ok("very long question ~5,400 chars ending with the real question", 5000 < len(long_q) < 6000
   and long_q.endswith("Can you tell me what you can help me with?"))
ok("CRLF like the rest of the repo", all(b"\r\n" in open(os.path.join(out_dir, f), "rb").read() for f in files))

print("deterministic + safe:")
before = {f: open(os.path.join(out_dir, f), "rb").read() for f in files}
with contextlib.redirect_stdout(io.StringIO()):
    ga.main(["configurable", "--out-root", root])
ok("re-generating gives byte-identical files", all(open(os.path.join(out_dir, f), "rb").read() == before[f] for f in files))
open(os.path.join(out_dir, "adv_gibberish.yaml"), "ab").write(b"# my edit\r\n")
try:
    with contextlib.redirect_stdout(io.StringIO()):
        ga.main(["configurable", "--out-root", root])
    ok("hand-edited file not overwritten", False)
except SystemExit as e:
    ok("hand-edited file not overwritten", "adv_gibberish.yaml" in str(e) and "--force" in str(e))
with contextlib.redirect_stdout(io.StringIO()):
    ga.main(["configurable", "--out-root", root, "--force"])
ok("--force regenerates it", open(os.path.join(out_dir, "adv_gibberish.yaml"), "rb").read() == before["adv_gibberish.yaml"])

print("universal: adapts to another description:")
coder = dict(ga.build("a coding assistant that helps developers write Python code"))
ok("coding assistant gets the cake question as out-of-scope",
   "chocolate cake" in coder["adv_out_of_scope"], coder["adv_out_of_scope"].splitlines()[2][:80])
try:
    ga.build("")
    ok("no description -> refused", False)
except ValueError as e:
    ok("no description -> refused", "description" in str(e))
try:
    ga.main(["strong", "--out-root", root])
    ok("target without description in targets.yaml -> exit", False)
except SystemExit as e:
    ok("target without description in targets.yaml -> exit", "description" in str(e))

print("run_experiment --scenario-dir (fake target):")
open(os.path.join(root, "adv_target.py"), "w").write(textwrap.dedent('''
    SEEN = []
    def run(question, settings):
        SEEN.append(question)
        return "I can only help within my scope."
'''))
sys.path.insert(0, root)
import adv_target  # noqa: E402

open(os.path.join(root, "targets.yaml"), "w").write("adv_t:\n  module: adv_target\n  function: run\n  accepts_settings: true\n")
open(os.path.join(root, "adv_exp.yaml"), "w").write("name: adv_exp\ntarget: adv_t\nsettings: {x: 1}\n")
real_targets = run_suite.TARGETS_FILE
run_suite.TARGETS_FILE = os.path.join(root, "targets.yaml")
run_experiment.RUNS_DIR = os.path.join(root, "runs")
run_experiment.PRICING_FILE = os.path.join(root, "none.yaml")
with contextlib.redirect_stdout(io.StringIO()):
    run_dir = run_experiment.main([os.path.join(root, "adv_exp.yaml"), "--scenario-dir", out_dir, "--yes"])
meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
ok("ran exactly the 7 adversarial scenarios", sorted(meta["scenario_ids"]) == sorted(d["id"] for d in docs.values())
   and len(adv_target.SEEN) == 7, meta["scenario_ids"])
ok("scenario_dir recorded in run_meta", meta["scenario_dir"] == out_dir.replace("\\", "/"))
ok("empty and very long inputs reached the target", "" in adv_target.SEEN and any(len(q) > 5000 for q in adv_target.SEEN))
adv_target.SEEN.clear()
with contextlib.redirect_stdout(io.StringIO()):
    run_dir = run_experiment.main([os.path.join(root, "adv_exp.yaml"), "--scenarios", "normal_01", "normal_02", "--yes"])
meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
ok("default = scenarios/ exactly as before", meta["scenario_dir"] == "scenarios" and meta["scenario_ids"] == ["normal_01", "normal_02"])
try:
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment.main([os.path.join(root, "adv_exp.yaml"), "--scenario-dir", os.path.join(root, "nope"), "--yes"])
    ok("missing folder -> clear exit", False)
except SystemExit as e:
    ok("missing folder -> clear exit", "scenario folder not found" in str(e))
run_suite.TARGETS_FILE = real_targets

print("main suite untouched:")
ok("scenarios/*.yaml byte-identical, still 31", {p: open(p, "rb").read() for p in glob.glob(os.path.join("scenarios", "*.yaml"))}
   == main_before and len(main_before) == 31)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
