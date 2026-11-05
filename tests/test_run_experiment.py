"""
Tests for run_experiment.py with fake targets (plain functions, no API).
    python tests/test_run_experiment.py
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import run_experiment as rx  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


tmp = tempfile.mkdtemp()
# --- fake target module
open(os.path.join(tmp, "fake_targets.py"), "w", encoding="utf-8").write(textwrap.dedent('''
    import time
    CALLS = []
    def plain(question):
        CALLS.append(("plain", question))
        time.sleep(0.05)
        return "a plain string answer"
    def with_settings(question, settings):
        CALLS.append(("with_settings", question, settings))
        if "BOOM" in question:
            raise RuntimeError("target blew up")
        time.sleep(0.02)
        return {"answer": "dict answer", "tools_used": ["t"], "tool_outputs": ["o"],
                "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}}
    def validate_settings(settings):
        if "bad_key" in settings:
            raise ValueError("settings.bad_key: unknown key")
'''))
sys.path.insert(0, tmp)
import fake_targets  # noqa: E402

# --- fake targets.yaml, scenarios, experiments, pricing
open(os.path.join(tmp, "targets.yaml"), "w").write(
    "plain_t:\n  module: fake_targets\n  function: plain\n"
    "settings_t:\n  module: fake_targets\n  function: with_settings\n  accepts_settings: true\n"
    "settings_off:\n  module: fake_targets\n  function: plain\n  accepts_settings: false\n")
os.makedirs(os.path.join(tmp, "scenarios"))
for sid, q in [("s1", "first q"), ("s2", "second q BOOM"), ("s3", "third q")]:
    open(os.path.join(tmp, "scenarios", f"{sid}.yaml"), "w").write(f'id: "{sid}"\nquestion: "{q}"\ncategory: "normal"\n')
exp_dir = os.path.join(tmp, "experiments")
os.makedirs(exp_dir)
open(os.path.join(exp_dir, "base_e.yaml"), "w").write(
    "name: base_e\ndescription: base\ntarget: settings_t\nsettings:\n  model: fake-model\n  k: 1\n")
open(os.path.join(exp_dir, "child_e.yaml"), "w").write("name: child_e\nextends: base_e\nsettings:\n  k: 2\n")
open(os.path.join(exp_dir, "plain_e.yaml"), "w").write("name: plain_e\ntarget: plain_t\nsettings:\n  model: fake-model\n")
open(os.path.join(exp_dir, "off_e.yaml"), "w").write("name: off_e\ntarget: settings_off\nsettings:\n  x: 1\n")
open(os.path.join(exp_dir, "bad_e.yaml"), "w").write("name: bad_e\ntarget: settings_t\nsettings:\n  bad_key: 1\n")
open(os.path.join(tmp, "pricing.yaml"), "w").write("fake-model: {input: 2.0, output: 10.0}\n")

run_suite.TARGETS_FILE = os.path.join(tmp, "targets.yaml")
run_suite.SCENARIOS_DIR = os.path.join(tmp, "scenarios")
rx.RUNS_DIR = os.path.join(tmp, "runs")
rx.PRICING_FILE = os.path.join(tmp, "pricing.yaml")
repo_runs_before = os.path.exists(os.path.join(REPO, "runs"))


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = rx.main(argv)
    return result, buf.getvalue()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


print("settings target, child experiment, crash continues:")
fake_targets.CALLS.clear()
run_dir, out = run([os.path.join(exp_dir, "child_e.yaml"), "--yes"])
print("    " + "\n    ".join(out.strip().splitlines()))
ok("folder structure", sorted(os.listdir(run_dir)) == ["run_meta.json", "traces"]
   and sorted(os.listdir(os.path.join(run_dir, "traces"))) == ["s1.json", "s2.json", "s3.json"])
ok("folder name <experiment>_<timestamp>", os.path.basename(run_dir).startswith("child_e_2"))
meta = load(os.path.join(run_dir, "run_meta.json"))
ok("meta: experiment/parent/changes", meta["experiment"] == "child_e" and meta["parent"] == "base_e"
   and meta["changes_vs_parent"] == ["settings.k: 1 -> 2"], meta["changes_vs_parent"])
ok("meta: resolved config", meta["resolved_config"] == {"target": "settings_t", "settings": {"model": "fake-model", "k": 2}})
ok("meta: git commit + dirty flag", isinstance(meta["git"]["commit"], str) and len(meta["git"]["commit"]) == 40
   and isinstance(meta["git"]["dirty"], bool), meta["git"])
ok("meta: python, timestamp, scenario ids, status", meta["python"].count(".") == 2 and meta["timestamp"]
   and meta["scenario_ids"] == ["s1", "s2", "s3"] and meta["status"] == "complete")
t = meta["totals"]
ok("totals: tokens from 2 successful scenarios", (t["input_tokens"], t["output_tokens"], t["total_tokens"],
   t["scenarios_with_usage"]) == (2000, 400, 2400, 2), t)
ok("totals: errors by type", t["errors"] == 1 and t["errors_by_type"] == {"crash": 1})
ok("totals: latency mean + p95", t["latency_mean_s"] > 0 and t["latency_p95_s"] >= t["latency_mean_s"])
ok("cost from pricing.yaml: (2000*2 + 400*10)/1e6", t["estimated_cost_usd"] == 0.008, (t["estimated_cost_usd"], t["cost_note"]))
s1 = load(os.path.join(run_dir, "traces", "s1.json"))
ok("trace: latency recorded", s1["latency_s"] >= 0.02, s1["latency_s"])
ok("trace: usage recorded", s1["usage"] == {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200})
ok("trace: Track 1 fields kept", all(k in s1 for k in ("question", "answer", "tools_used", "tool_outputs",
   "retries_needed", "timestamp", "error", "error_type")) and s1["experiment"] == "child_e")
s2 = load(os.path.join(run_dir, "traces", "s2.json"))
ok("crash recorded with error + error_type", s2["error"] == "target blew up" and s2["error_type"] == "crash"
   and s2["usage"] is None and s2["answer"] is None)
ok("run continued after the crash", os.path.exists(os.path.join(run_dir, "traces", "s3.json")))
ok("settings passed (accepts_settings: true)", all(c[0] == "with_settings" and c[2] == {"model": "fake-model", "k": 2}
   for c in fake_targets.CALLS) and len(fake_targets.CALLS) == 3)

print("plain-string target:")
fake_targets.CALLS.clear()
run_dir, out = run([os.path.join(exp_dir, "plain_e.yaml"), "--yes", "--scenarios", "s1", "s3"])
meta = load(os.path.join(run_dir, "run_meta.json"))
s1 = load(os.path.join(run_dir, "traces", "s1.json"))
ok("called as fn(question) only", fake_targets.CALLS == [("plain", "first q"), ("plain", "third q")], fake_targets.CALLS)
ok("usage None for plain string", s1["usage"] is None and s1["answer"] == "a plain string answer")
ok("latency still recorded", s1["latency_s"] >= 0.05, s1["latency_s"])
ok("--scenarios filter", meta["scenario_ids"] == ["s1", "s3"] and sorted(os.listdir(os.path.join(run_dir, "traces"))) == ["s1.json", "s3.json"])
ok("no usage -> tokens None, cost None with reason", meta["totals"]["total_tokens"] is None
   and meta["totals"]["estimated_cost_usd"] is None, meta["totals"]["cost_note"])

print("accepts_settings: false:")
fake_targets.CALLS.clear()
run_dir, out = run([os.path.join(exp_dir, "off_e.yaml"), "--yes", "--scenarios", "s1"])
ok("settings NOT passed + warning shown", fake_targets.CALLS == [("plain", "first q")] and "settings are IGNORED" in out)

print("no pricing.yaml:")
rx.PRICING_FILE = os.path.join(tmp, "empty_dir", "pricing.yaml")   # does not exist
run_dir, out = run([os.path.join(exp_dir, "child_e.yaml"), "--yes", "--scenarios", "s1"])
meta = load(os.path.join(run_dir, "run_meta.json"))
ok("tokens only, no cost", meta["totals"]["estimated_cost_usd"] is None and meta["totals"]["total_tokens"] == 1200
   and "no pricing.yaml" in meta["totals"]["cost_note"], meta["totals"]["cost_note"])

print("fail fast before any call:")
fake_targets.CALLS.clear()
for label, argv, expect in [
    ("validate_settings hook stops a bad experiment", [os.path.join(exp_dir, "bad_e.yaml"), "--yes"], "settings.bad_key"),
    ("unknown scenario id", [os.path.join(exp_dir, "child_e.yaml"), "--yes", "--scenarios", "nope"], "Unknown scenario ids: nope"),
    ("missing experiment file", [os.path.join(exp_dir, "missing.yaml"), "--yes"], "not found"),
]:
    try:
        run(argv)
        ok(label, False, "did not exit")
    except SystemExit as e:
        ok(label, expect in str(e), str(e).replace("\n", " | "))
ok("...and the target was never called", fake_targets.CALLS == [])

print("confirmation (subprocess, stdin from NUL, no --yes):")
env = {**os.environ, "GROQ_API_KEY": "offline-test"}
r = subprocess.run([sys.executable, os.path.join(REPO, "run_experiment.py"), "experiments_do_not_exist.yaml"],
                   cwd=REPO, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
ok("bad path exits non-zero", r.returncode != 0, r.stderr.strip())
probe = os.path.join(tmp, "probe.py")
open(probe, "w").write(textwrap.dedent(f'''
    import sys; sys.path[:0] = [{REPO!r}, {tmp!r}]
    import run_suite, run_experiment as rx
    run_suite.TARGETS_FILE = {run_suite.TARGETS_FILE!r}; run_suite.SCENARIOS_DIR = {run_suite.SCENARIOS_DIR!r}
    rx.RUNS_DIR = {rx.RUNS_DIR!r}
    import fake_targets
    rx.main([{os.path.join(exp_dir, "child_e.yaml")!r}])
    print("target calls:", len(fake_targets.CALLS))
'''))
r = subprocess.run([sys.executable, probe], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
ok("no --yes + no input -> aborted, 0 calls", "Aborted. No API calls were made." in r.stdout
   and "target calls: 0" in r.stdout, r.stdout.strip().splitlines()[-2:])

print("helpers:")
ok("p95 nearest rank", rx.p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 10 and rx.p95(list(range(1, 101))) == 95
   and rx.p95([3.0]) == 3.0 and rx.p95([]) is None)
ok("repo runs/ untouched by tests", os.path.exists(os.path.join(REPO, "runs")) == repo_runs_before)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
