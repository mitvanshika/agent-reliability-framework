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
import time
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

    # A fake LLM library at groq's real import path: the token meter wraps it like the real one
    import sys, types
    class _Usage:
        def __init__(self, i, o):
            self.prompt_tokens, self.completion_tokens, self.total_tokens = i, o, i + o
    class _Reply:
        def __init__(self, i, o):
            self.usage = _Usage(i, o)
    class Completions:
        def create(self, model=None, response_format=None, tin=0, tout=0):
            return _Reply(tin, tout)
    for name in ("groq", "groq.resources", "groq.resources.chat"):
        sys.modules.setdefault(name, types.ModuleType(name))
    _llm = types.ModuleType("groq.resources.chat.completions")
    _llm.Completions = Completions
    sys.modules["groq.resources.chat.completions"] = _llm

    def metered(question):
        """Reports no usage: its LLM calls are counted by the harness."""
        fmt = {"type": "json_schema", "json_schema": {"name": "grade"}}
        Completions().create(model="m1", response_format=fmt, tin=300, tout=20)
        Completions().create(model="m1", tin=500, tout=80)
        return "metered answer"
    CANCELLED = []
    def slow(question):
        """Gets stuck: never answers in time."""
        if "first" in question:
            time.sleep(3)
        return "late answer"
    def cancel():
        CANCELLED.append(True)
    def remote(question):
        """Like the Financial DD worker: its own process counted the calls and sends the list."""
        return {"answer": "remote answer", "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                "llm_calls": [{"label": "x", "model": "m2", "input_tokens": 7, "output_tokens": 3,
                               "total_tokens": 10, "seconds": 0.5}]}
'''))
sys.path.insert(0, tmp)
import fake_targets  # noqa: E402

# --- fake targets.yaml, scenarios, experiments, pricing
open(os.path.join(tmp, "targets.yaml"), "w").write(
    "plain_t:\n  module: fake_targets\n  function: plain\n"
    "settings_t:\n  module: fake_targets\n  function: with_settings\n  accepts_settings: true\n"
    "settings_off:\n  module: fake_targets\n  function: plain\n  accepts_settings: false\n"
    "metered_t:\n  module: fake_targets\n  function: metered\n"
    "remote_t:\n  module: fake_targets\n  function: remote\n"
    "slow_t:\n  module: fake_targets\n  function: slow\n  max_seconds: 0.3\n"
    "limited_t:\n  module: fake_targets\n  function: with_settings\n  accepts_settings: true\n  max_seconds: 5\n")
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
open(os.path.join(exp_dir, "metered_e.yaml"), "w").write("name: metered_e\ntarget: metered_t\nsettings: {}\n")
open(os.path.join(exp_dir, "remote_e.yaml"), "w").write("name: remote_e\ntarget: remote_t\nsettings: {}\n")
open(os.path.join(exp_dir, "slow_e.yaml"), "w").write("name: slow_e\ntarget: slow_t\nsettings: {}\n")
open(os.path.join(exp_dir, "limited_e.yaml"), "w").write("name: limited_e\ntarget: limited_t\nsettings: {k: 1}\n")
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

print("per-call breakdown (token meter):")
run_dir, out = run([os.path.join(exp_dir, "metered_e.yaml"), "--yes", "--scenarios", "s1"])
m1 = load(os.path.join(run_dir, "traces", "s1.json"))
ok("target reports no usage: the meter's count is used", m1["usage"] == {
    "input_tokens": 800, "output_tokens": 100, "total_tokens": 900} and m1["usage_source"] == "meter", m1["usage"])
ok("llm_calls: one entry per call, in order, with label/model/tokens/seconds",
   [(c["label"], c["model"], c["total_tokens"]) for c in m1["llm_calls"]] == [("grade", "m1", 320), (None, "m1", 580)]
   and all(c["seconds"] >= 0 for c in m1["llm_calls"]), m1["llm_calls"])
ok("printed: tokens in 2 calls", "900 tokens in 2 calls" in out, out.strip().splitlines()[-6:])
run_dir, out = run([os.path.join(exp_dir, "remote_e.yaml"), "--yes", "--scenarios", "s1"])
r1 = load(os.path.join(run_dir, "traces", "s1.json"))
ok("a target that sends its own llm_calls (own process): kept as sent", r1["llm_calls"] == [
    {"label": "x", "model": "m2", "input_tokens": 7, "output_tokens": 3, "total_tokens": 10, "seconds": 0.5}]
   and r1["usage_source"] == "target" and "10 tokens in 1 call " in out, (r1["llm_calls"], out))
s1 = load(os.path.join(rx.RUNS_DIR, sorted(d for d in os.listdir(rx.RUNS_DIR) if d.startswith("child_e"))[0],
                       "traces", "s1.json"))
ok("its own usage wins over the meter; no LLM calls seen -> llm_calls None",
   s1["usage_source"] == "target" and s1["llm_calls"] is None, (s1["usage_source"], s1["llm_calls"]))
p1 = load(os.path.join(rx.RUNS_DIR, sorted(d for d in os.listdir(rx.RUNS_DIR) if d.startswith("plain_e"))[0],
                       "traces", "s1.json"))
ok("no usage anywhere: usage None, usage_source None", p1["usage"] is None and p1["usage_source"] is None
   and p1["llm_calls"] is None)

print("time limit (max_seconds, 6c):")
t0 = time.perf_counter()
run_dir, out = run([os.path.join(exp_dir, "slow_e.yaml"), "--yes", "--scenarios", "s1", "s3"])
waited = time.perf_counter() - t0
st1 = load(os.path.join(run_dir, "traces", "s1.json"))
st3 = load(os.path.join(run_dir, "traces", "s3.json"))
ok("a stuck question stops at max_seconds, recorded as error_type timeout",
   st1["error_type"] == "timeout" and st1["answer"] is None and "max_seconds=0.3" in st1["error"]
   and 0.3 <= st1["latency_s"] < 1.5, (st1["error_type"], st1["latency_s"]))
ok("the run moved on: the next question answered normally", st3["error"] is None and st3["answer"] == "late answer")
ok("did not wait for the stuck call (3 s)", waited < 2.5, round(waited, 2))
ok("the target's cancel() was called once to stop the work", fake_targets.CANCELLED == [True], fake_targets.CANCELLED)
ok("timed out: no token count (it would be too low)", st1["usage"] is None and st1["usage_source"] is None)
ok("run_meta records the limit", load(os.path.join(run_dir, "run_meta.json")).get("max_seconds") == 0.3)
ok("printed as an error, counted by type", "ERROR timeout" in out and "'timeout': 1" in out, out[-400:])
run_dir, out = run([os.path.join(exp_dir, "limited_e.yaml"), "--yes", "--scenarios", "s1"])
lt = load(os.path.join(run_dir, "traces", "s1.json"))
ok("a limit that is not reached changes nothing (settings, usage, answer)", lt["error"] is None
   and lt["answer"] == "dict answer" and lt["usage"]["total_tokens"] == 1200, lt)
try:
    rx._call_with_limit(lambda: 1 / 0, 5)
    ok("an error inside a time-limited call is raised as itself", False)
except ZeroDivisionError:
    ok("an error inside a time-limited call is raised as itself", True)
plain_meta = load(os.path.join(rx.RUNS_DIR, sorted(d for d in os.listdir(rx.RUNS_DIR) if d.startswith("child_e"))[0],
                               "run_meta.json"))
ok("no max_seconds in targets.yaml: no limit recorded (as before)", "max_seconds" not in plain_meta)

print("Financial DD adapter: cancel() kills a stuck worker:")
from adapters import financial_dd as fdd  # noqa: E402
stuck = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, encoding="utf-8")
fdd._proc = stuck
t0 = time.perf_counter()
reply, timed_out = rx._call_with_limit(lambda: fdd._ask({"op": "answer", "question": "q", "settings": {}}),
                                       0.5, fdd.cancel)
time.sleep(0.5)
ok("timed out, worker process killed, next question gets a fresh worker",
   timed_out and stuck.poll() is not None and fdd._proc is None, (timed_out, stuck.poll()))
ok("...within seconds, not the worker's 60", time.perf_counter() - t0 < 10)

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