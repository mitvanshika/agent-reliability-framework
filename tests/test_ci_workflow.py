"""
Tests for .github/workflows/experiment-check.yml (no API calls, no GitHub):
structure, gating, the key-missing skip (its real shell script run in bash),
and experiment_pipeline.py --summary-md.
    python tests/test_ci_workflow.py
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""   # console-safe
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


WF = os.path.join(REPO, ".github", "workflows", "experiment-check.yml")
wf = yaml.safe_load(open(WF, encoding="utf-8"))
on = wf[True]   # PyYAML reads the key `on` as boolean True (YAML 1.1); GitHub reads it as "on"
job = wf["jobs"]["experiment-check"]
steps = {s.get("name") or s.get("uses"): s for s in job["steps"]}

print("structure:")
paths = on["pull_request"]["paths"]
for p in ["prompts/**", "docstrings/**", "experiments/**", "scenarios/**", "retrieval.py", "agent_configurable.py",
          "judges/**", "baselines/**"]:
    ok(f"PR paths include {p}", p in paths)
ok("manual trigger", "workflow_dispatch" in on)
ok("runs experiment_pipeline on experiments/baseline.yaml", job["env"]["EXPERIMENT"] == "experiments/baseline.yaml"
   and "experiment_pipeline.py \"$EXPERIMENT\"" in steps["Run experiment pipeline vs committed baseline"]["run"])
ok("report -> job summary", "--summary-md \"$GITHUB_STEP_SUMMARY\"" in steps["Run experiment pipeline vs committed baseline"]["run"])
up = steps["Upload comparison"]
ok("comparison.json uploaded as an artifact, even after a failure",
   up["uses"].startswith("actions/upload-artifact@") and "runs/*/comparison.json" in up["with"]["path"]
   and up["if"].startswith("always()"))
ok("call budget: repo variable with a 10-scenario default", "vars.EXPERIMENT_CI_SCENARIOS" in job["env"]["CI_SCENARIOS"]
   and len(job["env"]["CI_SCENARIOS"].split("||")[1].strip(" '}").split()) == 10)
ok("pip cache + model cache", steps["actions/setup-python@v5"]["with"]["cache"] == "pip"
   and steps["Cache embedding model + FAQ vectors"]["with"]["path"] == ".cache")
ok("every step after the key check is gated on the key",
   all("steps.key.outputs.available == 'true'" in s.get("if", "") for s in job["steps"][2:]),
   [s.get("name") for s in job["steps"][2:] if "available" not in s.get("if", "")])
fail_step = steps["Fail if DECLINED or a step failed"]["run"]
ok("fails on DECLINED (1) and on step failure (2)", '"$code" = "1"' in fail_step and '"$code" != "0"' in fail_step)
default_ids = job["env"]["CI_SCENARIOS"].split("||")[1].strip(" '}").split()
import run_suite  # noqa: E402

known = {s["id"]: s for s in run_suite.load_scenarios()}
ok("default subset: real, test-split scenarios", all(i in known and known[i].get("split", "test") == "test" for i in default_ids))
old = open(os.path.join(REPO, ".github", "workflows", "regression-check.yml"), "rb").read()
ok("old workflow untouched (still the Track 1 pipeline)", b"python pipeline.py" in old and b"experiment" not in old)

print("key missing -> skip with a notice (the step's real script, run in bash):")
bash = shutil.which("bash")
script = steps["Check for API key"]["run"]


def run_key_step(key):
    tmp = tempfile.mkdtemp()
    out_f, sum_f = os.path.join(tmp, "out"), os.path.join(tmp, "summary")
    open(out_f, "w").close()
    open(sum_f, "w").close()
    env = {**os.environ, "GROQ_API_KEY": key, "GITHUB_OUTPUT": out_f, "GITHUB_STEP_SUMMARY": sum_f}
    r = subprocess.run([bash, "-e", "-c", script], env=env, capture_output=True, text=True)
    return r, open(out_f).read(), open(sum_f).read()


if bash:
    r, out, summary = run_key_step("")
    ok("no key: exit 0 (job not failed)", r.returncode == 0, r.stderr.strip())
    ok("no key: ::notice printed", "::notice title=Experiment check skipped::" in r.stdout)
    ok("no key: available=false, so every later step is skipped", out.strip() == "available=false")
    ok("no key: explained in the job summary", "Experiment check skipped" in summary and "forks" in summary)
    r, out, summary = run_key_step("gsk_dummy")
    ok("with key: available=true, no notice", r.returncode == 0 and out.strip() == "available=true" and "::notice" not in r.stdout)
else:
    print("  SKIP  bash not found")

print("experiment_pipeline.py --summary-md:")
import compare  # noqa: E402
import diagnose  # noqa: E402
import experiment_pipeline as ep  # noqa: E402
import fake_judges  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402
import run_judges  # noqa: E402

root = tempfile.mkdtemp()
run_judges.time.sleep = lambda s: None
open(os.path.join(root, "ci_target.py"), "w").write(textwrap.dedent('''
    def run(question, settings):
        return {"answer": "BAD answer" if settings.get("mode") == "bad" else "good answer",
                "tools_used": ["faq_search"], "tool_outputs": ["FAQ"]}
'''))
sys.path.insert(0, root)
open(os.path.join(root, "targets.yaml"), "w").write("ci_t:\n  module: ci_target\n  function: run\n  accepts_settings: true\n")
os.makedirs(os.path.join(root, "scen"))
for i in range(1, 9):
    open(os.path.join(root, "scen", f"c{i}.yaml"), "w").write(
        f'id: "c{i}"\nquestion: "q{i}?"\ncategory: "normal"\nexpected_answer: "good"\nexpected_tools: [faq_search]\n')
exp = os.path.join(root, "exp")
os.makedirs(exp)
open(os.path.join(exp, "cbase.yaml"), "w").write("name: cbase\ntarget: ci_t\nsettings: {mode: good}\n")
open(os.path.join(exp, "cbad.yaml"), "w").write("name: cbad\nextends: cbase\nsettings: {mode: bad}\n")
run_suite.TARGETS_FILE, run_suite.SCENARIOS_DIR = os.path.join(root, "targets.yaml"), os.path.join(root, "scen")
run_experiment.RUNS_DIR = diagnose.RUNS_DIR = os.path.join(root, "runs")
run_experiment.PRICING_FILE = os.path.join(root, "none.yaml")
compare.BASELINES_DIR, compare.NOISE_DIR = os.path.join(root, "baselines"), os.path.join(root, "noise")
fake_judges.install(lambda name, prompt: (2, 0.9) if "BAD answer" in prompt else (9, 0.9))
md = os.path.join(root, "summary.md")


def pipe(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ep.run_pipeline(argv + ["--summary-md", md])


code = pipe([os.path.join(exp, "cbase.yaml"), "--yes"])
text = open(md, encoding="utf-8").read()
ok("no baseline: summary says what to do", code == 0 and "not compared" in text and "No committed baseline" in text)
base_run = sorted(os.listdir(run_experiment.RUNS_DIR))[0]
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([os.path.join(run_experiment.RUNS_DIR, base_run), "--yes"])
open(md, "w").close()
code = pipe([os.path.join(exp, "cbad.yaml"), "--yes", "--scenarios", "c1", "c2", "c3", "c4", "c5", "c6", "c7"])
text = open(md, encoding="utf-8").read()
ok("DECLINED: verdict, table of down questions, causes, subset noted", code == 1 and "❌ DECLINED" in text
   and "| ▼ | c1 |" in text and "likely cause" in text and "Not run (call budget / subset): 1 scenario(s)" in text
   and "McNemar p=" in text, text.splitlines()[0])
open(md, "w").close()
code = pipe([os.path.join(exp, "missing.yaml"), "--yes"])
text = open(md, encoding="utf-8").read()
ok("failed step: says which step stopped", code == 2 and "Pipeline stopped at step `run`" in text, text.splitlines()[-2:])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
