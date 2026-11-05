"""
Tests for experiment_pipeline.py with a fake target and fake judges (no API calls).
    python tests/test_experiment_pipeline.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import compare  # noqa: E402
import diagnose  # noqa: E402
import experiment_pipeline as ep  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402
import run_judges  # noqa: E402
import run_suite  # noqa: E402
from fake_runs import make_run  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
run_judges.time.sleep = lambda s: None
open(os.path.join(root, "pipe_target.py"), "w", encoding="utf-8").write(textwrap.dedent('''
    def run(question, settings):
        bad = settings.get("mode") == "bad"
        return {"answer": "BAD answer" if bad else "good answer", "tools_used": ["faq_search"],
                "tool_outputs": ["FAQ text"], "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
'''))
sys.path.insert(0, root)
open(os.path.join(root, "targets.yaml"), "w").write("pipe_t:\n  module: pipe_target\n  function: run\n  accepts_settings: true\n")
os.makedirs(os.path.join(root, "scen"))
IDS = [f"p{i:02d}" for i in range(1, 11)]
for sid in IDS:
    open(os.path.join(root, "scen", f"{sid}.yaml"), "w").write(
        f'id: "{sid}"\nquestion: "{sid}?"\ncategory: "normal"\nexpected_answer: "a good answer"\nexpected_tools: [faq_search]\n')
exp = os.path.join(root, "experiments")
os.makedirs(exp)
open(os.path.join(exp, "pbase.yaml"), "w").write("name: pbase\ntarget: pipe_t\nsettings:\n  mode: good\n  prompt: p1\n")
open(os.path.join(exp, "pbad.yaml"), "w").write("name: pbad\nextends: pbase\nsettings:\n  mode: bad\n")
open(os.path.join(exp, "psame.yaml"), "w").write("name: psame\nextends: pbase\nsettings:\n  prompt: p1\n")
open(os.path.join(exp, "broken.yaml"), "w").write("name: broken\ntarget: pipe_t\nsetings: {}\n")

run_suite.TARGETS_FILE = os.path.join(root, "targets.yaml")
run_suite.SCENARIOS_DIR = os.path.join(root, "scen")
run_experiment.RUNS_DIR = diagnose.RUNS_DIR = os.path.join(root, "runs")
run_experiment.PRICING_FILE = os.path.join(root, "none", "pricing.yaml")
compare.BASELINES_DIR = os.path.join(root, "baselines")
compare.NOISE_DIR = os.path.join(root, "noise")
sent = []
compare.send_slack_alert = lambda text: sent.append(text)
fake_judges.install(lambda name, prompt: (2, 0.9) if "BAD answer" in prompt else (9, 0.9))


def pipe(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ep.run_pipeline(argv)
    return code, out.getvalue(), err.getvalue()


def record(run_dir):
    return json.load(open(os.path.join(run_dir, "pipeline.json"), encoding="utf-8"))


latest = lambda prefix: sorted(os.path.join(run_experiment.RUNS_DIR, d) for d in os.listdir(run_experiment.RUNS_DIR)
                               if d.startswith(prefix + "_"))[-1]

print("no baseline yet:")
code, out, _ = pipe([os.path.join(exp, "pbase.yaml"), "--yes"])
base_run = latest("pbase")
ok("stops after judging, exit 0", code == 0 and "No baseline yet" in out
   and f"python promote_baseline.py {base_run}" in out, code)
ok("judged, not compared, nothing promoted", os.path.isdir(os.path.join(base_run, "judged"))
   and not os.path.exists(os.path.join(base_run, "comparison.json")) and not os.path.isdir(compare.BASELINES_DIR))
ok("pipeline.json records it", record(base_run)["steps"] == {"run": "ok", "judge": "ok", "compare": "skipped: no baseline"})
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([base_run, "--yes"])

print("full pipeline, same config (A/A):")
code, out, _ = pipe([os.path.join(exp, "psame.yaml"), "--yes"])
r = record(latest("psame"))
ok("all 4 steps, NO REAL DIFFERENCE, exit 0", code == 0 and r["steps"] == {s: "ok" for s in ep.STEPS}
   and r["verdict"] == "NO REAL DIFFERENCE", (code, r.get("verdict")))
ok("final verdict printed", "VERDICT: NO REAL DIFFERENCE" in out)

print("planted regression:")
code, out, _ = pipe([os.path.join(exp, "pbad.yaml"), "--yes", "--alert"])
bad_run = latest("pbad")
r = record(bad_run)
ok("DECLINED -> exit 1", code == 1 and r["verdict"] == "DECLINED", (code, r.get("verdict")))
ok("diagnose still ran after DECLINED", r["steps"]["diagnose"] == "ok" and len(r["causes"]) == 10
   and os.path.isfile(os.path.join(bad_run, "diagnosis.json")))
ok("final block lists down questions + causes", "VERDICT: DECLINED" in out and "Down: p01" in out)
ok("--alert passed through to compare (Slack)", len(sent) == 1 and "DECLINED" in sent[0])

print("step failures:")
code, _, err = pipe([os.path.join(exp, "broken.yaml"), "--yes"])
ok("bad experiment file -> stops at step 1, exit 2", code == 2 and "stopped at step 1/4 (run)" in err
   and "setings" in err, err.strip().splitlines()[-1][:110])
real_isatty = sys.stdin.isatty
sys.stdin.isatty = lambda: False
code, _, err = pipe([os.path.join(exp, "psame.yaml")])
sys.stdin.isatty = real_isatty
ok("not confirmed -> exit 2 (a CI job without --yes can't go green by doing nothing)",
   code == 2 and "not confirmed" in err)


def boom(name, prompt):
    raise RuntimeError("Groq rejected the API key (AuthenticationError)")


fake_judges.install(boom)
code, _, err = pipe([os.path.join(exp, "psame.yaml"), "--yes"])
dead = latest("psame")
ok("every judge failing -> stops at step 2, exit 2, resume hint", code == 2 and "stopped at step 2/4 (judge)" in err
   and f"Resume with: python judge_run.py {dead} --yes" in err, err.strip().splitlines()[-2][:110])
ok("...and the run itself is kept", os.path.isdir(os.path.join(dead, "traces"))
   and record(dead)["steps"]["judge"].startswith("failed"))
fake_judges.install(lambda name, prompt: (9, 0.9))

classic = make_run(tempfile.mkdtemp(dir=root), "pbase", {sid: 9.0 for sid in IDS}, target="pipe_t",
                   timestamp="20261101_000000")
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([classic, "--yes"])          # baseline now judged with the CLASSIC judge set
code, _, err = pipe([os.path.join(exp, "psame.yaml"), "--yes"])
ok("compare refused (judge sets differ) -> stops at step 3, exit 2", code == 2 and "stopped at step 3/4 (compare)" in err,
   err.strip().splitlines()[-2][:110])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
