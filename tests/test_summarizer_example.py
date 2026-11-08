"""
The example target `summarizer` (examples/summarizer/) through the whole
framework, offline: check_target -> pipeline (run + judge) -> promote ->
pipeline (run + judge + compare + diagnose). Judges faked, no API calls.
Also proves the framework needed no edits for it.
    python tests/test_summarizer_example.py
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import check_target  # noqa: E402
import compare  # noqa: E402
import diagnose  # noqa: E402
import experiment_pipeline as ep  # noqa: E402
import judge_sets  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


EXP = os.path.join("examples", "summarizer", "summarizer_baseline.yaml")
real_baseline = open(os.path.join("baselines", "configurable.json"), "rb").read()
root = tempfile.mkdtemp()
run_experiment.RUNS_DIR = diagnose.RUNS_DIR = os.path.join(root, "runs")
run_experiment.PRICING_FILE = os.path.join(root, "none.yaml")
compare.BASELINES_DIR = os.path.join(root, "baselines")
compare.NOISE_DIR = os.path.join(root, "noise")


def score(name, prompt):
    """A stand-in judge: low if the output misses a fact the reference calls key."""
    before, _, output = prompt.partition("Its output:")
    reference = before.partition("Reference answer")[2]        # empty for groundedness (no reference)
    return (3, 0.9) if "Maya Ortiz" in reference and "Maya Ortiz" not in output else (9, 0.9)


def quiet(fn, *a):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        r = fn(*a)
    return r, out.getvalue(), err.getvalue()


print("1. check_target:")
rep = check_target.check("summarizer", call=True)
ok("passes, one call returned a string, generic_v1 with tool judges N/A", not rep.failed and any(
   s == "OK" and c == "test call" and "returned a string" in d for s, c, d in rep.rows) and any(
   c == "judges" and "generic_v1" in d and "N/A: tool_efficiency, context_relevance, tool_match" in d
   for s, c, d in rep.rows), rep.rows)

print("2. first pipeline: run + judge, stops (no baseline yet):")
fake_judges.install(score)
code, out, err = quiet(ep.run_pipeline, [EXP, "--yes"])
runs = sorted(os.listdir(run_experiment.RUNS_DIR))
run1 = os.path.join(run_experiment.RUNS_DIR, runs[0])
ok("exit 0, stopped after judging: no baseline", code == 0 and "No baseline yet for target 'summarizer'" in out,
   (code, err[-300:]))
meta = json.load(open(os.path.join(run1, "run_meta.json"), encoding="utf-8"))
ok("3 scenarios from the target's own folder, suite main", sorted(meta["scenario_ids"]) == ["news_01", "news_02", "news_03"]
   and meta["suite"] == "main" and meta["scenario_dir"] == "examples/summarizer/scenarios")
tr = json.load(open(os.path.join(run1, "traces", "news_03.json"), encoding="utf-8"))
ok("a plain string return became a trace: no tools, the real summary",
   tr["tools_used"] == [] and tr["answer"].startswith("The annual marathon") and tr["error"] is None)
ok("9 judge calls (3 answers x 3 judges), all generic wording", len(fake_judges.CALLS) == 9
   and {n for n, _ in fake_judges.CALLS} == {"generic"}, [n for n, _ in fake_judges.CALLS])
g = [p for _, p in fake_judges.CALLS if "supported by the INPUT" in p]
ok("groundedness got the article as its source", len(g) == 3 and all("(none: only the input above)" in p
   and "summarises a short news article" in p for p in g))
j = json.load(open(os.path.join(run1, "judged", "news_03.json"), encoding="utf-8"))
ok("judged with generic_v1 (3 judges), tool judges N/A", j["judge_set"]["name"] == "generic_v1"
   and j["judge_set"]["llm_judges"] == ["correctness_gt", "completeness", "groundedness"]
   and all(j[f"{n}_result"].get("na") for n in ("tool_efficiency", "context_relevance", "tool_match")))
summ = json.load(open(os.path.join(run1, "summary.json"), encoding="utf-8"))
ok("news_03 (misses the winner) fails, the other two pass: 2/3 at threshold 7",
   summ["passed"] == 2 and summ["threshold"] == 7 and j["final_score"] < 7, (summ["passed"], j["final_score"]))

print("3. promote:")
quiet(promote_baseline.main, [run1, "--yes", "--note", "summarizer baseline"])
b = json.load(open(os.path.join(compare.BASELINES_DIR, "summarizer.json"), encoding="utf-8"))
ok("baselines/summarizer.json, generic_v1, suite main", b["judge_set"]["name"] == "generic_v1"
   and (b.get("suite") or "main") == "main" and len(b["scenarios"]) == 3)

print("4. second pipeline: run + judge + compare + diagnose:")
fake_judges.install(score)
code, out, err = quiet(ep.run_pipeline, [EXP, "--yes"])
run2 = os.path.join(run_experiment.RUNS_DIR, sorted(os.listdir(run_experiment.RUNS_DIR))[-1])
rec = json.load(open(os.path.join(run2, "pipeline.json"), encoding="utf-8"))
ok("all 4 steps ok", rec["steps"] == {"run": "ok", "judge": "ok", "compare": "ok", "diagnose": "ok"}, rec["steps"])
ok("same system twice -> not DECLINED (3 pairs: no real difference / not enough data)",
   rec["verdict"] in ("NO REAL DIFFERENCE", "NOT ENOUGH DATA"), rec["verdict"])
cj = json.load(open(os.path.join(run2, "comparison.json"), encoding="utf-8"))
ok("compared against baselines/summarizer.json, per-judge = the 3 that apply",
   cj["baseline_file"].endswith("summarizer.json") and set(cj["per_judge"]) == {"correctness_gt", "completeness",
                                                                                 "groundedness"}, set(cj["per_judge"]))

print("5. the framework needed no edits for this target:")
ALLOWED = ("examples/", "tests/", "evidence/")
ALLOWED_FILES = {"targets.yaml", "README.md"}


def framework(paths):
    return sorted(p for p in paths if p and not p.startswith(ALLOWED) and p not in ALLOWED_FILES)


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout


added_in = git("log", "--diff-filter=A", "--format=%H", "--", "examples/summarizer/summarizer.py").split()
if added_in:
    files = git("show", "--name-only", "--format=", added_in[-1]).splitlines()
    ok(f"the commit that added the summarizer ({added_in[-1][:7]}) touches no framework file",
       framework(files) == [], framework(files))
else:
    changed = git("diff", "--name-only", "HEAD").splitlines() + git("ls-files", "--others", "--exclude-standard").splitlines()
    ok("uncommitted: only examples/, tests/, targets.yaml changed (no framework file)", framework(changed) == [],
       framework(changed))

print("support bot untouched:")
ok("real baselines/configurable.json unchanged", open(os.path.join("baselines", "configurable.json"), "rb").read()
   == real_baseline)
ok("configurable still judged with the full experiment_v2", judge_sets.for_run(
   run_experiment.load_scenarios(None), __import__("target_config").get("configurable")) is judge_sets.EXPERIMENT_V2)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
