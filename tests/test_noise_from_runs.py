"""
Tests for calibrate_noise.py --from-runs (no API calls).
    python tests/test_noise_from_runs.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import calibrate_noise  # noqa: E402
import compare  # noqa: E402
import experiment_config  # noqa: E402
from fake_runs import make_run  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
calibrate_noise.NOISE_DIR = compare.NOISE_DIR = os.path.join(root, "noise")
runs = os.path.join(root, "runs")
SETTINGS = {"model": "m1", "prompt": "prompts/v1.txt"}

a = make_run(runs, "baseline", {"s1": 9.0, "s2": 8.0, "s3": 7.5, "s4": 9.0, "s5": 9.0},
             settings=SETTINGS, timestamp="20261106_100000")
b = make_run(runs, "baseline", {"s1": 8.5, "s2": 5.5, "s3": 6.5,
                                "s4": {"score": 9.0, "judge_error": "429 tokens per day"},
                                "s5": {"error": "Recursion limit", "error_type": "loop_limit"}},
             settings=SETTINGS, timestamp="20261106_110000")


def run(argv):
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rep = calibrate_noise.main(argv)
        except SystemExit as e:
            rep, code = None, e.code
    return rep, out.getvalue(), code


print("bands from two existing runs:")
rep, out, code = run(["--from-runs", a, b])
S = rep["scenarios"]
ok("s1: 9.0 vs 8.5 -> range 0.5, no flip", S["s1"]["range"] == 0.5 and S["s1"]["flips"] == 0)
ok("s2: 8.0 vs 5.5 -> range 2.5, pass->fail flip, jumpy", S["s2"]["range"] == 2.5 and S["s2"]["flips"] == 1 and S["s2"]["jumpy"])
ok("s3: 7.5 vs 6.5 -> range 1.0, flip (7 is the line)", S["s3"]["range"] == 1.0 and S["s3"]["flips"] == 1)
ok("s4: judge error in run b skipped -> only 1 scored run -> no band", S["s4"]["range"] is None
   and S["s4"]["n_scored"] == 1 and S["s4"]["skipped_runs"] == [b.replace("\\", "/")])
ok("s5: crash in run b counts as a fail flip, but no score -> no band", S["s5"]["flips"] == 1 and S["s5"]["pass_fail"] == "PF"
   and S["s5"]["range"] is None)
ok("suite per run: judge errors not counted, crash counted", [p["total"] for p in rep["suite"]["per_run"]] == [5, 4]
   and [p["passed"] for p in rep["suite"]["per_run"]] == [5, 1])
ok("records mode, source runs, config hash, judge set", rep["mode"] == "from_runs" and rep["runs"] == 2
   and rep["config_hash"] == experiment_config.config_hash({"target": "configurable", "settings": SETTINGS})
   and rep["judge_set"]["name"] == "classic_v1")
ok("prints which scenarios got no band + the caveat", "No band" in out and "s4 s5" in out and "underestimates" in out)

print("compare uses it (fresh), with the 0.5 floor; stale after a config change:")
baseline = {"experiment": "baseline", "resolved_config": {"target": "configurable", "settings": SETTINGS},
            "judge_set": rep["judge_set"]}
bands, note = compare.load_bands("configurable", baseline=baseline)
ok("fresh: bands used", not note.startswith("WARNING") and bands == {"s1": 0.5, "s2": 2.5, "s3": 1.0}, bands)
ok("scenarios without a band fall back to the default", "s4" not in bands and "s5" not in bands)
changed = {**baseline, "resolved_config": {"target": "configurable", "settings": {**SETTINGS, "prompt": "prompts/weak.txt"}}}
bands, note = compare.load_bands("configurable", baseline=changed)
ok("baseline config changed -> stale, default band", bands == {} and note.startswith("WARNING: noise file is stale"))

print("refusals:")
other_cfg = make_run(runs, "exp_x", {"s1": 9.0}, settings={**SETTINGS, "prompt": "prompts/weak.txt"}, timestamp="20261106_120000")
_, _, code = run(["--from-runs", a, other_cfg])
ok("different config -> refused", isinstance(code, str) and "different config" in code, code)
other_target = make_run(runs, "fdd", {"s1": 9.0}, target="financial_dd", settings=SETTINGS, timestamp="20261106_130000")
_, _, code = run(["--from-runs", a, other_target])
ok("different target -> refused", isinstance(code, str) and "target" in code, code)
_, _, code = run(["--from-runs", a])
ok("fewer than 2 runs -> refused", isinstance(code, str) and "at least 2" in code)
unjudged = make_run(runs, "baseline", {"s1": 9.0}, settings=SETTINGS, judged=False, timestamp="20261106_140000")
_, _, code = run(["--from-runs", a, unjudged])
ok("unjudged run -> refused", isinstance(code, str) and "not been judged" in code)
exp_js = make_run(runs, "baseline", {"s1": 9.0}, settings=SETTINGS, timestamp="20261106_150000")
jf = os.path.join(exp_js, "judged", "s1.json")
t = json.load(open(jf, encoding="utf-8"))
t["judge_set"] = {"name": "experiment_v1", "llm_judges": ["correctness_gt"], "checks": [], "judge_model": "m"}
json.dump(t, open(jf, "w", encoding="utf-8"))
_, _, code = run(["--from-runs", a, exp_js])
ok("different judge set -> refused", isinstance(code, str) and "judged with" in code, code)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
