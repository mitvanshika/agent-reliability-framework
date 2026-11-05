"""
Tests for promote_baseline.py and judge_run's next-step hint (no API calls).
    python tests/test_promote_baseline.py
"""

import builtins
import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import compare  # noqa: E402
import judge_run  # noqa: E402
import promote_baseline as pb  # noqa: E402
import run_experiment  # noqa: E402
from fake_runs import make_run  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
compare.BASELINES_DIR = os.path.join(root, "baselines")
compare.NOISE_DIR = os.path.join(root, "noise")
run_experiment.PRICING_FILE = os.path.join(root, "none", "pricing.yaml")
IDS = [f"s{i:02d}" for i in range(1, 8)]
dest = os.path.join(compare.BASELINES_DIR, "configurable.json")


def promote(argv):
    buf, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            result = pb.main(argv)
        except SystemExit as e:
            code, result = e.code, None
    return code, result, buf.getvalue(), err.getvalue()


def hint(run_dir):
    return judge_run.next_step_hint(run_dir, judge_run.load_meta(run_dir))


print("refusals:")
unjudged = make_run(os.path.join(root, "runs"), "baseline", {s: 8.0 for s in IDS}, judged=False, timestamp="20261101_000000")
code, _, _, err = promote([unjudged, "--yes"])
ok("unjudged run refused (exit 2)", code == 2 and "7 not judged" in err and not os.path.exists(dest), err.strip())
rows = {s: 8.0 for s in IDS}
rows["s03"] = {"score": 8.0, "judge_error": "429"}
jerr = make_run(os.path.join(root, "runs"), "baseline", rows, timestamp="20261102_000000")
code, _, _, err = promote([jerr, "--yes"])
ok("judge errors refused (exit 2)", code == 2 and "1 judge errors (s03)" in err and not os.path.exists(dest), err.strip())

print("judge_run hint before any baseline:")
good = make_run(os.path.join(root, "runs"), "baseline", {s: 8.0 + i * 0.1 for i, s in enumerate(IDS)},
                timestamp="20261103_000000")
ok("'No baseline yet'", hint(good) == f"No baseline yet: python promote_baseline.py {good}", hint(good))

print("confirmation required:")
real_input, real_isatty = builtins.input, sys.stdin.isatty
builtins.input = lambda prompt="": "n"
sys.stdin.isatty = lambda: True
code, result, out, _ = promote([good])
ok("answering 'n' does not promote", result is None and "Not promoted." in out and not os.path.exists(dest))
builtins.input = lambda prompt="": "y"
code, result, out, _ = promote([good, "--note", "first baseline"])
builtins.input, sys.stdin.isatty = real_input, real_isatty
ok("answering 'y' promotes", os.path.isfile(dest) and "No current baseline for 'configurable'" in out)

print("baseline contents:")
b = json.load(open(dest, encoding="utf-8"))
ok("target / experiment / run / note / date", b["target"] == "configurable" and b["experiment"] == "baseline"
   and b["run_dir"].endswith("baseline_20261103_000000") and b["note"] == "first baseline" and b["promoted_at"][:2] == "20")
ok("config + changes vs parent + git", b["resolved_config"]["target"] == "configurable" and b["changes_vs_parent"] == []
   and b["git"]["commit"] == "f" * 40)
s1 = b["scenarios"]["s01"]
ok("per-scenario fields", s1["final_score"] == 8.0 and s1["passed"] is True and s1["judges"] == {
   "correctness": 8, "tool_efficiency": 8, "completeness": 8, "groundedness": 8} and s1["error"] is None
   and s1["error_type"] is None and s1["latency_s"] == 1.0 and s1["usage"]["total_tokens"] == 1100, s1)
ok("summary stored", b["summary"]["passed"] == 7 and b["summary"]["scenarios"] == 7)
ok("hint now 'Compare with'", hint(good) == f"Compare with: python compare.py {good}")

print("re-promote: comparison shown + old baseline backed up:")
better = make_run(os.path.join(root, "runs"), "exp_x", {s: 9.0 for s in IDS}, timestamp="20261104_000000")
code, result, out, _ = promote([better, "--yes", "--note", "second"])
hist = os.path.join(compare.BASELINES_DIR, "history")
backups = os.listdir(hist)
ok("comparison vs current baseline printed", "Compared with the current baseline (baseline" in out
   and "Verdict vs current baseline:" in out, [l for l in out.splitlines() if "Verdict" in l])
ok("old baseline in history/", len(backups) == 1 and backups[0].startswith("configurable_")
   and json.load(open(os.path.join(hist, backups[0]), encoding="utf-8"))["note"] == "first baseline", backups)
ok("new baseline written", json.load(open(dest, encoding="utf-8"))["note"] == "second")

print("--force:")
code, result, out, _ = promote([jerr, "--yes", "--force"])
b = json.load(open(dest, encoding="utf-8"))
ok("judge errors allowed with --force, stored as such", code == 0 and b["scenarios"]["s03"]["status"] == "judge_error"
   and "--force: 1 scenarios without a score" in out)
ok("second backup kept (no overwrite)", len(os.listdir(hist)) == 2)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
