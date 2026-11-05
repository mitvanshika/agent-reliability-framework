"""
Tests for compare.py on fake runs with known numbers (no API calls).
    python tests/test_compare.py
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

import compare  # noqa: E402
import promote_baseline  # noqa: E402
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
sent = []
compare.send_slack_alert = lambda text: sent.append(text)

IDS = [f"s{i:02d}" for i in range(1, 26)]
BASE = {sid: 8.0 + (i % 3) * 0.5 for i, sid in enumerate(IDS)}       # 8.0 / 8.5 / 9.0, all pass


def records(rows, name="run", **kw):
    d = make_run(tempfile.mkdtemp(dir=root), name, rows, **kw)
    return d, compare.load_run_records(d)[1]


def run_main(argv):
    buf, code = io.StringIO(), 0
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        try:
            compare.main(argv)
        except SystemExit as e:
            code = e.code
    return code, buf.getvalue(), err.getvalue()


# promote the fake baseline once (also exercises promote_baseline)
base_dir = make_run(os.path.join(root, "runs"), "baseline", BASE)
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([base_dir, "--yes", "--note", "test baseline"])
_, base_rec = compare.load_run_records(base_dir)

print("McNemar exact:")
for b, c, expect in [(7, 0, 0.015625), (6, 1, 0.125), (4, 0, 0.125), (0, 0, 1.0), (0, 7, 0.015625)]:
    p = compare.mcnemar_exact(b, c)
    ok(f"b={b} c={c} -> {expect}", abs(p - expect) < 1e-9, f"{p:.6f}")

print("Wilcoxon:")
p, n, med, d, note = compare.wilcoxon_paired([0.0] * 10)
ok("all zero -> p=1", p == 1.0 and n == 0)
p, n, med, d, note = compare.wilcoxon_paired([-1, -2, -1.5, 0, 0, 0])
ok("fewer than 6 non-zero -> warning", n == 3 and "cannot reach significance" in note, note)
ok("median over all pairs reported", med == -0.5, med)

print("A/A (identical):")
code, out, _ = run_main([base_dir])
r = json.load(open(os.path.join(base_dir, "comparison.json"), encoding="utf-8"))
ok("NO REAL DIFFERENCE, all same, both p=1", r["verdict"] == "NO REAL DIFFERENCE" and r["counts"]["same"] == 25
   and r["mcnemar"]["p"] == 1.0 and r["wilcoxon"]["p"] == 1.0 and code == 0, r["counts"])
ok("default band announced", "using default band 1.0" in out)

print("A/A with small symmetric wobble (+-0.3):")
jitter = {sid: BASE[sid] + (0.3 if i % 2 else -0.3) for i, sid in enumerate(IDS)}
_, rec = records(jitter, "baseline")
r = compare.compare_records(base_rec, rec)
ok("NO REAL DIFFERENCE, nothing down", r["verdict"] == "NO REAL DIFFERENCE" and r["down"] == [],
   f"p_w={r['wilcoxon']['p']:.3f}")

print("planted regression (6 questions drop to 3.0):")
planted = ["s03", "s07", "s11", "s15", "s19", "s23"]
reg = {sid: (3.0 if sid in planted else BASE[sid]) for sid in IDS}
reg_dir, rec = records(reg, "exp_01_weak_prompt")
code, out, _ = run_main([reg_dir, "--alert"])
r = json.load(open(os.path.join(reg_dir, "comparison.json"), encoding="utf-8"))
ok("exactly the planted questions are down", r["down"] == planted, r["down"])
ok("McNemar 6/0 = 0.03125", r["mcnemar"]["b_pass_to_fail"] == 6 and abs(r["mcnemar"]["p"] - 0.03125) < 1e-9)
ok("DECLINED, exit code 1", r["verdict"] == "DECLINED" and code == 1, (r["verdict"], code))
ok("--alert sent to Slack", len(sent) == 1 and "DECLINED" in sent[0])
row = next(x for x in r["rows"] if x["scenario_id"] == "s03")
ok("judges moved listed", "correctness 8->3" in row["judges_moved"] or "correctness 9->3" in row["judges_moved"], row["judges_moved"])
ok("config diff vs baseline shown", "settings.prompt: prompts/baseline.txt -> prompts/exp_01_weak_prompt.txt" in out)
print("    --- printed table (first lines) ---")
lines = out.splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("scenario"))
print("    " + "\n    ".join(lines[:4] + lines[start:start + 5] + [l for l in lines if l.startswith(("McNemar", "Wilcoxon", "VERDICT"))]))

sent.clear()
code, out, _ = run_main([base_dir, "--alert"])
ok("--alert not sent when nothing declined", sent == [] and "no Slack alert sent" in out)

print("improvement:")
imp_ids = IDS[:8]
low_base = {sid: (5.0 if sid in imp_ids else BASE[sid]) for sid in IDS}
_, low_rec = records(low_base, "baseline")
_, rec = records({sid: (9.0 if sid in imp_ids else BASE[sid]) for sid in IDS})
r = compare.compare_records(low_rec, rec)
ok("IMPROVED (8 fail->pass, nothing down)", r["verdict"] == "IMPROVED", (r["verdict"], r["verdict_reason"]))
# small regression (s19 8.0 -> 6.5): beyond the band and a pass->fail flip, but the smallest Wilcoxon rank
_, rec = records({sid: (9.0 if sid in imp_ids else 6.5 if sid == "s19" else BASE[sid]) for sid in IDS})
r = compare.compare_records(low_rec, rec)
ok("IMPROVED WITH REGRESSIONS (8 up, small s19 regression)",
   r["verdict"] == "IMPROVED WITH REGRESSIONS" and r["down"] == ["s19"], (r["verdict"], r["verdict_reason"]))
# one LARGE regression (s19 8.0 -> 3.5) outweighs 8 moderate wins in Wilcoxon -> cautious verdict
_, rec = records({sid: (9.0 if sid in imp_ids else 3.5 if sid == "s19" else BASE[sid]) for sid in IDS})
r = compare.compare_records(low_rec, rec)
ok("large regression: McNemar improved, Wilcoxon not -> NO REAL DIFFERENCE, s19 listed",
   r["mcnemar"]["verdict"] == "IMPROVED WITH REGRESSIONS" and r["wilcoxon"]["verdict"] == "NO REAL DIFFERENCE"
   and r["verdict"] == "NO REAL DIFFERENCE" and r["down"] == ["s19"], r["verdict_reason"])

print("tests disagree -> more cautious verdict:")
_, rec = records({sid: (BASE[sid] - 1.5 if i < 10 else BASE[sid]) for i, sid in enumerate(IDS)})   # drop, all still pass
r = compare.compare_records(base_rec, rec)
ok("Wilcoxon DECLINED + McNemar NO REAL DIFFERENCE -> DECLINED",
   r["wilcoxon"]["verdict"] == "DECLINED" and r["mcnemar"]["verdict"] == "NO REAL DIFFERENCE"
   and r["verdict"] == "DECLINED" and "disagree" in r["verdict_reason"], r["verdict_reason"])
six_up = {sid: (5.0 if i < 6 else BASE[sid]) for i, sid in enumerate(IDS)}     # s01-s06 fail, s07 passes (8.0)
_, six_base = records(six_up, "baseline")
_, rec = records({sid: (9.0 if i < 6 else 6.5 if i == 6 else BASE[sid]) for i, sid in enumerate(IDS)})
r = compare.compare_records(six_base, rec)
ok("6 up / 1 down: McNemar p=0.125 -> cautious NO REAL DIFFERENCE, s07 still listed down",
   abs(r["mcnemar"]["p"] - 0.125) < 1e-9 and r["verdict"] == "NO REAL DIFFERENCE" and r["down"] == ["s07"],
   (r["verdict"], r["verdict_reason"]))

print("crash counts as fail:")
crash = dict(BASE)
crash["s05"] = {"error": "Recursion limit of 25 reached", "error_type": "loop_limit"}
crash_dir, rec = records(crash)
r = compare.compare_records(base_rec, rec)
row = next(x for x in r["rows"] if x["scenario_id"] == "s05")
ok("crash: score 0, down, P->F, note", row["run_score"] == 0.0 and row["direction"] == "down"
   and row["flip"] == "pass->fail" and row["notes"] == ["run crashed (loop_limit)"], row)
code, out, _ = run_main([crash_dir])
r = json.load(open(os.path.join(crash_dir, "comparison.json"), encoding="utf-8"))
ok("error delta reported", r["resources"]["agent_errors"]["delta"] == 1
   and r["resources"]["errors_by_type"]["loop_limit"] == {"baseline": 0, "run": 1}, r["resources"]["errors_by_type"])
ok("token/latency deltas", r["resources"]["total_tokens"]["delta"] == -1100 and r["resources"]["latency_mean_s"]["run"] == 1.0)

print("missing scenarios + judge errors:")
miss = {sid: BASE[sid] for sid in IDS[:-1]}
miss["s99"] = 8.0
miss["s02"] = {"score": 8.0, "judge_error": "429 rate limit"}
_, rec = records(miss)
r = compare.compare_records(base_rec, rec)
ok("missing listed both ways", r["missing_in_run"] == ["s25"] and r["missing_in_baseline"] == ["s99"])
ok("judge error excluded, not counted as fail", [e["scenario_id"] for e in r["excluded"]] == ["s02"]
   and r["pairs"] == 23 and r["flips"]["pass_to_fail"] == 0, r["excluded"])

print("noise file vs default band:")
drop = {sid: (BASE[sid] - 2.5 if sid == "s10" else BASE[sid]) for sid in IDS}
_, rec = records(drop)
ok("default band 1.0: -2.5 is down", compare.compare_records(base_rec, rec)["down"] == ["s10"])
os.makedirs(compare.NOISE_DIR, exist_ok=True)
json.dump({"mode": "full", "runs": 3, "scenarios": {"s10": {"range": 3.0}, "s11": {"range": 0.1}}},
          open(os.path.join(compare.NOISE_DIR, "configurable_noise.json"), "w"))
bands, note = compare.load_bands("configurable")
r = compare.compare_records(base_rec, rec, bands)
ok("noise band 3.0: -2.5 is within noise", r["down"] == [] and r["rows"][9]["band"] == 3.0)
ok("floor 0.5 for a tiny range, default for missing", bands["s11"] == 0.5 and "s12" not in bands
   and next(x for x in r["rows"] if x["scenario_id"] == "s12")["band"] == 1.0)
ok("noise source described", "configurable_noise.json (full mode, 3 runs" in note, note)
os.remove(os.path.join(compare.NOISE_DIR, "configurable_noise.json"))

print("not enough data / refusals:")
_, rec = records({sid: BASE[sid] for sid in IDS[:4]})
ok("4 pairs -> NOT ENOUGH DATA", compare.compare_records(base_rec, rec)["verdict"] == "NOT ENOUGH DATA")
other = make_run(tempfile.mkdtemp(dir=root), "x", BASE, target="financial_dd")
os.makedirs(os.path.join(root, "fd"), exist_ok=True)
code, _, err = run_main([other])
ok("no baseline for target -> refused, exit 2", code == 2 and "No baseline at" in err, err.strip())
code, _, err = run_main([other, "--baseline", os.path.join(compare.BASELINES_DIR, "configurable.json")])
ok("targets differ -> refused, exit 2", code == 2 and "Targets differ" in err, err.strip())
unj = make_run(tempfile.mkdtemp(dir=root), "y", BASE, judged=False)
code, _, err = run_main([unj])
ok("unjudged run -> refused, exit 2", code == 2 and "not been judged" in err, err.strip())

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
