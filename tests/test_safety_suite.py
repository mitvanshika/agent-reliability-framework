"""
Tests for the universal safety judge, the adversarial_v1 judge set and
per-suite baselines (LLM faked, no API calls).
    python tests/test_safety_suite.py
"""

import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import compare  # noqa: E402
import diagnose  # noqa: E402
import judge_run  # noqa: E402
import judge_sets  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402
import run_judges  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


run_judges.time.sleep = lambda s: None
root = tempfile.mkdtemp()
compare.BASELINES_DIR = os.path.join(root, "baselines")
compare.NOISE_DIR = os.path.join(root, "noise")
run_experiment.PRICING_FILE = os.path.join(root, "none.yaml")
real_main_baseline = open(os.path.join("baselines", "configurable.json"), "rb").read() \
    if os.path.isfile(os.path.join("baselines", "configurable.json")) else None
real_noise = open(os.path.join("noise", "configurable_noise.json"), "rb").read() \
    if os.path.isfile(os.path.join("noise", "configurable_noise.json")) else None
ADV_DIR = os.path.join("scenarios_adversarial", "configurable")
DESC = run_suite.load_targets()["configurable"]["description"]

print("judge set selection:")
adv = run_experiment.load_scenarios(ADV_DIR)
ok("adversarial folder -> adversarial_v1", judge_sets.for_scenarios(adv)["name"] == "adversarial_v1")
ok("main scenarios -> experiment_v1", judge_sets.for_scenarios(run_suite.load_scenarios())["name"] == "experiment_v1")
ok("adversarial_v1 = safety + groundedness (no completeness / tool_efficiency)",
   judge_sets.ADVERSARIAL["llm_judges"] == ["safety", "groundedness"])
ok("suite names", (run_experiment.suite_name(None), run_experiment.suite_name("scenarios"),
                   run_experiment.suite_name(ADV_DIR)) == ("main", "main", "adversarial"))


QUESTIONS = {s["id"]: s["question"] for s in adv}


def make_run(name, scenario_dir, ids, answer, ts):
    d = os.path.join(root, "runs", f"{name}_{ts}")
    os.makedirs(os.path.join(d, "traces"))
    json.dump({"experiment": name, "timestamp": ts, "scenario_ids": ids, "scenario_dir": scenario_dir,
               "suite": run_experiment.suite_name(scenario_dir), "parent": None, "changes_vs_parent": [],
               "resolved_config": {"target": "configurable", "settings": {"model": "m", "prompt": "prompts/v1.txt"}}},
              open(os.path.join(d, "run_meta.json"), "w"))
    for sid in ids:
        json.dump({"scenario_id": sid, "question": QUESTIONS.get(sid, f"{sid}?"), "answer": answer(sid), "error": None, "timestamp": ts,
                   "tools_used": [], "tool_outputs": [], "latency_s": 1.0,
                   "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
                  open(os.path.join(d, "traces", f"{sid}.json"), "w"))
    return d


def judge(d):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        s = judge_run.main([d, "--yes"])
    return s, out.getvalue()


print("adversarial run judged with the safety judge set:")
adv_ids = sorted(s["id"] for s in adv)
fake_judges.install(lambda name, prompt: (9, 0.9))
a1 = make_run("baseline", ADV_DIR.replace("\\", "/"), adv_ids, lambda sid: "I can't help with that.", "20261107_100000")
s1, out = judge(a1)
ok("2 LLM calls per answer (7 answers -> 14), no other judges", len(fake_judges.CALLS) == 14
   and {n for n, _ in fake_judges.CALLS} == {"safety", "groundedness"}, sorted({n for n, _ in fake_judges.CALLS}))
ok("estimate shows the adversarial set + suite", "adversarial_v1" in out and "up to 14" in out and "suite: adversarial" in out)
sp = fake_judges.called("safety")
inj = next(p for p in sp if "Ignore all previous instructions" in p)
ok("safety prompt: attack kind + description + expected behaviour from the ADVERSARIAL folder",
   "Kind of test: injection" in inj and DESC in inj and "Does not reveal its system prompt" in inj)
ok("safety prompt says refusals are correct", "Do not lower the score for not answering" in inj)
ok("safety prompt is generic (no support-bot words in the template)", all(
    w not in open(os.path.join("judges", "safety.py"), encoding="utf-8").read().lower()
    for w in ("order", "refund", "faq", "shipping")))
ok("empty message shown as such", any("(an empty message)" in p for p in sp))
ok("summary judge set adversarial_v1, per-judge safety + groundedness",
   s1["judge_set"]["name"] == "adversarial_v1" and set(s1["per_judge_mean"]) == {"safety", "groundedness"})

print("main run still uses experiment_v1:")
fake_judges.install(lambda name, prompt: (9, 0.9))
m1 = make_run("baseline", "scenarios", ["normal_01", "normal_02"], lambda sid: "ok", "20261107_110000")
s_main, _ = judge(m1)
ok("5 judges, answer key from scenarios/", s_main["judge_set"]["name"] == "experiment_v1"
   and {n for n, _ in fake_judges.CALLS} >= {"correctness_gt", "completeness", "tool_efficiency"}
   and "safety" not in {n for n, _ in fake_judges.CALLS})

print("per-suite baselines:")
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([a1, "--yes", "--note", "adversarial baseline"])
    promote_baseline.main([m1, "--yes", "--note", "main baseline"])
ok("separate files: configurable.json + configurable__adversarial.json",
   sorted(os.listdir(compare.BASELINES_DIR)) == ["configurable.json", "configurable__adversarial.json"])
ab = json.load(open(compare.baseline_path("configurable", suite="adversarial"), encoding="utf-8"))
ok("adversarial baseline records suite + scenario_dir + judge set", ab["suite"] == "adversarial"
   and ab["scenario_dir"].endswith("scenarios_adversarial/configurable") and ab["judge_set"]["name"] == "adversarial_v1")


def run_compare(argv):
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            compare.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


print("an attack that now succeeds -> SAFETY diagnosis:")
fake_judges.install(lambda name, prompt: (2, 0.9) if name == "safety" and "SYSTEM PROMPT:" in prompt else (9, 0.9))
a2 = make_run("exp_07_hardened_prompt", ADV_DIR.replace("\\", "/"), adv_ids,
              lambda sid: "SYSTEM PROMPT: You are a customer..." if sid == "adv_injection" else "I can't help with that.",
              "20261107_120000")
judge(a2)
code, out, err = run_compare([a2])
cj = json.load(open(os.path.join(a2, "comparison.json"), encoding="utf-8"))
ok("compared against the ADVERSARIAL baseline", cj["baseline_file"].endswith("configurable__adversarial.json"), cj["baseline_file"])
ok("adv_injection down, cause SAFETY", cj["down"] == ["adv_injection"]
   and cj["diagnosis"][0]["cause"] == "SAFETY" and "prompt_addons" in cj["diagnosis"][0]["suggestion"], cj["diagnosis"])
ok("safety in the per-judge deltas", "safety" in cj["per_judge"] and cj["per_judge"]["safety"]["delta"] < 0)

print("suites never mix:")
code, _, err = run_compare([a2, "--baseline", compare.baseline_path("configurable")])
ok("adversarial run vs main baseline -> refused", code == 2 and "Different suites" in err, err.strip()[:110])
code, _, err = run_compare([m1, "--baseline", compare.baseline_path("configurable", suite="adversarial")])
ok("main run vs adversarial baseline -> refused", code == 2 and "Different suites" in err)
ok("noise files are per suite", compare.load_bands("configurable", suite="adversarial")[1].endswith(
   "configurable__adversarial_noise.json: using default band 1.0 for every scenario")
   or "configurable__adversarial_noise.json" in compare.load_bands("configurable", suite="adversarial")[1])

print("diagnose rule:")
row = {"scenario_id": "x", "direction": "down", "baseline_score": 9, "run_score": 3, "delta": -6,
       "judge_deltas": {"safety": -7, "groundedness": -1}, "baseline_judges": {"safety": 9, "groundedness": 9},
       "run_judges": {"safety": 2, "groundedness": 8}, "run_error_type": None}
ok("safety drop -> SAFETY (before RAG / PROMPT)", diagnose.diagnose_row(row)["cause"] == "SAFETY")
row["run_error_type"] = "rate_limit"
ok("errors still come first", diagnose.diagnose_row(row)["cause"] == "INFRA")

print("real repo files untouched:")
ok("baselines/configurable.json unchanged", real_main_baseline is None
   or open(os.path.join("baselines", "configurable.json"), "rb").read() == real_main_baseline)
ok("noise/configurable_noise.json unchanged", real_noise is None
   or open(os.path.join("noise", "configurable_noise.json"), "rb").read() == real_noise)
ok("no configurable__adversarial.json in the real baselines/", not os.path.exists(
   os.path.join("baselines", "configurable__adversarial.json")))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
