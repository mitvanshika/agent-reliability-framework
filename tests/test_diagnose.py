"""
Phase 5 tests (no API calls): diagnosis rules and their order, the
fine-tuning signal, stale noise files, draft experiments, and
calibrate_noise.py --experiment with a fake target + fake judges.
    python tests/test_diagnose.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import calibrate_noise  # noqa: E402
import compare  # noqa: E402
import diagnose  # noqa: E402
import experiment_config  # noqa: E402
import judge_sets  # noqa: E402
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


run_judges.time.sleep = lambda s: None
root = tempfile.mkdtemp()
compare.BASELINES_DIR = os.path.join(root, "baselines")
compare.NOISE_DIR = os.path.join(root, "noise")
run_experiment.PRICING_FILE = os.path.join(root, "none", "pricing.yaml")
JUDGES = ["correctness_gt", "completeness", "groundedness", "tool_efficiency", "context_relevance", "tool_match"]


def row(deltas=None, base=None, run_err=None, run_error=None, disagree=False, lowconf=(), sid="s01", na_both=()):
    base = base or {n: 9 for n in JUDGES if n not in na_both}
    deltas = {n: d for n, d in (deltas or {}).items() if n not in na_both}
    run = {n: base[n] + deltas.get(n, 0) for n in base} if run_err is None else {}
    common = {n: run[n] - base[n] for n in base if n in run}
    return {"scenario_id": sid, "direction": "down", "baseline_score": 9.0, "run_score": 3.0, "delta": -6.0,
            "judge_deltas": common, "baseline_judges": base, "run_judges": run,
            "run_error_type": run_err, "run_error": run_error if run_err else None,
            "run_disagreement": disagree, "run_low_confidence": list(lowconf)}


cause = lambda r: diagnose.diagnose_row(r)["cause"]

print("one fake comparison row per rule:")
ok("rate_limit -> INFRA", cause(row(run_err="rate_limit")) == "INFRA")
ok("timeout -> INFRA", cause(row(run_err="timeout")) == "INFRA")
ok("auth -> CONFIG", cause(row(run_err="auth")) == "CONFIG")
ok("loop_limit -> TOOLS/PROMPT", cause(row(run_err="loop_limit")) == "TOOLS/PROMPT")
ok("other crash -> AGENT CRASH", cause(row(run_err="crash", run_error="KeyError 'x'")) == "AGENT CRASH")
ok("context_relevance dropped -> RAG", cause(row({"context_relevance": -6, "correctness_gt": -1})) == "RAG")
ok("context fine + correctness_gt dropped -> PROMPT (used info badly)",
   cause(row({"correctness_gt": -5, "context_relevance": 0})) == "PROMPT (used info badly)")
ok("context fine + groundedness dropped -> PROMPT (used info badly)",
   cause(row({"groundedness": -4})) == "PROMPT (used info badly)")
ok("no retrieval on either side + correctness dropped -> PROMPT (used info badly)",
   cause(row({"correctness_gt": -5}, na_both=("context_relevance",))) == "PROMPT (used info badly)")
ok("classic judge set 'correctness' counts too",
   cause(row({"correctness": -5}, base={"correctness": 9, "completeness": 9, "groundedness": 9, "tool_efficiency": 9})) == "PROMPT (used info badly)")
ok("tool_efficiency dropped -> TOOL DESCRIPTIONS", cause(row({"tool_efficiency": -5})) == "TOOL DESCRIPTIONS")
ok("tool_match dropped -> TOOL DESCRIPTIONS", cause(row({"tool_match": -6})) == "TOOL DESCRIPTIONS")
ok("completeness dropped -> PROMPT (missed part)", cause(row({"completeness": -4})) == "PROMPT (missed part)")
ok("small drops + low confidence -> MEASUREMENT UNSURE",
   cause(row({"correctness_gt": -1, "completeness": -1}, lowconf=["correctness_gt"])) == "MEASUREMENT UNSURE")
ok("small drops + disagreement -> MEASUREMENT UNSURE", cause(row({"groundedness": -1}, disagree=True)) == "MEASUREMENT UNSURE")
d = diagnose.diagnose_row(row({"correctness_gt": -1.5, "completeness": -1, "groundedness": -1}))
ok("several small drops -> UNCLEAR, lists them", d["cause"] == "UNCLEAR" and "correctness_gt -1.5" in d["evidence"]
   and "completeness -1" in d["evidence"], d["evidence"][:90])
d = diagnose.diagnose_row(row({"correctness_gt": -5, "completeness": -5, "groundedness": -4.5, "tool_efficiency": -5}))
ok("several judges dropped about equally -> UNCLEAR (no single judge stands out)",
   d["cause"] == "UNCLEAR" and "about equally" in d["evidence"], d["evidence"][:80])

print("order of rules:")
ok("infra error beats judge drops", cause(row({"context_relevance": -9, "correctness_gt": -9}, run_err="rate_limit")) == "INFRA")
ok("loop_limit beats context drop", cause(row({"context_relevance": -9}, run_err="loop_limit")) == "TOOLS/PROMPT")
ok("RAG beats PROMPT when context also dropped", cause(row({"context_relevance": -4, "correctness_gt": -6})) == "RAG")
ok("PROMPT beats TOOL DESCRIPTIONS", cause(row({"correctness_gt": -6, "tool_match": -6})) == "PROMPT (used info badly)")
ok("TOOL DESCRIPTIONS beats completeness", cause(row({"tool_efficiency": -3, "completeness": -6})) == "TOOL DESCRIPTIONS")
ok("judge drop beats MEASUREMENT UNSURE", cause(row({"completeness": -5}, lowconf=["completeness"])) == "PROMPT (missed part)")
ok("threshold is a constant at the top (DROP=2.0): -1.9 is not a drop",
   diagnose.DROP == 2.0 and cause(row({"completeness": -1.9})) == "UNCLEAR")
d = diagnose.diagnose_row(row({"context_relevance": -6}))
ok("evidence has the judge deltas + suggested experiment", "context_relevance -6" in d["evidence"]
   and "final_score 9 -> 3" in d["evidence"] and "retrieval" in d["suggestion"], d["suggestion"])
try:
    diagnose.diagnose({"rows": [{"scenario_id": "x", "direction": "down"}]})
    ok("old comparison.json -> clear error", False)
except ValueError as e:
    ok("old comparison.json -> clear error", "re-run compare.py" in str(e))

print("end to end: compare -> diagnosis section -> diagnose.py:")
IDS = [f"s{i:02d}" for i in range(1, 11)]
R = lambda score, **j: {"score": score, "judges": [j.get("c", 9), j.get("t", 9), j.get("m", 9), j.get("g", 9)]}
base_rows = {sid: R(9.0) for sid in IDS}
base_dir = make_run(os.path.join(root, "runs"), "baseline", base_rows, timestamp="20261101_000000")
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([base_dir, "--yes"])
run_rows = dict(base_rows)
run_rows["s02"] = {"error": "429 Too Many Requests", "error_type": "rate_limit"}
run_rows["s05"] = R(4.0, c=3)          # classic correctness dropped
run_rows["s08"] = R(5.0, t=2)          # tool_efficiency dropped
run_dir = make_run(os.path.join(root, "runs"), "exp_01_weak_prompt", run_rows,
                   settings={"model": "m1", "prompt": "prompts/weak.txt"}, timestamp="20261102_000000")
out = io.StringIO()
with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
    try:
        compare.main([run_dir])
    except SystemExit:
        pass
cj = json.load(open(os.path.join(run_dir, "comparison.json"), encoding="utf-8"))
causes = {d["scenario_id"]: d["cause"] for d in cj["diagnosis"]}
ok("comparison.json has a diagnosis per down question", causes == {"s02": "INFRA", "s05": "PROMPT (used info badly)",
   "s08": "TOOL DESCRIPTIONS"}, causes)
ok("compare prints the short diagnosis section", "Diagnosis of the 3 down question(s)" in out.getvalue()
   and "python diagnose.py" in out.getvalue())
diagnose.RUNS_DIR = os.path.join(root, "runs")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = diagnose.main([run_dir])
ok("diagnose.py prints causes + evidence + next, saves diagnosis.json",
   "INFRA" in buf.getvalue() and "evidence:" in buf.getvalue() and "next:" in buf.getvalue()
   and os.path.isfile(os.path.join(run_dir, "diagnosis.json")))

print("fine-tuning signal:")
ft_root = os.path.join(root, "ft_runs")
for i, prompt in enumerate(["prompts/v1.txt", "prompts/v2.txt", "prompts/v3.txt"]):
    make_run(ft_root, f"exp_p{i}", {"q_hard": 4.0, "q_same": 4.0 if i == 0 else 9.0, "q_ok": 9.0},
             settings={"model": "m1", "prompt": prompt}, timestamp=f"2026110{i + 1}_000000")
for i in range(3):   # q_same fails 3 more times, but always with the SAME prompt
    make_run(ft_root, f"exp_same{i}", {"q_hard": 9.0, "q_same": 3.0, "q_ok": 9.0},
             settings={"model": "m1", "prompt": "prompts/v1.txt"}, timestamp=f"2026110{i + 4}_000000")
make_run(ft_root, "other_target", {"q_ok": 1.0}, target="financial_dd",
         settings={"prompt": "a"}, timestamp="20261108_000000")
make_run(ft_root, "unjudged", {"q_ok": 1.0}, judged=False, settings={"prompt": "z"}, timestamp="20261109_000000")
cands, _ = diagnose.fine_tune_candidates("configurable", ft_root)
ok("failed under 3 DIFFERENT prompts -> fine-tune candidate", list(cands) == ["q_hard"], list(cands))
ok("same prompt 3 times -> no signal (q_same failed 4x, but only 1 distinct prompt)", "q_same" not in cands)
ok("evidence lists the runs", len(cands["q_hard"]) == 3 and {r["prompt"] for r in cands["q_hard"]}
   == {"prompts/v1.txt", "prompts/v2.txt", "prompts/v3.txt"})
ok("other targets and unjudged runs ignored", "q_ok" not in cands)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    diagnose.print_fine_tune(cands, False)
ok("printed: 'prompting isn't fixing this, fine-tuning candidate'", "fine-tuning candidate" in buf.getvalue())

print("stale noise file:")
base = json.load(open(compare.baseline_path("configurable"), encoding="utf-8"))
os.makedirs(compare.NOISE_DIR, exist_ok=True)
noise_path = os.path.join(compare.NOISE_DIR, "configurable_noise.json")


def write_noise(**kw):
    json.dump({"mode": "full", "runs": 3, "experiment": "baseline", "scenarios": {"s01": {"range": 2.5}}, **kw},
              open(noise_path, "w", encoding="utf-8"))


write_noise(config_hash=experiment_config.config_hash(base["resolved_config"]), judge_set=base.get("judge_set"))
bands, note = compare.load_bands("configurable", baseline=base)
ok("matching config_hash + judge set -> bands used", bands == {"s01": 2.5} and not note.startswith("WARNING"), note[:60])
write_noise(config_hash="0000000000000000", judge_set=base.get("judge_set"))
bands, note = compare.load_bands("configurable", baseline=base)
ok("different config_hash -> stale, default band", bands == {} and note.startswith("WARNING: noise file is stale")
   and "different config" in note, note[:110])
write_noise(config_hash=experiment_config.config_hash(base["resolved_config"]),
            judge_set={**judge_sets.EXPERIMENT, "judge_model": "x"})
bands, note = compare.load_bands("configurable", baseline=base)
ok("different judge set -> stale", bands == {} and "judge set" in note, note[:110])
write_noise()
bands, note = compare.load_bands("configurable", baseline=base)
ok("old noise file without config_hash -> stale", bands == {} and "no config_hash" in note)
err = io.StringIO()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
    try:
        compare.main([run_dir])
    except SystemExit:
        pass
ok("compare.py warns on stderr", "noise file is stale" in err.getvalue())
os.remove(noise_path)

print("draft experiment:")
exp_dir = os.path.join(root, "experiments")
os.makedirs(exp_dir)
shutil.copy(os.path.join(REPO, "experiments", "baseline.yaml"), exp_dir)
shutil.copy(os.path.join(REPO, "experiments", "exp_01_weak_prompt.yaml"), exp_dir)
cmp = {"experiment": "exp_01_weak_prompt", "target": "configurable", "run_dir": "runs/exp_01_x",
       "baseline_experiment": "baseline"}
diags = [diagnose.diagnose_row(row({"correctness_gt": -5}, sid=s)) for s in ("a", "b")] + \
        [diagnose.diagnose_row(row(run_err="rate_limit", sid="c"))]
now = datetime(2026, 11, 6, 12, 0, 0)
path = diagnose.draft_experiment(diags, cmp, experiments_dir=exp_dir, now=now)
text = open(path, encoding="utf-8").read()
raw = open(path, "rb").read()
ok("file name draft_<cause>_<timestamp>.yaml", os.path.basename(path) == "draft_prompt_used_info_badly_20261106_120000.yaml",
   os.path.basename(path))
ok("extends the run's experiment, TODO on the setting, not-run note",
   "extends: exp_01_weak_prompt" in text and "# TODO" in text and "prompt: prompts/TODO_new_prompt.txt" in text
   and "NOT run automatically" in text and "2 of 3: a b" in text)
ok("CRLF like the rest of the repo", raw.count(b"\r\n") == raw.count(b"\n"))
resolved = experiment_config.load_experiment(path)
ok("it is a valid experiment file (name = file name) changing only the prompt",
   resolved["parent"] == "exp_01_weak_prompt" and resolved["changes_vs_parent"]
   == ["settings.prompt: prompts/weak.txt -> prompts/TODO_new_prompt.txt"], resolved["changes_vs_parent"])
import agent_configurable  # noqa: E402

try:
    agent_configurable.validate_settings(resolved["settings"])
    ok("...but cannot run until the TODO is filled in", False)
except agent_configurable.SettingsError as e:
    ok("...but cannot run until the TODO is filled in", "file not found: prompts/TODO_new_prompt.txt" in str(e))
try:
    diagnose.draft_experiment(diags, cmp, experiments_dir=exp_dir, now=now)
    ok("never overwrites an existing draft", False)
except FileExistsError:
    ok("never overwrites an existing draft", True)
ok("only infra causes -> no draft", diagnose.draft_experiment([diags[2]], cmp, experiments_dir=exp_dir) is None)

print("calibrate_noise.py --experiment (fake target + fake judges):")
open(os.path.join(root, "fake_target.py"), "w", encoding="utf-8").write(textwrap.dedent('''
    N = {"n": 0}
    def run(question, settings):
        N["n"] += 1
        return {"answer": "a", "tools_used": ["faq_search"], "tool_outputs": ["FAQ text"],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
'''))
sys.path.insert(0, root)
open(os.path.join(root, "targets.yaml"), "w").write("fake_t:\n  module: fake_target\n  function: run\n  accepts_settings: true\n")
os.makedirs(os.path.join(root, "scen"))
for sid in ("c1", "c2", "c3"):
    open(os.path.join(root, "scen", f"{sid}.yaml"), "w").write(
        f'id: "{sid}"\nquestion: "{sid}?"\ncategory: "normal"\nexpected_answer: "x"\nexpected_tools: [faq_search]\n')
open(os.path.join(exp_dir, "cal_base.yaml"), "w").write("name: cal_base\ntarget: fake_t\nsettings:\n  prompt: p1\n")
run_suite.TARGETS_FILE = os.path.join(root, "targets.yaml")
run_suite.SCENARIOS_DIR = os.path.join(root, "scen")
calibrate_noise.NOISE_RUNS_DIR = os.path.join(root, "noise_runs")
calibrate_noise.NOISE_DIR = os.path.join(root, "noise_cal")
seq = iter([8, 6, 9, 7, 8, 8] * 20)
fake_judges.install(lambda name, prompt: (next(seq) if name == "correctness_gt" else 8, 0.9))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rep = calibrate_noise.main(["--experiment", os.path.join(exp_dir, "cal_base.yaml"), "--runs", "2", "--yes"])
exp = experiment_config.load_experiment(os.path.join(exp_dir, "cal_base.yaml"))
saved = json.load(open(os.path.join(calibrate_noise.NOISE_DIR, "fake_t_noise.json"), encoding="utf-8"))
ok("full mode: 2 runs x 3 scenarios through run_experiment + judge_experiment",
   rep["mode"] == "full" and sorted(rep["scenarios"]) == ["c1", "c2", "c3"]
   and os.path.isfile(os.path.join(rep["session_dir"], "run_2", "run_meta.json")), rep["session_dir"])
ok("noise file keyed by target, with config_hash + judge set", saved["target"] == "fake_t"
   and saved["config_hash"] == experiment_config.config_hash(exp) and saved["judge_set"]["name"] == "experiment_v2")
ok("estimate uses the experiment judge set (5 per trace)", "judges up to 30" in buf.getvalue(),
   [l for l in buf.getvalue().splitlines() if "Estimated" in l])
ok("scores wobbled -> non-zero range recorded", any(st["range"] for st in saved["scenarios"].values()))
fresh_base = {"experiment": "cal_base", "resolved_config": {"target": "fake_t", "settings": exp["settings"]},
              "judge_set": saved["judge_set"]}
bands, note = compare.load_bands("fake_t", noise_dir=calibrate_noise.NOISE_DIR, baseline=fresh_base)
ok("compare accepts it for the same baseline config", bool(bands) and not note.startswith("WARNING"), note[:70])
src = os.path.join(rep["session_dir"], "run_1", "traces")
with contextlib.redirect_stdout(io.StringIO()):
    rep2 = calibrate_noise.main(["--experiment", os.path.join(exp_dir, "cal_base.yaml"), "--runs", "2", "--judge-only",
                                 "--from-traces", src, "--yes"])
ok("judge-only mode re-judges one fixed set", rep2["mode"] == "judge_only"
   and os.path.isdir(os.path.join(rep2["session_dir"], "source_traces")))
code = 0
try:
    with contextlib.redirect_stderr(io.StringIO()):
        calibrate_noise.main(["strong", "--experiment", "x.yaml", "--yes"])
except SystemExit as e:
    code = e.code
ok("variant AND --experiment -> refused", "not both" in str(code))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
