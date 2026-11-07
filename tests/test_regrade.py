"""
Tests for regrade.py (LLM faked, no API calls).
    python tests/test_regrade.py
"""

import contextlib
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
import judge_experiment  # noqa: E402
import judge_sets  # noqa: E402
import rate_limit  # noqa: E402
import regrade  # noqa: E402
import run_judges  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


IDS = ["gen_015", "normal_01", "rag_06", "edge_01"]
V1 = {**judge_sets.EXPERIMENT, "judge_model": "openai/gpt-oss-120b"}
root = tempfile.mkdtemp()


def verdict(score, conf=0.9):
    return {"score": score, "reason": "r", "confidence": conf}


def make_run():
    d = os.path.join(root, "runs", "baseline_20261106_000000")
    for sub in ("traces", "judged"):
        os.makedirs(os.path.join(d, sub))
    json.dump({"experiment": "baseline", "timestamp": "20261106_000000", "scenario_ids": IDS, "scenario_dir": "scenarios",
               "suite": "main", "parent": None, "changes_vs_parent": [],
               "resolved_config": {"target": "configurable", "settings": {"model": "m"}}},
              open(os.path.join(d, "run_meta.json"), "w"))
    for sid in IDS:
        tr = {"scenario_id": sid, "question": f"question {sid}?", "answer": f"answer {sid}", "error": None,
              "timestamp": "20261106_000000", "tools_used": ["faq_search"], "tool_outputs": [f"output {sid}"],
              "latency_s": 1.0, "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
        json.dump(tr, open(os.path.join(d, "traces", f"{sid}.json"), "w"))
        j = {**tr, "judge_set": V1, "answer_key_hash": "k", "judge_error": None,
             "correctness_gt_result": verdict(9), "completeness_result": verdict(1), "groundedness_result": verdict(9),
             "tool_efficiency_result": verdict(8), "context_relevance_result": verdict(9),
             "tool_match_result": {"score": 10}}
        if sid == "edge_01":   # a judge error in v1: no scores to copy
            j = {**tr, "judge_set": V1, "final_score": None, "disagreement": None, "judge_error": "429 TPD"}
        else:
            j.update(judge_experiment.combine({n: j[f"{n}_result"] for n in V1["llm_judges"]}))
        json.dump(j, open(os.path.join(d, "judged", f"{sid}.json"), "w"))
    return d


def run(argv):
    out, code = io.StringIO(), 0
    with contextlib.redirect_stdout(out):
        try:
            regrade.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def load(d, folder, sid):
    return json.load(open(os.path.join(d, folder, f"{sid}.json"), encoding="utf-8"))


print("changed judges:")
ok("v1 -> v2: only completeness", judge_sets.changed_judges(judge_sets.EXPERIMENT, judge_sets.EXPERIMENT_V2)
   == ["completeness"])
ok("v1 -> v1: nothing", judge_sets.changed_judges(judge_sets.EXPERIMENT, judge_sets.EXPERIMENT) == [])

print("re-grade v1 -> v2 (completeness only):")
d = make_run()
v1_bytes = {sid: open(os.path.join(d, "judged", f"{sid}.json"), "rb").read() for sid in IDS}
fake_judges.install(lambda name, prompt: (10, 0.9))
code, out = run([d, "--to", "experiment_v2", "--yes"])
ok("exit ok", code in (0, None), out[-300:])
ok("only the completeness judge was called, once per OK answer (3)",
   [n for n, _ in fake_judges.CALLS] == ["completeness"] * 3, [n for n, _ in fake_judges.CALLS])
ok("it was the v2 prompt (tool outputs + answer key)", all("Reference answer" in p and "- output " in p
   for p in fake_judges.called("completeness")))
ok("backup judged_experiment_v1/ = the v1 files, byte for byte", all(
   open(os.path.join(d, "judged_experiment_v1", f"{sid}.json"), "rb").read() == v1_bytes[sid] for sid in IDS))
g = load(d, "judged", "gen_015")
ok("completeness 1 -> 10; the other 4 judges copied unchanged", g["completeness_result"]["score"] == 10 and all(
   g[f"{n}_result"] == verdict(s) for n, s in (("correctness_gt", 9), ("groundedness", 9), ("tool_efficiency", 8),
                                                ("context_relevance", 9))))
expect = judge_experiment.combine({"correctness_gt": verdict(9), "completeness": verdict(10, 0.9),
                                   "groundedness": verdict(9), "tool_efficiency": verdict(8),
                                   "context_relevance": verdict(9)})["final_score"]
ok("final_score recomputed with combine()", abs(g["final_score"] - expect) < 1e-9 and g["final_score"] > load(
   d, "judged_experiment_v1", "gen_015")["final_score"], (g["final_score"], expect))
ok("records judge_set v2 + regraded_from / regraded_judges", g["judge_set"]["name"] == "experiment_v2"
   and g["regraded_from"] == "experiment_v1" and g["regraded_judges"] == ["completeness"])
e = load(d, "judged", "edge_01")
ok("judge-error trace: kept as an error, marked v2 (folder not mixed)", e["judge_error"] == "429 TPD"
   and e["judge_set"]["name"] == "experiment_v2")
ok("folder is one judge set", judge_sets.of_judged_traces(
   json.load(open(os.path.join(d, "judged", f), encoding="utf-8")) for f in os.listdir(os.path.join(d, "judged")))["name"]
   == "experiment_v2")
ok("summary.json recomputed with v2", json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
   ["judge_set"]["name"] == "experiment_v2")
ok("table printed with old -> new and a flip marker", "1 -> 10" in out and "pass old->new" in out)

print("re-running is a no-op; --judge-errors full judges only the error:")
fake_judges.install(lambda name, prompt: (10, 0.9))
code, out = run([d, "--to", "experiment_v2", "--yes"])
ok("nothing left -> 0 calls", fake_judges.CALLS == [], out[-200:])
code, out = run([d, "--to", "experiment_v2", "--judge-errors", "full", "--yes"])
ok("full: all 5 judges for edge_01 only", sorted(n for n, _ in fake_judges.CALLS) == sorted(
   judge_sets.EXPERIMENT_V2["llm_judges"]), [n for n, _ in fake_judges.CALLS])
e = load(d, "judged", "edge_01")
ok("edge_01 now scored, no judge_error", e["judge_error"] is None and e["final_score"] is not None)

print("daily limit mid-way -> stop, resume:")
shutil.rmtree(os.path.join(root, "runs"))
d = make_run()
n = {"k": 0}


def quota(name, prompt):
    n["k"] += 1
    if n["k"] == 2:
        rate_limit.STATS.daily_limit = True
        raise RuntimeError("Rate limit reached on tokens per day (TPD)")
    return (10, 0.9)


rate_limit.STATS.reset()
fake_judges.install(quota)
code, out = run([d, "--to", "experiment_v2", "--yes"])
ok("stops with a resume command", isinstance(code, str) and "Continue later" in code and "regrade.py" in code, code)
done = [s for s in IDS if load(d, "judged", s).get("judge_set", {}).get("name") == "experiment_v2"]
ok("before the stop: edge_01 marked (no call) + gen_015 re-graded; the rest untouched (still v1)",
   sorted(done) == ["edge_01", "gen_015"], done)
rate_limit.STATS.reset()
fake_judges.install(lambda name, prompt: (10, 0.9))
code, out = run([d, "--to", "experiment_v2", "--yes"])
ok("resume: only the 2 missing answers re-graded", len(fake_judges.CALLS) == 2, len(fake_judges.CALLS))

print("compare after re-grading:")
ok("v2 run vs v2 baseline -> same judge set", judge_sets.same(load(d, "judged", "gen_015")["judge_set"],
   {**judge_sets.EXPERIMENT_V2, "judge_model": "openai/gpt-oss-120b"}))
ok("v2 run vs v1 baseline -> refused by compare's check", not judge_sets.same(
   load(d, "judged", "gen_015")["judge_set"], V1))

print("guards:")
code, _ = run([d, "--to", "nope", "--yes"])
ok("unknown judge set refused", isinstance(code, str) and "Unknown judge set" in code)

shutil.rmtree(root)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
