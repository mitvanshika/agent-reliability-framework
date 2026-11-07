"""
Phase 3 tests, LLM faked (no API calls):
  tool_match cases, N/A excluded from final_score, correctness_gt fallback,
  context_relevance N/A, judging a run end to end, judge-set / answer-key
  mismatch refusals, per-judge deltas incl. the new judges.
    python tests/test_judge_experiment.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import fake_judges  # noqa: E402

fake_judges.install()
import compare  # noqa: E402
import judge_experiment as je  # noqa: E402
import judge_run  # noqa: E402
import promote_baseline  # noqa: E402
import run_experiment  # noqa: E402
import run_judges  # noqa: E402
import run_suite  # noqa: E402
from fake_runs import make_run  # noqa: E402
from judges.context_relevance import retrieval_outputs  # noqa: E402
from judges.tool_match import tool_match  # noqa: E402

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

print("tool_match (plain code):")
cases = [
    ("perfect", ["order_lookup", "refund_calculator"], ["order_lookup", "refund_calculator"], 10.0),
    ("one of two missing", ["order_lookup"], ["order_lookup", "refund_calculator"], 5.0),
    ("one extra tool", ["order_lookup", "faq_search"], ["order_lookup"], 9.0),
    ("wrong order", ["refund_calculator", "order_lookup"], ["order_lookup", "refund_calculator"], 8.0),
    ("expected but nothing called", [], ["faq_search"], 0.0),
    ("repeat not penalised", ["faq_search", "faq_search"], ["faq_search"], 10.0),
    ("extras capped at -3", ["a", "b", "c", "d", "faq_search"], ["faq_search"], 7.0),
    ("no tools expected, none used", [], [], 10.0),
    ("missing + extra + order", ["escalate_to_human", "faq_search"], ["order_lookup", "refund_calculator", "escalate_to_human"], 2.3),
]
for label, used, expected, score in cases:
    r = tool_match(used, expected)
    ok(f"{label}: {score}", r["score"] == score, f"{r['score']} ({r['reason']})")
ok("no expected_tools -> N/A", tool_match(["x"], None) == {"na": True, "reason": "scenario has no expected_tools"})

print("final_score: N/A left out, not counted as 0:")
v = lambda s, c=0.9: {"score": s, "confidence": c, "reason": "x"}
four = {"correctness_gt": v(8), "completeness": v(8), "groundedness": v(6), "tool_efficiency": v(10)}
with_na = je.combine({**four, "context_relevance": {"na": True, "reason": "no retrieval"}})
ok("N/A excluded: (8+8+6+10)/4 = 8.0", abs(with_na["final_score"] - 8.0) < 1e-9 and with_na["na_judges"] == ["context_relevance"],
   with_na)
as_zero = je.combine({**four, "context_relevance": v(0)})
ok("...whereas a real 0 would give 6.4", abs(as_zero["final_score"] - 6.4) < 1e-9, as_zero["final_score"])
weighted = je.combine({"correctness_gt": v(10, 1.0), "completeness": v(0, 0.25)})
ok("confidence-weighted: (10*1 + 0*0.25)/1.25 = 8.0", abs(weighted["final_score"] - 8.0) < 1e-9)
ok("disagreement and low confidence over applicable judges only",
   weighted["disagreement"] is True and weighted["low_confidence_judges"] == ["completeness"])
ok("everything N/A -> no score", je.combine({"x": {"na": True}})["final_score"] is None)

print("correctness_gt fallback:")
fake_judges.install()
from judges.correctness_gt import judge_correctness_gt  # noqa: E402

verdict, mode = judge_correctness_gt("Status of A1001?", "Delivered, $89.99", "A1001 has been delivered. $89.99, 12 days.")
prompt = fake_judges.called("correctness_gt")[0]
ok("with answer key: mode answer_key, reference in prompt", mode == "answer_key" and "Reference answer" in prompt
   and "12 days" in prompt and fake_judges.called("correctness") == [])
fake_judges.install()
verdict, mode = judge_correctness_gt("Status of A1001?", "Delivered", None)
ok("no answer key: classic correctness judge used", mode == "fallback_no_answer_key"
   and len(fake_judges.called("correctness")) == 1 and fake_judges.called("correctness_gt") == [])

print("context_relevance N/A:")
ok("no tool outputs -> nothing to judge", retrieval_outputs([], []) == [])
ok("only escalate_to_human -> N/A", retrieval_outputs(["escalate_to_human"], ["Escalated..."]) == [])
ok("escalation output dropped, retrieval kept",
   retrieval_outputs(["faq_search", "escalate_to_human"], ["FAQ text", "Escalated"]) == ["FAQ text"])
ok("unpairable lists -> all outputs count", retrieval_outputs(["order_lookup"], ["a", "b"]) == ["a", "b"])

print("judging a run end to end (real scenario ids, so the answer key is used):")
SCORES = {"correctness_gt": (9, 0.9), "completeness": (8, 0.9), "groundedness": (10, 0.9),
          "tool_efficiency": (7, 0.9), "context_relevance": (6, 0.9), "correctness": (4, 0.9)}
fake_judges.install(lambda name, prompt: SCORES[name])
rows = {
    "normal_02": {"score": 0, "tools_used": ["order_lookup"],
                  "tool_outputs": ["Order A1001: status=delivered, total=$89.99, days_since_purchase=12"]},
    "gen_007": {"score": 0, "tools_used": ["escalate_to_human"], "tool_outputs": ["Escalated to human support. Reason: login"]},
    "normal_05": {"score": 0, "tools_used": ["refund_calculator", "order_lookup"], "tool_outputs": ["Eligible", "Order A1003"]},
    "zz_no_key": {"score": 0, "tools_used": [], "tool_outputs": []},
    "edge_02": {"error": "boom", "error_type": "crash"},
}
run1 = make_run(os.path.join(root, "runs"), "baseline", rows, judged=False, timestamp="20261106_100000")
with contextlib.redirect_stdout(io.StringIO()):
    s1 = judge_run.main([run1, "--yes"])
J = lambda sid: json.load(open(os.path.join(run1, "judged", f"{sid}.json"), encoding="utf-8"))
n02, g07, n05, nokey, crash = J("normal_02"), J("gen_007"), J("normal_05"), J("zz_no_key"), J("edge_02")
expected_final = (9 + 8 + 10 + 7 + 6) / 5
ok("5 judges in final_score when retrieval happened", abs(n02["final_score"] - expected_final) < 1e-9, n02["final_score"])
ok("answer key used + its hash recorded", n02["correctness_gt_result"]["mode"] == "answer_key"
   and n02["answer_key_hash"] == je.answer_key_hash({s["id"]: s for s in run_suite.load_scenarios()}["normal_02"]))
ok("escalation only -> context_relevance N/A, final over 4 judges",
   g07["context_relevance_result"].get("na") is True and abs(g07["final_score"] - (9 + 8 + 10 + 7) / 4) < 1e-9)
ok("tool_match recorded but NOT in final_score", n05["tool_match_result"]["score"] == 8.0
   and abs(n05["final_score"] - expected_final) < 1e-9, n05["tool_match_result"])
ok("no answer key -> classic fallback, tool_match N/A", nokey["correctness_gt_result"]["mode"] == "fallback_no_answer_key"
   and nokey["correctness_gt_result"]["score"] == 4 and nokey["tool_match_result"].get("na") is True
   and nokey["answer_key_hash"] is None)
ok("crash: no judge calls, judge_set still recorded", crash["final_score"] is None and crash["judge_set"]["name"] == "experiment_v2")
ok("summary.json: judge set, per-judge incl. new judges, tool_match, N/A counts",
   s1["judge_set"]["name"] == "experiment_v2" and set(s1["per_judge_mean"]) == set(je.LLM_JUDGES)
   and s1["per_judge_na"] == {"context_relevance": 2} and s1["tool_match_mean"] is not None, s1["per_judge_mean"])

print("judge-set mismatch refusal:")
classic = make_run(os.path.join(root, "runs"), "baseline", {sid: 8.0 for sid in rows}, timestamp="20261101_000000")
with contextlib.redirect_stdout(io.StringIO()):
    promote_baseline.main([classic, "--yes"])


def run_compare(argv):
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            compare.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


code, _, err = run_compare([run1])
ok("classic baseline vs experiment run -> refused, exit 2", code == 2 and "Judge sets differ" in err
   and "re-judge or re-promote the baseline" in err, err.strip()[:160])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    promote_baseline.main([run1, "--yes", "--note", "re-judged with experiment_v2"])
ok("re-promote shows 'not comparable' instead of a bogus comparison", "Not comparable: the current baseline was judged with classic_v1" in buf.getvalue())
base = json.load(open(compare.baseline_path("configurable"), encoding="utf-8"))
ok("baseline stores judge_set + answer_key_hash", base["judge_set"]["name"] == "experiment_v2"
   and base["scenarios"]["normal_02"]["answer_key_hash"] == n02["answer_key_hash"])

print("per-judge deltas include the new judges:")
SCORES.update(correctness_gt=(4, 0.9), context_relevance=(9, 0.9))
fake_judges.install(lambda name, prompt: SCORES[name])
run2 = make_run(os.path.join(root, "runs"), "exp_01_weak_prompt", rows, judged=False, timestamp="20261106_110000")
with contextlib.redirect_stdout(io.StringIO()):
    judge_run.main([run2, "--yes"])
code, out, err = run_compare([run2])
r = json.load(open(os.path.join(run2, "comparison.json"), encoding="utf-8"))
pj = r["per_judge"]
ok("same judge set -> compared", code in (0, 1) and r["judge_set"]["name"] == "experiment_v2", err.strip())
# correctness_gt: 3 answer-key scenarios 9 -> 4 (-5) + zz_no_key on the classic fallback both times (4 -> 4)
ok("correctness_gt delta (-5*3 + 0)/4 = -3.75, context_relevance +3 (n=2: N/A left out), tool_match marked as check",
   pj["correctness_gt"]["delta"] == -3.75 and pj["context_relevance"]["delta"] == 3 and pj["context_relevance"]["n"] == 2
   and pj["tool_match"]["in_final_score"] is False, {k: (v["delta"], v["n"]) for k, v in pj.items()})
ok("printed per-judge section", "Per-judge mean" in out and "tool_match (code check, not in final_score)" in out)
ok("judges moved lists correctness_gt", any("correctness_gt 9->4" in m for x in r["rows"] for m in x["judges_moved"]))

print("answer key changed -> refused:")
judged_file = os.path.join(run2, "judged", "normal_02.json")
t = json.load(open(judged_file, encoding="utf-8"))
t["answer_key_hash"] = "000000000000"
json.dump(t, open(judged_file, "w", encoding="utf-8"))
code, _, err = run_compare([run2])
ok("refused, names the scenario", code == 2 and "answer key changed" in err and "normal_02" in err, err.strip()[:140])

print("mixed judge sets -> refused:")
t["judge_set"] = None
json.dump(t, open(judged_file, "w", encoding="utf-8"))
code, _, err = run_compare([run2])
ok("refused: re-judge it", code == 2 and "mix of judge sets" in err, err.strip()[:120])

print("judge errors / crashes don't count as an answer-key change (real run bug, 2026-11-06):")
# baseline judged with experiment_v1 (run1 promoted above); new run: normal_02 judge error, gen_007 crash
SCORES.update(correctness_gt=(9, 0.9), context_relevance=(6, 0.9))
fake_judges.install(lambda name, prompt: SCORES[name])
rows3 = dict(rows)
rows3["gen_007"] = {"error": "Recursion limit reached", "error_type": "loop_limit"}
run3 = make_run(os.path.join(root, "runs"), "exp_02", rows3, judged=False, timestamp="20261106_120000")
with contextlib.redirect_stdout(io.StringIO()):
    judge_run.main([run3, "--yes"])
jf = os.path.join(run3, "judged", "normal_02.json")
t3 = json.load(open(jf, encoding="utf-8"))
t3.update(final_score=None, judge_error="Error code: 429 - tokens per day (TPD)")
for k in [k for k in t3 if k.endswith("_result") or k == "answer_key_hash"]:
    t3.pop(k)
json.dump(t3, open(jf, "w", encoding="utf-8"))
code, out, err = run_compare([run3])
r3 = json.load(open(os.path.join(run3, "comparison.json"), encoding="utf-8"))
ok("judge error + crash -> compared (not refused), judge error excluded",
   code in (0, 1) and "answer key changed" not in err and [e["scenario_id"] for e in r3["excluded"]] == ["normal_02"], err.strip()[:120])
ok("crash still counts as a fail (score 0)", next(x for x in r3["rows"] if x["scenario_id"] == "gen_007")["run_score"] == 0.0)
t5 = json.load(open(os.path.join(run3, "judged", "normal_05.json"), encoding="utf-8"))
t5["answer_key_hash"] = "ffffffffffff"
json.dump(t5, open(os.path.join(run3, "judged", "normal_05.json"), "w", encoding="utf-8"))
code, _, err = run_compare([run3])
ok("a real answer-key change on a scored scenario is still refused", code == 2 and "normal_05" in err
   and "normal_02" not in err.split("for:")[1], err.strip()[:120])

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
