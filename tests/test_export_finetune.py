"""
Tests for export_finetune.py on fake judged runs (no API calls).
    python tests/test_export_finetune.py
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
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import export_finetune as ef  # noqa: E402
import judge_sets  # noqa: E402
import run_suite  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


SC = {s["id"]: s for s in run_suite.load_scenarios()}
TRAIN = sorted(s for s, v in SC.items() if v.get("split") == "train")
ok("the 5 approved train scenarios", TRAIN == ["edge_02", "edge_03", "gen_007", "gen_013", "gen_015"], TRAIN)

root = tempfile.mkdtemp()
ef.RUNS_DIR = os.path.join(root, "runs")
ef.FINETUNE_DIR = os.path.join(root, "finetune")


def conv(answer, call_id="c1"):
    return [{"role": "system", "content": "You are a customer support agent."},
            {"role": "user", "content": "q?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function",
                                                                 "function": {"name": "escalate_to_human",
                                                                              "arguments": '{"reason": "x"}'}}]},
            {"role": "tool", "tool_call_id": call_id, "content": "Escalated to human support. Reason: x"},
            {"role": "assistant", "content": answer}]


def judged(sid, score=9.5, **kw):
    t = {"scenario_id": sid, "timestamp": kw.pop("timestamp", "20261106_100000"), "question": "q?", "answer": "a",
         "error": None, "judge_error": None, "final_score": score, "disagreement": False, "low_confidence_judges": [],
         "answer_key_hash": judge_sets.answer_key_hash(SC[sid]) if sid in SC else None,
         "messages": conv(kw.pop("answer", f"good answer for {sid}"), kw.pop("call_id", "c1")),
         "correctness_gt_result": {"score": 10, "confidence": 0.9, "reason": "x"}}
    t.update(kw)
    return t


def make(name, traces, target="configurable", ts="20261106_100000"):
    d = os.path.join(ef.RUNS_DIR, f"{name}_{ts}")
    os.makedirs(os.path.join(d, "judged"))
    json.dump({"experiment": name, "timestamp": ts,
               "resolved_config": {"target": target, "settings": {"model": "m", "prompt": "prompts/v1.txt",
                                                                  "docstrings": "docstrings/detailed.yaml"}}},
              open(os.path.join(d, "run_meta.json"), "w"))
    for t in traces:
        json.dump(t, open(os.path.join(d, "judged", f"{t['scenario_id']}.json"), "w"))
    return d


make("baseline", [
    judged("gen_007"),                                         # good -> exported
    judged("gen_013", score=8.9),                              # below 9
    judged("gen_015", error="boom", final_score=None),         # agent error
    judged("edge_02", judge_error="429", final_score=None),    # judge error
    judged("edge_03", disagreement=True),                      # judges disagreed
    judged("normal_01", score=10.0),                           # TEST split: never
    judged("rag_06", score=10.0),                              # TEST split: never
], ts="20261106_100000")
make("exp_x", [
    judged("gen_007", score=9.8, call_id="zz9", timestamp="20261106_110000"),   # same conversation, other id -> dup
    judged("gen_013", low_confidence_judges=["completeness"], timestamp="20261106_110000"),
    judged("gen_015", answer_key_hash="000000000000", timestamp="20261106_110000"),   # older answer key
    judged("edge_02", messages=None, timestamp="20261106_110000"),                      # no conversation
    judged("edge_03", answer="a DIFFERENT good answer", timestamp="20261106_110000"),   # good, distinct
], ts="20261106_110000")
make("other", [judged("gen_007", answer="other target")], target="financial_dd", ts="20261106_120000")

print("filters + dedupe:")
ex, rej = ef.collect("configurable", scenarios=SC)
ok("exactly the 2 clean, distinct train examples", [(e["scenario_id"], e["experiment"]) for e in ex]
   == [("edge_03", "exp_x"), ("gen_007", "exp_x")], [(e["scenario_id"], e["experiment"]) for e in ex])
ok("duplicate kept once, at the higher score", next(e for e in ex if e["scenario_id"] == "gen_007")["final_score"] == 9.8
   and rej.get("duplicate conversation") == 1)
expected = {"final_score < 9": 1, "agent error": 1, "judge error": 1, "judges disagreed": 1, "low-confidence judge": 1,
            "judged against an older answer key": 1, "trace has no full conversation (older run)": 1,
            "split: test (the exam: never exported)": 2, "duplicate conversation": 1}
ok("every rejection counted by its reason", rej == expected, rej)
ok("other targets ignored", all(e["run_dir"].find("other_") < 0 for e in ex))
ok("tool schemas from that run's docstrings", [t["function"]["name"] for t in ex[0]["tools"]]
   == ["order_lookup", "refund_calculator", "faq_search", "escalate_to_human"]
   and ex[0]["tools"][0]["function"]["description"].startswith("Looks up an order"))

print("leakage guard:")
err, code = io.StringIO(), 0
with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
    try:
        ef.main(["--scenarios", "gen_007", "normal_01", "--yes"])
    except SystemExit as e:
        code = e.code
ok("asking for a test scenario is refused, with the reason", code == 2 and "normal_01" in err.getvalue()
   and "memorise" in err.getvalue(), err.getvalue().strip()[:110])
try:
    ef.guard([{"scenario_id": "normal_01"}], SC)
    ok("final guard catches a test example that slipped through", False)
except ef.LeakageError as e:
    ok("final guard catches a test example that slipped through", "normal_01" in str(e))

print("review first, JSONL only after confirming:")
with contextlib.redirect_stdout(io.StringIO()):
    r = ef.main(["--dry-run"])
review = open(r["review"], encoding="utf-8").read()
ok("dry run: review.md written, no JSONL", r["jsonl"] is None and os.listdir(ef.FINETUNE_DIR) == [os.path.basename(r["review"])])
ok("review lists examples, judges, conversation and rejections", "2 example(s)" in review and "## 1. edge_03" in review
   and "escalate_to_human" in review and "split: test (the exam: never exported): 2" in review)
real_input, real_isatty = builtins.input, sys.stdin.isatty
builtins.input, sys.stdin.isatty = (lambda prompt="": "n"), (lambda: True)
with contextlib.redirect_stdout(io.StringIO()):
    r = ef.main([])
ok("answering 'n' writes no JSONL", r["jsonl"] is None and not any(f.endswith(".jsonl") for f in os.listdir(ef.FINETUNE_DIR)))
builtins.input = lambda prompt="": "y"
with contextlib.redirect_stdout(io.StringIO()):
    r = ef.main([])
builtins.input, sys.stdin.isatty = real_input, real_isatty
lines = [json.loads(line) for line in open(r["jsonl"], encoding="utf-8")]
ok("answering 'y' writes finetune/train_<ts>.jsonl", os.path.basename(r["jsonl"]).startswith("train_") and len(lines) == 2)
ok("chat format: messages with tool calls + tool results, plus tools",
   [m["role"] for m in lines[0]["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
   and lines[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "escalate_to_human"
   and len(lines[0]["tools"]) == 4)
ok("no test-split scenario in the file", all("normal_01" not in json.dumps(x) and "rag_06" not in json.dumps(x) for x in lines))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
