"""
Tests for judge_run.py: summary statistics on fake judged files with known
numbers, plus judging end to end with the judges mocked (no API calls).
    python tests/test_judge_run.py
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

import judge_run  # noqa: E402
import run_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def approx(a, b):
    return a is not None and abs(a - b) < 1e-9


def dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def res(score):
    return {"score": score, "confidence": 0.9, "reason": "x"}


def make_run(judged_rows):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "traces"))
    ids = [r["id"] for r in judged_rows]
    dump(os.path.join(d, "run_meta.json"), {
        "experiment": "exp_01_weak_prompt", "parent": "baseline",
        "changes_vs_parent": ["settings.prompt: prompts/v1.txt -> prompts/weak.txt"],
        "resolved_config": {"target": "configurable", "settings": {"model": "m1"}}, "scenario_ids": ids})
    for r in judged_rows:
        trace = {"scenario_id": r["id"], "question": "q", "answer": None if r.get("error") else "a",
                 "timestamp": "20261105_120000", "error": r.get("error"), "error_type": r.get("error_type"),
                 "latency_s": r["latency"], "usage": r.get("usage"), "tools_used": [], "tool_outputs": []}
        dump(os.path.join(d, "traces", f"{r['id']}.json"), trace)
    return d


U = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
rows = [
    {"id": "s1", "final": 9.0, "judges": [9, 9, 9, 9], "latency": 1.0, "usage": U},
    {"id": "s2", "final": 7.0, "judges": [8, 6, 7, 7], "latency": 2.0, "usage": U},   # exactly 7 -> pass
    {"id": "s3", "final": 5.0, "judges": [4, 6, 5, 5], "latency": 3.0, "usage": U},
    {"id": "s4", "error": "loop", "error_type": "loop_limit", "latency": 10.0},          # crashed agent
]

print("summary statistics (known numbers):")
d = make_run(rows)
os.makedirs(os.path.join(d, "judged"))
for r in rows:
    j = {"scenario_id": r["id"], "timestamp": "20261105_120000", "error": r.get("error"), "judge_error": None}
    if "final" in r:
        j["final_score"] = r["final"]
        for name, sc in zip(["correctness", "tool_efficiency", "completeness", "groundedness"], r["judges"]):
            j[f"{name}_result"] = res(sc)
    else:
        j.update(final_score=None, not_judged_reason="agent_error")
    dump(os.path.join(d, "judged", f"{r['id']}.json"), j)

run_experiment.PRICING_FILE = os.path.join(d, "pricing.yaml")
open(run_experiment.PRICING_FILE, "w").write("m1: {input: 1.0, output: 5.0}\n")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s = judge_run.main([d, "--summary-only"])
print("    " + "\n    ".join(buf.getvalue().strip().splitlines()))
ok("pass rate 2/4 (7.0 passes, crash fails)", s["passed"] == 2 and s["scenarios"] == 4 and s["pass_rate"] == 0.5)
ok("mean final_score over the 3 scored = 7.0", approx(s["mean_final_score"], 7.0) and s["scored"] == 3)
ok("per-judge means", s["per_judge_mean"] == {"correctness": 7.0, "tool_efficiency": 7.0,
                                             "completeness": 7.0, "groundedness": 7.0}, s["per_judge_mean"])
ok("tokens summed (3 x 1000 in / 100 out)", s["tokens"] == {"input_tokens": 3000, "output_tokens": 300,
                                                            "total_tokens": 3300} and s["scenarios_with_usage"] == 3)
ok("latency mean 4.0, p95 10.0", approx(s["latency_mean_s"], 4.0) and s["latency_p95_s"] == 10.0)
ok("errors by type", s["agent_errors"] == 1 and s["errors_by_type"] == {"loop_limit": 1})
ok("cost (3000*1 + 300*5)/1e6 = 0.0045", approx(s["estimated_cost_usd"], 0.0045), s["cost_note"])
ok("what changed", s["what_changed"] == "settings.prompt: prompts/v1.txt -> prompts/weak.txt   (vs baseline)")
saved = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
ok("summary.json saved and equal", saved == s)

print("per-judge mean with different judges:")
d2 = make_run(rows[:2])
os.makedirs(os.path.join(d2, "judged"))
for r, judges in zip(rows[:2], ([10, 2, 6, 8], [6, 4, 8, 9])):
    j = {"scenario_id": r["id"], "timestamp": "20261105_120000", "error": None, "judge_error": None, "final_score": 6.5}
    for name, sc in zip(["correctness", "tool_efficiency", "completeness", "groundedness"], judges):
        j[f"{name}_result"] = res(sc)
    dump(os.path.join(d2, "judged", f"{r['id']}.json"), j)
with contextlib.redirect_stdout(io.StringIO()):
    s2 = judge_run.main([d2, "--summary-only"])
ok("means per judge", s2["per_judge_mean"] == {"correctness": 8.0, "tool_efficiency": 3.0,
                                              "completeness": 7.0, "groundedness": 8.5}, s2["per_judge_mean"])
ok("6.5 fails", s2["passed"] == 0)

print("judging end to end (experiment judge set, LLM faked):")
import run_judges  # noqa: E402

import fake_judges  # noqa: E402

run_judges.time.sleep = lambda s: None
fake_judges.install(lambda name, prompt: (8, 0.9))
d3 = make_run(rows)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    s3 = judge_run.main([d3, "--yes"])
out = buf.getvalue()
ok("estimate printed: up to 3 answered x 5 judges = 15", "Estimated judge API calls: up to 15" in out)
ok("judged into runs/<run>/judged", sorted(os.listdir(os.path.join(d3, "judged"))) == ["s1.json", "s2.json", "s3.json", "s4.json"])
# fake traces have no tool outputs -> context_relevance N/A; ids s1..s4 have no answer key -> classic fallback
ok("LLM calls: 3 traces x 4 (context_relevance N/A), none for the crash", len(fake_judges.CALLS) == 12
   and fake_judges.called("context_relevance") == [] and len(fake_judges.called("correctness")) == 3,
   sorted({n for n, _ in fake_judges.CALLS}))
crashed = json.load(open(os.path.join(d3, "judged", "s4.json"), encoding="utf-8"))
ok("crashed trace handled like run_judges.py", crashed["final_score"] is None and crashed["not_judged_reason"] == "agent_error")
ok("summary after judging", s3["passed"] == 3 and s3["scenarios"] == 4 and s3["not_judged"] == [])
ok("judge set saved in summary.json", json.load(open(os.path.join(d3, "summary.json"), encoding="utf-8"))["judge_set"]["name"]
   == "experiment_v2" and "Judge set:    experiment_v2" in out)
ok("N/A counted, fallbacks listed", s3["per_judge_na"] == {"context_relevance": 3}
   and s3["correctness_fallbacks"] == ["s1", "s2", "s3"] and s3["per_judge_mean"]["context_relevance"] is None, s3["per_judge_na"])
ok("latency/usage survive judging (judged files keep trace fields)",
   json.load(open(os.path.join(d3, "judged", "s1.json"), encoding="utf-8"))["latency_s"] == 1.0)
ok("legacy (classic) judged run summarised with classic judges", s["judge_set"]["name"] == "classic_v1")

print("confirmation without --yes (no input):")
d4 = make_run(rows)
fake_judges.CALLS.clear()
import builtins  # noqa: E402

real_input = builtins.input
builtins.input = lambda prompt="": (_ for _ in ()).throw(EOFError())
judge_run.confirm.__globals__["sys"].stdin.isatty = lambda: True
with contextlib.redirect_stdout(io.StringIO()) as b:
    r = judge_run.main([d4])
builtins.input = real_input
ok("aborted, no judge calls, no judged folder", r is None and fake_judges.CALLS == [] and not os.path.exists(os.path.join(d4, "judged")))

print("bad folder:")
try:
    with contextlib.redirect_stdout(io.StringIO()):
        judge_run.main([tempfile.mkdtemp()])
    ok("missing run_meta.json -> clear exit", False)
except SystemExit as e:
    ok("missing run_meta.json -> clear exit", "run_meta.json not found" in str(e), str(e))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
