"""
Tests for the step timeline on the dashboard (/runs/<run>/<scenario> + its API). Flask test client, no API calls.
    python tests/test_timeline.py
"""

import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import app as appmod  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
appmod.RUNS_DIR = root
run = os.path.join(root, "exp_x_20261108_100000")
for sub in ("traces", "judged", "judged_steps"):
    os.makedirs(os.path.join(run, sub))
json.dump({"experiment": "exp_x", "timestamp": "20261108_100000", "scenario_ids": ["s1", "s2"],
           "resolved_config": {"target": "configurable", "settings": {}}}, open(os.path.join(run, "run_meta.json"), "w"))
recorded = [
    {"index": 0, "type": "llm", "name": "ChatGroq", "tool_calls": [{"name": "faq_search", "args": {"query": "ship"}}],
     "output": "", "latency_s": 1.2, "tokens": {"input": 900, "output": 20, "total": 920}, "error": None},
    {"index": 1, "type": "tool", "name": "faq_search", "args": {"query": "ship"}, "output": "5-7 days",
     "latency_s": 0.01, "error": None},
    {"index": 2, "type": "tool", "name": "faq_search", "args": {"query": "shipping"}, "output": "5-7 days",
     "latency_s": 0.01, "error": None},
    {"index": 3, "type": "llm", "name": "ChatGroq", "tool_calls": [], "output": "It takes <b>5-7</b> days.",
     "latency_s": 12.5, "tokens": {"input": 1000, "output": 15, "total": 1015}, "error": None}]
json.dump({"scenario_id": "s1", "question": "How long is shipping?", "answer": "It takes 5-7 days.", "error": None,
           "timestamp": "t", "latency_s": 13.8, "usage": {"total_tokens": 1935}, "trajectory": recorded},
          open(os.path.join(run, "traces", "s1.json"), "w"))
json.dump({"scenario_id": "s2", "question": "q", "answer": None, "error": "boom", "timestamp": "t", "latency_s": 0.1},
          open(os.path.join(run, "traces", "s2.json"), "w"))
json.dump({"scenario_id": "s1", "final_score": 8.4, "judge_error": None,
           "groundedness_result": {"score": 9, "reason": "ok", "confidence": 0.9},
           "tool_efficiency_result": {"score": 6, "reason": "one extra search", "confidence": 0.8},
           "context_relevance_result": {"na": True, "reason": "x"}},
          open(os.path.join(run, "judged", "s1.json"), "w"))
json.dump({"scenario_id": "s1", "judge_set": {"name": "steps_v1"}, "na": False,
           "steps": [{"index": 1, "name": "faq_search", "score": 10, "reason": "needed"},
                     {"index": 2, "name": "faq_search", "score": 4, "reason": "near-repeat of step 1"}]},
          open(os.path.join(run, "judged_steps", "s1.json"), "w"))

c = appmod.app.test_client()
RID = "exp_x_20261108_100000"

print("the step API:")
d = c.get(f"/api/experiments/runs/{RID}/steps/s1").get_json()
ok("question, answer, latency, tokens", d["question"] == "How long is shipping?" and d["latency_s"] == 13.8
   and d["usage"]["total_tokens"] == 1935)
ok("the recorded steps, in order, with timing", d["source"] == "recorded" and [s["type"] for s in d["steps"]]
   == ["llm", "tool", "tool", "llm"] and d["steps"][3]["latency_s"] == 12.5)
ok("step-check findings: same_result_repeat + slow_step (12.5 s)", sorted(f["check"] for f in d["findings"])
   == ["same_result_repeat", "slow_step"], d["findings"])
ok("answer-level judges incl. N/A", d["final_score"] == 8.4 and d["judges"]["tool_efficiency"]["score"] == 6
   and d["judges"]["context_relevance"]["na"])
ok("per-step judge scores keyed by step index", d["step_scores"] == {"1": {"score": 10, "reason": "needed"},
   "2": {"score": 4, "reason": "near-repeat of step 1"}})
d2 = c.get(f"/api/experiments/runs/{RID}/steps/s2").get_json()
ok("an answer that crashed: error shown, no steps, no judges", d2["error"] == "boom" and d2["steps"] == []
   and d2["source"] == "none" and d2["step_scores"] is None and d2["final_score"] is None)

print("safety of the routes:")
for bad in ("../x", "..", "s1;rm", "s 1"):
    r = c.get(f"/api/experiments/runs/{RID}/steps/{bad}")
    ok(f"scenario id {bad!r} -> 404", r.status_code == 404)
ok("unknown run -> 404", c.get("/api/experiments/runs/nope/steps/s1").status_code == 404)
ok("unknown scenario -> 404", c.get(f"/api/experiments/runs/{RID}/steps/zzz").status_code == 404)

print("the pages:")
r = c.get(f"/runs/{RID}/s1")
html = r.data.decode()
ok("/runs/<run>/<scenario> renders the timeline page", r.status_code == 200 and "Step timeline" in html
   and "/steps/" in html)
ok("everything from the trace is HTML-escaped (esc), so an answer with <b> can't inject markup",
   "const esc" in html and "${esc(s.output)}" in html and "${esc(d.answer" in html)
run_detail = c.get(f"/api/experiments/runs/{RID}").get_json()
ok("run detail lists every scenario with step counts + findings for the links",
   [(s["scenario_id"], s["steps"], s["findings"], s["error"]) for s in run_detail["scenarios"]]
   == [("s1", 4, 2, None), ("s2", 0, 0, "boom")], run_detail["scenarios"])
exp_html = c.get("/experiments").data.decode()
ok("experiments page links to the timelines", "Step timelines" in exp_html and "/runs/${encodeURIComponent(d.run_id)}/" in exp_html)

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
