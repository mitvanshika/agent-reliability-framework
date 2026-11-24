"""
Tests for adapters/financial_dd_steps.py: Financial DD's self-healing attempts, its
progress events and the token meter's calls turned into timeline steps. Uses the
event order and record fields of Financial DD's own code; no Financial DD, no API calls.
    python tests/test_fdd_steps.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "adapters")]

import financial_dd_steps as fs  # noqa: E402
import step_checks  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


# One question, as Financial DD runs it: search 1 -> grade -> rewrite -> search 2 -> grade ->
# stagnated (same chunks again, so it stops) -> write the answer. Fields = dd.models.AttemptRecord.
attempts = [
    {"attempt": 1, "query": "Apple R&D spending", "strategy": "original", "n_retrieved": 6, "n_relevant": 1,
     "mean_relevance": 0.31, "chunk_ids": ["c07", "c12", "c40"], "stagnated": False, "note": ""},
    {"attempt": 2, "query": "research and development expenses", "strategy": "rewrite", "n_retrieved": 6,
     "n_relevant": 1, "mean_relevance": 0.33, "chunk_ids": ["c07", "c12", "c40"], "stagnated": True, "note": ""},
]
events = [(0.00, "lookup", {"facts": []}),
          (0.01, "search", {"attempt": 1, "query": "Apple R&D spending"}),
          (0.90, "grade", {"attempt": 1, "passages": 6, "facts": 0}),
          (6.20, "graded", {"attempt": 1, "relevant": 1, "total": 6}),
          (8.10, "rewrite", {"attempt": 2, "query": "research and development expenses"}),
          (8.11, "search", {"attempt": 2, "query": "research and development expenses"}),
          (8.80, "grade", {"attempt": 2, "passages": 6, "facts": 0}),
          (14.0, "graded", {"attempt": 2, "relevant": 1, "total": 6}),
          (14.0, "stagnated", {"attempt": 2}),
          (14.1, "write", {"passages": 1, "facts": 0})]
calls = [  # token_meter.calls(): label = the JSON format's name; the rewrite call has none
    {"label": "chunk_grades", "model": "openai/gpt-oss-120b", "input_tokens": 1800, "output_tokens": 300,
     "total_tokens": 2100, "seconds": 5.2, "started_at": 0.95},
    {"label": None, "model": "openai/gpt-oss-120b", "input_tokens": 350, "output_tokens": 50,
     "total_tokens": 400, "seconds": 1.8, "started_at": 6.25},
    {"label": "chunk_grades", "model": "openai/gpt-oss-120b", "input_tokens": 1800, "output_tokens": 280,
     "total_tokens": 2080, "seconds": 5.1, "started_at": 8.85},
    {"label": "grounded_answer", "model": "openai/gpt-oss-120b", "input_tokens": 900, "output_tokens": 600,
     "total_tokens": 1500, "seconds": 9.8, "started_at": 14.2},
]

print("steps in the order they happened:")
steps = fs.build(attempts, events, calls)
order = [(s["type"], s.get("label") or s["name"]) for s in steps]
ok("retrieve -> grade -> rewrite call -> retrieve -> grade -> answer", order == [
    ("tool", "retrieve"), ("llm", "chunk_grades"), ("llm", "LLM call"), ("tool", "retrieve"),
    ("llm", "chunk_grades"), ("llm", "grounded_answer")], order)
ok("indexed 0..5", [s["index"] for s in steps] == list(range(6)))
r1, r2 = steps[0], steps[3]
ok("retrieve step: args = query + strategy, output = the chunks it found",
   r2["args"] == {"query": "research and development expenses", "strategy": "rewrite"}
   and r2["output"] == "chunks: c07, c12, c40", r2)
ok("retrieve timing: from its search event to its grade event", r1["latency_s"] == 0.89 and r1["started_at"] == 0.01,
   (r1["started_at"], r1["latency_s"]))
ok("note: relevance + Financial DD's own stagnated flag", "1 of 6 relevant" in r2["note"]
   and "stagnated" in r2["note"] and "stagnated" not in r1["note"], r2["note"])
ok("llm step: tokens and seconds from the meter", steps[5]["tokens"] == {"input": 900, "output": 600, "total": 1500}
   and steps[5]["latency_s"] == 9.8)

print("step_checks works on it:")
findings = step_checks.check_steps(steps)
same = [f for f in findings if f["check"] == "same_result_repeat"]
ok("the rewrite that brought back the same chunks is flagged (same_result_repeat)",
   len(same) == 1 and same[0]["tool"] == "retrieve", findings)
ok("the grounded answer (9.8 s) is not a slow step at the default 10 s", not any(
   f["check"] == "slow_step" for f in findings), findings)

print("edge cases:")
ok("model call failed before any attempt: only the llm steps", [s["type"] for s in fs.build([], [], calls[:1])] == ["llm"])
fail = fs.build([], [], [{"label": "chunk_grades", "model": "m", "input_tokens": None, "output_tokens": None,
                          "total_tokens": None, "seconds": 0.4, "started_at": 0.1, "error": "RateLimitError"}])
ok("a failed call: tokens None, error kept", fail[0]["tokens"] is None and fail[0]["error"] == "RateLimitError")
nochunks = fs.build([{**attempts[0], "chunk_ids": [], "n_relevant": 0}], [], [])
ok("no chunks found: says so; no events -> no timing, still listed", nochunks[0]["output"] == "no chunks found"
   and nochunks[0]["latency_s"] is None and nochunks[0]["started_at"] is None)

print("the worker returns it:")
with open(os.path.join(REPO, "adapters", "financial_dd_worker.py"), encoding="utf-8") as f:
    src = f.read()
ok("passes on_progress to answer() and builds the steps", "answer(req[\"question\"], cfg, on_progress=on_progress)" in src
   and "financial_dd_steps.build(" in src)
ok("trajectory returned on success and on failure", src.count('"trajectory": steps') == 2)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
