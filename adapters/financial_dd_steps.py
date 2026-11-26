"""
Financial DD's answer as steps (trajectory format, see trajectory.py), so the dashboard
step timeline and step_checks.py work for it like for the support bot. Standard library
only: adapters/financial_dd_worker.py imports it inside Financial DD's own environment.

Financial DD already keeps what happened, it just doesn't return it:
  outcome.attempts   one record per pass of its self-healing loop: query, strategy
                     (original / rewrite / regrade / decompose), chunks found, how many
                     were relevant, stagnated (= the same chunks as the pass before)
  on_progress        events as they happen ("search", "grade", "rewrite", ...), in order
and token_meter's call list adds every LLM call with its label, tokens and seconds.

    retrieve  tool step per attempt: args {query, strategy}, output = the chunk ids it found
              (so step_checks' same_result_repeat fires when a rewritten query brings back
              exactly the same chunks), note = relevance + Financial DD's own flags
    llm       one step per LLM call (chunk_grades, grounded_answer, ...), with its tokens
Sorted by start time. Nothing in Financial DD changes.
"""

SEARCH_STAGES = ("search", "split")   # the event that starts an attempt's retrieval


def _retrieve_step(a, events):
    n = a.get("attempt")
    mine = [(t, stage) for t, stage, d in events if (d or {}).get("attempt") == n]
    start = next((t for t, stage in mine if stage in SEARCH_STAGES), mine[0][0] if mine else None)
    graded = next((t for t, stage in mine if stage == "grade"), None)
    ids = a.get("chunk_ids") or []
    note = f"{a.get('n_relevant', 0)} of {a.get('n_retrieved', 0)} relevant"
    if a.get("mean_relevance") is not None:
        note += f" (mean relevance {a['mean_relevance']:.2f})"
    if a.get("stagnated"):
        note += "; stagnated: essentially the same chunks as the attempt before"
    if a.get("note"):
        note += f"; {a['note']}"
    return {"type": "tool", "name": "retrieve",
            "args": {"query": a.get("query"), "strategy": a.get("strategy")},
            "output": "chunks: " + ", ".join(ids) if ids else "no chunks found",
            "note": note, "retrieval_attempt": n,
            "started_at": start,
            "latency_s": round(graded - start, 3) if start is not None and graded is not None else None,
            "error": None}


def _llm_step(c):
    tokens = None
    if c.get("total_tokens") is not None:
        tokens = {"input": c.get("input_tokens"), "output": c.get("output_tokens"), "total": c.get("total_tokens")}
    return {"type": "llm", "name": c.get("model") or "llm", "label": c.get("label") or "LLM call",
            "tool_calls": [], "output": c.get("reason") or "", "tokens": tokens,
            "started_at": c.get("started_at"), "latency_s": c.get("seconds"),
            "error": c.get("error")}


def build(attempts, events, calls):
    """attempts: outcome.attempts as dicts; events: [(seconds, stage, details)] from on_progress;
    calls: token_meter calls. Returns the step list, indexed in start order."""
    steps = [_retrieve_step(a, events) for a in attempts or []] + [_llm_step(c) for c in calls or []]
    order = {id(s): i for i, s in enumerate(steps)}
    steps.sort(key=lambda s: (s["started_at"] is None, s["started_at"] or 0.0, order[id(s)]))
    for i, s in enumerate(steps):
        s["index"] = i
    return steps
