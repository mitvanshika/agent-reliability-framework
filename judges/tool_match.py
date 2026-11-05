"""
tool_match: compares the tools an agent called with the scenario's
expected_tools in plain code. No LLM, no API call, fully deterministic.

Display only: it is NOT part of final_score.

Score (0-10):
    10 x (share of expected tools that were called)
    - 1 per unexpected extra tool (max -3)
    - 2 if the expected tools were called in a different order
      (only the first call of each tool counts; repeats are not penalised,
      they can be retries)
No expected_tools in the scenario -> N/A.
"""

EXTRA_PENALTY = 1.0
MAX_EXTRA_PENALTIES = 3
ORDER_PENALTY = 2.0


def tool_match(tools_used, expected_tools):
    if expected_tools is None:
        return {"na": True, "reason": "scenario has no expected_tools"}
    used, expected = list(tools_used or []), list(expected_tools)
    missing = [t for t in expected if t not in used]
    extras = sorted(set(used) - set(expected))
    coverage = 1.0 if not expected else (len(expected) - len(missing)) / len(expected)
    first_calls = [used.index(t) for t in expected if t in used]
    order_ok = first_calls == sorted(first_calls)
    repeated = sorted(t for t in set(used) if used.count(t) > 1)

    score = 10 * coverage - EXTRA_PENALTY * min(len(extras), MAX_EXTRA_PENALTIES) - (0 if order_ok else ORDER_PENALTY)
    score = round(max(0.0, min(10.0, score)), 1)

    parts = [f"{len(expected) - len(missing)}/{len(expected)} expected tools called" if expected
             else "no tools expected"]
    if missing:
        parts.append(f"missing: {', '.join(missing)}")
    if extras:
        parts.append(f"extra: {', '.join(extras)}")
    if not order_ok:
        parts.append("expected tools called out of order")
    if repeated:
        parts.append(f"repeated (not penalised): {', '.join(repeated)}")
    return {"score": score, "reason": "; ".join(parts), "missing": missing, "extras": extras,
            "order_ok": order_ok, "repeated": repeated}
