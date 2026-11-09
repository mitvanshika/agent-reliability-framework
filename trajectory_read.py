"""
The read-only half of trajectory.py: steps rebuilt from a saved trace.
No LangChain import, so the dashboard (also in demo mode) can use it cheaply.
trajectory.py re-exports everything here.
"""

import json

MAX_TEXT = 500


def _text(x, n=MAX_TEXT):
    s = x if isinstance(x, str) else getattr(x, "content", None)
    if s is None:
        try:
            s = json.dumps(x, default=str)
        except Exception:
            s = str(x)
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... ({len(s)} chars)"


def from_messages(messages):
    """Steps rebuilt from a chat-format conversation (no timing, no per-step tokens)."""
    steps, calls = [], {}
    for m in messages or []:
        role = m.get("role")
        if role == "assistant":
            tcs = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"raw": fn.get("arguments")}
                tcs.append({"name": fn.get("name"), "args": args})
                calls[c.get("id")] = (fn.get("name"), args)
            steps.append({"type": "llm", "name": "llm", "tool_calls": tcs, "output": _text(m.get("content") or ""),
                          "latency_s": None, "tokens": None, "error": None})
        elif role == "tool":
            name, args = calls.get(m.get("tool_call_id"), ("tool", {}))
            steps.append({"type": "tool", "name": name, "args": args, "output": _text(m.get("content") or ""),
                          "latency_s": None, "error": None})
    return [{"index": i, **s, "source": "messages"} for i, s in enumerate(steps)]


def of_trace(trace):
    """(steps, source): the recorded trajectory if the trace has one, else rebuilt from its messages."""
    if isinstance(trace.get("trajectory"), list) and trace["trajectory"]:
        return trace["trajectory"], "recorded"
    if isinstance(trace.get("messages"), list) and trace["messages"]:
        return from_messages(trace["messages"]), "messages"
    return [], "none"
