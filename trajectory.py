"""
Step recorder: what an agent did, step by step. Works for any LangChain /
LangGraph app - nothing here knows what the agent is for.

    from trajectory import TrajectoryRecorder
    rec = TrajectoryRecorder()
    app.invoke(inputs, config={"callbacks": [rec]})
    rec.steps()     # [{"index", "type": "llm"|"tool", "name", ..., "latency_s", "tokens", "error"}, ...]

A step is either
  llm   one model call: latency, tokens (input/output/total) if the provider
        reports them, the tool calls it asked for, its text (truncated), error
  tool  one tool call: name, arguments, output (truncated), latency, error
in the order they started. The recorder only observes; the agent behaves the
same with or without it.

Traces may carry the list as an optional "trajectory" field (additive: old
traces without it keep working). from_messages() rebuilds the same step list,
without timing or per-step tokens, from a saved chat-format conversation
(trace["messages"]), so older runs can be checked too.
"""

import json
import threading
import time

from langchain_core.callbacks import BaseCallbackHandler

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


def _tokens(response):
    """Token usage of one LLM call from an LLMResult (usage_metadata, else llm_output)."""
    try:
        msg = response.generations[0][0].message
        um = getattr(msg, "usage_metadata", None)
        if um:
            return {"input": um.get("input_tokens"), "output": um.get("output_tokens"),
                    "total": um.get("total_tokens")}
    except (AttributeError, IndexError):
        pass
    tu = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage") or {}
    if tu:
        i, o = tu.get("prompt_tokens", tu.get("input_tokens")), tu.get("completion_tokens", tu.get("output_tokens"))
        return {"input": i, "output": o, "total": tu.get("total_tokens", (i or 0) + (o or 0))}
    return None


class TrajectoryRecorder(BaseCallbackHandler):
    raise_error = False           # a recorder bug must never break the agent

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._t0 = clock()
        self._open = {}
        self._done = []
        self._lock = threading.Lock()
        self.attempt = 0

    # ---- helpers
    def _start(self, run_id, step):
        with self._lock:
            step["started_at"] = round(self._clock() - self._t0, 4)
            step["_t"] = self._clock()
            step["attempt"] = self.attempt
            self._open[run_id] = step

    def _end(self, run_id, **fields):
        with self._lock:
            step = self._open.pop(run_id, None)
            if step is None:
                return
            step["latency_s"] = round(self._clock() - step.pop("_t"), 4)
            step.update(fields)
            self._done.append(step)

    # ---- LLM
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or ((serialized or {}).get("id") or ["llm"])[-1]
        self._start(run_id, {"type": "llm", "name": name, "n_messages_in": len(messages[0]) if messages else 0})

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or ((serialized or {}).get("id") or ["llm"])[-1]
        self._start(run_id, {"type": "llm", "name": name, "n_messages_in": len(prompts)})

    def on_llm_end(self, response, *, run_id, **kwargs):
        calls, text = [], ""
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = (msg.content if msg is not None else gen.text) or ""
            calls = [{"name": c["name"], "args": c["args"]} for c in (getattr(msg, "tool_calls", None) or [])]
        except (AttributeError, IndexError):
            pass
        self._end(run_id, tokens=_tokens(response), tool_calls=calls, output=_text(text), error=None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, tokens=None, tool_calls=[], output="", error=f"{type(error).__name__}: {error}")

    # ---- tools
    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        args = inputs
        if args is None:
            try:
                args = json.loads(input_str)
            except (TypeError, ValueError):
                args = {"input": input_str}
        self._start(run_id, {"type": "tool", "name": (serialized or {}).get("name") or kwargs.get("name") or "tool",
                             "args": args})

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id, output=_text(output), error=None)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, output="", error=f"{type(error).__name__}: {error}")

    # ---- result
    def steps(self):
        with self._lock:
            done = sorted(self._done, key=lambda s: s["started_at"])
        return [{"index": i, **s} for i, s in enumerate(done)]


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
