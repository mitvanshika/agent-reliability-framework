"""
Counts the LLM tokens a system uses, for systems that don't report usage themselves.

It wraps the `create` method of whichever LLM client libraries are installed, so any
code that calls them is counted, whatever the system is:

    groq       chat.completions.create            (sync + async)
    openai     chat.completions.create            (also OpenAI-compatible servers:
               responses.create                    vLLM, Ollama, Together ...)
    anthropic  messages.create

    import token_meter
    with token_meter.measure() as m:
        answer = my_system(question)
    m.usage()     # {"input_tokens", "output_tokens", "total_tokens"} or None
    m.summary()   # usage + llm_calls, unmetered_calls, failed_calls, unmetered_reasons

Honesty rule: usage() is None ("not measured") when a call returned no usage
(e.g. stream=True) or no LLM call was seen at all - never a number that is too low.
Not seen: other libraries, raw HTTP calls, and retries a library makes inside one
create(). Calls that raise (a 429, a 400) are counted in failed_calls, not in tokens.

Standard library only and imports nothing from this repo, so a system's own
environment (e.g. adapters/financial_dd_worker.py) can import it.
"""

import contextlib
import functools
import importlib
import inspect
import threading

# (module, classes whose .create is wrapped)
TARGETS = (
    ("groq.resources.chat.completions", ("Completions", "AsyncCompletions")),
    ("openai.resources.chat.completions", ("Completions", "AsyncCompletions")),
    ("openai.resources.responses", ("Responses", "AsyncResponses")),
    ("anthropic.resources.messages", ("Messages", "AsyncMessages")),
)

_lock = threading.Lock()     # several threads may finish calls at the same moment
_open = []                   # meters inside a `with measure()` block right now (nesting allowed)


class Meter:
    def __init__(self):
        self.llm_calls = self.unmetered_calls = self.failed_calls = 0
        self.input_tokens = self.output_tokens = self.total_tokens = 0
        self.unmetered_reasons = []     # why each unmetered call could not be counted

    def usage(self):
        """Tokens of every call inside the block, or None when that can't be known."""
        with _lock:
            if self.llm_calls == 0 or self.unmetered_calls:
                return None
            return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens}

    def summary(self):
        with _lock:
            counts = {"llm_calls": self.llm_calls, "unmetered_calls": self.unmetered_calls,
                      "failed_calls": self.failed_calls, "unmetered_reasons": list(self.unmetered_reasons)}
        return {"usage": self.usage(), **counts}


def _read(usage, *names):
    """First name present on the usage object (or dict) as an int."""
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return None


def _tokens(response):
    """(input, output, total) from a response, or None if it carries no usage."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    tin = _read(usage, "prompt_tokens", "input_tokens")         # chat style / Responses + Anthropic style
    tout = _read(usage, "completion_tokens", "output_tokens")
    if tin is None or tout is None:
        return None
    total = _read(usage, "total_tokens")
    return tin, tout, total if total is not None else tin + tout


def _record(response=None, failed=False, kwargs=None):
    tokens = None if failed else _tokens(response)
    reason = "stream=True" if (kwargs or {}).get("stream") else "reply had no usage"
    with _lock:
        for m in _open:
            if failed:
                m.failed_calls += 1
                continue
            m.llm_calls += 1
            if tokens is None:
                m.unmetered_calls += 1
                if len(m.unmetered_reasons) < 10:
                    m.unmetered_reasons.append(reason)
            else:
                m.input_tokens += tokens[0]
                m.output_tokens += tokens[1]
                m.total_tokens += tokens[2]


def _wrap(create):
    """Same signature as create. A sync create returns the response; an async one returns
    something to await, so the counting then waits for the real response."""
    async def finish(pending, kwargs):
        try:
            response = await pending
        except BaseException:
            _record(failed=True)
            raise
        _record(response, kwargs=kwargs)
        return response

    @functools.wraps(create)
    def metered(*args, **kwargs):
        try:
            result = create(*args, **kwargs)
        except BaseException:
            _record(failed=True)
            raise
        if inspect.isawaitable(result):
            return finish(result, kwargs)
        _record(result, kwargs=kwargs)
        return result

    metered._token_meter = True
    return metered


def install():
    """Wrap every installed library once (calling it again does nothing). Missing ones are skipped."""
    wrapped = []
    for module_name, class_names in TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name in class_names:
            cls = getattr(module, class_name, None)
            create = getattr(cls, "create", None)
            if create is None or getattr(create, "_token_meter", False):
                continue
            cls.create = _wrap(create)
            wrapped.append(f"{module_name}.{class_name}")
    return wrapped


@contextlib.contextmanager
def measure():
    """Count only the LLM calls made inside this block (from any thread)."""
    install()
    meter = Meter()
    with _lock:
        _open.append(meter)
    try:
        yield meter
    finally:
        with _lock:
            _open.remove(meter)
