"""
Tests for adapters/token_meter.py: counting tokens of LLM calls a system makes
without reporting usage itself. Fake groq / openai / anthropic modules are put in
sys.modules at the real import paths, so no library, key or network is needed.
    python tests/test_token_meter.py
"""

import asyncio
import os
import re
import sys
import threading
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adapters"))

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def chat_reply(tin, tout):
    """What groq / openai chat.completions.create return: usage.prompt_tokens/completion_tokens."""
    return Obj(usage=Obj(prompt_tokens=tin, completion_tokens=tout, total_tokens=tin + tout))


def register(path, **classes):
    """Fake module at a real path (and its parents), e.g. groq.resources.chat.completions."""
    parts = path.split(".")
    for i in range(1, len(parts)):
        sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    mod = types.ModuleType(path)
    mod.__dict__.update(classes)
    sys.modules[path] = mod
    return mod


class GroqCompletions:
    def create(self, tin=10, tout=5, stream=False):
        return Obj(usage=None) if stream else chat_reply(tin, tout)


class GroqAsyncCompletions:
    async def create(self, tin=10, tout=5):
        return chat_reply(tin, tout)


class OpenAICompletions:
    def create(self, tin=1, tout=1):
        return chat_reply(tin, tout)


class OpenAIResponses:
    def create(self):          # Responses API: input_tokens/output_tokens/total_tokens
        return Obj(usage=Obj(input_tokens=7, output_tokens=3, total_tokens=10))


class Boom:
    def create(self):
        raise RuntimeError("429 rate limit")


class AsyncBoom:
    async def create(self):
        raise RuntimeError("429 rate limit")


register("groq.resources.chat.completions", Completions=GroqCompletions, AsyncCompletions=GroqAsyncCompletions)
register("openai.resources.chat.completions", Completions=OpenAICompletions)
register("openai.resources.responses", Responses=OpenAIResponses, AsyncResponses=Boom)  # sync create on an Async* class: still works
sys.modules["anthropic.resources.messages"] = None     # "not installed": import fails

import token_meter  # noqa: E402

print("install:")
first = token_meter.install()
ok("wraps what is installed; a missing library (anthropic) is skipped", sorted(first) == sorted([
    "groq.resources.chat.completions.Completions", "groq.resources.chat.completions.AsyncCompletions",
    "openai.resources.chat.completions.Completions", "openai.resources.responses.Responses",
    "openai.resources.responses.AsyncResponses"]), first)
ok("installing again wraps nothing (no double counting)", token_meter.install() == [])

groq, groq_async = GroqCompletions(), GroqAsyncCompletions()
print("counting:")
with token_meter.measure() as m:
    groq.create(100, 20)
    groq.create(50, 10)
ok("sums two calls", m.usage() == {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180}, m.usage())
ok("summary counts calls", m.summary()["llm_calls"] == 2 and m.summary()["unmetered_calls"] == 0, m.summary())

groq.create(999, 999)                                   # outside any block
with token_meter.measure() as m2:
    groq.create(1, 2)
ok("each measure starts at zero; calls outside a block are not counted",
   m2.usage() == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}, m2.usage())

with token_meter.measure() as m:
    OpenAIResponses().create()
ok("Responses / Anthropic naming (input_tokens/output_tokens)", m.usage() == {
    "input_tokens": 7, "output_tokens": 3, "total_tokens": 10}, m.usage())

u = token_meter._tokens(Obj(usage={"input_tokens": 4, "output_tokens": 6}))
ok("no total_tokens given: total = in + out (Anthropic)", u == (4, 6, 10), u)


async def two_async_calls():
    await groq_async.create(3, 4)
    await groq_async.create(3, 4)

with token_meter.measure() as m:
    asyncio.run(two_async_calls())
ok("async calls counted", m.usage() == {"input_tokens": 6, "output_tokens": 8, "total_tokens": 14}, m.usage())

print("honesty:")
with token_meter.measure() as m:
    groq.create(100, 20)
    groq.create(stream=True)
ok("a call without usage (stream) -> None, not a low number", m.usage() is None and
   m.summary()["unmetered_calls"] == 1, m.summary())
ok("the reason is recorded: stream=True", m.summary()["unmetered_reasons"] == ["stream=True"], m.summary())
r = token_meter._wrap(lambda: Obj(usage=None))     # a reply with no receipt, not streamed
with token_meter.measure() as m:
    r()
ok("no receipt without streaming: 'reply had no usage'", m.summary()["unmetered_reasons"] == ["reply had no usage"],
   m.summary())
with token_meter.measure() as m:
    pass
ok("no LLM call seen -> None (maybe a library it can't see)", m.usage() is None)
with token_meter.measure() as m:
    groq.create(10, 10)
    try:
        sys.modules["openai.resources.responses"].AsyncResponses().create()
    except RuntimeError:
        pass
ok("a call that raises counts as failed, not as tokens", m.summary() == {
    "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    "llm_calls": 1, "unmetered_calls": 0, "failed_calls": 1, "unmetered_reasons": []}, m.summary())

failing_async = token_meter._wrap(AsyncBoom.create)


async def call_failing():
    try:
        await failing_async(AsyncBoom())
    except RuntimeError:
        pass

with token_meter.measure() as m:
    asyncio.run(call_failing())
ok("an async call that raises counts as failed too", m.summary()["failed_calls"] == 1 and m.usage() is None,
   m.summary())

print("per-call list (6a):")
fmt = {"type": "json_schema", "json_schema": {"name": "chunk_grades", "schema": {}}}
labelled = token_meter._wrap(lambda model=None, response_format=None: chat_reply(100, 20))
with token_meter.measure() as m:
    labelled(model="gpt-oss-120b", response_format=fmt)
    groq.create(50, 10)
    groq.create(stream=True)
    try:
        sys.modules["openai.resources.responses"].AsyncResponses().create()
    except RuntimeError:
        pass
calls = m.calls()
ok("one entry per call, in order (failed ones too)", len(calls) == 4, calls)
ok("label = the JSON format's name; model kept", calls[0]["label"] == "chunk_grades"
   and calls[0]["model"] == "gpt-oss-120b" and calls[0]["total_tokens"] == 120, calls[0])
ok("no response_format -> label None", calls[1]["label"] is None and calls[1]["total_tokens"] == 60, calls[1])
ok("no receipt: tokens None + the reason", calls[2]["total_tokens"] is None and calls[2]["reason"] == "stream=True",
   calls[2])
ok("a call that raised: tokens None + the error type", calls[3]["total_tokens"] is None
   and calls[3]["error"] == "RuntimeError", calls[3])
ok("every call has its seconds", all(isinstance(c["seconds"], float) and c["seconds"] >= 0 for c in calls))
ok("calls() is a copy (editing it changes nothing)", (m.calls().clear() or len(m.calls())) == 4)
with token_meter.measure() as m:
    for _ in range(token_meter.MAX_CALLS + 5):
        groq.create(1, 1)
ok(f"list capped at {token_meter.MAX_CALLS}; counts and tokens still include every call",
   len(m.calls()) == token_meter.MAX_CALLS and m.summary()["llm_calls"] == token_meter.MAX_CALLS + 5
   and m.usage()["total_tokens"] == 2 * (token_meter.MAX_CALLS + 5), m.summary())

print("nesting and threads:")
with token_meter.measure() as outer:
    groq.create(1, 1)
    with token_meter.measure() as inner:
        groq.create(2, 2)
ok("nested meters: inner sees its call, outer sees both",
   inner.usage()["total_tokens"] == 4 and outer.usage()["total_tokens"] == 6, (inner.usage(), outer.usage()))

with token_meter.measure() as m:
    threads = [threading.Thread(target=lambda: [groq.create(1, 1) for _ in range(10)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
ok("8 threads x 10 calls = 80 calls, none lost", m.summary()["llm_calls"] == 80 and
   m.usage()["total_tokens"] == 160, m.summary())

print("Financial DD worker uses it:")
with open(os.path.join(REPO, "adapters", "financial_dd_worker.py"), encoding="utf-8") as f:
    src = f.read()
ok("imports token_meter", "import token_meter" in src)
ok("answer() runs inside a meter", re.search(r"with token_meter\.measure\(\) as meter:[^\n]*\n\s+a = answer\(", src) is not None)
ok("usage returned on success and on failure", src.count('"usage": meter.usage()') == 2)
ok("the call list returned on success and on failure", src.count('"llm_calls": meter.calls()') == 2)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)