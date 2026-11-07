"""
Tests for rate_limit.py and the daily-limit stop (fake HTTP transport + fake clock, no API calls).
    python tests/test_rate_limit.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import groq  # noqa: E402
import httpx  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

import rate_limit as rl  # noqa: E402
from agent_retry import classify_agent_error, retry_delay  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


CLOCK = FakeClock()
REAL_NOW, REAL_SLEEP = rl._now, rl._sleep


def fake_time():
    global CLOCK
    CLOCK = FakeClock()
    rl._now, rl._sleep = CLOCK.now, CLOCK.sleep
    rl.STATS.reset()


CFG = {"max_concurrent": 2, "requests_per_minute": 1000, "tokens_per_minute": 1_000_000, "reply_tokens": 600,
       "max_retries": 4, "max_wait_s": 120, "jitter_s": 0.5}

COMPLETION = {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def scripted(responses):
    """A fake server: returns the given (status, headers, body) in order, then 200s."""
    calls = []

    def handler(request):
        calls.append(request)
        status, headers, body = responses[len(calls) - 1] if len(calls) <= len(responses) else (200, {}, COMPLETION)
        return httpx.Response(status, headers=headers, json=body)

    return handler, calls


def client(responses, **cfg):
    handler, calls = scripted(responses)
    lim = rl.Limiter({**CFG, **cfg})
    return httpx.Client(transport=rl.LimitedTransport(lim, httpx.MockTransport(handler))), calls, lim


def post(c):
    return c.post("https://api.example/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})


ERR = {"error": {"message": "Rate limit reached for model on tokens per minute (TPM)", "type": "tokens"}}

print("429 handling:")
fake_time()
c, calls, _ = client([(429, {"retry-after": "3"}, ERR)])
r = post(c)
ok("retry-after: 3 -> waits 3s (+ jitter <= 0.5), then succeeds", r.status_code == 200 and len(calls) == 2
   and len(CLOCK.sleeps) == 1 and 3.0 <= CLOCK.sleeps[0] <= 3.5, CLOCK.sleeps)
ok("counted: 2 requests, 1 retry, 0 failures", (rl.STATS.requests, rl.STATS.retries, rl.STATS.failures) == (2, 1, 0))

fake_time()
c, calls, _ = client([(429, {"retry-after-ms": "1500"}, ERR)])
post(c)
ok("retry-after-ms: 1500 -> 1.5s", 1.5 <= CLOCK.sleeps[0] <= 2.0, CLOCK.sleeps)

fake_time()
c, calls, _ = client([(429, {"x-ratelimit-reset-tokens": "1m2.5s"}, ERR)])
post(c)
ok("only x-ratelimit-reset-tokens: 1m2.5s -> 62.5s", 62.5 <= CLOCK.sleeps[0] <= 63.0, CLOCK.sleeps)

fake_time()
c, calls, _ = client([(429, {}, ERR), (429, {}, ERR), (429, {}, ERR)])
r = post(c)
ok("no headers -> exponential backoff 2s, 4s, 8s (+ jitter)", r.status_code == 200 and len(CLOCK.sleeps) == 3
   and [int(s) for s in CLOCK.sleeps] == [2, 4, 8], CLOCK.sleeps)
ok("3 retries counted", rl.STATS.retries == 3)

fake_time()
c, calls, _ = client([(429, {}, ERR)] * 10, max_retries=2)
r = post(c)
ok("gives up after max_retries: returns the 429, 1 failure", r.status_code == 429 and len(calls) == 3
   and rl.STATS.failures == 1 and rl.STATS.retries == 2)

print("daily limit:")
DAILY = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                              "Limit 200000, Used 199950. Please try again in 7m12s.", "type": "tokens"}}
fake_time()
c, calls, _ = client([(429, {"retry-after": "432"}, DAILY)])
r = post(c)
ok("retry-after 432s > max_wait 120s -> no wait, no retry", r.status_code == 429 and len(calls) == 1 and not CLOCK.sleeps)
ok("STATS.daily_limit set, message kept, 1 failure", rl.STATS.daily_limit and "tokens per day" in rl.STATS.daily_limit_message
   and rl.STATS.failures == 1)
fake_time()
c, calls, _ = client([(429, {"retry-after": "5"}, DAILY)])
post(c)
ok("'per day' in the message -> daily even with a short retry-after", rl.STATS.daily_limit and len(calls) == 1)
ok("summary mentions it", "DAILY LIMIT reached" in rl.summary())

print("other statuses:")
fake_time()
c, calls, _ = client([(503, {}, {"error": "down"})])
ok("5xx retried with backoff", post(c).status_code == 200 and len(calls) == 2 and int(CLOCK.sleeps[0]) == 2)
fake_time()
c, calls, _ = client([(400, {}, {"error": "bad"})])
ok("400 returned at once, no retry", post(c).status_code == 400 and len(calls) == 1 and rl.STATS.retries == 0)
fake_time()
c, calls, _ = client([(401, {}, {"error": "key"})])
ok("401 returned at once, no retry", post(c).status_code == 401 and len(calls) == 1)
fake_time()
n = {"k": 0}


def flaky(request):
    n["k"] += 1
    if n["k"] == 1:
        raise httpx.ConnectError("reset")
    return httpx.Response(200, json=COMPLETION)


c = httpx.Client(transport=rl.LimitedTransport(rl.Limiter(CFG), httpx.MockTransport(flaky)))
ok("connection error retried", post(c).status_code == 200 and rl.STATS.retries == 1)

print("budgets (waiting BEFORE sending):")
fake_time()
b = rl.TokenBucket(per_minute=6500)
w1 = b.acquire(6000)
w2 = b.acquire(1000)
ok("6,500 tok/min: 6,000 goes at once, the next 1,000 waits ~4.6s for the refill", w1 == 0 and 4.5 < w2 < 4.7, w2)
fake_time()
b = rl.TokenBucket(per_minute=6500)
ok("a request bigger than the bucket waits for a full bucket (no deadlock)", b.acquire(7000) == 0 and b.level < 0)
fake_time()
c, calls, lim = client([], requests_per_minute=2)
for _ in range(3):
    post(c)
ok("2 req/min: the 3rd request waits ~30s", len(CLOCK.sleeps) == 1 and 29.9 < CLOCK.sleeps[0] < 30.1, CLOCK.sleeps)
ok("pre-send waits are reported as pacing, not retries", rl.STATS.retries == 0 and rl.STATS.throttle_wait_s > 29)
fake_time()
c, calls, lim = client([(200, {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "7.5s"}, COMPLETION)])
post(c)
post(c)
ok("server says remaining-tokens 100 (reset 7.5s) -> the next request waits 7.5s", any(7.4 < s < 7.6 for s in CLOCK.sleeps),
   CLOCK.sleeps)
fake_time()
c, calls, lim = client([], tokens_per_minute=10_000)
post(c)
ok("estimate corrected with the real usage (15 tokens): budget given back",
   lim.tokens.level > 10_000 - 50, round(lim.tokens.level))
ok("real usage counted", rl.STATS.tokens == 15)

print("max_concurrent (real threads):")
rl._now, rl._sleep = REAL_NOW, REAL_SLEEP
rl.STATS.reset()


def slow(request):
    time.sleep(0.05)
    return httpx.Response(200, json=COMPLETION)


lim = rl.Limiter(CFG)
c = httpx.Client(transport=rl.LimitedTransport(lim, httpx.MockTransport(slow)))
threads = [threading.Thread(target=post, args=(c,)) for _ in range(5)]
[t.start() for t in threads]
[t.join() for t in threads]
ok("5 parallel callers, never more than 2 in flight", lim.max_in_flight == 2 and rl.STATS.requests == 5, lim.max_in_flight)

print("duration parsing:")
ok("parse_duration", [rl.parse_duration(x) for x in ("7.66s", "2m59.56s", "1h2m3s", "250ms", "12", "soon", None)]
   == [7.66, 179.56, 3723.0, 0.25, 12.0, None, None])

print("end to end with ChatGroq (fake server, no network):")
fake_time()
handler, calls = scripted([(429, {"retry-after": "2"}, ERR)])
llm = ChatGroq(model="m", api_key="x", max_retries=0,
               http_client=httpx.Client(transport=rl.LimitedTransport(rl.Limiter(CFG), httpx.MockTransport(handler))))
ok("429 then 200 -> 'hello', 1 retry", llm.invoke("hi").content == "hello" and rl.STATS.retries == 1 and len(calls) == 2)
fake_time()
handler, calls = scripted([(429, {"retry-after": "3600"}, DAILY)] * 5)
llm = ChatGroq(model="m", api_key="x", max_retries=0,
               http_client=httpx.Client(transport=rl.LimitedTransport(rl.Limiter(CFG), httpx.MockTransport(handler))))
try:
    llm.invoke("hi")
    ok("daily -> RateLimitError", False)
except groq.RateLimitError as e:
    ok("daily -> groq.RateLimitError after ONE request (SDK retries off)", len(calls) == 1)
    ok("agent classifies it rate_limit_daily -> never retried", classify_agent_error(e) == "rate_limit_daily"
       and retry_delay("rate_limit_daily", 0) is None)
rl.STATS.reset()
ok("an ordinary 429 is still 'rate_limit' (retried)", classify_agent_error(
   groq.RateLimitError("Rate limit reached on tokens per minute", response=httpx.Response(
       429, request=httpx.Request("POST", "https://x")), body=None)) == "rate_limit")

print("wired in:")
import judges.base as jb  # noqa: E402
import agent_configurable as ac  # noqa: E402
judge_client = jb.llm.client._client
ok("judge client: max_retries 0 + LimitedTransport", judge_client.max_retries == 0
   and isinstance(judge_client._client._transport, rl.LimitedTransport), type(judge_client._client._transport).__name__)
agent_llm = ac._make_llm("openai/gpt-oss-120b", 0)
ok("agent client: max_retries 0 + LimitedTransport, SAME limiter as the judges",
   agent_llm.client._client.max_retries == 0
   and agent_llm.client._client._client._transport.limiter is judge_client._client._transport.limiter)
ok("limits read from rate_limits.yaml", rl.limiter("groq").cfg["tokens_per_minute"] == 6500
   and rl.limiter("groq").cfg["max_concurrent"] == 2)
ok("min_minutes: 110k tokens at 6,500/min ~ 17 min", round(rl.min_minutes(110_000)) == 17)

print("run_experiment stops cleanly at the daily limit:")
rl._now, rl._sleep = REAL_NOW, REAL_SLEEP
rl.STATS.reset()
import run_experiment  # noqa: E402

calls_made = []


def target(question):
    calls_made.append(question)
    if len(calls_made) == 3:
        rl.STATS.daily_limit, rl.STATS.daily_limit_message = True, "tokens per day (TPD)"
        return {"answer": None, "error": "429 per day", "error_type": "rate_limit_daily"}
    return "an answer"


scen = [{"id": f"s{i}", "question": f"q{i}?"} for i in range(1, 6)]
exp = {"name": "t", "description": "", "parent": None, "chain": ["t"], "changes_vs_parent": [],
       "target": "x", "settings": {}}
tmp = tempfile.mkdtemp()
with contextlib.redirect_stdout(io.StringIO()) as out:
    totals = run_experiment.execute(exp, target, False, scen, tmp, "t.yaml")
meta = json.load(open(os.path.join(tmp, "run_meta.json")))
ok("stops at scenario 3, no more calls", len(calls_made) == 3)
ok("2 traces saved, the quota-stopped one NOT saved (it isn't a fail)",
   sorted(os.listdir(os.path.join(tmp, "traces"))) == ["s1.json", "s2.json"])
ok("run_meta: status stopped_daily_limit + not_run", meta["status"] == "stopped_daily_limit"
   and meta["stopped"]["not_run"] == ["s3", "s4", "s5"])
ok("printed", "DAILY API LIMIT reached" in out.getvalue())
shutil.rmtree(tmp)

print("judging stops cleanly and resumes without re-judging:")
import run_judges  # noqa: E402
import judge_sets  # noqa: E402
tmp = tempfile.mkdtemp()
tdir, jdir = os.path.join(tmp, "traces"), os.path.join(tmp, "judged")
os.makedirs(tdir)
for i in range(1, 5):
    json.dump({"scenario_id": f"s{i}", "question": "q", "answer": "a", "error": None, "timestamp": "20260101_000000",
               "tools_used": [], "tool_outputs": []}, open(os.path.join(tdir, f"s{i}.json"), "w"))
judged_calls = []
quota = {"hit_once": False}
JS = {"judge_set": {**judge_sets.EXPERIMENT_V2, "judge_model": "m"}}


def evaluator(trace):
    judged_calls.append(trace["scenario_id"])
    if trace["scenario_id"] == "s3" and not quota["hit_once"]:   # the quota runs out once, at s3
        quota["hit_once"] = True
        rl.STATS.daily_limit = True
        raise groq.RateLimitError("tokens per day", response=httpx.Response(
            429, request=httpx.Request("POST", "https://x")), body=None)
    return {"final_score": 8.0}


rl.STATS.reset()
try:
    with contextlib.redirect_stdout(io.StringIO()):
        run_judges.judge_all_traces("t", tdir, jdir, evaluator, JS, skip_judged=True)
    ok("raises DailyLimitReached", False)
except run_judges.DailyLimitReached:
    ok("raises DailyLimitReached", True)
ok("s1, s2 judged + saved; s3 NOT saved (no judge_error file)", sorted(os.listdir(jdir)) == ["s1.json", "s2.json"])
rl.STATS.reset()
judged_calls.clear()
with contextlib.redirect_stdout(io.StringIO()):
    run_judges.judge_all_traces("t", tdir, jdir, evaluator, JS, skip_judged=True)
ok("re-run judges only s3, s4 (s1, s2 skipped)", judged_calls == ["s3", "s4"], judged_calls)
judged_calls.clear()
with contextlib.redirect_stdout(io.StringIO()):
    run_judges.judge_all_traces("t", tdir, jdir, evaluator, {"judge_set": {**judge_sets.EXPERIMENT, "judge_model": "m"}},
                                skip_judged=True)
ok("a different judge set re-judges everything", judged_calls == ["s1", "s2", "s3", "s4"], judged_calls)
judged_calls.clear()
with contextlib.redirect_stdout(io.StringIO()):
    run_judges.judge_all_traces("t", tdir, jdir, evaluator, JS)
ok("skip_judged off (default) -> unchanged old behaviour, re-judges all", len(judged_calls) == 4)
shutil.rmtree(tmp)

print("agent settings untouched:")
from experiment_config import config_hash, load_experiment  # noqa: E402
base = load_experiment(os.path.join("experiments", "baseline.yaml"))
ok("baseline settings have no rate-limit keys (limits live in rate_limits.yaml)",
   not any("rate" in k for k in base["settings"]))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
