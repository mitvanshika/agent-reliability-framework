"""
Client-side rate limiting + retries for LLM API calls, for any SDK built on httpx
(groq, openai, anthropic, ...). Settings: rate_limits.yaml.

    import rate_limit
    llm = ChatGroq(model=..., http_client=rate_limit.http_client("groq"), max_retries=0)
    client = OpenAI(http_client=rate_limit.http_client("openai"), max_retries=0)
    ...
    print(rate_limit.summary())   # "API: 263 requests, 2 retries (waited 14s), 0 failures"

Every request first waits for a free slot (max_concurrent), a request token
(requests_per_minute) and enough token budget (tokens_per_minute, estimated
from the request size and corrected with the real usage afterwards). Waiting
BEFORE sending is what avoids most 429s.

If a 429 still happens, the wait the server asks for is used: retry-after-ms /
retry-after, else x-ratelimit-reset-tokens (Groq / OpenAI style, e.g. "7.66s",
"2m59.5s", "250ms"), plus a little jitter. Without those headers: exponential
backoff (2s, 4s, 8s, ...). A wait longer than max_wait_s, or a message about a
per-day limit, means the daily quota is used up: no waiting, STATS.daily_limit
is set and the 429 is returned, so the caller can stop cleanly and resume later.

Pass max_retries=0 to the SDK so this is the only retry layer (and every retry
is counted).
"""

import contextlib
import email.utils
import json
import os
import random
import re
import threading
import time

import httpx
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_limits.yaml")
_now = time.monotonic      # replaced by a fake clock in tests
_sleep = time.sleep
_DAILY_WORDS = ("per day", "tokens per day", "requests per day", "(tpd)", "(rpd)")


def load_config(provider, path=None):
    with open(path or CONFIG_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {**(cfg.get("default") or {}), **(cfg.get(provider) or {})}


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.requests = self.retries = self.failures = 0
        self.retry_wait_s = self.throttle_wait_s = 0.0
        self.tokens = 0
        self.daily_limit = False
        self.daily_limit_message = None

    def add(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)


STATS = Stats()


def summary(stats=None) -> str:
    s = stats or STATS
    text = (f"API: {s.requests} requests, {s.retries} retries (waited {s.retry_wait_s:.0f}s), "
            f"{s.failures} failures; paced {s.throttle_wait_s:.0f}s before sending to stay under the limits")
    if s.daily_limit:
        text += f"\nDAILY LIMIT reached: {s.daily_limit_message}"
    return text


class TokenBucket:
    """`per_minute` units refill continuously, up to `capacity` (default: one minute's worth)."""

    def __init__(self, per_minute, capacity=None):
        self.rate = per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else per_minute)
        self.level = self.capacity
        self.updated = _now()
        self.lock = threading.Lock()

    def _refill(self):
        t = _now()
        self.level = min(self.capacity, self.level + (t - self.updated) * self.rate)
        self.updated = t

    def acquire(self, n=1.0) -> float:
        """Blocks until n units are available, takes them, returns seconds waited.
        A request bigger than the capacity waits for a full bucket and goes into debt."""
        waited = 0.0
        while True:
            with self.lock:
                self._refill()
                need = min(n, self.capacity)
                if self.level >= need:
                    self.level -= n
                    return waited
                wait = (need - self.level) / self.rate
            _sleep(wait)
            waited += wait

    def adjust(self, delta):
        """Give back (delta > 0) or take more (delta < 0) after the real usage is known."""
        with self.lock:
            self._refill()
            self.level = min(self.capacity, self.level + delta)

    def drain(self):
        with self.lock:
            self._refill()
            self.level = min(self.level, 0.0)


class Limiter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.slots = threading.BoundedSemaphore(int(cfg["max_concurrent"]))
        self.requests = TokenBucket(cfg["requests_per_minute"])
        self.tokens = TokenBucket(cfg["tokens_per_minute"])
        self.blocked_until = 0.0     # set from x-ratelimit-remaining-tokens = low
        self.in_flight = self.max_in_flight = 0
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def slot(self, est_tokens):
        self.slots.acquire()
        try:
            waited = 0.0
            pause = self.blocked_until - _now()
            if pause > 0:
                _sleep(pause)
                waited += pause
            waited += self.requests.acquire(1)
            waited += self.tokens.acquire(est_tokens)
            if waited:
                STATS.add(throttle_wait_s=waited)
            with self._lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield
            finally:
                with self._lock:
                    self.in_flight -= 1
        finally:
            self.slots.release()

    def sync(self, headers, est_tokens, actual_tokens):
        if actual_tokens is not None:
            self.tokens.adjust(est_tokens - actual_tokens)
        remaining = _number(headers.get("x-ratelimit-remaining-tokens"))
        reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if remaining is not None and reset is not None and remaining < self.cfg["reply_tokens"]:
            self.tokens.drain()
            self.blocked_until = max(self.blocked_until, _now() + reset)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_duration(value):
    """'7.66s' / '2m59.56s' / '1h2m3s' / '250ms' / '12' -> seconds (None if unreadable)."""
    if value is None:
        return None
    value = str(value).strip()
    if _number(value) is not None:
        return float(value)
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value)
    if not parts or "".join(n + u for n, u in parts) != value.replace(" ", ""):
        return None
    scale = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(n) * scale[u] for n, u in parts)


def server_wait(headers):
    """Seconds the server asked us to wait, or None."""
    ms = _number(headers.get("retry-after-ms"))
    if ms is not None:
        return ms / 1000
    ra = headers.get("retry-after")
    if ra is not None:
        if _number(ra) is not None:
            return float(ra)
        parsed = email.utils.parsedate_tz(ra)
        if parsed is not None:
            return max(0.0, email.utils.mktime_tz(parsed) - time.time())
    return parse_duration(headers.get("x-ratelimit-reset-tokens"))


def _usage_tokens(resp):
    try:
        return int(json.loads(resp.content)["usage"]["total_tokens"])
    except Exception:
        return None


def _error_text(resp):
    try:
        return resp.text[:300]
    except Exception:
        return ""


class LimitedTransport(httpx.BaseTransport):
    def __init__(self, limiter, inner=None):
        self.limiter = limiter
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request):
        cfg = self.limiter.cfg
        est = len(request.content or b"") // 4 + cfg["reply_tokens"]
        resp = None
        for attempt in range(cfg["max_retries"] + 1):
            try:
                with self.limiter.slot(est):
                    STATS.add(requests=1)
                    resp = self.inner.handle_request(request)
                    resp.read()
            except httpx.TransportError:          # connection reset, timeout, DNS
                if attempt == cfg["max_retries"]:
                    STATS.add(failures=1)
                    raise
                wait = 2 ** (attempt + 1) + random.uniform(0, cfg["jitter_s"])
                STATS.add(retries=1, retry_wait_s=wait)
                _sleep(wait)
                continue

            if resp.status_code != 429 and resp.status_code < 500:
                actual = _usage_tokens(resp)
                if actual:
                    STATS.add(tokens=actual)
                self.limiter.sync(resp.headers, est, actual)
                return resp                        # success, or a 4xx that retrying can't fix

            wait = server_wait(resp.headers) if resp.status_code == 429 else None
            text = _error_text(resp)
            if resp.status_code == 429 and ((wait is not None and wait > cfg["max_wait_s"])
                                            or any(w in text.lower() for w in _DAILY_WORDS)):
                STATS.daily_limit = True
                STATS.daily_limit_message = text or f"server asked to wait {wait:.0f}s"
                STATS.add(failures=1)
                return resp                        # surfaces as a RateLimitError; the run stops cleanly
            if attempt == cfg["max_retries"]:
                break
            wait = (wait if wait is not None else 2 ** (attempt + 1)) + random.uniform(0, cfg["jitter_s"])
            if resp.status_code == 429:
                self.limiter.tokens.drain()        # our estimate was off: don't let others rush in
            STATS.add(retries=1, retry_wait_s=wait)
            print(f"  [rate limit] HTTP {resp.status_code}, retry {attempt + 1}/{cfg['max_retries']} in {wait:.1f}s")
            _sleep(wait)
        STATS.add(failures=1)
        return resp


_LIMITERS = {}
_LIMITERS_LOCK = threading.Lock()


def limiter(provider="groq"):
    """One shared limiter per provider, so the agent and the judges share one budget."""
    with _LIMITERS_LOCK:
        if provider not in _LIMITERS:
            _LIMITERS[provider] = Limiter(load_config(provider))
        return _LIMITERS[provider]


def http_client(provider="groq", inner=None, timeout=60.0) -> httpx.Client:
    return httpx.Client(transport=LimitedTransport(limiter(provider), inner), timeout=timeout)


def is_daily_limit_error(e) -> bool:
    text = str(e).lower()
    return STATS.daily_limit or any(w in text for w in _DAILY_WORDS)


def min_minutes(est_tokens, provider="groq") -> float:
    """Lower bound on wall time: the tokens-per-minute budget, not the retries, is what limits a run."""
    return est_tokens / float(load_config(provider)["tokens_per_minute"])
