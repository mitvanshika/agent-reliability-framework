"""
Shared error classification + retry policy for the demo agents'
run_agent() (test_agent.py and agent_v2_weak.py).

Not every failure deserves a retry. A tool loop or a bad API key will
fail the same way every time, so re-running the whole agent graph just
burns calls. Rate limits and timeouts are temporary, so they get an
exponential backoff. Anything unrecognised is retried immediately, as
before (e.g. Groq's random 400 "tool_use_failed" on a malformed tool call).

Note: the groq SDK already retries 429/timeout/5xx twice internally
with a short backoff, so by the time one reaches here it has persisted.
"""

import random

import groq
from langgraph.errors import GraphRecursionError

RETRYABLE_WITH_BACKOFF = {"rate_limit", "timeout"}
NEVER_RETRY = {"loop_limit", "auth", "rate_limit_daily"}   # daily quota used up: waiting minutes won't help


def classify_agent_error(e: Exception) -> str:
    """Returns one of: loop_limit, auth, rate_limit_daily, rate_limit, timeout, crash."""
    if isinstance(e, GraphRecursionError):
        return "loop_limit"   # stuck in a tool loop: same input loops again
    if isinstance(e, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return "auth"         # bad / unauthorised key: retrying can't fix it
    if isinstance(e, groq.RateLimitError):
        import rate_limit
        return "rate_limit_daily" if rate_limit.is_daily_limit_error(e) else "rate_limit"
    if isinstance(e, (groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError)):
        return "timeout"      # transient network / server problem
    return "crash"


def retry_delay(error_type: str, attempt: int):
    """None = don't retry, 0 = retry now, >0 = seconds to wait first.

    Backoff: attempt 0 -> ~2s, 1 -> ~4s, 2 -> ~8s, plus jitter.
    """
    if error_type in NEVER_RETRY:
        return None
    if error_type in RETRYABLE_WITH_BACKOFF:
        return 2 ** (attempt + 1) + random.uniform(0, 0.5)
    return 0
