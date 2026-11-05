"""
Shared retry + structured-output logic used by all 4 judges.

Previously each judge file (correctness.py, completeness.py,
tool_efficiency.py, groundedness.py) had its own copy-pasted retry
loop and its own ChatGroq/structured_output setup. Now each judge
file only builds its own prompt text and calls call_judge() — one
place to fix if the retry logic or model ever needs to change.
"""

import random
import time

import groq
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import ValidationError
from judges.schema import JudgeVerdict

load_dotenv()

# temperature=0 so judges score the same answer (nearly) the same way
# every run. ChatGroq defaults to 0.7 otherwise. Only the judges are
# pinned here; the agents keep their own settings.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
structured_llm = llm.with_structured_output(JudgeVerdict, method="json_mode")

# Temporary problems on Groq's side or the network: worth waiting and
# retrying. Note the groq SDK already retries these twice internally
# with a short backoff, so by the time one reaches us it has persisted.
TRANSIENT_ERRORS = (
    groq.RateLimitError,       # 429
    groq.APITimeoutError,      # request timed out
    groq.APIConnectionError,   # network / DNS / connection reset
    groq.InternalServerError,  # 5xx
)
# The key itself is wrong or not allowed: retrying can never fix this.
AUTH_ERRORS = (groq.AuthenticationError, groq.PermissionDeniedError)
# The model answered, but not with a valid verdict (bad JSON, missing
# field, score outside 0-10). Pydantic errors arrive wrapped in
# OutputParserException; ValidationError is listed in case one doesn't.
BAD_OUTPUT_ERRORS = (OutputParserException, ValidationError)


def _classify(e: Exception) -> str:
    if isinstance(e, TRANSIENT_ERRORS):
        return "transient"
    if isinstance(e, BAD_OUTPUT_ERRORS):
        return "bad_output"
    # In json_mode Groq itself rejects invalid JSON with a 400 whose
    # error code is json_validate_failed. Same problem as above.
    if isinstance(e, groq.BadRequestError) and "json_validate_failed" in str(e):
        return "bad_output"
    return "other"


def _backoff_seconds(attempt: int) -> float:
    # attempt 0 -> ~2s, 1 -> ~4s, 2 -> ~8s. The random jitter stops the
    # 4 parallel judges from all retrying at the exact same moment.
    return 2 ** (attempt + 1) + random.uniform(0, 0.5)


def call_judge(prompt: str, max_retries: int = 2) -> JudgeVerdict:
    # json_mode requires the word "JSON" to appear in the prompt, or
    # Groq rejects the request outright — appended here once so every
    # judge file's prompt doesn't need to remember to include it.
    json_prompt = prompt + "\n\nRespond with a JSON object containing exactly these fields: score (integer 0-10), reason (string), confidence (float 0.0-1.0)."
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke(json_prompt)
        except AUTH_ERRORS as e:
            raise RuntimeError(
                f"Groq rejected the API key ({type(e).__name__}). Check GROQ_API_KEY "
                "in your .env or CI secrets. Not retrying: a bad key won't fix itself."
            ) from e
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break
            kind = _classify(e)
            retry = f"retry {attempt + 1}/{max_retries}"
            if kind == "transient":
                wait = _backoff_seconds(attempt)
                print(f"  [judge] {type(e).__name__} (rate limit/timeout/connection), {retry} in {wait:.1f}s")
                time.sleep(wait)
            elif kind == "bad_output":
                print(f"  [judge] {type(e).__name__} (invalid verdict from model), {retry} now")
            else:
                print(f"  [judge] unexpected {type(e).__name__}: {e} -- {retry} now")
    raise last_error