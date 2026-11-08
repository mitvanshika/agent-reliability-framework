"""
Runs INSIDE the separate DeepEval environment (.venv-deepeval), never in the
main project's Python. judges/deepeval_bridge.py starts it and talks to it
over stdin/stdout, one JSON object per line:

    {"op": "ping"}                                              -> {"ok": true, "telemetry_off": true, ...}
    {"op": "measure", "metric": "faithfulness" | "answer_relevancy",
     "input": ..., "actual_output": ..., "retrieval_context": [...],
     "model": "openai/gpt-oss-120b"}                            -> {"score": 0.0-1.0, "reason": ...}
                                                                or {"error": "..."}

DeepEval's metrics use Groq through DeepEval's custom-model interface
(GroqModel below), with the same groq SDK version and the same client-side
rate limiter (rate_limit.py, rate_limits.yaml) as the rest of the framework.

Telemetry is switched OFF before DeepEval is imported (DEEPEVAL_TELEMETRY_OPT_OUT=1,
unless you set that variable yourself), and DeepEval's local state
(DEEPEVAL_HOME, which holds its anonymous telemetry id) is kept inside the
environment folder instead of your home folder.

For tests: DEEPEVAL_FAKE_MODEL=module:Class swaps in a fake DeepEvalBaseLLM.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_HOME", os.path.join(sys.prefix, "deepeval_home"))
WORK_DIR = os.path.join(sys.prefix, "deepeval_work")      # DeepEval may write cache files into the cwd
sys.path.insert(0, REPO)
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))

import deepeval  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.telemetry import telemetry_opt_out  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

METRICS = {"faithfulness": FaithfulnessMetric, "answer_relevancy": AnswerRelevancyMetric}
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqModel(DeepEvalBaseLLM):
    """DeepEval custom model backed by Groq (temperature 0, JSON mode when a schema is asked for)."""

    def __init__(self, model=DEFAULT_MODEL, client=None):
        self.model_name = model
        self._client = client
        super().__init__(model)

    def load_model(self):
        if self._client is None:
            import groq
            import rate_limit
            self._client = groq.Groq(http_client=rate_limit.http_client("groq"), max_retries=0)
        return self._client

    def generate(self, prompt, schema=None, **kwargs):
        content = prompt
        extra = {}
        if schema is not None:
            content += ("\n\nRespond with a JSON object matching this JSON schema exactly:\n"
                        + json.dumps(schema.model_json_schema()))
            extra["response_format"] = {"type": "json_object"}
        resp = self.load_model().chat.completions.create(
            model=self.model_name, temperature=0, messages=[{"role": "user", "content": content}], **extra)
        text = resp.choices[0].message.content or ""
        if schema is not None:
            try:
                return schema.model_validate_json(text)
            except Exception:
                return text                  # DeepEval parses JSON text itself as a fallback
        return text

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self):
        return f"groq:{self.model_name}"


def make_model(name):
    fake = os.environ.get("DEEPEVAL_FAKE_MODEL")
    if fake:
        import importlib
        module, _, cls = fake.partition(":")
        return getattr(importlib.import_module(module), cls)()
    return GroqModel(name or DEFAULT_MODEL)


def handle(req, models):
    if req.get("op") == "ping":
        return {"ok": True, "deepeval": deepeval.__version__, "telemetry_off": bool(telemetry_opt_out()),
                "python": sys.executable, "deepeval_home": os.environ["DEEPEVAL_HOME"]}
    if req.get("op") != "measure":
        return {"error": f"unknown op {req.get('op')!r}"}
    if req.get("metric") not in METRICS:
        return {"error": f"unknown metric {req.get('metric')!r}; known: {sorted(METRICS)}"}
    name = req.get("model") or DEFAULT_MODEL
    model = models.setdefault(name, make_model(name))
    metric = METRICS[req["metric"]](model=model, async_mode=False, include_reason=True, threshold=0.5)
    case = LLMTestCase(input=req.get("input") or "", actual_output=req.get("actual_output") or "",
                       retrieval_context=req.get("retrieval_context") or None)
    metric.measure(case)
    return {"score": metric.score, "reason": metric.reason}


def main():
    models = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = handle(json.loads(line), models)
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
