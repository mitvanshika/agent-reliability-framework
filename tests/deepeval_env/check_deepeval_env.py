"""
Runs INSIDE .venv-deepeval (tests/test_deepeval.py starts it): the real DeepEval
metrics with a fake model, telemetry off, and no network at all.
    .venv-deepeval/Scripts/python tests/deepeval_env/check_deepeval_env.py
"""

import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, HERE]
os.environ.pop("DEEPEVAL_TELEMETRY_OPT_OUT", None)      # prove the worker switches it off by itself

attempts = []


def no_network(self, *a, **k):
    attempts.append(a)
    raise OSError("network blocked by the test")


socket.socket.connect = no_network                        # any telemetry / API call would land here
socket.create_connection = lambda *a, **k: no_network(None, *a)

import deepeval_worker as w  # noqa: E402
import fake_model  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


print("environment:")
ok("running in the separate environment, not the main Python", ".venv-deepeval" in sys.prefix, sys.prefix)
ok("telemetry switched off by the worker itself", w.telemetry_opt_out() is True
   and os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "1")
ok("DeepEval's state folder is inside the environment, not ~/.deepeval",
   os.environ["DEEPEVAL_HOME"].startswith(sys.prefix))
ping = w.handle({"op": "ping"}, {})
ok("ping", ping["ok"] and ping["deepeval"] == "4.3.3" and ping["telemetry_off"])

print("real DeepEval metrics, fake model:")
os.environ["DEEPEVAL_FAKE_MODEL"] = "fake_model:FakeModel"
models = {}
ctx = ["Order A1001: status=delivered, delivery took 5 days"]
r = w.handle({"op": "measure", "metric": "faithfulness", "input": "Where is A1001?",
              "actual_output": "It was delivered after 5 days.", "retrieval_context": ctx}, models)
ok("faithfulness, a supported answer -> 1.0", r.get("score") == 1.0 and r.get("reason") == "fake reason", r)
r = w.handle({"op": "measure", "metric": "faithfulness", "input": "Where is A1001?",
              "actual_output": "Delivered. INVENTED: it comes with a free gift.", "retrieval_context": ctx}, models)
ok("faithfulness, one invented claim of two -> 0.5", r.get("score") == 0.5, r)
schemas = [s for s, _ in fake_model.CALLS]
ok("faithfulness asked for truths, claims, verdicts, reason", {"Truths", "Claims", "Verdicts"} <= set(schemas)
   and any(s and s.endswith("ScoreReason") for s in schemas), schemas)
ok("the retrieved context reached the model", any("status=delivered" in p for _, p in fake_model.CALLS))
r = w.handle({"op": "measure", "metric": "answer_relevancy", "input": "Where is A1001?",
              "actual_output": "It was delivered."}, models)
ok("answer relevancy, on topic -> 1.0", r.get("score") == 1.0, r)
r = w.handle({"op": "measure", "metric": "answer_relevancy", "input": "Where is A1001?",
              "actual_output": "Delivered. OFFTOPIC: the weather is nice."}, models)
ok("answer relevancy, one irrelevant statement of two -> 0.5", r.get("score") == 0.5, r)
ok("unknown metric -> error, not a crash", "error" in w.handle({"op": "measure", "metric": "toxicity"}, models))

print("GroqModel (the custom model) without network:")


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kw):
        self.kwargs = kw

        class M:
            content = '{"claims": ["a", "b"]}'

        class C:
            message = M()

        class R:
            choices = [C()]
        return R()


class FakeGroq:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions()


fg = FakeGroq()
gm = w.GroqModel("openai/gpt-oss-120b", client=fg)
from deepeval.metrics.faithfulness.schema import Claims  # noqa: E402
out = gm.generate("Extract the claims. JSON.", schema=Claims)
ok("schema requested -> JSON mode + the schema in the prompt, returns a Claims instance",
   isinstance(out, Claims) and out.claims == ["a", "b"] and fg.chat.completions.kwargs["response_format"]
   == {"type": "json_object"} and '"claims"' in fg.chat.completions.kwargs["messages"][0]["content"])
ok("temperature 0, the configured Groq model", fg.chat.completions.kwargs["temperature"] == 0
   and fg.chat.completions.kwargs["model"] == "openai/gpt-oss-120b" and gm.get_model_name() == "groq:openai/gpt-oss-120b")
real = w.GroqModel("openai/gpt-oss-120b").load_model()
import rate_limit  # noqa: E402
ok("the real client: groq SDK, SDK retries off, the framework's rate limiter", real.max_retries == 0
   and isinstance(real._client._transport, rate_limit.LimitedTransport))

print("no network was touched:")
ok("zero connection attempts (no telemetry, no API)", attempts == [], attempts)

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
