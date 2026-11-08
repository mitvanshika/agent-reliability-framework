"""
A fake DeepEval model for tests (runs inside .venv-deepeval). It answers every
schema DeepEval's faithfulness / answer-relevancy metrics ask for, from markers:
  INVENTED in the answer -> one of two claims is not supported   (faithfulness 0.5)
  OFFTOPIC in the answer -> one of two statements is irrelevant  (answer relevancy 0.5)
No network.
"""

from deepeval.models import DeepEvalBaseLLM

CALLS = []


class FakeModel(DeepEvalBaseLLM):
    def __init__(self):
        super().__init__("fake")

    def load_model(self):
        return self

    def get_model_name(self):
        return "fake"

    def generate(self, prompt, schema=None, **kwargs):
        CALLS.append((schema.__name__ if schema else None, prompt))
        if schema is None:
            return "{}"
        name, module = schema.__name__, schema.__module__
        if name == "Truths":
            return schema(truths=["The order was delivered.", "Delivery took 5 days."])
        if name == "Claims":
            return schema(claims=["The order was delivered."] + (["INVENTED: it comes with a free gift."]
                                                                   if "INVENTED" in prompt else ["It took 5 days."]))
        if name == "Statements":
            return schema(statements=["The order was delivered."] + (["OFFTOPIC: the weather is nice."]
                                                                       if "OFFTOPIC" in prompt else ["It took 5 days."]))
        if name == "Verdicts":
            bad = ("INVENTED" in prompt) if "faithfulness" in module else ("OFFTOPIC" in prompt)
            return schema.model_validate({"verdicts": [{"verdict": "yes", "reason": None},
                                                       {"verdict": "no" if bad else "yes", "reason": "fake"}]})
        if name.endswith("ScoreReason"):
            return schema(reason="fake reason")
        raise AssertionError(f"unexpected schema {module}.{name}")

    async def a_generate(self, prompt, schema=None, **kwargs):
        return self.generate(prompt, schema=schema, **kwargs)
