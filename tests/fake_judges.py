"""
Replaces every LLM judge with a fake so tests never reach the Groq API.

install(score_for) patches:
  - judges.base.call_judge used by every judge module (so the REAL judge
    functions run, including correctness_gt's fallback and context_relevance's
    N/A logic, but the LLM call is faked)
score_for(judge_name, prompt) -> (score, confidence)

Calls are recorded in CALLS as (judge_name, prompt).
"""

import importlib

from judges.schema import JudgeVerdict

CALLS = []
_MODULES = {
    "correctness": "judges.correctness",
    "correctness_gt": "judges.correctness_gt",
    "completeness": "judges.completeness",
    "groundedness": "judges.groundedness",
    "tool_efficiency": "judges.tool_efficiency",
    "context_relevance": "judges.context_relevance",
    "safety": "judges.safety",
    "generic": "judges.generic",   # all five generic judges: tell them apart by the prompt's first line
    "step": "judges.step",         # step_quality (judge_steps.py)
}


def _guard(*args, **kwargs):
    raise AssertionError("a test reached the real LLM (judges.base.call_judge was not faked)")


def install(score_for=lambda name, prompt: (8, 0.9)):
    CALLS.clear()
    import judges.base
    judges.base.call_judge = _guard
    for name, modname in _MODULES.items():
        mod = importlib.import_module(modname)

        def fake(prompt, max_retries=2, _name=name):
            CALLS.append((_name, prompt))
            score, conf = score_for(_name, prompt)
            return JudgeVerdict(score=score, reason=f"fake {_name}", confidence=conf)

        mod.call_judge = fake


def called(name):
    return [p for n, p in CALLS if n == name]
