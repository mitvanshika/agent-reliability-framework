"""
Step-level judge (judge set steps_v1, judge_steps.py): was ONE tool step of an
answer necessary and correct? Domain-neutral: the only system-specific input
is the target's one-line description.
"""

import json

from judges.base import call_judge


def _fmt_step(i, s):
    if s.get("type") == "tool":
        return f"{i}. tool {s.get('name')}({json.dumps(s.get('args'), default=str)}) -> {s.get('output')}"
    calls = ", ".join(c.get("name", "?") for c in s.get("tool_calls") or [])
    return f"{i}. model turn" + (f": asked for {calls}" if calls else ": answered")


def judge_step(question, history, step, description=None, max_retries=2):
    """history: the steps before this one (trajectory format); step: the tool step being judged."""
    before = "\n".join(_fmt_step(i + 1, s) for i, s in enumerate(history)) or "(none: this is the first step)"
    prompt = f"""You are checking ONE step an AI system took while handling a request, not its final answer.

The system being evaluated: {description or "(not given)"}
Request it received: {question if question else "(an empty request)"}
Steps before this one:
{before}
THIS step: it called {step.get("name")} with arguments {json.dumps(step.get("args"), default=str)}
What it returned: {step.get("output")}

Was this step necessary and correct?
- necessary: it was needed to handle the request; it did not fetch something already known from the request or the
  earlier steps, and it was not unrelated to the request
- correct: the right tool, with arguments that follow from the request and the earlier steps (not guessed or made up)
Score 0-10:
- 10: necessary and correct.
- 5: useful but avoidable (e.g. a near-repeat of an earlier call), or arguments partly off.
- 0-2: unnecessary or wrong (wrong tool, wrong record, invented arguments).
A step that correctly finds that the information does not exist is still necessary and correct.
Explain why, and state your confidence."""
    return call_judge(prompt, max_retries)
