"""
Domain-neutral judges (judge set generic_v1, opt-in per target with
`judge_set: generic_v1` in targets.yaml). Same five judges as experiment_v2,
worded so they fit any system: a chatbot, a summarizer, a RAG pipeline, a
classifier, an agent with or without tools.

What changes vs the original wording:
- the system is described by its own one-line `description` from targets.yaml
- "source material" = what the system was given or retrieved: tool outputs if
  it has tools, otherwise the input itself (a summarizer's source is the text
  it was asked to summarise)
- an honest "I don't know / I can't do that / passing this on" is correct
  behaviour when the source doesn't have the answer, whatever form the
  hand-off takes
- completeness: small unsupported extras cost a little; only an invented core
  answer scores 0 (the v2 over-correction on normal_01, see
  evidence/judge_fix_completeness.md)
"""

from judges.base import call_judge
from judges.context_relevance import retrieval_outputs

HONEST = "says it doesn't know, says it can't do it, or passes the request on"


def _header(description, question, sources):
    src = "\n".join(f"- {s}" for s in sources) if sources else "(none: only the input above)"
    return f"""The system being evaluated: {description or "(not given)"}
Input it received: {question if question else "(an empty input)"}
Source material it was given or retrieved:
{src}"""


def judge_correctness(question, answer, expected_answer, description=None, sources=None, max_retries=2):
    """Returns (verdict, mode): against the reference answer if there is one."""
    if not expected_answer:
        prompt = f"""You are checking whether a system's output is correct.

{_header(description, question, sources or [])}
Its output: {answer}

No reference answer is available. Judge correctness from the input and the source material only; do not reward
specific facts that neither contains. Score correctness from 0 to 10, explain why, and state your confidence."""
        return call_judge(prompt, max_retries), "fallback_no_answer_key"
    prompt = f"""You are checking a system's output against a reference answer written for this system.

{_header(description, question, sources or [])}
Reference answer (the facts a correct output must contain, and what it must NOT claim): {expected_answer}
Its output: {answer}

Rules:
- The reference is the ground truth for this system. Do not use outside knowledge to overrule it.
- Every required fact in the reference must be present and correct. Wording does not matter.
- Missing a required fact lowers the score; contradicting the reference scores very low (0-3).
- Claims the reference says must not be made score very low.
- Extra information is fine if it does not contradict the reference.
Score correctness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries), "answer_key"


def judge_completeness(question, answer, sources, expected_answer, description=None, max_retries=2):
    prompt = f"""You are checking whether a system's output fully handles everything the input asked for.

{_header(description, question, sources)}
Reference answer (what a correct output covers): {expected_answer or "(not given)"}
Its output: {answer}

Split the request into its parts. For each part:
- COMPLETE if the output handles it correctly (consistent with the input, the source material and the reference).
- COMPLETE if the information needed is in neither the input nor the source material and the output honestly
  {HONEST}.
- NOT complete if the part is skipped, if the output {HONEST} although the needed information WAS there,
  or if the core of the answer is made up (not in the input, the source material or the reference).
Small extra details that are not in the source but don't change the answer: a small deduction, not a zero.

Score 0-10 = the share of parts that are complete. Say which parts are complete and why, and state your confidence."""
    return call_judge(prompt, max_retries)


def judge_groundedness(question, answer, sources, description=None, max_retries=2):
    prompt = f"""You are checking whether a system's output is supported by what it was given, rather than made up.

{_header(description, question, sources)}
Its output: {answer}

Is every specific claim in the output supported by the INPUT or the SOURCE MATERIAL above?
- 10: everything specific comes from the input or the source material (rephrasing and summarising are fine).
- If the needed information is in neither and the output honestly {HONEST}, that is correct: score high (8-10).
- 0-2: it states specific facts (names, numbers, dates, rules, prices) found in neither, however plausible.
- In between: mostly supported with some unsupported details.
Score groundedness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


def judge_tool_efficiency(question, answer, tools_used, description=None, max_retries=2):
    prompt = f"""You are checking whether a system used its tools sensibly.

The system being evaluated: {description or "(not given)"}
Input it received: {question if question else "(an empty input)"}
Tools it called, in order: {tools_used or "(none)"}
Its output: {answer}

Did it call the tools the input actually needed, avoid unnecessary or wrong calls, and call them in a sensible
order? Calling no tool is right when none was needed. Score tool use from 0 to 10, explain why, and state your
confidence."""
    return call_judge(prompt, max_retries)


def judge_context_relevance(question, tools_used, tool_outputs, expected_answer=None, description=None,
                            non_retrieval=None, max_retries=2):
    """None = N/A: nothing was retrieved."""
    context = retrieval_outputs(tools_used, tool_outputs, non_retrieval)
    if not context:
        return None
    reference = (f"\nFor reference, a correct output is: {expected_answer}\n"
                 "If the reference says the information is not available, retrieved material that correctly shows "
                 "it is missing is relevant and should score high.\n") if expected_answer else ""
    outputs = "\n".join(f"- {c}" for c in context)
    prompt = f"""You are checking the RETRIEVAL step of a system, not its final output.

The system being evaluated: {description or "(not given)"}
Input it received: {question if question else "(an empty input)"}
What it retrieved:
{outputs}
{reference}
Did the retrieved material contain what is needed to handle the input correctly and completely?
- 10: everything needed is there. 5: some of it; important parts missing or the wrong thing fetched. 0: nothing relevant.
Score context relevance from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)
