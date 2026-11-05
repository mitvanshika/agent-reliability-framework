"""
Context relevance: did the retrieved / tool outputs contain the
information needed to answer the question?

This scores the RETRIEVAL step, not the answer: an agent can write a
perfect answer from bad context (lucky guess) or a bad answer from good
context. For a RAG system plugged in later, this is the "did we fetch
the right chunks" judge.

N/A (no LLM call, left out of final_score - not counted as 0) when no
retrieval happened: no tool outputs at all, or only tools that do not
retrieve anything (escalate_to_human just echoes its input).
"""

from judges.base import call_judge

NON_RETRIEVAL_TOOLS = {"escalate_to_human"}


def retrieval_outputs(tools_used, tool_outputs):
    """Tool outputs that came from retrieval tools. tools_used and
    tool_outputs are in call order; if they can't be paired, all outputs count."""
    tools_used, tool_outputs = list(tools_used or []), list(tool_outputs or [])
    if len(tools_used) == len(tool_outputs):
        return [out for name, out in zip(tools_used, tool_outputs) if name not in NON_RETRIEVAL_TOOLS]
    if tools_used and all(name in NON_RETRIEVAL_TOOLS for name in tools_used):
        return []
    return tool_outputs


def judge_context_relevance(question: str, tools_used, tool_outputs, expected_answer: str | None = None,
                            max_retries: int = 2):
    """Returns a verdict, or None for N/A (no retrieval happened)."""
    context = retrieval_outputs(tools_used, tool_outputs)
    if not context:
        return None
    reference = (f"\nFor reference, a correct answer is: {expected_answer}\n"
                 "If the reference says the information is not available, outputs that correctly show it is "
                 "missing (e.g. 'not found') are relevant and should score high.\n") if expected_answer else ""
    outputs_text = "\n".join(f"- {c}" for c in context)
    prompt = f"""You are evaluating the RETRIEVAL step of an AI agent, not its final answer.

Question: {question}
What the agent's tools retrieved:
{outputs_text}
{reference}
Did the retrieved information contain what is needed to answer the question correctly and completely?
- 10: everything needed is there.
- 5: some of it is there, important parts are missing or the wrong record/topic was fetched.
- 0: nothing relevant was retrieved.

Score context relevance from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)
