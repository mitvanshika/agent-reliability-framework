from judges.base import call_judge


def judge_groundedness(question: str, answer: str, tool_outputs: list[str], max_retries: int = 2):
    outputs_text = "\n".join(tool_outputs) if tool_outputs else "(no tool outputs returned)"
    prompt = f"""You are evaluating whether an AI agent's answer is honestly grounded in the actual data its tools returned, rather than made up.

Question: {question}
What the agent's tools actually returned: {outputs_text}
Agent's final answer: {answer}

Rules for scoring:
- If the tools returned no useful information (empty, "not found", "no matching FAQ", etc.) and the agent honestly said it doesn't know or escalated to a human, that is CORRECT and GOOD behavior. Score this high (8-10).
- If the tools returned no useful information but the agent confidently answered anyway with specific claims not present in the tool output, that is a HALLUCINATION. Score this very low (0-2), regardless of how well-written or plausible the answer sounds.
- If the tools did return useful information and the agent's answer accurately reflects it, score high.
- If the tools returned useful information but the agent's answer contradicts or ignores it, score low.

Score groundedness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


if __name__ == "__main__":
    result = judge_groundedness(
        "What's your policy on international customs fees?",
        "Our policy is buyers pay all customs fees.",
        ["No matching FAQ found for this query"]
    )
    print(result)