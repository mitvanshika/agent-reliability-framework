from judges.base import call_judge


def judge_tool_efficiency(question: str, answer: str, tools_used: list[str], max_retries: int = 2):
    prompt = f"""You are evaluating whether an AI agent used tools sensibly to answer a question.

Question: {question}
Tools the agent called: {tools_used}
Agent's final answer: {answer}

Consider: did it call the tools actually needed for this question? Did it avoid unnecessary or wrong tool calls? Did it use tools in a sensible order?

Score the tool usage from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


if __name__ == "__main__":
    result = judge_tool_efficiency(
        "What is 47 times 12, and convert the result from km to miles?",
        "564 km is approximately 350.4 miles.",
        ["calculator", "unit_converter"]
    )
    print(result)