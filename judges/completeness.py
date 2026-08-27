from judges.base import call_judge


def judge_completeness(question: str, answer: str, max_retries: int = 2):
    prompt = f"""You are evaluating whether an AI agent's answer fully addresses everything the question asked.

Question: {question}
Agent's answer: {answer}

Consider: if the question had multiple parts, did the answer address all of them? Did it leave anything unanswered or incomplete?

Score the completeness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


if __name__ == "__main__":
    result = judge_completeness(
        "What is 47 times 12, and convert the result from km to miles?",
        "47 times 12 is 564."
    )
    print(result)