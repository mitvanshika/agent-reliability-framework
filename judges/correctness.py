from judges.base import call_judge


def judge_correctness(question: str, answer: str, max_retries: int = 2):
    prompt = f"""You are evaluating whether an AI agent's answer to a question is factually correct.

Question: {question}
Agent's answer: {answer}

Score how correct this answer is from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


if __name__ == "__main__":
    result = judge_correctness("What is 47 times 12?", "47 times 12 is 564.")
    print(result)