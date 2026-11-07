from judges.base import call_judge


def judge_completeness(question: str, answer: str, max_retries: int = 2):
    prompt = f"""You are evaluating whether an AI agent's answer fully addresses everything the question asked.

Question: {question}
Agent's answer: {answer}

Consider: if the question had multiple parts, did the answer address all of them? Did it leave anything unanswered or incomplete?

Score the completeness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries)


def judge_completeness_v2(question: str, answer: str, tool_outputs, expected_answer: str | None,
                          max_retries: int = 2):
    """v2 (judge set experiment_v2): also sees the tool outputs and the answer key, so an honest
    "that information isn't available" counts as complete and an invented answer does not."""
    outputs_text = "\n".join(f"- {o}" for o in tool_outputs) if tool_outputs else "(no tool outputs)"
    prompt = f"""You are evaluating whether an AI agent's answer fully handles everything the question asked.

Question: {question}
What the agent's tools returned:
{outputs_text}
Reference answer (what a correct answer covers): {expected_answer or "(not given)"}
Agent's answer: {answer}

Split the question into its parts. For each part:
- COMPLETE if the answer addresses it correctly (consistent with the tool outputs and the reference answer).
- COMPLETE if the information needed was NOT in the tool outputs and the answer honestly says it does not have it,
  or hands it to a human / escalates.
- NOT complete if the part is skipped; if it is answered with details that are in neither the tool outputs nor the
  reference (invented); or if the answer escalates or says it doesn't know although the needed information WAS in
  the tool outputs.

Score 0-10 = the share of parts that are complete (10 = every part). An invented answer is never complete, however
thorough it looks. Say which parts are complete and why, and state your confidence."""
    return call_judge(prompt, max_retries)


if __name__ == "__main__":
    result = judge_completeness(
        "What is 47 times 12, and convert the result from km to miles?",
        "47 times 12 is 564."
    )
    print(result)