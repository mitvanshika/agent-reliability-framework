"""
Correctness against the scenario's answer key (expected_answer).

The classic correctness judge only sees the question and the answer, so
it cannot verify store-specific facts: it called a correct, tool-grounded
"A1001 was delivered, $89.99" answer "likely fabricated" (score 2). With
an answer key the judge knows what the right answer is.

No expected_answer for the scenario -> falls back to the classic
correctness judge (judges/correctness.py), and says so in "mode".
"""

from judges.base import call_judge
from judges.correctness import judge_correctness


def judge_correctness_gt(question: str, answer: str, expected_answer: str | None, max_retries: int = 2):
    """Returns (verdict, mode) with mode "answer_key" or "fallback_no_answer_key"."""
    if not expected_answer:
        return judge_correctness(question, answer, max_retries), "fallback_no_answer_key"
    prompt = f"""You are checking an AI customer-support agent's answer against a reference answer written by the store.

Question: {question}
Reference answer (the facts a correct answer must contain, and what it must NOT claim): {expected_answer}
Agent's answer: {answer}

Rules for scoring:
- The reference is the ground truth for this store. Do not use outside knowledge about other companies.
- Every required fact (numbers, statuses, dates, eligibility, whether to escalate) must be present and correct. Wording does not matter.
- Missing a required fact lowers the score; contradicting the reference scores very low (0-3).
- Claims the reference says must not be made (e.g. inventing details, promising refunds) score very low.
- Extra information is fine if it does not contradict the reference.

Score correctness from 0 to 10, explain why, and state your confidence."""
    return call_judge(prompt, max_retries), "answer_key"
