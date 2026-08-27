"""
Shared retry + structured-output logic used by all 4 judges.

Previously each judge file (correctness.py, completeness.py,
tool_efficiency.py, groundedness.py) had its own copy-pasted retry
loop and its own ChatGroq/structured_output setup. Now each judge
file only builds its own prompt text and calls call_judge() — one
place to fix if the retry logic or model ever needs to change.
"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from judges.schema import JudgeVerdict

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
structured_llm = llm.with_structured_output(JudgeVerdict, method="json_mode")


def call_judge(prompt: str, max_retries: int = 2) -> JudgeVerdict:
    # json_mode requires the word "JSON" to appear in the prompt, or
    # Groq rejects the request outright — appended here once so every
    # judge file's prompt doesn't need to remember to include it.
    json_prompt = prompt + "\n\nRespond with a JSON object containing exactly these fields: score (integer 0-10), reason (string), confidence (float 0.0-1.0)."
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke(json_prompt)
        except Exception as e:
            last_error = e
    raise last_error