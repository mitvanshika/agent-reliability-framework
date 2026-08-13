from langchain_groq import ChatGroq
from dotenv import load_dotenv
from judges.schema import JudgeVerdict

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(JudgeVerdict)

def judge_completeness(question: str, answer: str) -> JudgeVerdict:
    prompt = f"""You are evaluating whether an AI agent's answer fully addresses everything the question asked.

Question: {question}
Agent's answer: {answer}

Consider: if the question had multiple parts, did the answer address all of them? Did it leave anything unanswered or incomplete?

Score the completeness from 0 to 10, explain why, and state your confidence."""
    return structured_llm.invoke(prompt)

if __name__ == "__main__":
    result = judge_completeness(
        "What is 47 times 12, and convert the result from km to miles?",
        "47 times 12 is 564."
    )
    print(result)