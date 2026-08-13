from langchain_groq import ChatGroq
from dotenv import load_dotenv
from judges.schema import JudgeVerdict

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(JudgeVerdict)

def judge_correctness(question: str, answer: str) -> JudgeVerdict:
    prompt = f"""You are evaluating whether an AI agent's answer to a question is factually correct.

Question: {question}
Agent's answer: {answer}

Score how correct this answer is from 0 to 10, explain why, and state your confidence."""
    return structured_llm.invoke(prompt)

if __name__ == "__main__":
    result = judge_correctness("What is 47 times 12?", "47 times 12 is 550.")
    print(result)
    result = judge_correctness("What is 47 times 12?", "47 times 12 is 550.")
    print(result)