from langchain_groq import ChatGroq
from dotenv import load_dotenv
from judges.schema import JudgeVerdict

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(JudgeVerdict)

def judge_tool_efficiency(question: str, answer: str, tools_used: list[str]) -> JudgeVerdict:
    prompt = f"""You are evaluating whether an AI agent used tools sensibly to answer a question.

Question: {question}
Tools the agent called: {tools_used}
Agent's final answer: {answer}

Consider: did it call the tools actually needed for this question? Did it avoid unnecessary or wrong tool calls? Did it use tools in a sensible order?

Score the tool usage from 0 to 10, explain why, and state your confidence."""
    return structured_llm.invoke(prompt)

if __name__ == "__main__":
    result = judge_tool_efficiency(
        "What is 47 times 12, and convert the result from km to miles?",
        "564 km is approximately 350.4 miles.",
        ["calculator", "unit_converter"]
    )
    print(result)