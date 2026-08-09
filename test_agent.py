from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression and returns the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a value from one unit to another, e.g. km to miles."""
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
    }
    factor = conversions.get((from_unit, to_unit))
    if factor is None:
        return f"Conversion from {from_unit} to {to_unit} not supported"
    return str(value * factor)

tools = [calculator, unit_converter]
tool_node = ToolNode(tools)
llm = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

def run_agent(question: str):
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content

if __name__ == "__main__":
    print(run_agent("What is 47 times 12, and convert the result from km to miles?"))