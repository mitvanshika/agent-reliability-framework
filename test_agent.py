import operator
import os
import time
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent_retry import classify_agent_error, retry_delay
from store import get_order, search_faq

load_dotenv()


@tool
def order_lookup(order_id: str) -> str:
    """Looks up an order by its ID and returns its status, total, and purchase date info."""
    order = get_order(order_id)
    if order is None:
        return f"No order found with ID {order_id}"
    return f"Order {order_id}: status={order['status']}, total=${order['total']}, days_since_purchase={order['days_since_purchase']}"


@tool
def refund_calculator(order_total: float, days_since_purchase: int) -> str:
    """Determines refund eligibility and amount based on order total and days since purchase."""
    if days_since_purchase <= 30:
        return f"Eligible for full refund of ${order_total}"
    return "Not eligible for refund: purchase was made more than 30 days ago"


@tool
def faq_search(query: str) -> str:
    """Searches FAQ topics: shipping, return policy, payment methods, PO box, or customs."""
    answer = search_faq(query)
    return answer if answer else "No matching FAQ found for this query"


@tool
def escalate_to_human(reason: str) -> str:
    """Flags that this case should be handed off to a human agent, with a reason."""
    return f"Escalated to human support. Reason: {reason}"


tools = [order_lookup, refund_calculator, faq_search, escalate_to_human]
tool_node = ToolNode(tools)

AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(model=AGENT_MODEL, temperature=0).bind_tools(tools)

SYSTEM_PROMPT = """You are a customer support agent for an online store.
Use order_lookup to check order details, refund_calculator to determine refund eligibility,
faq_search for general policy questions, and escalate_to_human for anything you cannot resolve
yourself or that requires human judgment. Always use the actual tools rather than guessing."""


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


def run_agent(question: str, max_retries: int = 2):
    last_error = None
    error_type = None
    for attempt in range(max_retries + 1):
        try:
            # recursion_limit=25 caps a stuck tool loop at ~12 rounds.
            # langgraph 1.2's default is 10007 (~5000 LLM calls per loop).
            result = app.invoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]},
                config={"recursion_limit": 25},
            )
            tools_used = []
            tool_outputs = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for call in msg.tool_calls:
                        tools_used.append(call["name"])
                if isinstance(msg, ToolMessage):
                    tool_outputs.append(msg.content)
            return {
                "answer": result["messages"][-1].content,
                "tools_used": tools_used,
                "tool_outputs": tool_outputs,
                "error": None,
                "error_type": None,
                "retries_needed": attempt
            }
        except Exception as e:
            last_error = str(e)
            error_type = classify_agent_error(e)
            delay = retry_delay(error_type, attempt)
            if delay is None or attempt == max_retries:
                break
            print(f"  [agent] {type(e).__name__} ({error_type}), retry {attempt + 1}/{max_retries}"
                  + (f" in {delay:.1f}s" if delay else " now"))
            time.sleep(delay)
    return {
        "answer": None,
        "tools_used": [],
        "tool_outputs": [],
        "error": last_error,
        "error_type": error_type,
        "retries_needed": attempt
    }


if __name__ == "__main__":
    print(run_agent("Can I get a refund for order A1002?"))