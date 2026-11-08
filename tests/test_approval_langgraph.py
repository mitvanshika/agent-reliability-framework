"""
LangGraph mode of approval.py: a real graph (agent -> tools -> agent, like the
support bot) pauses at a checkpoint before an action and resumes with a
decision. The LLM is faked; no API calls.
    python tests/test_approval_langgraph.py
"""

import os
import shutil
import sys
import tempfile
from typing import Annotated, TypedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command  # noqa: E402

import approval as ap  # noqa: E402
import idempotent as idem  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
real_db_existed = os.path.exists("actions.db")


class S(TypedDict):
    messages: Annotated[list, add_messages]


def build(orders, store, saver=None, effects=None):
    """agent asks for issue_refund on each order in one turn, then answers with what the tools said."""
    effects = effects if effects is not None else []

    @tool
    def issue_refund(order_id: str) -> str:
        """Refund an order."""
        effects.append(order_id)
        return f"refunded {order_id}"

    safe = ap.safe_action(issue_refund, approver="langgraph", store=store,
                          describe=lambda a: f"Refund order {a['order_id']}")

    def agent(state):
        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not tool_msgs:
            return {"messages": [AIMessage(content="", tool_calls=[
                {"name": "issue_refund", "args": {"order_id": o}, "id": f"call_{o}"} for o in orders])]}
        return {"messages": [AIMessage(content="FINAL: " + " | ".join(m.content for m in tool_msgs))]}

    g = StateGraph(S)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([safe]))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END,
                            {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=saver or MemorySaver()), effects


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


def store(name):
    return idem.ActionStore(os.path.join(root, f"{name}.db"))


print("pauses BEFORE acting:")
st = store("pause")
app, effects = build(["A1001"], st)
r = app.invoke({"messages": [HumanMessage("refund A1001")]}, cfg("t1"))
pend = r.get("__interrupt__") or []
ok("the run returns paused (__interrupt__), one pending approval", len(pend) == 1)
v = pend[0].value if pend else {}
ok("it carries tool, args, description, request id (= thread_id) and key", v.get("type") == "approval"
   and v.get("tool") == "issue_refund" and v.get("args") == {"order_id": "A1001"}
   and v.get("description") == "Refund order A1001" and v.get("request_id") == "t1" and v.get("key"), v)
ok("nothing done yet, nothing claimed", effects == [] and st.executions() == 0 and st.get(v.get("key")) is None)

print("resume with approve -> acts once:")
r = app.invoke(Command(resume={pend[0].id: {"approved": True, "by": "alice", "reason": "ok", "key": v["key"]}}),
               cfg("t1"))
ok("finished, exactly one refund", "__interrupt__" not in r and effects == ["A1001"] and st.executions() == 1)
ok("the model got the tool's result", r["messages"][-1].content == "FINAL: refunded A1001", r["messages"][-1].content)
log = ap.approval_log(st)
ok("decision logged with who + why", any(x["approved"] == 1 and x["by_whom"] == "alice" and x["reason"] == "ok"
   for x in log), log)

print("re-running the same thread does not refund again:")
app2, _ = build(["A1001"], st, effects=effects)
r = app2.invoke({"messages": [HumanMessage("refund A1001 again")]}, cfg("t1"))
ok("same thread + same order -> stored result, no pause, no second refund", "__interrupt__" not in r
   and effects == ["A1001"] and st.executions() == 1, effects)

print("resume with reject -> no effect, the model is told:")
st = store("reject")
app, effects = build(["A1002"], st)
r = app.invoke({"messages": [HumanMessage("refund A1002")]}, cfg("t2"))
p2 = r["__interrupt__"][0]
r = app.invoke(Command(resume={p2.id: {"approved": False, "by": "bob", "reason": "outside 30 days",
                                       "key": p2.value["key"]}}), cfg("t2"))
ok("no refund", effects == [] and st.executions() == 0)
ok("the model reads 'not approved ... Nothing was done'", r["messages"][-1].content
   == "FINAL: Error: issue_refund was not approved by bob (outside 30 days). Nothing was done.",
   r["messages"][-1].content)
ok("nothing stored for the key (a later request may be approved)", st.get(idem.action_key(
   "issue_refund", {"order_id": "A1002"}, "t2")) is None)

print("resume values that can't be trusted -> reject:")
for label, value in (("'sure, why not'", "sure, why not"), ("plain True (no key)", True),
                     ("{'approved': True} (no key)", {"approved": True})):
    st = store(f"unclear_{len(label)}")
    app, effects = build(["A1003"], st)
    r = app.invoke({"messages": [HumanMessage("x")]}, cfg(f"t3-{len(label)}"))
    r = app.invoke(Command(resume={r["__interrupt__"][0].id: value}), cfg(f"t3-{len(label)}"))
    ok(f"{label} -> refused (no key to match it to an action), nothing done", effects == []
       and "not approved" in r["messages"][-1].content and "has no 'key'" in r["messages"][-1].content)
st = store("unclear_keyed")
app, effects = build(["A1003"], st)
r = app.invoke({"messages": [HumanMessage("x")]}, cfg("t3k"))
p3 = r["__interrupt__"][0]
r = app.invoke(Command(resume={p3.id: {"approved": "yes", "key": p3.value["key"]}}), cfg("t3k"))
ok("keyed but approved='yes' (not a bool) -> unclear -> reject", effects == []
   and "unclear answer" in r["messages"][-1].content)

print("restart while paused (new app object, same checkpointer):")
st = store("restart")
saver = MemorySaver()
effects = []
app, _ = build(["A1004"], st, saver, effects)
r = app.invoke({"messages": [HumanMessage("x")]}, cfg("t4"))
iid, ikey = r["__interrupt__"][0].id, r["__interrupt__"][0].value["key"]
del app
app_after_restart, _ = build(["A1004"], st, saver, effects)
r = app_after_restart.invoke(Command(resume={iid: {"approved": True, "by": "carol", "key": ikey}}), cfg("t4"))
ok("resumed by the new app: one refund", effects == ["A1004"] and st.executions() == 1)

print("two actions in one turn, decided separately (run_with_approvals):")
st = store("two")
app, effects = build(["A1", "A2"], st)
asked = []


def approver(req):
    asked.append(req.args["order_id"])
    return req.args["order_id"] == "A1"          # approve A1, reject A2


r = ap.run_with_approvals(app, {"messages": [HumanMessage("refund both")]}, approver, cfg("t5"))
ok("both were asked, once each", sorted(asked) == ["A1", "A2"], asked)
ok("A1 refunded once, A2 not", effects == ["A1"] and st.executions() == 1, effects)
ok("the model saw both outcomes", "refunded A1" in r["messages"][-1].content
   and "issue_refund was not approved" in r["messages"][-1].content, r["messages"][-1].content)
ok("A1 executed once although LangGraph re-ran the tools node on each resume",
   len(st.events("issue_refund", "executed")) == 1 and len(st.events("issue_refund", "replayed")) >= 0)

print("regression: resume values are matched by key, not position (3 actions, mixed decisions):")
for pattern in ("YNY", "NYN", "NNY", "YYN"):
    st = store(f"mix_{pattern}")
    orders = ["B1", "B2", "B3"]
    want = dict(zip(orders, (c == "Y" for c in pattern)))
    app, effects = build(orders, st)
    asked = []

    def decide(req, want=want, asked=asked):
        asked.append(req.args["order_id"])
        return want[req.args["order_id"]]

    r = ap.run_with_approvals(app, {"messages": [HumanMessage("refund all")]}, decide, cfg(f"mix-{pattern}"))
    ok(f"{pattern}: exactly the approved ones refunded, each once; each asked once",
       sorted(effects) == sorted(o for o in orders if want[o]) and sorted(asked) == orders
       and st.executions() == sum(want.values()), (effects, asked))

st = store("keyless")
app, effects = build(["C1", "C2"], st)
r = app.invoke({"messages": [HumanMessage("x")]}, cfg("kl"))
first = r["__interrupt__"][0]
r = app.invoke(Command(resume={first.id: {"approved": True}}), cfg("kl"))   # keyless: can't be matched
while r.get("__interrupt__"):
    p = r["__interrupt__"][0]
    r = app.invoke(Command(resume={p.id: {"approved": False, "key": p.value["key"]}}), cfg("kl"))
ok("the dangerous case: a keyless approve can no longer land on another action (nothing refunded)",
   effects == [], effects)

print("manual resumes WITH keys, approving the second of two:")
st = store("manual2")
app, effects = build(["D1", "D2"], st)
r = app.invoke({"messages": [HumanMessage("x")]}, cfg("m2"))
while r.get("__interrupt__"):
    p = r["__interrupt__"][0]
    r = app.invoke(Command(resume={p.id: {"approved": p.value["args"]["order_id"] == "D2", "key": p.value["key"]}}),
                   cfg("m2"))
ok("only D2 refunded, once", effects == ["D2"] and st.executions() == 1, effects)

print("run_with_approvals with the default-safe approvers:")
st = store("cli")
app, effects = build(["A9"], st)
shown = []
r = ap.run_with_approvals(app, {"messages": [HumanMessage("x")]},
                          ap.CliApprover(input=lambda p: "y", output=shown.append), cfg("t6"))
ok("CLI approver: shown the request, typed y -> one refund", effects == ["A9"] and any("Refund order A9" in s
   for s in shown))
st = store("deny")
app, effects = build(["A8"], st)
r = ap.run_with_approvals(app, {"messages": [HumanMessage("x")]}, "deny", cfg("t7"))
ok("deny approver -> no refund, model told", effects == [] and "not approved by deny" in r["messages"][-1].content)
try:
    ap.run_with_approvals(app, {"messages": [HumanMessage("x")]}, "cli", {})
    ok("no thread_id -> clear error", False)
except ValueError as e:
    ok("no thread_id -> clear error", "thread_id" in str(e))
try:
    ap.run_with_approvals(app, {"messages": [HumanMessage("x")]}, "langgraph", cfg("t8"))
    ok("'langgraph' as the outside approver is refused", False)
except ValueError as e:
    ok("'langgraph' as the outside approver is refused", "in-graph side" in str(e))

print("outside a graph:")
st = store("outside")
effects = []


def plain_refund(order_id: str) -> str:
    effects.append(order_id)
    return "refunded"


fn = ap.safe_action(plain_refund, approver="langgraph", store=st)
with idem.request_context("r"):
    out = fn("A1")
ok("LangGraphApprover with no graph -> reject with a clear reason, nothing done", effects == []
   and "needs a LangGraph run with a checkpointer" in out, out)

print("request ids inside a graph:")
ok("thread_id became the request id (checked above: 't1')", True)
seen = {}


@tool
def probe(x: str) -> str:
    """probe"""
    seen["rid"] = idem.current_request_id()
    return "ok"


g = StateGraph(S)
g.add_node("tools", ToolNode([probe]))
g.set_entry_point("tools")
g.add_edge("tools", END)
papp = g.compile(checkpointer=MemorySaver())
msg = AIMessage(content="", tool_calls=[{"name": "probe", "args": {"x": "1"}, "id": "p"}])
papp.invoke({"messages": [msg]}, {"configurable": {"thread_id": "th", "request_id": "run9/scen3"}})
ok("configurable.request_id wins over thread_id", seen.get("rid") == "run9/scen3", seen)
with idem.request_context("ctx-id"):
    papp.invoke({"messages": [msg]}, {"configurable": {"thread_id": "th2", "request_id": "cfg-id"}})
ok("request_context (set by the harness) wins over the config", seen.get("rid") == "ctx-id", seen)
ok("tests never touched a real actions.db", os.path.exists("actions.db") == real_db_existed)

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
