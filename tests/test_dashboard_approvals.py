"""
Tests for DashboardApprover + the /approvals page and API (Flask test client, no API calls).
    python tests/test_dashboard_approvals.py
"""

import os
import shutil
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

root = tempfile.mkdtemp()
os.environ["ACTIONS_DB"] = os.path.join(root, "actions.db")     # the dashboard reads the same store

import app as appmod  # noqa: E402
import approval as ap  # noqa: E402
import idempotent as idem  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


client = appmod.app.test_client()
remote = {"REMOTE_ADDR": "203.0.113.9"}
st = idem.default_store()


def wait_pending(n=1, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        items = client.get("/api/approvals").get_json()["pending"]
        if len(items) >= n:
            return items
        time.sleep(0.02)
    return client.get("/api/approvals").get_json()["pending"]


def guarded(effects, approver):
    def issue_refund(order_id: str) -> str:
        effects.append(order_id)
        return f"refunded {order_id}"
    return ap.safe_action(issue_refund, approver=approver, store=st, describe=lambda a: f"Refund order {a['order_id']}")


def call_in_thread(fn, *args, rid):
    out = {}

    def run():
        with idem.request_context(rid):
            out["result"] = fn(*args)
    t = threading.Thread(target=run)
    t.start()
    return t, out


print("the page:")
r = client.get("/approvals")
ok("/approvals renders, with Approve / Reject and HTML escaping", r.status_code == 200 and b"Approve" in r.data
   and b"Reject" in r.data and b"esc(" in r.data)
ok("empty queue", client.get("/api/approvals").get_json()["pending"] == [])

print("approve from the dashboard -> acts once:")
effects = []
fn = guarded(effects, ap.DashboardApprover(poll_s=0.01))
t, out = call_in_thread(fn, "A1001", rid="run1/s1")
items = wait_pending()
ok("the request shows up as pending, nothing done yet", len(items) == 1 and items[0]["tool"] == "issue_refund"
   and items[0]["description"] == "Refund order A1001" and items[0]["request_id"] == "run1/s1" and effects == [])
r = client.post(f"/api/approvals/{items[0]['id']}", json={"approved": True, "by": "alice", "reason": "checked"})
t.join(5)
ok("POST approve -> 200, the waiting tool acts once", r.status_code == 200 and out.get("result") == "refunded A1001"
   and effects == ["A1001"] and st.executions("issue_refund") == 1, r.get_json())
ok("decision logged as 'dashboard: alice' with the reason", any(
   x["by_whom"] == "dashboard: alice" and x["reason"] == "checked" and x["approved"] == 1 for x in ap.approval_log(st)))
ok("queue: approved, by alice", ap.queue(st)[0]["status"] == "approved" and ap.queue(st)[0]["by_whom"] == "alice")
r = client.post(f"/api/approvals/{items[0]['id']}", json={"approved": False, "by": "bob"})
ok("a second decision on the same request -> 409, first decision stands", r.status_code == 409
   and ap.queue(st)[0]["status"] == "approved")

print("reject from the dashboard -> no effect:")
effects = []
fn = guarded(effects, ap.DashboardApprover(poll_s=0.01))
t, out = call_in_thread(fn, "A1002", rid="run1/s2")
items = wait_pending()
client.post(f"/api/approvals/{items[0]['id']}", json={"approved": False, "by": "bob", "reason": "outside 30 days"})
t.join(5)
ok("rejected: nothing done, the model reads who and why", effects == [] and
   out.get("result") == "Error: issue_refund was not approved by dashboard: bob (outside 30 days). Nothing was done.",
   out.get("result"))

print("nobody decides -> expired = reject:")
effects = []
fake_now = [0.0]
slow = ap.DashboardApprover(poll_s=1, timeout_s=3, sleep=lambda s: fake_now.__setitem__(0, fake_now[0] + s),
                            clock=lambda: fake_now[0])
fn = guarded(effects, slow)
with idem.request_context("run1/s3"):
    res = fn("A1003")
ok("after the timeout: rejected, nothing done", effects == [] and "no decision within 3s (expired)" in res, res)
ok("queue shows it as expired", next(x for x in ap.queue(st) if x["request_id"] == "run1/s3")["status"] == "expired")
ok("a late click on an expired request -> 409", client.post(
   f"/api/approvals/{next(x for x in ap.queue(st) if x['request_id'] == 'run1/s3')['id']}",
   json={"approved": True, "by": "late"}).status_code == 409)

print("who may decide:")
effects = []
fn = guarded(effects, ap.DashboardApprover(poll_s=0.01, timeout_s=10))
t, out = call_in_thread(fn, "A1004", rid="run1/s4")
qid = wait_pending()[0]["id"]
r = client.post(f"/api/approvals/{qid}", json={"approved": True, "by": "mallory"}, environ_base=remote)
ok("no token configured: a request from another machine -> 403", r.status_code == 403 and effects == [])
os.environ["APPROVALS_TOKEN"] = "s3cret"
ok("token configured -> the page is told to show the token box", client.get("/api/approvals").get_json()["token_required"])
r = client.post(f"/api/approvals/{qid}", json={"approved": True, "by": "mallory"})
ok("token configured: even localhost without the token -> 403", r.status_code == 403)
r = client.post(f"/api/approvals/{qid}", json={"approved": True, "by": "mallory"}, headers={"X-Approval-Token": "nope"},
                environ_base=remote)
ok("wrong token -> 403", r.status_code == 403)
r = client.post(f"/api/approvals/{qid}", json={"approved": True, "by": "dana"}, headers={"X-Approval-Token": "s3cret"},
                environ_base=remote)
t.join(5)
ok("right token from another machine -> approved, acts once", r.status_code == 200 and effects == ["A1004"])
del os.environ["APPROVALS_TOKEN"]

print("bad requests:")
ok("approved must be a bool and by is required -> 400", client.post("/api/approvals/x", json={"approved": "yes",
   "by": "a"}).status_code == 400 and client.post("/api/approvals/x", json={"approved": True}).status_code == 400)
ok("unknown id -> 404", client.post("/api/approvals/nope", json={"approved": True, "by": "a"}).status_code == 404)

print("with run_with_approvals (LangGraph graph, dashboard as the outside approver):")
from typing import Annotated, TypedDict  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

lg_effects = []


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    lg_effects.append(order_id)
    return f"cancelled {order_id}"


class S(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state):
    if not any(isinstance(m, ToolMessage) for m in state["messages"]):
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "cancel_order", "args": {"order_id": "Z1"},
                                                                 "id": "c"}])]}
    return {"messages": [AIMessage(content="done")]}


g = StateGraph(S)
g.add_node("agent", agent)
g.add_node("tools", ToolNode([ap.safe_action(cancel_order, approver="langgraph", store=st)]))
g.set_entry_point("agent")
g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, {"tools": "tools", END: END})
g.add_edge("tools", "agent")
lg = g.compile(checkpointer=MemorySaver())
res = {}
th = threading.Thread(target=lambda: res.update(r=ap.run_with_approvals(
    lg, {"messages": [HumanMessage("cancel Z1")]}, ap.DashboardApprover(store=st, poll_s=0.01),
    {"configurable": {"thread_id": "dash-1"}})))
th.start()
item = wait_pending()[0]
ok("the paused graph's action appears on the dashboard", item["tool"] == "cancel_order" and lg_effects == [])
client.post(f"/api/approvals/{item['id']}", json={"approved": True, "by": "erin"})
th.join(10)
ok("approved on the dashboard -> the graph resumed and cancelled once", lg_effects == ["Z1"]
   and res["r"]["messages"][-1].content == "done")
ok("get_approver('dashboard') works", isinstance(ap.get_approver("dashboard"), ap.DashboardApprover))

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
