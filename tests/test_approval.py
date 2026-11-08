"""
Tests for approval.py: requires_approval, safe_action, approvers, the log (no LLM, no API calls).
    python tests/test_approval.py
"""

import os
import shutil
import sys
import tempfile
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from langchain_core.tools import tool  # noqa: E402

import approval as ap  # noqa: E402
from idempotent import ActionStore, MissingRequestId, request_context  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
real_db_existed = os.path.exists("actions.db")


def make(approver, name="s", describe=None):
    st = ActionStore(os.path.join(root, f"{name}.db"))
    effects = []

    def issue_refund(order_id: str) -> str:
        effects.append(order_id)
        return f"refunded {order_id}"

    return ap.safe_action(issue_refund, approver=approver, store=st, describe=describe), st, effects


class Scripted:
    """An approver that answers from a list and counts how often it was asked."""
    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def decide(self, req):
        self.asked.append(req)
        return self.answers.pop(0)


print("default = deny:")
fn, st, effects = make(None, "deny")
with request_context("r1"):
    out = fn("A1001")
ok("blocked: nothing happened, the model reads why", effects == [] and st.executions() == 0
   and out == "Error: issue_refund was not approved by deny (default) (no approver configured). Nothing was done.", out)
log = ap.approval_log(st)
ok("logged: rejected, by deny", len(log) == 1 and log[0]["approved"] == 0 and log[0]["by_whom"] == "deny (default)")

print("reject, then approve:")
appr = Scripted(ap.Decision(False, "alice", "customer already got a voucher"), ap.Decision(True, "bob", "checked"))
fn, st, effects = make(appr, "rej", describe=lambda a: f"Refund order {a['order_id']}")
with request_context("r1"):
    first = fn("A1001")
    effects_after_reject = list(effects)
    second = fn("A1001")
ok("reject -> no effect, who + why passed to the model", effects_after_reject == [] and "alice" in first
   and "customer already got a voucher" in first and first.endswith("Nothing was done."), first)
ok("the rejection was NOT stored: the same request asked again and approved -> acts once",
   second == "refunded A1001" and effects == ["A1001"] and len(appr.asked) == 2)
ok("approver saw the description", appr.asked[0].description == "Refund order A1001" and appr.asked[0].args == {
   "order_id": "A1001"} and appr.asked[0].request_id == "r1")

print("approve, then retries:")
appr = Scripted(ap.Decision(True, "alice"))
fn, st, effects = make(appr, "retry")
with request_context("r7"):
    a, b, c = fn("A1003"), fn("A1003"), fn(" A1003 ")
ok("one approval, one execution, three identical results", (a, b, c) == ("refunded A1003",) * 3
   and effects == ["A1003"] and len(appr.asked) == 1 and st.executions("issue_refund") == 1)
log = ap.approval_log(st)
ok("log: 1 approval + 2 'not needed: already done'", [r["approved"] for r in log] == [1, None, None]
   and all("already done" in r["reason"] for r in log[1:]))
with request_context("r8"):
    fn("A1003")
ok("a NEW request asks again (and fails: the scripted approver has no answer left -> reject)",
   effects == ["A1003"] and len(appr.asked) == 2)

print("fail closed:")
for label, answer in (("raises", RuntimeError("chat service down")), ("says 'maybe'", "maybe"), ("returns None", None)):
    def bad(req, answer=answer):
        if isinstance(answer, Exception):
            raise answer
        return answer
    fn, st, effects = make(bad, f"bad_{label.split()[0]}")
    with request_context("r"):
        out = fn("A1001")
    ok(f"approver {label} -> reject, nothing done, logged", effects == [] and "Nothing was done" in out
       and ap.approval_log(st)[0]["approved"] == 0, out)
fn, st, effects = make(lambda req: True, "bool")
with request_context("r"):
    fn("A1001")
ok("a callback returning True approves", effects == ["A1001"])

print("CLI approver:")
for typed, approved in (("y", True), ("YES ", True), ("", False), ("n", False), ("nope", False)):
    shown = []
    cli = ap.CliApprover(input=lambda prompt, t=typed: t, output=shown.append)
    fn, st, effects = make(cli, f"cli_{len(typed)}_{typed.strip() or 'empty'}", describe=lambda a: "Refund A1001")
    with request_context("r"):
        fn("A1001")
    ok(f"typed {typed!r} -> {'approve' if approved else 'reject'}", (effects == ["A1001"]) == approved
       and any("issue_refund" in s for s in shown) and any("Refund A1001" in s for s in shown))

print("request id required:")
fn, st, effects = make("auto", "rid")
try:
    fn("A1001")
    ok("no request id -> MissingRequestId", False)
except MissingRequestId:
    ok("no request id -> MissingRequestId, and nobody was asked", effects == [] and ap.approval_log(st) == [])

print("concurrency (auto approver, 5 threads, same request):")
fn, st, effects = make("auto", "race")
gate = threading.Event()


def worker():
    gate.wait()
    with request_context("same"):
        fn("A1001")


ts = [threading.Thread(target=worker) for _ in range(5)]
[t.start() for t in ts]
gate.set()
[t.join() for t in ts]
ok("executed exactly once", effects == ["A1001"] and st.executions() == 1, effects)
ok("AutoApprover is labelled in the log", all(r["by_whom"] in ("auto (tests only)", "-") for r in ap.approval_log(st)))

print("requires_approval alone (no idempotency) asks every time:")
st = ActionStore(os.path.join(root, "plain.db"))
n = {"k": 0}
asked = Scripted(ap.Decision(True, "a"), ap.Decision(True, "a"))


def act(x: int) -> str:
    n["k"] += 1
    return "ok"


plain = ap.requires_approval(act, approver=asked, store=st)
with request_context("r"):
    plain(1)
    plain(1)
ok("2 calls -> 2 approvals, 2 executions (this is why safe_action exists)", n["k"] == 2 and len(asked.asked) == 2)

print("LangChain tools:")
st = ActionStore(os.path.join(root, "lc.db"))
lc = []


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    lc.append(order_id)
    return f"cancelled {order_id}"


safe = ap.safe_action(cancel_order, approver="auto", store=st)
ok("same name, description, schema", (safe.name, safe.description, safe.args)
   == (cancel_order.name, cancel_order.description, cancel_order.args))
with request_context("t"):
    safe.invoke({"order_id": "A1"})
    safe.invoke({"order_id": "A1"})
ok("invoke twice -> once", lc == ["A1"])
ok("is_safe_action: wrapped yes, original no", ap.is_safe_action(safe) and not ap.is_safe_action(cancel_order))
denied = ap.safe_action(cancel_order, store=st)
with request_context("t2"):
    out = denied.invoke({"order_id": "A2"})
ok("default deny through .invoke -> text for the model, nothing done", "not approved" in out and lc == ["A1"])

print("approver names:")
ok("get_approver: deny / auto / cli / callable / object", [type(ap.get_approver(s)).__name__ for s in (
   None, "deny", "auto", "cli", lambda r: True, ap.DenyApprover())] ==
   ["DenyApprover", "DenyApprover", "AutoApprover", "CliApprover", "CallbackApprover", "DenyApprover"])
try:
    ap.get_approver("always")
    ok("unknown name rejected", False)
except ValueError as e:
    ok("unknown name rejected", "unknown approver 'always'" in str(e))
ok("tests never touched a real actions.db", os.path.exists("actions.db") == real_db_existed)

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
