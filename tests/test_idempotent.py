"""
Tests for idempotent.py (no LLM, no API calls).
    python tests/test_idempotent.py
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

from langchain_core.tools import tool  # noqa: E402

import idempotent as idem  # noqa: E402
from idempotent import ActionStore, MissingRequestId, idempotent, request_context  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
real_db_existed = os.path.exists("actions.db")


def fresh(name):
    return ActionStore(os.path.join(root, f"{name}.db"))


print("acts once per request:")
st = fresh("basic")
effects = []


@idempotent(store=st)
def send_invoice(customer_id: str, amount: float) -> str:
    effects.append((customer_id, amount))
    return f"invoice {len(effects)} sent to {customer_id} for {amount}"


with request_context("run1/s1"):
    r1 = send_invoice("C1", 10)
    r2 = send_invoice("C1", 10)
    r3 = send_invoice(" C1 ", 10.0)                  # normalized: same key
    r4 = send_invoice(customer_id="C1", amount=10)   # keywords: same key
ok("first call acts", r1 == "invoice 1 sent to C1 for 10" and len(effects) == 1, effects)
ok("repeats return the STORED result, no second effect", r2 == r3 == r4 == r1 and len(effects) == 1)
ok("events: 1 executed, 3 replayed", st.executions("send_invoice") == 1
   and len(st.events("send_invoice", "replayed")) == 3)
with request_context("run1/s2"):
    send_invoice("C1", 10)
ok("a different request id is a different action", len(effects) == 2 and st.executions("send_invoice") == 2)
with request_context("run1/s1"):
    send_invoice("C2", 10)
ok("different arguments are a different action", len(effects) == 3)
send_invoice("C1", 10, request_id="run1/s1")
ok("explicit request_id=... works and matches the context id", len(effects) == 3)

print("needs a request id:")
try:
    send_invoice("C9", 1)
    ok("no request id -> MissingRequestId", False)
except MissingRequestId as e:
    ok("no request id -> MissingRequestId, nothing done", "needs a request id" in str(e) and len(effects) == 3)

print("normalizing:")
ok("strings stripped, 10.0 == 10, dict keys sorted, tuple == list",
   idem.action_key("t", {"a": " x ", "b": 10.0, "c": {"z": 1, "y": (1, 2)}}, "r")
   == idem.action_key("t", {"c": {"y": [1, 2], "z": 1}, "b": 10, "a": "x"}, "r"))
ok("10.5 != 10, True != 1 stays distinct", idem.action_key("t", {"a": 10.5}, "r") != idem.action_key("t", {"a": 10}, "r")
   and idem.action_key("t", {"a": True}, "r") != idem.action_key("t", {"a": 1}, "r"))
ok("tool name is part of the key", idem.action_key("t1", {}, "r") != idem.action_key("t2", {}, "r"))

print("a failure can be retried:")
st = fresh("fail")
calls = {"n": 0}


@idempotent(store=st)
def flaky(x: int) -> str:
    calls["n"] += 1
    if calls["n"] == 1:
        raise ConnectionError("payment API down")
    return f"done {x}"


with request_context("r"):
    try:
        flaky(1)
        ok("raises the first time", False)
    except ConnectionError:
        ok("the exception reaches the caller", True)
    row = st.get(idem.action_key("flaky", {"x": 1}, "r"))
    ok("recorded as failed, with the error", row["status"] == "failed" and "payment API down" in row["error"])
    ok("retry acts (the failed claim is released)", flaky(1) == "done 1" and calls["n"] == 2)
    ok("...and then it's done: a third call replays", flaky(1) == "done 1" and calls["n"] == 2)
ok("events: failed, executed, replayed", [e["event"] for e in st.events("flaky")] == ["failed", "executed", "replayed"])

print("concurrent callers: still once:")
st = fresh("race")
count = {"n": 0}
gate = threading.Event()


@idempotent(store=st)
def charge(card: str) -> str:
    count["n"] += 1
    time.sleep(0.2)          # a slow side effect: others arrive while it runs
    return "charged"


results = []


def worker():
    gate.wait()
    with request_context("order-77"):
        results.append(charge("4242"))


threads = [threading.Thread(target=worker) for _ in range(10)]
[t.start() for t in threads]
gate.set()
[t.join() for t in threads]
ok("10 simultaneous calls -> the side effect ran exactly once", count["n"] == 1 and st.executions("charge") == 1,
   count["n"])
ok("the others were told 'in progress' or got the stored result, none acted",
   sum(1 for r in results if r == "charged") >= 1 and all(r == "charged" or "already in progress" in r for r in results))

print("LangChain tools:")
st = fresh("lc")
lc_effects = []


@tool
def issue_credit(account: str, amount: float) -> str:
    """Give an account store credit."""
    lc_effects.append(account)
    return f"credited {account}"


safe = idempotent(issue_credit, store=st)
ok("same name, description, argument schema", (safe.name, safe.description, safe.args)
   == (issue_credit.name, issue_credit.description, issue_credit.args))
with request_context("chat-5"):
    a = safe.invoke({"account": "X", "amount": 5})
    b = safe.invoke({"account": "X", "amount": 5})
ok("invoke twice -> one effect, same result", a == b == "credited X" and lc_effects == ["X"])
ok("is_idempotent: wrapped yes, original no", idem.is_idempotent(safe) and not idem.is_idempotent(issue_credit))

print("persistence:")
st2 = ActionStore(st.path)
with request_context("chat-5"):
    c = idempotent(issue_credit, store=st2).invoke({"account": "X", "amount": 5})
ok("a NEW store object on the same file (e.g. after a restart) still replays", c == "credited X" and lc_effects == ["X"])

print("store location:")
os.environ["ACTIONS_DB"] = os.path.join(root, "env.db")


@idempotent
def ping(x: int) -> str:
    return "pong"


with request_context("e"):
    ping(1)
ok("$ACTIONS_DB picked at call time", os.path.exists(os.path.join(root, "env.db"))
   and idem.default_store().executions("ping") == 1)
del os.environ["ACTIONS_DB"]
ok("tests never touched a real actions.db", os.path.exists("actions.db") == real_db_existed)

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
