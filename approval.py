"""
Human approval for any action tool, plus the safe combination with
@idempotent. Works for plain functions and LangChain tools.

    from approval import safe_action, CliApprover

    refund = safe_action(issue_refund_tool, approver=CliApprover(),
                         describe=lambda a: f"Refund order {a['order_id']}")

safe_action(...) runs in this order, on purpose:
  1. already done for this request + arguments?  -> the stored result: no
     approval asked, no action (a retry never bothers the approver twice)
  2. ask the approver                             -> rejected: a clear "not
     approved, nothing was done" text; NOTHING is stored, so a later request
     may still be approved
  3. claim the key, act once, store the result    (idempotent.py's store)
Asking before claiming matters for LangGraph: a paused graph re-runs the tool
from the start when it resumes, and must not find its own claim.

requires_approval(...) is the approval step alone (it asks every time).

Approvers turn an ActionRequest into a Decision:
  DenyApprover      the default: rejects ("no approver configured")
  CliApprover       asks on the terminal; only y / yes approves
  CallbackApprover  any fn(request) -> bool | Decision (a chat bot, a policy check...)
  AutoApprover      approves everything, logged as "auto (tests only)"
An approver that raises or answers something else counts as a REJECT.

Every request and decision goes into approval_log (same SQLite file as the
action store), including replays that needed no approval.
"""

import json
import time
from dataclasses import dataclass, field

import idempotent as idem


@dataclass
class ActionRequest:
    tool: str
    args: dict
    request_id: str | None
    description: str
    key: str | None = None


@dataclass
class Decision:
    approved: bool
    by: str
    reason: str = ""


class DenyApprover:
    name = "deny"

    def decide(self, request):
        return Decision(False, "deny (default)", "no approver configured")


class AutoApprover:
    name = "auto"

    def decide(self, request):
        return Decision(True, "auto (tests only)", "")


class CallbackApprover:
    name = "callback"

    def __init__(self, fn, by="callback"):
        self.fn, self.by = fn, by

    def decide(self, request):
        return self.fn(request)


@dataclass
class CliApprover:
    input: object = field(default=input)
    output: object = field(default=print)
    by: str = "cli"
    name = "cli"

    def decide(self, request):
        self.output(f"\nAPPROVAL NEEDED  {request.tool}({json.dumps(request.args, default=str)})")
        if request.description:
            self.output(f"  {request.description}")
        answer = str(self.input("Approve? [y/N] ")).strip().lower()
        return Decision(answer in ("y", "yes"), self.by, f"typed {answer!r}")


APPROVERS = {"deny": DenyApprover, "auto": AutoApprover, "cli": CliApprover}


def get_approver(spec=None):
    if spec is None:
        return DenyApprover()
    if isinstance(spec, str):
        if spec not in APPROVERS:
            raise ValueError(f"unknown approver '{spec}'. Known: {', '.join(APPROVERS)}, or an object with decide()")
        return APPROVERS[spec]()
    if hasattr(spec, "decide"):
        return spec
    if callable(spec):
        return CallbackApprover(spec)
    raise ValueError(f"not an approver: {spec!r}")


def _decide(approver, request) -> Decision:
    """Fail closed: anything but a clear yes is a no."""
    try:
        d = approver.decide(request)
    except Exception as e:  # noqa: BLE001
        return Decision(False, getattr(approver, "name", "approver"), f"approver failed: {type(e).__name__}: {e}")
    if isinstance(d, Decision):
        return d
    if d is True or d is False:
        return Decision(d, getattr(approver, "by", getattr(approver, "name", "approver")))
    return Decision(False, getattr(approver, "name", "approver"), f"unclear answer {d!r}: treated as reject")


# ---------------------------------------------------------------- log

def _ensure_log(store):
    with store._conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS approval_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, tool TEXT, args TEXT, request_id TEXT,
            description TEXT, approved INTEGER, by_whom TEXT, reason TEXT, at REAL)""")


def log_decision(store, request, approved, by, reason):
    _ensure_log(store)
    with store._conn() as c:
        c.execute("INSERT INTO approval_log (key, tool, args, request_id, description, approved, by_whom, reason, at) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (request.key, request.tool, json.dumps(idem.normalize(request.args), default=str), request.request_id,
                   request.description, None if approved is None else int(approved), by, reason, time.time()))


def approval_log(store=None, tool=None):
    store = store or idem.default_store()
    _ensure_log(store)
    q, params = "SELECT * FROM approval_log", []
    if tool:
        q, params = q + " WHERE tool=?", [tool]
    with store._conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id", params)]


def rejected_text(tool, decision):
    return (f"Error: {tool} was not approved by {decision.by}"
            + (f" ({decision.reason})" if decision.reason else "") + ". Nothing was done.")


# ---------------------------------------------------------------- wrappers

def _wrap_tool(target, make):
    if hasattr(target, "func") and hasattr(target, "name") and hasattr(target, "model_copy"):   # LangChain tool
        return target.model_copy(update={"func": make(target.func, target.name)})
    return make(target, getattr(target, "__name__", "action"))


def _description(describe, args):
    if describe is None:
        return ""
    try:
        return str(describe(args))
    except Exception as e:  # noqa: BLE001
        return f"(describe failed: {e})"


def requires_approval(target=None, *, approver=None, describe=None, store=None):
    """Ask before every call (no idempotency). Decorator or wrapper."""
    def make(fn, name):
        def wrapped(*args, request_id=None, **kwargs):
            st = store or idem.default_store()
            call_args = idem._bind(fn, args, kwargs)
            rid = request_id if request_id is not None else idem.current_request_id()
            req = ActionRequest(name, call_args, rid, _description(describe, call_args))
            d = _decide(get_approver(approver), req)
            log_decision(st, req, d.approved, d.by, d.reason)
            if not d.approved:
                return rejected_text(name, d)
            return fn(*args, **kwargs)

        wrapped.__wrapped__, wrapped.__requires_approval__, wrapped.__name__ = fn, True, getattr(fn, "__name__", name)
        return wrapped

    return (lambda t: _wrap_tool(t, make)) if target is None else _wrap_tool(target, make)


def safe_action(target=None, *, approver=None, describe=None, store=None):
    """Approval + idempotency in the right order (see the module docstring). Decorator or wrapper."""
    def make(fn, name):
        def wrapped(*args, request_id=None, **kwargs):
            rid = request_id if request_id is not None else idem.current_request_id()
            if rid is None:
                raise idem.MissingRequestId(f"{name} is a safe action and needs a request id: call it inside "
                                            f"request_context(...) or pass request_id=...")
            st = store or idem.default_store()
            call_args = idem._bind(fn, args, kwargs)
            key = idem.action_key(name, call_args, rid)
            req = ActionRequest(name, call_args, rid, _description(describe, call_args), key)

            row = st.get(key)                                   # 1. already done?
            if row and row["status"] == "done":
                st.event(key, name, rid, "replayed")
                log_decision(st, req, None, "-", "not needed: already done for this request")
                return json.loads(row["result"])

            d = _decide(get_approver(approver), req)           # 2. ask (nothing claimed yet)
            log_decision(st, req, d.approved, d.by, d.reason)
            if not d.approved:
                st.event(key, name, rid, "rejected", f"{d.by}: {d.reason}")
                return rejected_text(name, d)

            if not st.claim(key, name, call_args, rid):         # 3. claim, act once, store
                st.event(key, name, rid, "refused_in_progress")
                return f"Error: {name} is already in progress for this request; it was not repeated."
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                st.fail(key, f"{type(e).__name__}: {e}")
                st.event(key, name, rid, "failed", f"{type(e).__name__}: {e}")
                raise
            st.event(key, name, rid, "executed")
            st.finish(key, result)
            return result

        wrapped.__wrapped__, wrapped.__name__ = fn, getattr(fn, "__name__", name)
        wrapped.__safe_action__ = wrapped.__requires_approval__ = wrapped.__idempotent__ = True
        return wrapped

    return (lambda t: _wrap_tool(t, make)) if target is None else _wrap_tool(target, make)


def is_safe_action(t) -> bool:
    fn = getattr(t, "func", t)
    while fn is not None:
        if getattr(fn, "__safe_action__", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


__all__ = ["ActionRequest", "Decision", "DenyApprover", "AutoApprover", "CallbackApprover", "CliApprover",
           "get_approver", "requires_approval", "safe_action", "is_safe_action", "approval_log"]
