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
  DashboardApprover puts the request on the dashboard's /approvals page and waits;
                    no decision within timeout_s = reject ("expired")
  LangGraphApprover pauses a LangGraph run (interrupt) until run_with_approvals()
                    or your own code resumes it with a decision
An approver that raises or answers something else counts as a REJECT
(LangGraph's pause signal is the one exception that is let through).

Every request and decision goes into approval_log (same SQLite file as the
action store), including replays that needed no approval.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field

import idempotent as idem


@dataclass
class ActionRequest:
    tool: str
    args: dict
    request_id: str | None
    description: str
    key: str | None = None
    store: object = field(default=None, repr=False, compare=False)   # the action store (for LangGraph mode)


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


def parse_answer(answer, by="approver"):
    """A resume value / callback answer -> Decision. Anything unclear is a reject."""
    if isinstance(answer, Decision):
        return answer
    if answer is True or answer is False:
        return Decision(answer, by)
    if isinstance(answer, dict) and isinstance(answer.get("approved"), bool):
        return Decision(answer["approved"], str(answer.get("by") or by), str(answer.get("reason") or ""))
    return Decision(False, by, f"unclear answer {answer!r}: treated as reject")


class LangGraphApprover:
    """Inside a LangGraph tool: pause the graph (interrupt) and wait for the decision.
    The graph must be compiled with a checkpointer and run with a thread_id.

    Resume with {"approved": bool, "by": ..., "reason": ..., "key": <the pause payload's key>};
    run_with_approvals() does this. A resume value without "key" is REFUSED (rejected).

    Why the key and the decision memory: LangGraph hands resume values to interrupt() calls
    by POSITION within the tools step (interrupt() has no id, and every interrupt of one step
    shares one interrupt id). Positions shift when an action that is already done replays
    without interrupting, and parallel tool calls race for them. So whoever reads a resume
    value files it under its key (table `decisions`, scoped to this tools step), and every
    action looks in that memory before interrupting. A decision is never applied to the
    wrong action, never lost and never asked twice."""
    name = "langgraph"
    max_skips = 50
    _locks = {}
    _locks_guard = threading.Lock()

    def _lock(self, scope):
        with self._locks_guard:
            return self._locks.setdefault(scope, threading.Lock())

    def decide(self, request):
        try:
            from langgraph.config import get_config
            from langgraph.types import interrupt
        except ImportError:
            return Decision(False, self.name, "LangGraph is not installed")
        try:
            conf = get_config().get("configurable") or {}
        except RuntimeError as e:          # called outside a graph run: no checkpoint to pause at
            return Decision(False, self.name, f"needs a LangGraph run with a checkpointer ({e})")
        scope = f"{conf.get('thread_id')}|{conf.get('checkpoint_ns')}"
        store = request.store or idem.default_store()
        payload = {"type": "approval", "tool": request.tool, "args": request.args,
                   "request_id": request.request_id, "description": request.description, "key": request.key}
        with self._lock(scope):            # parallel tool calls take turns reading resume values
            known = _recall(store, scope, request.key)
            if known:
                return known
            for _ in range(self.max_skips):
                answer = interrupt(payload)    # pauses the graph if no resume value is left
                if not (isinstance(answer, dict) and answer.get("key")):
                    return Decision(False, self.name, f"resume value {answer!r} has no 'key': refused, since it can't "
                                                      "be matched to an action (use run_with_approvals, or resume "
                                                      "with the pause payload's key)")
                d = parse_answer(answer, "langgraph resume")
                _remember(store, scope, answer["key"], d)
                if answer["key"] == request.key:
                    return d
            return Decision(False, self.name, "no decision for this action arrived")


def _decisions_table(store):
    with store._conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS decisions (scope TEXT, key TEXT, approved INTEGER, by_whom TEXT,
                     reason TEXT, at REAL, PRIMARY KEY (scope, key))""")


def _remember(store, scope, key, d):
    _decisions_table(store)
    with store._conn() as c:
        c.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                  (scope, key, int(d.approved), d.by, d.reason, time.time()))


def _recall(store, scope, key):
    _decisions_table(store)
    with store._conn() as c:
        row = c.execute("SELECT * FROM decisions WHERE scope=? AND key=?", (scope, key)).fetchone()
    return Decision(bool(row["approved"]), row["by_whom"], row["reason"]) if row else None


# ---------------------------------------------------------------- dashboard queue

def _queue_table(store):
    with store._conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS approval_queue (
            id TEXT PRIMARY KEY, key TEXT, tool TEXT, args TEXT, request_id TEXT, description TEXT,
            status TEXT, by_whom TEXT, reason TEXT, created REAL, decided REAL)""")


def enqueue(store, request) -> str:
    _queue_table(store)
    qid = uuid.uuid4().hex
    with store._conn() as c:
        c.execute("INSERT INTO approval_queue VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, NULL)",
                  (qid, request.key, request.tool, json.dumps(idem.normalize(request.args), default=str),
                   request.request_id, request.description, time.time()))
    return qid


def queue_item(store, qid):
    _queue_table(store)
    with store._conn() as c:
        row = c.execute("SELECT * FROM approval_queue WHERE id=?", (qid,)).fetchone()
    return dict(row) if row else None


def queue(store=None, status=None, limit=100):
    store = store or idem.default_store()
    _queue_table(store)
    q, params = "SELECT * FROM approval_queue", []
    if status:
        q, params = q + " WHERE status=?", [status]
    with store._conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY created DESC LIMIT ?", params + [limit])]


def decide_pending(store, qid, approved, by, reason="") -> bool:
    """Record a decision for a pending request. False if it was not pending any more
    (already decided, or expired): the first decision wins."""
    _queue_table(store)
    with store._conn() as c:
        cur = c.execute("UPDATE approval_queue SET status=?, by_whom=?, reason=?, decided=? "
                        "WHERE id=? AND status='pending'",
                        ("approved" if approved else "rejected", by, reason, time.time(), qid))
        return cur.rowcount == 1


class DashboardApprover:
    """Shows the request on the dashboard (/approvals) and waits for Approve / Reject.
    No decision within timeout_s -> reject (status 'expired'), so an unattended run
    never acts by default."""
    name = "dashboard"

    def __init__(self, store=None, timeout_s=300.0, poll_s=0.5, sleep=time.sleep, clock=time.monotonic):
        self.store, self.timeout_s, self.poll_s, self._sleep, self._clock = store, timeout_s, poll_s, sleep, clock

    def decide(self, request):
        st = self.store or request.store or idem.default_store()
        qid = enqueue(st, request)
        deadline = self._clock() + self.timeout_s
        while True:
            item = queue_item(st, qid)
            if item["status"] in ("approved", "rejected"):
                return Decision(item["status"] == "approved", f"dashboard: {item['by_whom']}", item["reason"] or "")
            if self._clock() >= deadline:
                if decide_pending(st, qid, False, "timeout", f"no decision within {self.timeout_s:g}s"):
                    with st._conn() as c:
                        c.execute("UPDATE approval_queue SET status='expired' WHERE id=?", (qid,))
                    return Decision(False, "dashboard", f"no decision within {self.timeout_s:g}s (expired)")
                continue          # someone decided in the last instant: read it
            self._sleep(self.poll_s)


APPROVERS = {"deny": DenyApprover, "auto": AutoApprover, "cli": CliApprover, "langgraph": LangGraphApprover,
             "dashboard": DashboardApprover}


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
        if _is_graph_pause(e):
            raise                       # LangGraph pausing for approval: not a failure
        return Decision(False, getattr(approver, "name", "approver"), f"approver failed: {type(e).__name__}: {e}")
    if isinstance(d, Decision):
        return d
    if d is True or d is False:
        return Decision(d, getattr(approver, "by", getattr(approver, "name", "approver")))
    return Decision(False, getattr(approver, "name", "approver"), f"unclear answer {d!r}: treated as reject")


def _is_graph_pause(e):
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:
        return False
    return isinstance(e, GraphBubbleUp)


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
            req = ActionRequest(name, call_args, rid, _description(describe, call_args), key, st)

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


def run_with_approvals(app, graph_input, approver="cli", config=None, max_rounds=100):
    """Run a LangGraph app whose action tools use approver="langgraph". Each time the graph
    pauses for an approval, ask `approver` (cli / callback / dashboard / deny...; never
    "langgraph" itself) and resume with the decision, until the graph finishes.
    The app must have a checkpointer; config must carry a thread_id."""
    from langgraph.types import Command
    if not ((config or {}).get("configurable") or {}).get("thread_id"):
        raise ValueError("run_with_approvals needs config={'configurable': {'thread_id': ...}} "
                         "(the checkpoint the paused graph is resumed from)")
    outside = get_approver(approver)
    if isinstance(outside, LangGraphApprover):
        raise ValueError("the outside approver decides; 'langgraph' is the in-graph side")
    result = app.invoke(graph_input, config)
    for _ in range(max_rounds):
        pending = result.get("__interrupt__") if isinstance(result, dict) else None
        if not pending:
            return result
        resume = {}
        for it in pending:
            v = it.value
            if not (isinstance(v, dict) and v.get("type") == "approval"):
                raise ValueError(f"the graph paused for something that is not an approval: {v!r}")
            req = ActionRequest(v["tool"], v["args"], v.get("request_id"), v.get("description") or "", v.get("key"))
            d = _decide(outside, req)
            resume[it.id] = {"approved": d.approved, "by": d.by, "reason": d.reason, "key": req.key}
        result = app.invoke(Command(resume=resume), config)
    raise RuntimeError(f"still paused after {max_rounds} approval rounds")


def is_safe_action(t) -> bool:
    fn = getattr(t, "func", t)
    while fn is not None:
        if getattr(fn, "__safe_action__", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


__all__ = ["ActionRequest", "Decision", "DenyApprover", "AutoApprover", "CallbackApprover", "CliApprover",
           "LangGraphApprover", "DashboardApprover", "queue", "decide_pending", "get_approver", "parse_answer", "requires_approval", "safe_action",
           "run_with_approvals", "is_safe_action", "approval_log"]
