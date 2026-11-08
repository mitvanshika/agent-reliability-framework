"""
@idempotent: a side-effect tool acts at most once per request, however often
it is called (agent retries, a re-run graph, a double click, two workers).
Works for plain Python functions and LangChain tools; nothing here knows what
the tool does.

    from idempotent import idempotent, request_context

    @idempotent
    def send_invoice(customer_id: str, amount: float) -> str:
        ...                                  # the real side effect

    with request_context("run_42/scenario_7"):
        send_invoice("C1", 10)               # acts
        send_invoice(" C1 ", 10.0)           # same normalized args + request -> stored result, no second invoice

    safe_tool = idempotent(langchain_tool)   # a LangChain tool: same name / description / schema

Key = sha256(tool name + normalized arguments + request id). Normalizing:
strings are stripped, 10.0 == 10, dict keys sorted, tuples == lists. The
request id comes from request_id=..., else request_context(), else (inside a
LangGraph run) the run's config: configurable.request_id, or its thread_id, so a
paused and resumed graph keeps the same key. The harness sets one per scenario. Without one the call is refused (raises
MissingRequestId): "once ever" is almost never what you want, and silently
guessing is worse.

Storage: SQLite (default actions.db, or $ACTIONS_DB, or store=...). A call
first INSERTs its key (status in_progress); only the caller whose insert
succeeded acts, so two callers at the same time cannot both act. Then:
  done         -> later calls get the stored result, no new action
  in_progress  -> another caller is acting: "already in progress", no action
  failed       -> the function raised; the key may be claimed again (retry)
Every outcome is also appended to action_events, which is what
check_actions.py counts ("executed" = the side effect really ran).

Results should be JSON-serializable (tool results are usually strings); other
values are stored as their str().
"""

import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import os
import sqlite3
import threading
import time

DEFAULT_DB = "actions.db"
_REQUEST_ID = contextvars.ContextVar("idempotent_request_id", default=None)
_default_store = None
_lock = threading.Lock()


class MissingRequestId(RuntimeError):
    """An idempotent action was called with no request id."""


@contextlib.contextmanager
def request_context(request_id):
    """Every idempotent call inside uses this request id (unless given request_id=...)."""
    token = _REQUEST_ID.set(str(request_id))
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)


def current_request_id():
    """request_context(), else the LangGraph run's configurable.request_id / thread_id, else None."""
    rid = _REQUEST_ID.get()
    if rid is not None:
        return rid
    try:
        from langgraph.config import get_config
        conf = get_config().get("configurable") or {}
    except Exception:          # not inside a LangGraph run (or no LangGraph)
        return None
    rid = conf.get("request_id") or conf.get("thread_id")
    return str(rid) if rid is not None else None


def normalize(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def action_key(tool, args, request_id):
    blob = json.dumps({"tool": tool, "args": normalize(args), "request_id": str(request_id)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ActionStore:
    def __init__(self, path=None):
        self.path = path or os.environ.get("ACTIONS_DB") or DEFAULT_DB
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS actions (
                key TEXT PRIMARY KEY, tool TEXT, args TEXT, request_id TEXT,
                status TEXT, result TEXT, error TEXT, created REAL, updated REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS action_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, tool TEXT, request_id TEXT,
                event TEXT, detail TEXT, at REAL)""")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)   # autocommit: each statement is atomic
        c.row_factory = sqlite3.Row
        return contextlib.closing(c)

    def claim(self, key, tool, args, request_id) -> bool:
        """True if THIS caller may act (it inserted the key, or re-claimed a failed one)."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute("INSERT OR IGNORE INTO actions VALUES (?, ?, ?, ?, 'in_progress', NULL, NULL, ?, ?)",
                            (key, tool, json.dumps(normalize(args), default=str), request_id, now, now))
            if cur.rowcount == 1:
                return True
            cur = c.execute("UPDATE actions SET status='in_progress', error=NULL, updated=? "
                            "WHERE key=? AND status='failed'", (now, key))
            return cur.rowcount == 1

    def get(self, key):
        with self._conn() as c:
            row = c.execute("SELECT * FROM actions WHERE key=?", (key,)).fetchone()
        return dict(row) if row else None

    def finish(self, key, result):
        try:
            text = json.dumps(result)
        except (TypeError, ValueError):
            text = json.dumps(str(result))
        with self._conn() as c:
            c.execute("UPDATE actions SET status='done', result=?, updated=? WHERE key=?", (text, time.time(), key))

    def fail(self, key, error):
        with self._conn() as c:
            c.execute("UPDATE actions SET status='failed', error=?, updated=? WHERE key=?",
                      (str(error), time.time(), key))

    def event(self, key, tool, request_id, event, detail=""):
        with self._conn() as c:
            c.execute("INSERT INTO action_events (key, tool, request_id, event, detail, at) VALUES (?, ?, ?, ?, ?, ?)",
                      (key, tool, request_id, event, str(detail)[:500], time.time()))

    def events(self, tool=None, event=None):
        q, params = "SELECT * FROM action_events WHERE 1=1", []
        if tool:
            q, params = q + " AND tool=?", params + [tool]
        if event:
            q, params = q + " AND event=?", params + [event]
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id", params)]

    def executions(self, tool=None):
        """How many times the side effect really ran."""
        return len(self.events(tool, "executed"))


def default_store():
    global _default_store
    with _lock:
        want = os.environ.get("ACTIONS_DB") or DEFAULT_DB
        if _default_store is None or _default_store.path != want:
            _default_store = ActionStore(want)
        return _default_store


def _bind(fn, args, kwargs):
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}


def idempotent_function(fn, store=None, name=None):
    name = name or getattr(fn, "__name__", "action")

    @functools.wraps(fn)
    def wrapped(*args, request_id=None, **kwargs):
        rid = request_id if request_id is not None else current_request_id()
        if rid is None:
            raise MissingRequestId(f"{name} is idempotent and needs a request id: call it inside "
                                   f"request_context(...) or pass request_id=...")
        st = store if store is not None else default_store()
        call_args = _bind(fn, args, kwargs)
        key = action_key(name, call_args, rid)
        row = st.get(key)
        if row and row["status"] == "done":
            st.event(key, name, rid, "replayed")
            return json.loads(row["result"])
        if not st.claim(key, name, call_args, rid):
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

    wrapped.__wrapped__ = fn
    wrapped.__idempotent__ = True
    wrapped.idempotency_key = lambda *a, request_id=None, **kw: action_key(
        name, _bind(fn, a, kw), request_id if request_id is not None else current_request_id())
    return wrapped


def idempotent(target=None, *, store=None, name=None):
    """Decorator / wrapper for a plain function or a LangChain tool (also @idempotent(store=...))."""
    def apply(t):
        if hasattr(t, "func") and hasattr(t, "name") and hasattr(t, "model_copy"):   # LangChain tool
            return t.model_copy(update={"func": idempotent_function(t.func, store, name or t.name)})
        return idempotent_function(t, store, name)

    return apply(target) if target is not None else apply


def is_idempotent(t) -> bool:
    fn = getattr(t, "func", t)
    while fn is not None:
        if getattr(fn, "__idempotent__", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False
