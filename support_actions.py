"""
Sample-agent only: the support bot's one ACTION tool, issue_refund.

It is only given to the agent with the experiment setting `actions: enabled`
(default off, so the baseline is unchanged), and always wrapped with
approval.safe_action: it asks an approver first (default: deny) and refunds an
order at most once per request.

The refund is written to a ledger in its own SQLite file (refunds.db, or
$SUPPORT_REFUNDS_DB), never to store.db, so the demo data stays untouched.
sandbox() points both the ledger and the action store at a temporary folder;
check_actions.py calls it (targets.yaml `actions_sandbox:`) so its checks never
refund anything real.
"""

import contextlib
import os
import sqlite3
import time

from langchain_core.tools import tool

from store import get_order

REFUND_WINDOW_DAYS = 30


def ledger_path():
    return os.environ.get("SUPPORT_REFUNDS_DB") or "refunds.db"


def _conn():
    c = sqlite3.connect(ledger_path(), timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS refunds (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, "
              "amount REAL, at REAL)")
    return contextlib.closing(c)


def ledger():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM refunds ORDER BY id")]


def _issue_refund(order_id: str) -> str:
    order = get_order(order_id.strip())
    if order is None:
        return f"No order found with ID {order_id}: no refund issued."
    if order["days_since_purchase"] > REFUND_WINDOW_DAYS:
        return (f"Order {order_id} is outside the {REFUND_WINDOW_DAYS}-day refund window "
                f"({order['days_since_purchase']} days): no refund issued.")
    with _conn() as c:
        cur = c.execute("INSERT INTO refunds (order_id, amount, at) VALUES (?, ?, ?)",
                        (order_id.strip(), order["total"], time.time()))
    return f"Refund of ${order['total']} issued for order {order_id.strip()} (refund #{cur.lastrowid})."


@tool
def issue_refund(order_id: str) -> str:
    """Issues a full refund for an order (only within 30 days of purchase). This ACTS: it needs a human's
    approval, and an order is refunded at most once per conversation. Use it only when the customer asks for
    the refund and the order is eligible."""
    return _issue_refund(order_id)


def describe(args):
    """One line for the approver."""
    oid = str(args.get("order_id", "")).strip()
    order = get_order(oid)
    if order is None:
        return f"Refund order {oid} (order not found: it will refuse)"
    return f"Refund order {oid}: ${order['total']}, {order['days_since_purchase']} days since purchase"


def guarded(approver="deny", store=None):
    """issue_refund wrapped with approval + idempotency."""
    import approval
    return approval.safe_action(issue_refund, approver=approver, describe=describe, store=store)


def sandbox(tmpdir):
    """Point the ledger and the action store at tmpdir (for check_actions.py); returns a restore function."""
    old = {k: os.environ.get(k) for k in ("SUPPORT_REFUNDS_DB", "ACTIONS_DB")}
    os.environ["SUPPORT_REFUNDS_DB"] = os.path.join(tmpdir, "refunds.db")
    os.environ["ACTIONS_DB"] = os.path.join(tmpdir, "actions.db")

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore
