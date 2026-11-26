"""
Tests for store.py: a fresh clone has no store.db (it's gitignored), so the first use must
create and seed it; an existing database is left alone; FAQ matching uses a fixed order.
    python tests/test_store.py
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]

import store  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


real_path = store.DB_PATH
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    try:
        print("fresh clone (no store.db):")
        store.DB_PATH = Path(tmp) / "fresh.db"
        ok("no database yet", not store.DB_PATH.exists())
        ok("first lookup works", store.get_order("A1003") == {"order_id": "A1003", "status": "shipped",
                                                             "total": 210.0, "days_since_purchase": 3})
        ok("...and created the file", store.DB_PATH.exists())
        conn = store._connect()
        ok("all 5 FAQs seeded", len(conn.execute("SELECT * FROM faqs").fetchall()) == 5)
        conn.close()   # Windows can't delete a temp folder holding an open database file

        print("an existing database is left alone:")
        store.DB_PATH = Path(tmp) / "existing.db"
        store.init_db()
        store.seed_db()
        conn = sqlite3.connect(store.DB_PATH)
        conn.execute("UPDATE orders SET status = 'lost' WHERE order_id = 'A1001'")
        conn.commit()
        conn.close()
        ok("an edited row is not reset", store.get_order("A1001")["status"] == "lost")

        print("an empty database (tables, no rows) gets seeded:")
        store.DB_PATH = Path(tmp) / "empty.db"
        store.init_db()
        ok("seeded on first use", store.search_faq("what payment methods?") == "We accept Visa, Mastercard, and PayPal.")

        print("FAQ order is fixed (first matching topic wins, in insertion order):")
        store.DB_PATH = Path(tmp) / "order.db"
        ans = store.search_faq("how long is shipping to a po box, and what about customs?")
        ok("'shipping' is inserted first, so it wins over 'po box' and 'customs'",
           ans is not None and ans.startswith("Standard shipping"), ans)
        ok("unknown topic -> None", store.search_faq("do you sell gift cards?") is None)
    finally:
        store.DB_PATH = real_path

print("\nPASS" if not failures else f"\nFAIL: {failures}")
sys.exit(1 if failures else 0)
