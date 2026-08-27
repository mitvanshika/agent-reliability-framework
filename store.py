

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "store.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            total REAL NOT NULL,
            days_since_purchase INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faqs (
            topic TEXT PRIMARY KEY,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def seed_db():
    conn = get_connection()
    orders = [
        ("A1001", "delivered", 89.99, 12),
        ("A1002", "delivered", 45.50, 45),
        ("A1003", "shipped", 210.00, 3),
        ("A1004", "delivered", 15.99, 60),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO orders (order_id, status, total, days_since_purchase) VALUES (?, ?, ?, ?)",
        orders,
    )
    faqs = [
        ("shipping", "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days."),
        ("return policy", "Items can be returned within 30 days of purchase for a full refund."),
        ("payment methods", "We accept Visa, Mastercard, and PayPal."),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO faqs (topic, answer) VALUES (?, ?)",
        faqs,
    )
    conn.commit()
    conn.close()


def get_order(order_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def search_faq(query: str):
    query_lower = query.lower()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM faqs").fetchall()
    conn.close()
    for row in rows:
        if row["topic"] in query_lower:
            return row["answer"]
    return None


if __name__ == "__main__":
    init_db()
    seed_db()
    print(f"Database initialized and seeded at {DB_PATH}")

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "store.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            total REAL NOT NULL,
            days_since_purchase INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faqs (
            topic TEXT PRIMARY KEY,
            answer TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def seed_db():
    conn = get_connection()
    orders = [
        ("A1001", "delivered", 89.99, 12),
        ("A1002", "delivered", 45.50, 45),
        ("A1003", "shipped", 210.00, 3),
        ("A1004", "delivered", 15.99, 60),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO orders (order_id, status, total, days_since_purchase) VALUES (?, ?, ?, ?)",
        orders,
    )
    faqs = [
        ("shipping", "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days."),
        ("return policy", "Items can be returned within 30 days of purchase for a full refund."),
        ("payment methods", "We accept Visa, Mastercard, and PayPal."),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO faqs (topic, answer) VALUES (?, ?)",
        faqs,
    )
    conn.commit()
    conn.close()


def get_order(order_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def search_faq(query: str):
    query_lower = query.lower()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM faqs").fetchall()
    conn.close()
    for row in rows:
        if row["topic"] in query_lower:
            return row["answer"]
    return None


if __name__ == "__main__":
    init_db()
    seed_db()
    print(f"Database initialized and seeded at {DB_PATH}")