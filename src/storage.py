import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DB_PATH = os.getenv("FAKESCOPE_DB_PATH", "./data/predictions.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
    finally:
        con.close()


def _column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols


def ensure_schema() -> None:
    with _conn() as con:
        # Base table
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_type TEXT,
                url TEXT,
                title TEXT,
                text TEXT,
                model_fake REAL,
                model_true REAL,
                google_score REAL,
                explanation TEXT
            )
            """
        )
        # Additive migrations: ensure columns exist
        if not _column_exists(con, "predictions", "google_score"):
            con.execute("ALTER TABLE predictions ADD COLUMN google_score REAL")
        if not _column_exists(con, "predictions", "explanation"):
            con.execute("ALTER TABLE predictions ADD COLUMN explanation TEXT")
        con.commit()


def init_db() -> None:
    ensure_schema()


def insert_prediction(
    *,
    input_type: str,
    url: Optional[str],
    title: Optional[str],
    text: str,
    model_fake: float,
    model_true: float,
    google_score: Optional[float],
    explanation: str,
) -> None:
    with _conn() as con:
        con.execute(
            """
            INSERT INTO predictions (input_type, url, title, text, model_fake, model_true, google_score, explanation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                input_type,
                url,
                title,
                text,
                float(model_fake),
                float(model_true),
                None if google_score is None else float(google_score),
                explanation,
            ),
        )
        con.commit()


def fetch_recent(limit: int = 100) -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute(
            """
            SELECT id, ts, input_type, url, title, text, model_fake, model_true, google_score
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows
