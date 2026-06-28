from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config.settings import get_settings

settings = get_settings()


def _db_path() -> Path:
    path = Path(settings.chroma_persist_dir).parent / "analytics.sqlite3"
    return path if path.is_absolute() else Path.cwd() / path


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tenant_id TEXT NOT NULL DEFAULT 'default',
            query TEXT NOT NULL,
            strategy TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            status TEXT NOT NULL,
            source_count INTEGER NOT NULL DEFAULT 0,
            answer_preview TEXT NOT NULL DEFAULT '',
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            tenant_id TEXT NOT NULL DEFAULT 'default',
            query_log_id INTEGER,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT NOT NULL DEFAULT '',
            query TEXT,
            strategy TEXT,
            collection_name TEXT NOT NULL DEFAULT 'default',
            FOREIGN KEY(query_log_id) REFERENCES query_logs(id) ON DELETE SET NULL
        );
        """
    )
    _ensure_column(conn, "query_logs", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
    _ensure_column(conn, "feedback", "tenant_id", "TEXT NOT NULL DEFAULT 'default'")
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


async def log_query(
    *,
    tenant_id: str = "default",
    query: str,
    strategy: str,
    collection_name: str,
    latency_ms: float,
    status: str,
    source_count: int = 0,
    answer: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    answer_preview = answer[:500] if answer else ""
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO query_logs (
                tenant_id, query, strategy, collection_name, latency_ms, status, source_count, answer_preview, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, query, strategy, collection_name, latency_ms, status, source_count, answer_preview, error),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM query_logs WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _row_to_dict(row)


async def list_query_logs(limit: int = 50, tenant_id: str = "default") -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM query_logs WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


async def create_feedback(
    *,
    tenant_id: str = "default",
    rating: str,
    query_log_id: int | None = None,
    comment: str = "",
    query: str | None = None,
    strategy: str | None = None,
    collection_name: str = "default",
) -> dict[str, Any]:
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback (tenant_id, query_log_id, rating, comment, query, strategy, collection_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (tenant_id, query_log_id, rating, comment, query, strategy, collection_name),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _row_to_dict(row)


async def list_feedback(limit: int = 50, tenant_id: str = "default") -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
            (tenant_id, limit),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]
