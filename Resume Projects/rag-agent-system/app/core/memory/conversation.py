"""
Conversation memory backed by SQLite.

Short-term : last N messages kept per session.
Long-term  : summary of older messages, compressed by LLM.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config.settings import get_settings
from app.services.azure_openai import chat_completion

settings = get_settings()

SUMMARIZE_PROMPT = """Summarize the following conversation history into a brief paragraph
that captures the key context, decisions made, and unresolved questions.

Conversation:
{history}

Summary:
"""


def _db_path() -> Path:
    path = Path(settings.conversation_db_path)
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


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
        ON conversation_messages (tenant_id, session_id, id);

        CREATE TABLE IF NOT EXISTS conversation_summaries (
            tenant_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (tenant_id, session_id)
        );
        """
    )
    conn.commit()


def _get_summary(conn: sqlite3.Connection, tenant_id: str, session_id: str) -> str:
    row = conn.execute(
        "SELECT summary FROM conversation_summaries WHERE tenant_id = ? AND session_id = ?",
        (tenant_id, session_id),
    ).fetchone()
    return row["summary"] if row else ""


def _upsert_summary(conn: sqlite3.Connection, tenant_id: str, session_id: str, summary: str) -> None:
    conn.execute(
        """
        INSERT INTO conversation_summaries (tenant_id, session_id, summary, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(tenant_id, session_id)
        DO UPDATE SET summary = excluded.summary, updated_at = datetime('now')
        """,
        (tenant_id, session_id, summary),
    )


def _list_message_rows(conn: sqlite3.Connection, tenant_id: str, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, role, content
        FROM conversation_messages
        WHERE tenant_id = ? AND session_id = ?
        ORDER BY id ASC
        """,
        (tenant_id, session_id),
    ).fetchall()


# ── Short-term memory ─────────────────────────────────────────────────────────
def add_message(session_id: str, role: str, content: str, tenant_id: str = "default") -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages (tenant_id, session_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (tenant_id, session_id, role, content),
        )
        conn.commit()


def get_messages(session_id: str, tenant_id: str = "default", limit: int = 30) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM conversation_messages
            WHERE tenant_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (tenant_id, session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_session(session_id: str, tenant_id: str = "default") -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM conversation_messages WHERE tenant_id = ? AND session_id = ?",
            (tenant_id, session_id),
        )
        conn.execute(
            "DELETE FROM conversation_summaries WHERE tenant_id = ? AND session_id = ?",
            (tenant_id, session_id),
        )
        conn.commit()


# ── Long-term memory (summarization) ─────────────────────────────────────────
async def summarize_and_compress(session_id: str, keep_last: int = 6, tenant_id: str = "default") -> str:
    """Summarize older messages; keep only the last `keep_last` messages in the window."""
    with _connect() as conn:
        summary = _get_summary(conn, tenant_id, session_id)
        message_rows = _list_message_rows(conn, tenant_id, session_id)

    if len(message_rows) <= keep_last:
        return summary

    archived_rows = message_rows[:-keep_last]
    history_parts = []
    if summary:
        history_parts.append(f"Existing summary:\n{summary}")
    history_parts.append(
        "Recent archived messages:\n" + "\n".join(
            f"{row['role'].upper()}: {row['content']}" for row in archived_rows
        )
    )
    history_text = "\n\n".join(history_parts)

    msg = await chat_completion([
        {"role": "user", "content": SUMMARIZE_PROMPT.format(history=history_text)}
    ], temperature=0.2)

    summary = msg.content
    keep_ids = [row["id"] for row in message_rows[-keep_last:]]

    with _connect() as conn:
        _upsert_summary(conn, tenant_id, session_id, summary)
        if keep_ids:
            placeholders = ", ".join("?" for _ in keep_ids)
            conn.execute(
                f"DELETE FROM conversation_messages WHERE tenant_id = ? AND session_id = ? AND id NOT IN ({placeholders})",
                (tenant_id, session_id, *keep_ids),
            )
        else:
            conn.execute(
                "DELETE FROM conversation_messages WHERE tenant_id = ? AND session_id = ?",
                (tenant_id, session_id),
            )
        conn.commit()

    return summary


def build_context_messages(session_id: str, tenant_id: str = "default") -> list[dict]:
    """Build messages list including long-term summary as system note."""
    msgs = []
    with _connect() as conn:
        summary = _get_summary(conn, tenant_id, session_id)
    if summary:
        msgs.append({"role": "system", "content": f"Previous conversation summary:\n{summary}"})
    msgs.extend(get_messages(session_id, tenant_id=tenant_id))
    return msgs
