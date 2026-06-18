"""
Conversation memory — in-process store (swap for DB/file-backed in production).

Short-term : last N messages kept per session (sliding window).
Long-term   : summary of older messages, compressed by LLM.
"""
from __future__ import annotations
from collections import defaultdict, deque
from app.services.azure_openai import chat_completion

_sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))
_summaries: dict[str, str] = {}

SUMMARIZE_PROMPT = """Summarize the following conversation history into a brief paragraph
that captures the key context, decisions made, and unresolved questions.

Conversation:
{history}

Summary:
"""


# ── Short-term memory ─────────────────────────────────────────────────────────
def add_message(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})


def get_messages(session_id: str) -> list[dict]:
    return list(_sessions[session_id])


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    _summaries.pop(session_id, None)


# ── Long-term memory (summarization) ─────────────────────────────────────────
async def summarize_and_compress(session_id: str, keep_last: int = 6) -> str:
    """Summarize older messages; keep only the last `keep_last` messages in the window."""
    messages = list(_sessions[session_id])
    if len(messages) <= keep_last:
        return _summaries.get(session_id, "")

    to_summarize = messages[:-keep_last]
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_summarize)

    msg = await chat_completion([
        {"role": "user", "content": SUMMARIZE_PROMPT.format(history=history_text)}
    ], temperature=0.2)

    _summaries[session_id] = msg.content
    # Trim session to last keep_last messages
    _sessions[session_id] = deque(messages[-keep_last:], maxlen=30)
    return msg.content


def build_context_messages(session_id: str) -> list[dict]:
    """Build messages list including long-term summary as system note."""
    msgs = []
    summary = _summaries.get(session_id)
    if summary:
        msgs.append({"role": "system", "content": f"Previous conversation summary:\n{summary}"})
    msgs.extend(get_messages(session_id))
    return msgs
