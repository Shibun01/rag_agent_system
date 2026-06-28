"""
Chunking strategies:
  1. Recursive character splitter (default)
  2. Semantic chunking (sentence-level + embedding similarity)
  3. Parent-Child chunking (large parent + small child chunks)
  4. Sentence Window chunking (small chunk + surrounding context window)
"""
from __future__ import annotations
import re


# ── 1. Recursive Character Splitter ───────────────────────────────────────────
def recursive_split(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split by paragraph → sentence → word until chunks fit within chunk_size."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(text) <= chunk_size:
            return [text] if text.strip() else []
        sep = seps[0]
        splits = text.split(sep) if sep else list(text)
        chunks, current = [], ""
        for piece in splits:
            candidate = current + (sep if current else "") + piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(_split(current, seps[1:]))
                current = piece
        if current:
            chunks.extend(_split(current, seps[1:]))
        return chunks

    raw = _split(text, separators)

    # Apply overlap
    if chunk_overlap <= 0 or len(raw) < 2:
        return raw

    overlapped = [raw[0]]
    for chunk in raw[1:]:
        prev = overlapped[-1]
        overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
        overlapped.append(overlap_text + " " + chunk)
    return overlapped


# ── 2. Semantic Chunker ────────────────────────────────────────────────────────
def semantic_split(text: str, breakpoint_percentile: float = 0.85) -> list[str]:
    """
    Split at semantic breakpoints detected by cosine distance between
    consecutive sentence embeddings (requires sentence-transformers).
    Falls back to sentence-level split if model is unavailable.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 2:
        return sentences

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        distances = [
            1 - float(np.dot(embeddings[i], embeddings[i + 1]) /
                       (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-10))
            for i in range(len(embeddings) - 1)
        ]
        threshold = float(np.percentile(distances, breakpoint_percentile * 100))
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        chunks, prev = [], 0
        for bp in breakpoints + [len(sentences)]:
            chunks.append(" ".join(sentences[prev:bp]))
            prev = bp
        return [c for c in chunks if c.strip()]

    except ImportError:
        # Fallback: group every 5 sentences
        return [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]


# ── 3. Parent-Child Chunking ──────────────────────────────────────────────────
def parent_child_split(
    text: str,
    parent_size: int = 2048,
    child_size: int = 256,
    child_overlap: int = 32,
) -> list[dict]:
    """
    Returns list of {parent_id, child_chunk, parent_chunk}.
    Index child chunks; retrieve parent for richer context at generation time.
    """
    import uuid
    parents = recursive_split(text, parent_size, 0)
    result: list[dict] = []
    for parent in parents:
        parent_id = str(uuid.uuid4())
        children = recursive_split(parent, child_size, child_overlap)
        for child in children:
            result.append({
                "parent_id": parent_id,
                "parent_content": parent,
                "child_content": child,
            })
    return result


# ── 4. Sentence Window Chunking ───────────────────────────────────────────────
def sentence_window_split(
    text: str,
    window_size: int = 3,
) -> list[dict]:
    """
    Each chunk is a single sentence; metadata carries a ±window_size sentence window.
    Embed the sentence, retrieve by sentence, return the window for generation context.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result: list[dict] = []
    for i, sentence in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        window = " ".join(sentences[start:end])
        result.append({
            "sentence": sentence,
            "window": window,
            "sentence_index": i,
        })
    return [r for r in result if r["sentence"].strip()]
