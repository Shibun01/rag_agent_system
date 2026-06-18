"""
Document processor — handles PDF, DOCX, TXT ingestion
and delegates chunking to the core chunking strategies.
"""
from __future__ import annotations
import uuid
from pathlib import Path

from app.config.settings import get_settings

settings = get_settings()


async def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pymupdf  # fitz
        doc = pymupdf.open(file_path)
        return "\n".join(page.get_text() for page in doc)

    elif suffix in (".docx", ".doc"):
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)

    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8")

    else:
        # Fallback: unstructured
        from unstructured.partition.auto import partition
        elements = partition(filename=file_path)
        return "\n".join(str(e) for e in elements)


def build_documents(
    chunks: list[str],
    source_file: str,
    metadata: dict | None = None,
) -> list[dict]:
    base_meta = {"source": source_file, **(metadata or {})}
    return [
        {"id": str(uuid.uuid4()), "content": chunk, "metadata": {**base_meta, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
