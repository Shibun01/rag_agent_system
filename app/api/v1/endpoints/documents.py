import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.models.schemas import DocumentIngestRequest, DocumentIngestResponse
from app.services.document_processor import extract_text, build_documents
from app.services.vector_store import get_vector_store
from app.core.chunking.strategies import (
    recursive_split, semantic_split, parent_child_split, sentence_window_split
)

router = APIRouter()


@router.post("/ingest", response_model=DocumentIngestResponse, summary="Ingest a document")
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
):
    """Upload and chunk a document (PDF, DOCX, TXT) into the vector store."""
    import tempfile, os

    # Save upload to temp file
    suffix = "." + (file.filename or "file.txt").rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        print(f"Collection name: {collection_name}")
        text = await extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from document.")

    # Chunk
    if chunk_strategy == "recursive":
        chunks = recursive_split(text, chunk_size, chunk_overlap)
        docs = build_documents(chunks, file.filename or "upload", {})

    elif chunk_strategy == "semantic":
        chunks = semantic_split(text)
        docs = build_documents(chunks, file.filename or "upload", {})

    elif chunk_strategy == "parent_child":
        pc_chunks = parent_child_split(text, chunk_size * 4, chunk_size, chunk_overlap)
        docs = [
            {
                "id": str(uuid.uuid4()),
                "content": c["child_content"],
                "metadata": {"source": file.filename, "parent_id": c["parent_id"], "parent_content": c["parent_content"]},
            }
            for c in pc_chunks
        ]

    elif chunk_strategy == "sentence_window":
        sw_chunks = sentence_window_split(text)
        docs = [
            {
                "id": str(uuid.uuid4()),
                "content": c["sentence"],
                "metadata": {"source": file.filename, "window": c["window"], "sentence_index": c["sentence_index"]},
            }
            for c in sw_chunks
        ]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown chunk strategy: {chunk_strategy}")

    if not docs:
        raise HTTPException(status_code=422, detail="No chunks generated from document.")

    store = get_vector_store()
    doc_id = str(uuid.uuid4())
    await store.add_documents(docs, collection_name)

    return DocumentIngestResponse(
        document_id=doc_id,
        chunks_created=len(docs),
        collection_name=collection_name,
    )


@router.delete("/collection/{name}", summary="Delete a collection")
async def delete_collection(name: str):
    store = get_vector_store()
    await store.delete_collection(name)
    return {"deleted": name}
