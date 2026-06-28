import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.api.dependencies import get_tenant_id
from app.core.tenancy import is_tenant_collection, public_collection_name, scope_collection_name
from app.models.schemas import (
    CollectionSummary,
    DocumentChunk,
    DocumentIngestResponse,
    DocumentSummary,
)
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
    tenant_id: str = Depends(get_tenant_id),
):
    """Upload and chunk a document (PDF, DOCX, TXT) into the vector store."""
    import tempfile, os

    # Save upload to temp file
    suffix = "." + (file.filename or "file.txt").rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = await extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from document.")

    # Chunk
    doc_id = str(uuid.uuid4())
    base_metadata = {"document_id": doc_id, "source": file.filename or "upload"}
    if chunk_strategy == "recursive":
        chunks = recursive_split(text, chunk_size, chunk_overlap)
        docs = build_documents(chunks, file.filename or "upload", base_metadata)

    elif chunk_strategy == "semantic":
        chunks = semantic_split(text)
        docs = build_documents(chunks, file.filename or "upload", base_metadata)

    elif chunk_strategy == "parent_child":
        pc_chunks = parent_child_split(text, chunk_size * 4, chunk_size, chunk_overlap)
        docs = [
            {
                "id": str(uuid.uuid4()),
                "content": c["child_content"],
                "metadata": {
                    **base_metadata,
                    "parent_id": c["parent_id"],
                    "parent_content": c["parent_content"],
                },
            }
            for c in pc_chunks
        ]

    elif chunk_strategy == "sentence_window":
        sw_chunks = sentence_window_split(text)
        docs = [
            {
                "id": str(uuid.uuid4()),
                "content": c["sentence"],
                "metadata": {
                    **base_metadata,
                    "window": c["window"],
                    "sentence_index": c["sentence_index"],
                },
            }
            for c in sw_chunks
        ]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown chunk strategy: {chunk_strategy}")

    if not docs:
        raise HTTPException(status_code=422, detail="No chunks generated from document.")

    store = get_vector_store()
    scoped_collection = scope_collection_name(collection_name, tenant_id)
    await store.add_documents(docs, scoped_collection)

    return DocumentIngestResponse(
        document_id=doc_id,
        chunks_created=len(docs),
        collection_name=collection_name,
    )


@router.get("/collections", response_model=list[CollectionSummary], summary="List collections")
async def list_collections(tenant_id: str = Depends(get_tenant_id)):
    store = get_vector_store()
    collections = await store.list_collections()
    visible = []
    for item in collections:
        name = item["name"]
        if is_tenant_collection(name, tenant_id):
            visible.append({**item, "name": public_collection_name(name, tenant_id)})
    return visible


@router.get("/{collection_name}", response_model=list[DocumentSummary], summary="List documents")
async def list_documents(collection_name: str, tenant_id: str = Depends(get_tenant_id)):
    store = get_vector_store()
    return await store.list_documents(scope_collection_name(collection_name, tenant_id))


@router.get(
    "/{collection_name}/documents/{document_id}/chunks",
    response_model=list[DocumentChunk],
    summary="List document chunks",
)
async def list_document_chunks(
    collection_name: str,
    document_id: str,
    limit: int = 100,
    tenant_id: str = Depends(get_tenant_id),
):
    store = get_vector_store()
    return await store.get_document_chunks(scope_collection_name(collection_name, tenant_id), document_id, limit)


@router.delete("/{collection_name}/documents/{document_id}", summary="Delete a document")
async def delete_document(collection_name: str, document_id: str, tenant_id: str = Depends(get_tenant_id)):
    store = get_vector_store()
    deleted = await store.delete_document(scope_collection_name(collection_name, tenant_id), document_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id, "chunks_deleted": deleted}


@router.delete("/collection/{name}", summary="Delete a collection")
async def delete_collection(name: str, tenant_id: str = Depends(get_tenant_id)):
    store = get_vector_store()
    await store.delete_collection(scope_collection_name(name, tenant_id))
    return {"deleted": name}
