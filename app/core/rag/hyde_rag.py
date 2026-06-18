"""
HyDE (Hypothetical Document Embeddings) — from "Precise Zero-Shot Dense Retrieval
without Relevance Labels" (Gao et al., 2022).

Key idea:
  Instead of embedding the raw query, ask the LLM to generate a HYPOTHETICAL
  answer document first, then embed that document for retrieval.
  This narrows the semantic gap between query and document embeddings.
"""
from app.services.azure_openai import chat_completion, get_embedding
from app.services.vector_store import get_vector_store, SearchResult

HYPOTHETICAL_DOC_PROMPT = """Write a short, factual paragraph that directly answers the
following question. This will be used for document retrieval, so write it as if it were
an excerpt from a relevant reference document.

Question: {question}

Hypothetical answer document:"""


async def hyde_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    # Step 1: Generate hypothetical document
    msg = await chat_completion(
        [{"role": "user", "content": HYPOTHETICAL_DOC_PROMPT.format(question=query)}],
        temperature=0.7,
    )
    hypothetical_doc = msg.content.strip()

    # Step 2: Embed the hypothetical document (not the query)
    hyp_embedding = await get_embedding(hypothetical_doc)

    # Step 3: Retrieve using the hypothetical embedding
    store = get_vector_store()
    col = store._collection(collection)
    results_raw = col.query(query_embeddings=[hyp_embedding], n_results=top_k)

    results = []
    for i, doc_id in enumerate(results_raw["ids"][0]):
        results.append(SearchResult(
            id=doc_id,
            content=results_raw["documents"][0][i],
            metadata=results_raw["metadatas"][0][i],
            score=1 - results_raw["distances"][0][i],
        ))

    # Step 4: Generate final answer
    context = "\n\n".join(f"[Source {i+1}]: {r.content}" for i, r in enumerate(results))
    msg2 = await chat_completion([
        {"role": "system", "content": "Answer the question using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ])

    return {
        "answer": msg2.content,
        "hypothetical_document": hypothetical_doc,
        "sources": [{"id": r.id, "content": r.content, "score": r.score} for r in results],
    }
