"""
Self-RAG — from "Self-RAG: Learning to Retrieve, Generate, and Critique" (2023).

The model decides:
  1. Whether retrieval is needed at all (RETRIEVE token).
  2. Whether retrieved docs are relevant (ISREL token).
  3. Whether the generated response is supported by docs (ISSUP token).
  4. Whether the response is useful (ISUSE token).

Here we simulate these decisions via an LLM judge since we don't fine-tune a
Self-RAG model from scratch.
"""
from __future__ import annotations
import json
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store, SearchResult

RETRIEVE_DECISION_PROMPT = """Decide if the following question requires retrieving external documents.
Output JSON: {{"needs_retrieval": true/false, "reason": "..."}}

Question: {question}
"""

RELEVANCE_PROMPT = """Is the given document relevant to answering the question?
Output JSON: {{"relevant": true/false}}

Question: {question}
Document: {document}
"""

SUPPORT_PROMPT = """Is the generated response fully supported by the provided document?
Output JSON: {{"supported": "fully" | "partially" | "no", "reason": "..."}}

Document: {document}
Response: {response}
"""

GENERATE_PROMPT = """Answer the question based on the provided context.

Context: {context}
Question: {question}
"""


async def _needs_retrieval(question: str) -> bool:
    msg = await chat_completion(
        [{"role": "user", "content": RETRIEVE_DECISION_PROMPT.format(question=question)}],
        temperature=0.0,
    )
    try:
        return json.loads(msg.content).get("needs_retrieval", True)
    except Exception:
        return True


async def _is_relevant(question: str, document: str) -> bool:
    msg = await chat_completion(
        [{"role": "user", "content": RELEVANCE_PROMPT.format(question=question, document=document)}],
        temperature=0.0,
    )
    try:
        return json.loads(msg.content).get("relevant", True)
    except Exception:
        return True


async def _check_support(document: str, response: str) -> str:
    msg = await chat_completion(
        [{"role": "user", "content": SUPPORT_PROMPT.format(document=document, response=response)}],
        temperature=0.0,
    )
    try:
        return json.loads(msg.content).get("supported", "partially")
    except Exception:
        return "partially"


async def self_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    critiques = []

    # Token 1: RETRIEVE — does this query need retrieval?
    retrieve = await _needs_retrieval(query)
    critiques.append({"token": "RETRIEVE", "value": retrieve})

    if not retrieve:
        # Generate directly from LLM parametric knowledge
        msg = await chat_completion(
            [{"role": "user", "content": query}], temperature=0.3
        )
        return {"answer": msg.content, "critiques": critiques, "sources": []}

    # Token 2: ISREL — filter retrieved docs by relevance
    store = get_vector_store()
    candidates = await store.similarity_search(query, collection, top_k)
    relevant_docs = []
    for r in candidates:
        is_rel = await _is_relevant(query, r.content)
        critiques.append({"token": "ISREL", "doc_id": r.id, "value": is_rel})
        if is_rel:
            relevant_docs.append(r)

    if not relevant_docs:
        relevant_docs = candidates  # fallback: use all

    context = "\n\n".join(f"[Doc {i+1}]: {r.content}" for i, r in enumerate(relevant_docs))
    msg = await chat_completion(
        [{"role": "user", "content": GENERATE_PROMPT.format(context=context, question=query)}],
    )
    answer = msg.content

    # Token 3: ISSUP — check if answer is grounded
    support_checks = []
    for r in relevant_docs[:2]:  # check top-2 for efficiency
        sup = await _check_support(r.content, answer)
        support_checks.append(sup)
        critiques.append({"token": "ISSUP", "doc_id": r.id, "value": sup})

    overall_support = "fully" if all(s == "fully" for s in support_checks) else "partially"

    return {
        "answer": answer,
        "support": overall_support,
        "critiques": critiques,
        "sources": [{"id": r.id, "content": r.content} for r in relevant_docs],
    }
