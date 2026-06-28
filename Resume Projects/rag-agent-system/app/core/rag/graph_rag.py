"""
GraphRAG — builds a knowledge graph from ingested documents then uses graph
traversal alongside vector search to answer multi-hop questions.

Entities and relationships are extracted via LLM, stored in NetworkX (in-memory)
or optionally Neo4j. During retrieval, relevant entity sub-graphs are surfaced
together with the original chunks.
"""
from __future__ import annotations
import json
import networkx as nx
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store

# In-memory graph (swap for Neo4j in production)
_graph: nx.DiGraph = nx.DiGraph()

EXTRACT_PROMPT = """Extract entities and relationships from the following text.
Output a JSON object with two keys:
- "entities": list of {{"name": str, "type": str}}
- "relationships": list of {{"source": str, "relation": str, "target": str}}

Text: {text}
"""

GRAPH_QUERY_PROMPT = """Using the following knowledge graph context and document passages,
answer the question thoroughly.

Graph context (entities & relationships):
{graph_context}

Document passages:
{doc_context}

Question: {question}
"""


async def extract_graph(text: str) -> tuple[list[dict], list[dict]]:
    msg = await chat_completion(
        [{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:3000])}],
        temperature=0.0,
    )
    try:
        data = json.loads(msg.content)
        return data.get("entities", []), data.get("relationships", [])
    except Exception:
        return [], []


def add_to_graph(entities: list[dict], relationships: list[dict]) -> None:
    for e in entities:
        _graph.add_node(e["name"], type=e.get("type", "unknown"))
    for r in relationships:
        _graph.add_edge(r["source"], r["target"], relation=r["relation"])


def get_entity_context(entities: list[str], hops: int = 2) -> str:
    """Return sub-graph text for given seed entities up to N hops away."""
    lines = []
    for entity in entities:
        if entity not in _graph:
            continue
        neighbors = nx.single_source_shortest_path_length(_graph, entity, cutoff=hops)
        for neighbor, depth in neighbors.items():
            edges = _graph.get_edge_data(entity, neighbor) or {}
            rel = edges.get("relation", "related_to")
            lines.append(f"  {entity} --[{rel}]--> {neighbor} (depth={depth})")
    return "\n".join(lines) if lines else "No graph context found."


async def graph_rag(
    query: str,
    collection: str = "default",
    top_k: int = 5,
) -> dict:
    store = get_vector_store()

    # 1. Dense retrieval
    results = await store.similarity_search(query, collection, top_k)

    # 2. Extract entities from query (use first 3 words as seed heuristic + LLM)
    msg = await chat_completion([
        {"role": "user", "content": f"List the key named entities in this question as a JSON array of strings: {query}"}
    ], temperature=0.0)
    try:
        seed_entities: list[str] = json.loads(msg.content)
    except Exception:
        seed_entities = []

    # 3. Graph traversal context
    graph_ctx = get_entity_context(seed_entities)

    # 4. Document passage context
    doc_ctx = "\n\n".join(f"[Passage {i+1}]: {r.content}" for i, r in enumerate(results))

    # 5. Generate
    msg2 = await chat_completion([
        {"role": "user", "content": GRAPH_QUERY_PROMPT.format(
            graph_context=graph_ctx, doc_context=doc_ctx, question=query
        )}
    ])

    return {
        "answer": msg2.content,
        "graph_entities_used": seed_entities,
        "graph_nodes": _graph.number_of_nodes(),
        "graph_edges": _graph.number_of_edges(),
        "sources": [{"id": r.id, "content": r.content} for r in results],
    }
