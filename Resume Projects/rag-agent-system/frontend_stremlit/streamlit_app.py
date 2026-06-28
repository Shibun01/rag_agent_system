from __future__ import annotations

import html
import time
import uuid
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="RAG Agent Studio",
    page_icon="RA",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000/api/v1"
STRATEGIES = ["naive", "advanced", "corrective", "self_rag", "hyde", "multi_query", "graph_rag"]
AGENT_TYPES = ["react", "plan_execute", "reflection", "supervisor", "rag_agent", "multi_agent"]
CHUNK_STRATEGIES = ["recursive", "semantic", "parent_child", "sentence_window"]

st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --ink: #18202f;
        --muted: #657084;
        --line: #d8dee8;
        --accent: #0f766e;
        --accent-2: #b45309;
        --panel: #f7f9fc;
    }
    .stApp {
        background: linear-gradient(180deg, #fbfcfe 0%, #eef3f8 100%);
        color: var(--ink);
    }
    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    h1, h2, h3 {
        letter-spacing: 0 !important;
    }
    h1 {
        font-size: 2.1rem !important;
        font-weight: 760 !important;
        margin-bottom: .25rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: .75rem !important;
    }
    h3 {
        font-size: 1.02rem !important;
    }
    .studio-header {
        border: 1px solid var(--line);
        background: rgba(255,255,255,.9);
        padding: 1.15rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .studio-kicker {
        color: var(--accent);
        font-weight: 700;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .studio-subtitle {
        color: var(--muted);
        font-size: .98rem;
        max-width: 900px;
        margin-top: .35rem;
    }
    .metric-row [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .75rem .9rem;
    }
    .source-box {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .85rem .95rem;
        margin-bottom: .6rem;
    }
    .source-meta {
        color: var(--muted);
        font-size: .78rem;
        margin-bottom: .35rem;
    }
    .answer-box {
        background: #ffffff;
        border-left: 4px solid var(--accent);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        min-height: 160px;
    }
    .status-chip {
        display: inline-block;
        padding: .18rem .48rem;
        border-radius: 999px;
        background: #e7f4f2;
        color: #115e59;
        font-size: .75rem;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button p {
        font-size: .95rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_base() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{api_base()}{path}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            if response.content:
                return response.json()
            return None
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {url}: {exc}")
    return None


def stream_chat(payload: dict[str, Any]) -> str | None:
    url = f"{api_base()}/chat/stream"
    tokens: list[str] = []
    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    tokens.append(data)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
        return None
    except httpx.RequestError as exc:
        st.error(f"Could not reach API at {url}: {exc}")
        return None
    return "".join(tokens)


def health_check() -> bool:
    root_url = api_base().removesuffix("/api/v1")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{root_url}/health")
            response.raise_for_status()
        return True
    except httpx.RequestError:
        return False
    except httpx.HTTPStatusError:
        return False


def get_collections() -> list[dict[str, Any]]:
    data = request("GET", "/documents/collections")
    return data or []


def collection_names() -> list[str]:
    names = [item["name"] for item in get_collections()]
    return names or ["default"]


def rag_strategy_names() -> list[str]:
    data = request("GET", "/rag/strategies") or {}
    names = [item.get("name") for item in data.get("strategies", []) if item.get("name")]
    return names or STRATEGIES


def agent_type_names() -> list[str]:
    data = request("GET", "/agents/types") or {}
    names = [item.get("name") for item in data.get("agent_types", []) if item.get("name")]
    return names or AGENT_TYPES


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources returned.")
        return
    for index, source in enumerate(sources, start=1):
        score = source.get("score")
        score_text = f" | score {score:.3f}" if isinstance(score, (int, float)) else ""
        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-meta">Source {index}{score_text} | {source.get('id', 'unknown')}</div>
                <div>{html.escape(source.get('content', '')[:1200])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown("### Control Plane")
    st.text_input("API base URL", DEFAULT_API_URL, key="api_url")
    health = health_check() if st.button("Check API") else False
    if health:
        st.success("API is reachable")
    st.divider()
    st.markdown("### Defaults")
    available_strategies = rag_strategy_names()
    selected_collection = st.selectbox("Collection", collection_names(), key="active_collection")
    strategy_index = available_strategies.index("advanced") if "advanced" in available_strategies else 0
    selected_strategy = st.selectbox("RAG strategy", available_strategies, index=strategy_index)
    top_k = st.slider("Top K", 1, 20, 5)

st.markdown(
    """
    <div class="studio-header">
        <div class="studio-kicker">RAG Agent Studio</div>
        <h1>Production workspace for document-grounded AI</h1>
        <div class="studio-subtitle">
            Ingest documents, inspect collections, compare retrieval strategies, capture feedback, and monitor system quality from one operator-grade interface.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

collections = get_collections()
chunk_total = sum(item.get("chunk_count", 0) for item in collections)
document_total = sum(item.get("document_count", 0) for item in collections)

st.markdown('<div class="metric-row">', unsafe_allow_html=True)
metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Collections", len(collections))
metric_b.metric("Documents", document_total)
metric_c.metric("Chunks", chunk_total)
metric_d.metric("Active Strategy", selected_strategy)
st.markdown("</div>", unsafe_allow_html=True)

tab_ingest, tab_ask, tab_chat, tab_agents, tab_compare, tab_manage, tab_analytics = st.tabs(
    ["Ingest", "Ask", "Chat", "Agents", "Compare", "Manage", "Analytics"]
)

with tab_ingest:
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("Document Intake")
        upload = st.file_uploader("Upload PDF, DOCX, TXT, or MD", type=["pdf", "docx", "doc", "txt", "md"])
        collection_name = st.text_input("Target collection", selected_collection)
        chunk_strategy = st.selectbox("Chunking", CHUNK_STRATEGIES)
        col_a, col_b = st.columns(2)
        chunk_size = col_a.number_input("Chunk size", min_value=128, max_value=4096, value=512, step=64)
        chunk_overlap = col_b.number_input("Overlap", min_value=0, max_value=512, value=64, step=16)
        if st.button("Ingest document", type="primary", disabled=upload is None):
            files = {"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")}
            data = {
                "collection_name": collection_name,
                "chunk_strategy": chunk_strategy,
                "chunk_size": str(chunk_size),
                "chunk_overlap": str(chunk_overlap),
            }
            with st.spinner("Extracting, chunking, embedding, and storing..."):
                result = request("POST", "/documents/ingest", files=files, data=data)
            if result:
                st.success(f"Ingested {result['chunks_created']} chunks into {result['collection_name']}")
                st.json(result)
    with right:
        st.subheader("Collection Snapshot")
        if collections:
            st.dataframe(collections, use_container_width=True, hide_index=True)
        else:
            st.info("No collections found yet.")

with tab_ask:
    query = st.text_area("Question", height=120, placeholder="Ask a question grounded in your documents...")
    include_sources = st.toggle("Include sources", value=True)
    rerank = st.toggle("Rerank when supported", value=True)
    if st.button("Run RAG", type="primary", disabled=not query.strip()):
        payload = {
            "query": query,
            "strategy": selected_strategy,
            "collection_name": selected_collection,
            "top_k": top_k,
            "include_sources": include_sources,
            "rerank": rerank,
        }
        started = time.perf_counter()
        with st.spinner("Retrieving context and generating answer..."):
            result = request("POST", "/rag/query", json=payload)
        elapsed = (time.perf_counter() - started) * 1000
        if result:
            st.caption(f"Completed in {elapsed:.0f} ms")
            st.markdown(
                f"<div class='answer-box'>{html.escape(result.get('answer', ''))}</div>",
                unsafe_allow_html=True,
            )
            st.subheader("Sources")
            render_sources(result.get("sources", []))
            fb_a, fb_b, fb_c = st.columns([.2, .2, 1])
            if fb_a.button("Useful"):
                request("POST", "/analytics/feedback", json={
                    "rating": "up",
                    "query": query,
                    "strategy": selected_strategy,
                    "collection_name": selected_collection,
                })
                st.toast("Feedback recorded")
            if fb_b.button("Needs work"):
                request("POST", "/analytics/feedback", json={
                    "rating": "down",
                    "query": query,
                    "strategy": selected_strategy,
                    "collection_name": selected_collection,
                })
                st.toast("Feedback recorded")

with tab_compare:
    compare_query = st.text_area("Comparison question", height=110, placeholder="Run one question across several RAG pipelines...")
    selected_strategies = st.multiselect(
        "Strategies",
        available_strategies,
        default=[s for s in ["naive", "advanced", "multi_query"] if s in available_strategies] or available_strategies[:1],
    )
    if st.button("Compare strategies", type="primary", disabled=not compare_query.strip() or not selected_strategies):
        payload = {
            "query": compare_query,
            "strategies": selected_strategies,
            "collection_name": selected_collection,
            "top_k": top_k,
        }
        with st.spinner("Running strategy benchmark..."):
            comparison = request("POST", "/rag/compare", json=payload)
        if comparison:
            rows = [
                {
                    "strategy": item["strategy"],
                    "status": item["status"],
                    "latency_ms": round(item["latency_ms"], 1),
                    "sources": item["source_count"],
                    "error": item.get("error"),
                }
                for item in comparison.get("results", [])
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for item in comparison.get("results", []):
                with st.expander(f"{item['strategy']} | {item['status']} | {item['latency_ms']:.0f} ms", expanded=item["status"] == "success"):
                    if item["status"] == "error":
                        st.error(item.get("error", "Unknown error"))
                    else:
                        st.markdown(
                            f"<div class='answer-box'>{html.escape(item.get('answer', ''))}</div>",
                            unsafe_allow_html=True,
                        )
                        render_sources(item.get("sources", []))

with tab_chat:
    st.subheader("Multi-turn Chat")
    use_rag = st.toggle("Ground chat with RAG", value=True)
    stream_mode = st.toggle("Use streaming endpoint", value=False)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask follow-up questions, brainstorm, or plan tasks...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if stream_mode and not st.session_state.chat_session_id:
            st.session_state.chat_session_id = str(uuid.uuid4())

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "session_id": st.session_state.chat_session_id,
            "collection_name": selected_collection,
            "stream": False,
        }
        if use_rag:
            payload["rag_strategy"] = selected_strategy

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if stream_mode:
                    content = stream_chat(payload)
                    result = {"session_id": st.session_state.chat_session_id, "sources": []}
                else:
                    result = request("POST", "/chat/", json=payload)
                    content = result.get("message", {}).get("content", "") if result else ""
            if content is not None and result is not None:
                st.write(content)
                st.session_state.chat_messages.append({"role": "assistant", "content": content})
                if result.get("session_id"):
                    st.session_state.chat_session_id = result.get("session_id")
                sources = result.get("sources", [])
                if sources:
                    with st.expander("Retrieved sources"):
                        render_sources(sources)

    chat_a, chat_b = st.columns([.2, .8])
    if chat_a.button("Reset chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = None
        st.rerun()

with tab_agents:
    st.subheader("Agent Runner")
    available_agent_types = agent_type_names()
    agent_task = st.text_area(
        "Task",
        height=120,
        placeholder="Describe a task for an autonomous agent to execute...",
    )
    agent_type = st.selectbox("Agent type", available_agent_types, index=0)
    max_iterations = st.slider("Max iterations", 1, 30, 10)

    if st.button("Run agent", type="primary", disabled=not agent_task.strip()):
        payload = {
            "task": agent_task,
            "agent_type": agent_type,
            "max_iterations": max_iterations,
        }
        with st.spinner("Running agent..."):
            result = request("POST", "/agents/run", json=payload)
        if result:
            st.markdown(
                f"<div class='answer-box'>{html.escape(result.get('result', ''))}</div>",
                unsafe_allow_html=True,
            )
            stats_a, stats_b = st.columns(2)
            stats_a.metric("Iterations", result.get("iterations", 0))
            stats_b.metric("Tool Calls", len(result.get("tool_calls", [])))
            with st.expander("Steps", expanded=False):
                st.json(result.get("steps", []))
            with st.expander("Tool calls", expanded=False):
                st.json(result.get("tool_calls", []))

with tab_manage:
    st.subheader("Collections And Documents")
    manage_collection = st.selectbox("Inspect collection", collection_names(), key="manage_collection")
    delete_collection_col, _ = st.columns([.28, .72])
    if delete_collection_col.button("Delete collection", type="secondary"):
        result = request("DELETE", f"/documents/collection/{manage_collection}")
        if result:
            st.success(f"Deleted collection {result.get('deleted', manage_collection)}")
            st.rerun()

    documents = request("GET", f"/documents/{manage_collection}") or []
    if documents:
        st.dataframe(documents, use_container_width=True, hide_index=True)
        document_options = {f"{doc['source']} ({doc['chunk_count']} chunks)": doc["document_id"] for doc in documents}
        selected_doc_label = st.selectbox("Document", list(document_options.keys()))
        selected_doc_id = document_options[selected_doc_label]
        col_preview, col_delete = st.columns([1, .25])
        if col_preview.button("Preview chunks"):
            chunks = request("GET", f"/documents/{manage_collection}/documents/{selected_doc_id}/chunks") or []
            for chunk in chunks[:20]:
                with st.expander(chunk["id"]):
                    st.write(chunk.get("content", ""))
                    st.json(chunk.get("metadata", {}))
        if col_delete.button("Delete document"):
            result = request("DELETE", f"/documents/{manage_collection}/documents/{selected_doc_id}")
            if result:
                st.success(f"Deleted {result['chunks_deleted']} chunks")
    else:
        st.info("No documents found in this collection.")

with tab_analytics:
    st.subheader("Query Quality Loop")
    logs = request("GET", "/analytics/queries", params={"limit": 100}) or []
    feedback = request("GET", "/analytics/feedback", params={"limit": 100}) or []
    log_col, feedback_col = st.columns([1.2, .8], gap="large")
    with log_col:
        st.markdown("#### Recent Queries")
        if logs:
            st.dataframe(logs, use_container_width=True, hide_index=True)
        else:
            st.info("No query logs yet.")
    with feedback_col:
        st.markdown("#### Feedback")
        if feedback:
            st.dataframe(feedback, use_container_width=True, hide_index=True)
        else:
            st.info("No feedback captured yet.")
