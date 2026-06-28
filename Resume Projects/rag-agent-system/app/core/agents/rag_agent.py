"""
RAG Agent — a tool-using ReAct agent whose primary tool is a RAG retriever.

This combines retrieval-augmented generation with agentic tool use:
  - The agent can call the RAG retriever multiple times with different queries.
  - It can also call other tools (calculator, etc.).
  - It reasons about when retrieval is sufficient vs. when to refine the query.
"""
from __future__ import annotations
import json
import re
from app.services.azure_openai import chat_completion
from app.services.vector_store import get_vector_store
from app.core.tools.registry import execute_tool, list_tools

SYSTEM_PROMPT = """You are a RAG-augmented agent. You have access to a document retriever
and other tools. Use retrieval to ground your answers in facts.

Available tools: {tools}

Format each step as:
Thought: <reasoning>
Action: <tool_name>({"arg": "value"})

When you have the final answer:
Thought: I have enough information.
Final Answer: <answer>
"""


async def _retriever_tool(args: dict) -> str:
    query = args.get("query", args.get("input", ""))
    collection = args.get("collection", "default")
    top_k = int(args.get("top_k", 5))
    store = get_vector_store()
    results = await store.similarity_search(query, collection, top_k)
    if not results:
        return "No relevant documents found."
    return "\n---\n".join(f"[Score: {r.score:.3f}] {r.content}" for r in results)


async def rag_agent(
    task: str,
    tools: list[str] | None = None,
    max_iterations: int = 10,
    collection: str = "default",
    session_id: str | None = None,
    use_memory: bool = True,
) -> dict:
    available_tools = list_tools(tools)
    # Inject retriever as the primary tool
    available_tools["retrieve"] = {"description": "Search documents by semantic query", "args": {"query": "str", "top_k": "int (optional)"}}

    system = SYSTEM_PROMPT.format(tools=json.dumps(available_tools, indent=2))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {task}"},
    ]
    steps: list[dict] = []

    for iteration in range(max_iterations):
        msg = await chat_completion(messages, temperature=0.0)
        text = msg.content

        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            steps.append({"iteration": iteration, "type": "final", "content": text})
            return {
                "result": answer,
                "steps": steps,
                "iterations": iteration + 1,
                "tool_calls": [s for s in steps if s.get("type") == "action"],
            }

        action_match = re.search(r"Action:\s*(\w+)\((.+)\)", text, re.DOTALL)
        if not action_match:
            steps.append({"iteration": iteration, "type": "thought", "content": text})
            return {"result": text, "steps": steps, "iterations": iteration + 1, "tool_calls": []}

        tool_name = action_match.group(1)
        try:
            args = json.loads(action_match.group(2))
        except Exception:
            args = {"input": action_match.group(2).strip()}

        steps.append({"iteration": iteration, "type": "action", "tool": tool_name, "args": args})

        # Route to retriever or generic tool registry
        if tool_name == "retrieve":
            args["collection"] = collection
            observation = await _retriever_tool(args)
        else:
            observation = await execute_tool(tool_name, args)

        steps.append({"iteration": iteration, "type": "observation", "content": str(observation)})
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return {"result": "Max iterations reached.", "steps": steps, "iterations": max_iterations, "tool_calls": []}
