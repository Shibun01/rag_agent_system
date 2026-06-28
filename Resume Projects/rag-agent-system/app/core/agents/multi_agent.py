"""
Multi-Agent System — parallel execution of specialized agents with result fusion.

Pattern:
  1. A coordinator decomposes the task into sub-tasks.
  2. Each sub-task is dispatched to the best-suited agent in parallel.
  3. Results are aggregated into a final answer by a synthesizer.
"""
from __future__ import annotations
import json
import asyncio
import inspect
from app.services.azure_openai import chat_completion
from app.core.agents.react_agent import react_agent
from app.core.agents.reflection_agent import reflection_agent
from app.core.agents.rag_agent import rag_agent

DECOMPOSE_PROMPT = """Break the following task into independent sub-tasks that can be handled
by different specialized agents simultaneously.

Agents available:
- react_agent       : Tool-using agent for lookups, calculations, real-time data
- reflection_agent  : Deep reasoning, writing, analysis, critique
- rag_agent         : Knowledge retrieval from documents

Output JSON array of {{"sub_task": str, "agent": "react_agent|reflection_agent|rag_agent"}}

Task: {task}
"""

SYNTHESIZE_PROMPT = """Synthesize the following parallel agent outputs into a single
coherent, comprehensive answer for the original task.

Original task: {task}

Agent outputs:
{outputs}

Final answer:
"""

_AGENT_MAP = {
    "react_agent": react_agent,
    "reflection_agent": reflection_agent,
    "rag_agent": rag_agent,
}


async def multi_agent(
    task: str,
    max_iterations: int = 10,  # kept for API compatibility
    tools: list[str] | None = None,
    session_id: str | None = None,
    use_memory: bool = True,
) -> dict:
    # Step 1: Decompose
    decompose_msg = await chat_completion([
        {"role": "user", "content": DECOMPOSE_PROMPT.format(task=task)}
    ], temperature=0.0)

    try:
        sub_tasks: list[dict] = json.loads(decompose_msg.content)
    except Exception:
        sub_tasks = [{"sub_task": task, "agent": "react_agent"}]

    # Step 2: Parallel dispatch
    async def run_sub(item: dict) -> dict:
        agent_name = item.get("agent", "react_agent")
        agent_fn = _AGENT_MAP.get(agent_name, react_agent)
        agent_signature = inspect.signature(agent_fn)
        kwargs = {"task": item["sub_task"]}
        if "max_iterations" in agent_signature.parameters:
            kwargs["max_iterations"] = max_iterations
        if "tools" in agent_signature.parameters:
            kwargs["tools"] = tools
        if "session_id" in agent_signature.parameters:
            kwargs["session_id"] = session_id
        if "use_memory" in agent_signature.parameters:
            kwargs["use_memory"] = use_memory
        result = await agent_fn(**kwargs)
        return {"agent": agent_name, "sub_task": item["sub_task"], "result": result.get("result", str(result))}

    parallel_results = await asyncio.gather(*[run_sub(st) for st in sub_tasks])

    # Step 3: Synthesize
    outputs_text = "\n\n".join(
        f"[{r['agent']}] Sub-task: {r['sub_task']}\nOutput: {r['result']}"
        for r in parallel_results
    )
    synth_msg = await chat_completion([
        {"role": "user", "content": SYNTHESIZE_PROMPT.format(task=task, outputs=outputs_text)}
    ])

    return {
        "result": synth_msg.content,
        "sub_tasks": sub_tasks,
        "agent_outputs": list(parallel_results),
        "steps": [{"type": "sub_agent_result", **r} for r in parallel_results],
        "iterations": len(sub_tasks),
    }
