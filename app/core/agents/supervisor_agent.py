"""
Supervisor Agent — orchestrates multiple specialized sub-agents.

The supervisor decides which sub-agent to route a task to and aggregates results.
Sub-agents are registered with a name, description, and callable.
"""
from __future__ import annotations
import json
from app.services.azure_openai import chat_completion

# Sub-agent registry
_sub_agents: dict[str, dict] = {}


def register_sub_agent(name: str, description: str, handler) -> None:
    _sub_agents[name] = {"name": name, "description": description, "handler": handler}


ROUTE_PROMPT = """You are a supervisor that routes tasks to the most appropriate sub-agent.

Available sub-agents:
{agents}

Task: {task}

Output JSON: {{"agent": "<agent_name>", "reason": "<why>", "sub_task": "<refined task for the agent>"}}
"""

AGGREGATE_PROMPT = """Combine the outputs from multiple sub-agents into a single coherent answer.

Original task: {task}
Sub-agent outputs:
{outputs}

Final synthesized answer:
"""


async def supervisor_agent(
    task: str,
    max_iterations: int = 10,  # kept for API compatibility
) -> dict:
    if not _sub_agents:
        # Bootstrap with inline sub-agents if none registered
        from app.core.agents.react_agent import react_agent
        from app.core.agents.reflection_agent import reflection_agent
        register_sub_agent("react", "Tool-using agent for factual lookups & calculations", react_agent)
        register_sub_agent("reflection", "Self-critiquing agent for analysis & writing tasks", reflection_agent)

    agent_list = [{"name": a["name"], "description": a["description"]} for a in _sub_agents.values()]

    # Route
    route_msg = await chat_completion([
        {"role": "user", "content": ROUTE_PROMPT.format(
            agents=json.dumps(agent_list, indent=2), task=task
        )}
    ], temperature=0.0)

    try:
        routing = json.loads(route_msg.content)
    except Exception:
        routing = {"agent": list(_sub_agents.keys())[0], "sub_task": task}

    selected_agent = routing.get("agent", list(_sub_agents.keys())[0])
    sub_task = routing.get("sub_task", task)

    if selected_agent not in _sub_agents:
        selected_agent = list(_sub_agents.keys())[0]

    # Execute selected sub-agent
    handler = _sub_agents[selected_agent]["handler"]
    sub_result = await handler(sub_task)

    return {
        "result": sub_result.get("result", str(sub_result)),
        "routed_to": selected_agent,
        "routing_reason": routing.get("reason", ""),
        "sub_agent_output": sub_result,
        "steps": sub_result.get("steps", []),
        "iterations": sub_result.get("iterations", 1),
    }
