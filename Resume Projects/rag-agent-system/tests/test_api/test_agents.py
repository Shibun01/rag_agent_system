from __future__ import annotations

from app.api.v1.endpoints import agents as agents_endpoint
from app.models.schemas import AgentType


def test_agent_endpoint_forwards_supported_options(client):
    captured: dict[str, object] = {}

    async def fake_handler(task, max_iterations, tools=None, session_id=None, use_memory=True):
        captured.update({
            "task": task,
            "max_iterations": max_iterations,
            "tools": tools,
            "session_id": session_id,
            "use_memory": use_memory,
        })
        return {
            "result": "done",
            "steps": [],
            "tool_calls": [],
            "iterations": 1,
        }

    original = agents_endpoint._AGENT_MAP[AgentType.react]
    agents_endpoint._AGENT_MAP[AgentType.react] = fake_handler
    try:
        response = client.post(
            "/api/v1/agents/run",
            json={
                "task": "Summarize the report",
                "agent_type": "react",
                "max_iterations": 4,
                "tools": ["search"],
                "session_id": "session-123",
                "use_memory": False,
            },
        )
    finally:
        agents_endpoint._AGENT_MAP[AgentType.react] = original

    assert response.status_code == 200
    assert captured == {
        "task": "Summarize the report",
        "max_iterations": 4,
        "tools": ["search"],
        "session_id": "session-123",
        "use_memory": False,
    }