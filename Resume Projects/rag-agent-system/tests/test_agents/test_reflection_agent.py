from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.agents import reflection_agent as reflection_module


@pytest.mark.asyncio
async def test_reflection_agent_refines_response(monkeypatch):
    replies = iter([
        "Initial draft",
        "Add the missing constraints.",
        "Improved answer with constraints.",
    ])

    async def fake_chat_completion(messages, temperature=0.0, max_tokens=2048, tools=None):
        return SimpleNamespace(content=next(replies))

    monkeypatch.setattr(reflection_module, "chat_completion", fake_chat_completion)

    result = await reflection_module.reflection_agent("Explain the rollout plan", reflection_rounds=1)

    assert result["result"] == "Improved answer with constraints."
    assert [step["type"] for step in result["steps"]] == ["initial", "critique", "refined"]