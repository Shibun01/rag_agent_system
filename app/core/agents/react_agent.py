"""
ReAct Agent — Reasoning + Acting (Yao et al., 2022).

Loop:
  Thought → Action → Observation → Thought → … → Final Answer

The agent alternates between reasoning about the current state (Thought)
and calling a tool (Action). The tool result (Observation) feeds back
into the next reasoning step.
"""
from __future__ import annotations
import json
import re
from app.services.azure_openai import chat_completion
from app.core.tools.registry import execute_tool, list_tools

SYSTEM_PROMPT = """You are a ReAct agent. Solve tasks by interleaving Thought, Action, and Observation steps.

Available tools: {tools}

Format each step as:
Thought: <your reasoning>
Action: <tool_name>(<json_args>)

When you have the final answer output:
Thought: I now know the final answer.
Final Answer: <answer>
"""

OBSERVATION_TEMPLATE = "Observation: {result}"


def _parse_action(text: str) -> tuple[str, dict] | None:
    """Extract tool_name and args from 'Action: tool_name({"key": "val"})' line."""
    match = re.search(r"Action:\s*(\w+)\((.+)\)", text, re.DOTALL)
    if not match:
        return None
    tool_name = match.group(1)
    try:
        args = json.loads(match.group(2))
    except Exception:
        args = {"input": match.group(2).strip()}
    return tool_name, args


async def react_agent(
    task: str,
    tools: list[str] | None = None,
    max_iterations: int = 10,
    session_id: str | None = None,
) -> dict:
    available_tools = list_tools(tools)
    system = SYSTEM_PROMPT.format(tools=json.dumps(available_tools, indent=2))

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {task}"},
    ]
    steps: list[dict] = []

    for iteration in range(max_iterations):
        msg = await chat_completion(messages, temperature=0.0)
        assistant_text = msg.content

        # Check for final answer
        if "Final Answer:" in assistant_text:
            answer = assistant_text.split("Final Answer:")[-1].strip()
            steps.append({"iteration": iteration, "type": "final", "content": assistant_text})
            return {
                "result": answer,
                "steps": steps,
                "iterations": iteration + 1,
                "tool_calls": [s for s in steps if s["type"] == "action"],
            }

        # Parse action
        parsed = _parse_action(assistant_text)
        if not parsed:
            # No valid action — treat as final answer
            steps.append({"iteration": iteration, "type": "thought", "content": assistant_text})
            return {
                "result": assistant_text,
                "steps": steps,
                "iterations": iteration + 1,
                "tool_calls": [],
            }

        tool_name, args = parsed
        steps.append({"iteration": iteration, "type": "action", "tool": tool_name, "args": args, "content": assistant_text})

        # Execute tool
        observation = await execute_tool(tool_name, args)
        obs_text = OBSERVATION_TEMPLATE.format(result=str(observation))
        steps.append({"iteration": iteration, "type": "observation", "content": obs_text})

        # Append to conversation
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": obs_text})

    return {"result": "Max iterations reached.", "steps": steps, "iterations": max_iterations, "tool_calls": []}
