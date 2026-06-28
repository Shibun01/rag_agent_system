"""
Plan-and-Execute Agent — separates high-level planning from low-level execution.

Phase 1 (Plan):   LLM generates a numbered step-by-step plan.
Phase 2 (Execute): Each step is executed, potentially with tools.
Phase 3 (Replan):  After each step, the agent can replan if results deviate.

Based on the "Plan-and-Solve Prompting" paper and LangChain's agent executor.
"""
from __future__ import annotations
import json
import re
from app.services.azure_openai import chat_completion
from app.core.tools.registry import execute_tool, list_tools

PLANNER_PROMPT = """You are a planning agent. Break the given task into a numbered list of
concrete steps. Each step should be actionable and specific.

Available tools: {tools}

Task: {task}

Output a numbered plan as:
1. <step>
2. <step>
...
"""

EXECUTOR_PROMPT = """You are an execution agent. Complete the given step using the available tools.

Available tools: {tools}
Current step: {step}
Previous results: {previous}

If you need a tool, respond with:
Action: <tool_name>({"arg": "value"})

If the step can be answered directly, respond with:
Result: <answer>
"""

REPLAN_PROMPT = """Given the original task, original plan, and completed steps so far,
decide if the remaining plan steps are still valid or if re-planning is needed.
Output JSON: {{"replan": true/false, "updated_plan": [...]}}

Task: {task}
Original plan: {plan}
Completed steps: {completed}
"""


def _parse_plan(text: str) -> list[str]:
    steps = re.findall(r"^\d+\.\s+(.+)$", text, re.MULTILINE)
    return steps if steps else [text]


async def plan_execute_agent(
    task: str,
    tools: list[str] | None = None,
    max_iterations: int = 10,
    session_id: str | None = None,
    use_memory: bool = True,
) -> dict:
    available_tools = list_tools(tools)
    tools_json = json.dumps(available_tools, indent=2)

    # Phase 1: Plan
    plan_msg = await chat_completion([
        {"role": "user", "content": PLANNER_PROMPT.format(tools=tools_json, task=task)}
    ], temperature=0.0)
    plan = _parse_plan(plan_msg.content)

    all_steps = []
    completed_results: list[dict] = []

    for i, step in enumerate(plan[:max_iterations]):
        # Phase 2: Execute step
        exec_msg = await chat_completion([
            {"role": "user", "content": EXECUTOR_PROMPT.format(
                tools=tools_json,
                step=step,
                previous=json.dumps(completed_results[-3:], indent=2),
            )}
        ], temperature=0.0)

        step_result = {"step": step, "output": exec_msg.content}

        # Check for tool call
        action_match = re.search(r"Action:\s*(\w+)\((.+)\)", exec_msg.content, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            try:
                args = json.loads(action_match.group(2))
            except Exception:
                args = {"input": action_match.group(2).strip()}
            observation = await execute_tool(tool_name, args)
            step_result["tool"] = tool_name
            step_result["observation"] = str(observation)
        elif "Result:" in exec_msg.content:
            step_result["output"] = exec_msg.content.split("Result:")[-1].strip()

        all_steps.append(step_result)
        completed_results.append(step_result)

        # Phase 3: Replan check (every 3 steps)
        if i > 0 and i % 3 == 0 and i < len(plan) - 1:
            replan_msg = await chat_completion([
                {"role": "user", "content": REPLAN_PROMPT.format(
                    task=task, plan=plan, completed=completed_results
                )}
            ], temperature=0.0)
            try:
                replan_data = json.loads(replan_msg.content)
                if replan_data.get("replan") and replan_data.get("updated_plan"):
                    plan = replan_data["updated_plan"]
            except Exception:
                pass

    # Synthesize final answer
    summary_msg = await chat_completion([
        {"role": "user", "content": (
            f"Synthesize a final answer for the task:\n{task}\n\n"
            f"Based on these completed steps:\n{json.dumps(completed_results, indent=2)}"
        )}
    ])

    return {
        "result": summary_msg.content,
        "plan": plan,
        "steps": all_steps,
        "iterations": len(all_steps),
    }
