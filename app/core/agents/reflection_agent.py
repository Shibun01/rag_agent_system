"""
Reflection Agent — generates an initial response, then critiques and refines it.

Inspired by "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn 2023)
and "Constitutional AI" self-critique patterns.

Loop:
  1. Generate initial response.
  2. Critique (is it accurate, complete, helpful?).
  3. Refine based on critique.
  4. Repeat up to N rounds.
"""
from __future__ import annotations
from app.services.azure_openai import chat_completion

INITIAL_PROMPT = """Answer the following task as helpfully and accurately as possible.

Task: {task}
"""

CRITIQUE_PROMPT = """You are a critical reviewer. Evaluate the following response to a task.
Identify:
  - Factual errors or unsupported claims
  - Missing important information
  - Logical gaps or inconsistencies
  - Ways to make the response clearer or more useful

Task: {task}
Response: {response}

Provide a critique (or write "LGTM" if the response is excellent):
"""

REFINE_PROMPT = """Improve the following response based on the provided critique.
Keep what's good and fix the identified issues.

Task: {task}
Original response: {response}
Critique: {critique}

Improved response:
"""


async def reflection_agent(
    task: str,
    reflection_rounds: int = 2,
    max_iterations: int = 10,  # kept for API compatibility
) -> dict:
    reflections: list[dict] = []

    # Round 0: initial response
    msg = await chat_completion(
        [{"role": "user", "content": INITIAL_PROMPT.format(task=task)}],
        temperature=0.7,
    )
    current_response = msg.content
    reflections.append({"round": 0, "type": "initial", "content": current_response})

    for round_num in range(1, reflection_rounds + 1):
        # Critique
        critique_msg = await chat_completion(
            [{"role": "user", "content": CRITIQUE_PROMPT.format(task=task, response=current_response)}],
            temperature=0.3,
        )
        critique = critique_msg.content
        reflections.append({"round": round_num, "type": "critique", "content": critique})

        if "LGTM" in critique.upper():
            break

        # Refine
        refine_msg = await chat_completion([
            {"role": "user", "content": REFINE_PROMPT.format(
                task=task, response=current_response, critique=critique
            )}
        ], temperature=0.5)
        current_response = refine_msg.content
        reflections.append({"round": round_num, "type": "refined", "content": current_response})

    return {
        "result": current_response,
        "reflections": reflections,
        "rounds_completed": len([r for r in reflections if r["type"] == "refined"]),
        "steps": reflections,
        "iterations": len(reflections),
    }
