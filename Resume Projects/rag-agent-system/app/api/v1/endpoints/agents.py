import inspect
from fastapi import APIRouter, HTTPException
from app.models.schemas import AgentRequest, AgentResponse, AgentType
from app.core.agents.react_agent import react_agent
from app.core.agents.plan_execute_agent import plan_execute_agent
from app.core.agents.reflection_agent import reflection_agent
from app.core.agents.supervisor_agent import supervisor_agent
from app.core.agents.rag_agent import rag_agent
from app.core.agents.multi_agent import multi_agent

router = APIRouter()

_AGENT_MAP = {
    AgentType.react:         react_agent,
    AgentType.plan_execute:  plan_execute_agent,
    AgentType.reflection:    reflection_agent,
    AgentType.supervisor:    supervisor_agent,
    AgentType.rag_agent:     rag_agent,
    AgentType.multi_agent:   multi_agent,
}


async def _run_agent(req: AgentRequest) -> dict:
    handler = _AGENT_MAP.get(req.agent_type)
    if not handler:
        raise HTTPException(status_code=400, detail=f"Unknown agent type: {req.agent_type}")

    kwargs = {
        "task": req.task,
        "max_iterations": req.max_iterations,
    }
    signature = inspect.signature(handler)
    if "tools" in signature.parameters:
        kwargs["tools"] = req.tools
    if "session_id" in signature.parameters:
        kwargs["session_id"] = req.session_id
    if "use_memory" in signature.parameters:
        kwargs["use_memory"] = req.use_memory

    return await handler(**kwargs)


@router.post("/run", response_model=AgentResponse, summary="Run an agent")
async def run_agent(req: AgentRequest):
    """
    Execute a task with one of the available agent types:

    | Agent          | Description |
    |----------------|-------------|
    | `react`        | Reasoning + Acting loop with tools |
    | `plan_execute` | Two-phase: plan all steps, then execute |
    | `reflection`   | Generate → critique → refine (N rounds) |
    | `supervisor`   | Routes task to best sub-agent |
    | `rag_agent`    | ReAct agent whose primary tool is RAG retrieval |
    | `multi_agent`  | Decomposes task → parallel specialist agents |
    """
    try:
        result = await _run_agent(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AgentResponse(
        result=result.get("result", ""),
        agent_type=req.agent_type,
        steps=result.get("steps", []),
        tool_calls=result.get("tool_calls", []),
        iterations=result.get("iterations", 0),
    )


@router.get("/types", summary="List available agent types")
async def list_agent_types():
    return {
        "agent_types": [
            {"name": a.value, "description": desc}
            for a, desc in {
                AgentType.react:        "Reasoning + Acting loop with tool use",
                AgentType.plan_execute: "Two-phase: plan then execute each step",
                AgentType.reflection:   "Self-critique and refinement loop",
                AgentType.supervisor:   "Routes to best specialist sub-agent",
                AgentType.rag_agent:    "ReAct agent with RAG retrieval as primary tool",
                AgentType.multi_agent:  "Parallel specialist agents + synthesizer",
            }.items()
        ]
    }
