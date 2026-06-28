"""
Tool registry — agents look up and call tools through this interface.
"""
from __future__ import annotations
import math

# Registry: tool_name → {description, args, handler}
_REGISTRY: dict[str, dict] = {}


def register(name: str, description: str, args: dict):
    """Decorator to register a function as an agent tool."""
    def decorator(fn):
        _REGISTRY[name] = {"description": description, "args": args, "handler": fn}
        return fn
    return decorator


def list_tools(subset: list[str] | None = None) -> dict:
    """Return schema dict for a subset (or all) registered tools."""
    tools = _REGISTRY if subset is None else {k: v for k, v in _REGISTRY.items() if k in subset}
    return {k: {"description": v["description"], "args": v["args"]} for k, v in tools.items()}


async def execute_tool(name: str, args: dict) -> str:
    if name not in _REGISTRY:
        return f"Error: tool '{name}' not found."
    handler = _REGISTRY[name]["handler"]
    try:
        result = handler(**args)
        # Support both sync and async handlers
        if hasattr(result, "__await__"):
            result = await result
        return str(result)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"


# ── Built-in tools ─────────────────────────────────────────────────────────────
@register(
    "calculator",
    description="Evaluate a mathematical expression. Input must be a valid Python math expression.",
    args={"expression": "str"},
)
def calculator(expression: str) -> float:
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    return eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307


@register(
    "word_count",
    description="Count words in a given text.",
    args={"text": "str"},
)
def word_count(text: str) -> int:
    return len(text.split())


@register(
    "web_search",
    description="Search the web for current information. Returns top results.",
    args={"query": "str", "max_results": "int (optional, default 3)"},
)
async def web_search(query: str, max_results: int = 3) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return "\n".join(f"{r['title']}: {r['body']}" for r in results)
    except Exception as e:
        return f"Web search unavailable: {e}"
