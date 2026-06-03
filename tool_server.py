"""
AgentMesh MCP Tool Server
Exposes external tool integrations via the Model Context Protocol.
Agents call tools through this layer — decoupled from the tool implementation.

Supported tools (mock implementations — swap in real MCP SDK calls):
  - web_search     : DuckDuckGo / SerpAPI
  - github_search  : GitHub REST API
  - calculator     : Arithmetic evaluation
  - time           : Current datetime

Design: Each tool is a plain async function registered in TOOL_REGISTRY.
Agents invoke tools by name; the server handles dispatch + error normalisation.
"""
from __future__ import annotations
import asyncio
import datetime
import logging
import math
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Type alias
ToolFunc = Callable[..., Awaitable[dict[str, Any]]]


# ── Tool implementations ──────────────────────────────────────────────────────

async def tool_web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web for a query string.
    Replace with real SerpAPI / DuckDuckGo API call in production.
    """
    await asyncio.sleep(0.05)   # Simulate network latency
    return {
        "query":   query,
        "results": [
            {"title": f"Result {i+1} for '{query}'",
             "url":   f"https://example.com/result-{i+1}",
             "snippet": f"Relevant snippet {i+1} about {query}."}
            for i in range(max_results)
        ],
    }


async def tool_github_search(query: str, entity_type: str = "repositories") -> dict:
    """
    Search GitHub for repositories, issues, or users.
    Replace with real GitHub REST API call: GET /search/{entity_type}
    """
    await asyncio.sleep(0.05)
    return {
        "query":       query,
        "entity_type": entity_type,
        "items": [
            {"name": f"repo-{i+1}", "stars": (10 - i) * 100,
             "description": f"A {query}-related {entity_type} #{i+1}",
             "url": f"https://github.com/user/repo-{i+1}"}
            for i in range(3)
        ],
    }


async def tool_calculator(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    try:
        # Restrict to safe math ops only
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result  = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": result, "error": None}
    except Exception as exc:
        return {"expression": expression, "result": None, "error": str(exc)}


async def tool_current_time(timezone: str = "UTC") -> dict:
    """Return the current datetime."""
    now = datetime.datetime.utcnow()
    return {
        "timezone": timezone,
        "iso":      now.isoformat() + "Z",
        "readable": now.strftime("%A, %B %d %Y at %H:%M UTC"),
    }


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolFunc] = {
    "web_search":    tool_web_search,
    "github_search": tool_github_search,
    "calculator":    tool_calculator,
    "current_time":  tool_current_time,
}


# ── MCP Server ────────────────────────────────────────────────────────────────

class MCPToolServer:
    """
    Lightweight MCP-compatible tool dispatcher.

    Agents call: await mcp_server.call_tool("web_search", query="LangGraph")
    The server handles: lookup → async dispatch → normalised response.
    """

    def __init__(self, tools: dict[str, ToolFunc] | None = None) -> None:
        self._tools = tools or TOOL_REGISTRY.copy()

    def register_tool(self, name: str, func: ToolFunc) -> None:
        self._tools[name] = func
        logger.info("Registered MCP tool: %s", name)

    def list_tools(self) -> list[dict]:
        return [
            {"name": name, "description": func.__doc__  or ""}
            for name, func in self._tools.items()
        ]

    async def call_tool(self, name: str, **kwargs: Any) -> dict:
        """
        Invoke a tool by name. Returns a normalised dict with
        'success', 'result', and 'error' keys.
        """
        if name not in self._tools:
            return {
                "success": False,
                "result":  None,
                "error":   f"Tool '{name}' not found. Available: {list(self._tools)}",
            }
        try:
            result = await self._tools[name](**kwargs)
            logger.debug("MCP tool '%s' succeeded: %s", name, str(result)[:80])
            return {"success": True, "result": result, "error": None}
        except Exception as exc:
            logger.error("MCP tool '%s' failed: %s", name, exc)
            return {"success": False, "result": None, "error": str(exc)}


# Module-level singleton
mcp_server = MCPToolServer()
