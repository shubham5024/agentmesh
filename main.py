"""
AgentMesh API
FastAPI application exposing the registry, router, and invoke endpoints.

Endpoints:
  POST /invoke              — Route task to best agent and return result
  POST /agents/register     — Register a new agent
  DELETE /agents/{agent_id} — Deregister an agent
  GET  /agents              — List all agents
  GET  /tools               — List available MCP tools
  GET  /health              — Health check
"""
from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from agentmesh.a2a.protocol import A2AClient
from agentmesh.agents.builtin_agents import get_default_agents
from agentmesh.mcp.tool_server import mcp_server
from agentmesh.models import (
    AgentCard, AgentStatus, InvokeRequest, InvokeResponse, MessageStatus
)
from agentmesh.registry.agent_registry import registry
from agentmesh.router.intent_router import IntentRouter

logger = logging.getLogger(__name__)
router = IntentRouter()
a2a    = A2AClient(registry)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Register built-in agents
    for agent in get_default_agents():
        registry.register(agent.card)
    logger.info("AgentMesh started. Registry: %s", registry.stats())
    yield
    logger.info("AgentMesh shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AgentMesh",
    description = (
        "A2A Multi-Agent Orchestration Framework — "
        "register agents, route tasks semantically, chain agents via A2A protocol."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
    docs_url = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "registry": registry.stats()}


@app.post("/invoke", response_model=InvokeResponse, tags=["Orchestration"])
async def invoke(req: InvokeRequest):
    """
    Route a task to the best matching agent and return the result.

    1. IntentRouter embeds the task and finds the best agent via FAISS.
    2. A2AClient dispatches an A2AMessage to the agent's endpoint.
    3. Response is returned with routing metadata.
    """
    t0     = time.perf_counter()
    agents = registry.list_active()

    if not agents:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "No active agents available",
        )

    decision = router.route(req.task, agents)
    agent    = registry.get(decision.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Routed agent not found")

    # Build and dispatch A2A message
    message  = A2AClient.make_message(
        sender_id   = "agentmesh-router",
        receiver_id = decision.agent_id,
        task        = req.task,
        payload     = req.payload,
        context     = req.context,
    )
    a2a_resp = await a2a.send(message)

    if a2a_resp.status == MessageStatus.FAILED:
        raise HTTPException(
            status_code = status.HTTP_502_BAD_GATEWAY,
            detail      = f"Agent failed: {a2a_resp.error}",
        )

    total_ms = (time.perf_counter() - t0) * 1000
    return InvokeResponse(
        routed_to = agent.name,
        result    = a2a_resp.result,
        route     = decision,
        total_ms  = total_ms,
        status    = a2a_resp.status,
    )


@app.post("/agents/register", response_model=AgentCard, tags=["Registry"])
async def register_agent(card: AgentCard):
    """Register a new agent (or update an existing one)."""
    return registry.register(card)


@app.delete("/agents/{agent_id}", tags=["Registry"])
async def deregister_agent(agent_id: str):
    """Remove an agent from the registry."""
    if not registry.deregister(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": f"Agent '{agent_id}' deregistered"}


@app.get("/agents", tags=["Registry"])
async def list_agents(active_only: bool = True):
    agents = registry.list_active() if active_only else registry.list_all()
    return {"count": len(agents), "agents": agents}


@app.patch("/agents/{agent_id}/status", tags=["Registry"])
async def update_agent_status(agent_id: str, status: AgentStatus):
    if not registry.set_status(agent_id, status):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": f"Agent '{agent_id}' status updated to '{status}'"}


@app.get("/tools", tags=["MCP"])
async def list_tools():
    """List all available MCP tools."""
    return {"tools": mcp_server.list_tools()}


@app.post("/tools/{tool_name}", tags=["MCP"])
async def call_tool(tool_name: str, kwargs: dict = {}):
    """Invoke an MCP tool by name."""
    result = await mcp_server.call_tool(tool_name, **kwargs)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
