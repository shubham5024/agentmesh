"""
AgentMesh Core Data Models
All Pydantic schemas for agent registration, A2A messaging, and routing.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    ACTIVE   = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"


class MessageStatus(str, Enum):
    PENDING   = "pending"
    SUCCESS   = "success"
    FAILED    = "failed"
    DELEGATED = "delegated"


# ─── Agent Registration ───────────────────────────────────────────────────────

class AgentCard(BaseModel):
    """Descriptor each agent registers with the central registry."""
    agent_id:     str            = Field(default_factory=lambda: str(uuid4()))
    name:         str
    description:  str            # Used by the intent router for embedding
    capabilities: list[str]      # Fine-grained capability tags
    endpoint:     str            # URL the router dispatches to
    version:      str            = "1.0.0"
    status:       AgentStatus    = AgentStatus.ACTIVE
    metadata:     dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime      = Field(default_factory=datetime.utcnow)

    model_config = {"json_schema_extra": {
        "example": {
            "name": "ResearchAgent",
            "description": "Searches the web and summarises findings for research tasks",
            "capabilities": ["web_search", "summarisation", "citation"],
            "endpoint": "http://localhost:8001/invoke",
        }
    }}


# ─── A2A Protocol ─────────────────────────────────────────────────────────────

class A2AMessage(BaseModel):
    """Standardised inter-agent message envelope (Agent-to-Agent protocol)."""
    message_id:  str            = Field(default_factory=lambda: str(uuid4()))
    sender_id:   str
    receiver_id: str
    task:        str
    payload:     dict[str, Any] = Field(default_factory=dict)
    context:     dict[str, Any] = Field(default_factory=dict)  # Shared conversation state
    timestamp:   datetime       = Field(default_factory=datetime.utcnow)


class A2AResponse(BaseModel):
    """Standardised response returned by every agent."""
    message_id:  str
    agent_id:    str
    result:      Any
    status:      MessageStatus = MessageStatus.SUCCESS
    latency_ms:  float         = 0.0
    delegated_to: str | None   = None   # If agent delegated to another
    error:        str | None   = None
    timestamp:    datetime     = Field(default_factory=datetime.utcnow)


# ─── Routing ──────────────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    """Output of the RAG-based intent router."""
    agent_id:    str
    agent_name:  str
    confidence:  float          # Cosine similarity score
    reasoning:   str
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


# ─── API Contracts ────────────────────────────────────────────────────────────

class InvokeRequest(BaseModel):
    task:    str
    payload: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {
            "task": "Summarise the latest research on RAG systems",
            "payload": {},
            "context": {"user_id": "u_123", "session_id": "s_abc"},
        }
    }}


class InvokeResponse(BaseModel):
    request_id:   str = Field(default_factory=lambda: str(uuid4()))
    routed_to:    str
    result:       Any
    route:        RouteDecision
    total_ms:     float
    status:       MessageStatus = MessageStatus.SUCCESS
