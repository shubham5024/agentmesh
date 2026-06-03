"""
AgentMesh Base Agent
Abstract base class all agents inherit from.
Uses LangGraph for internal step execution with full state tracking.
"""
from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agentmesh.models import A2AMessage, A2AResponse, AgentCard, MessageStatus

logger = logging.getLogger(__name__)


# ── LangGraph State ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    task:        str
    payload:     dict[str, Any]
    context:     dict[str, Any]
    plan:        list[str]           # Steps the agent will execute
    observations: list[str]         # Intermediate results
    result:      str | None
    error:       str | None
    status:      str


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Every AgentMesh agent inherits this class.

    Subclasses implement:
      - card()       → AgentCard describing the agent for registry
      - _plan()      → decompose task into steps
      - _execute()   → run the plan and produce a result

    The LangGraph workflow handles: plan → execute → respond
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def card(self) -> AgentCard:
        """Return the AgentCard for registry registration."""
        ...

    @abstractmethod
    def _plan(self, state: AgentState) -> AgentState:
        """Decompose the task into a list of steps."""
        ...

    @abstractmethod
    def _execute(self, state: AgentState) -> AgentState:
        """Execute the plan steps and populate state['result']."""
        ...

    # ── LangGraph workflow ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Build the plan → execute → respond LangGraph."""
        g = StateGraph(AgentState)
        g.add_node("plan",    self._plan)
        g.add_node("execute", self._execute)
        g.add_node("respond", self._respond)

        g.set_entry_point("plan")
        g.add_edge("plan",    "execute")
        g.add_edge("execute", "respond")
        g.add_edge("respond", END)

        return g.compile()

    def _respond(self, state: AgentState) -> AgentState:
        """Final node — ensures status is set correctly."""
        if state.get("error"):
            state["status"] = MessageStatus.FAILED
        else:
            state["status"] = MessageStatus.SUCCESS
        return state

    # ── Public invoke ─────────────────────────────────────────────────────────

    def invoke(self, message: A2AMessage) -> A2AResponse:
        """
        Entry point called by the A2A dispatcher.
        Runs the LangGraph workflow and returns a normalised A2AResponse.
        """
        t0 = time.perf_counter()
        logger.info("[%s] Received task: %s", self.card.name, message.task[:80])

        initial_state: AgentState = {
            "task":         message.task,
            "payload":      message.payload,
            "context":      message.context,
            "plan":         [],
            "observations": [],
            "result":       None,
            "error":        None,
            "status":       "pending",
        }

        try:
            final_state = self._graph.invoke(initial_state)
            latency     = (time.perf_counter() - t0) * 1000
            status      = final_state.get("status", MessageStatus.SUCCESS)
            return A2AResponse(
                message_id = message.message_id,
                agent_id   = self.card.agent_id,
                result     = final_state.get("result"),
                status     = status,
                latency_ms = latency,
                error      = final_state.get("error"),
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.exception("[%s] Unhandled error: %s", self.card.name, exc)
            return A2AResponse(
                message_id = message.message_id,
                agent_id   = self.card.agent_id,
                result     = None,
                status     = MessageStatus.FAILED,
                latency_ms = latency,
                error      = str(exc),
            )
