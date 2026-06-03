"""
AgentMesh Registry
Thread-safe in-memory registry for agent discovery and health tracking.
Agents register via URL endpoints — no rewrites needed for existing services.
"""
from __future__ import annotations
import logging
from datetime import datetime
from threading import RLock

from agentmesh.models import AgentCard, AgentStatus

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry that stores AgentCards and exposes CRUD operations.

    Design decision: URL-based registration means any HTTP service can
    onboard as an agent without modifying internal logic.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentCard] = {}
        self._lock   = RLock()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, card: AgentCard) -> AgentCard:
        """Register or update an agent."""
        with self._lock:
            if card.agent_id in self._agents:
                logger.info("Re-registering agent '%s' (update)", card.name)
            else:
                logger.info("Registering new agent '%s' at %s", card.name, card.endpoint)
            self._agents[card.agent_id] = card
            return card

    def deregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry. Returns True if found."""
        with self._lock:
            if agent_id in self._agents:
                name = self._agents[agent_id].name
                del self._agents[agent_id]
                logger.info("Deregistered agent '%s'", name)
                return True
            return False

    def get(self, agent_id: str) -> AgentCard | None:
        with self._lock:
            return self._agents.get(agent_id)

    def list_active(self) -> list[AgentCard]:
        """Return all ACTIVE agents."""
        with self._lock:
            return [a for a in self._agents.values() if a.status == AgentStatus.ACTIVE]

    def list_all(self) -> list[AgentCard]:
        with self._lock:
            return list(self._agents.values())

    def set_status(self, agent_id: str, status: AgentStatus) -> bool:
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].status = status
                return True
            return False

    # ── Convenience ───────────────────────────────────────────────────────────

    def by_capability(self, capability: str) -> list[AgentCard]:
        """Filter active agents that expose a specific capability tag."""
        with self._lock:
            return [
                a for a in self._agents.values()
                if a.status == AgentStatus.ACTIVE and capability in a.capabilities
            ]

    def stats(self) -> dict:
        with self._lock:
            total  = len(self._agents)
            active = sum(1 for a in self._agents.values() if a.status == AgentStatus.ACTIVE)
            return {
                "total": total,
                "active": active,
                "inactive": total - active,
                "agents": [{"id": a.agent_id, "name": a.name, "status": a.status}
                           for a in self._agents.values()],
            }

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={len(self._agents)})"


# ── Module-level singleton (shared across the app) ────────────────────────────
registry = AgentRegistry()
