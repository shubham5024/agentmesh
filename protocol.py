"""
AgentMesh A2A Protocol
Standardised Agent-to-Agent communication layer.
Replaces point-to-point API coupling with a contract-based message exchange.

Design decision: Agents communicate via HTTP POST to each other's registered
endpoints using the A2AMessage envelope. This decouples caller from callee —
an agent can be replaced / upgraded without changing any consumer code.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx

from agentmesh.models import A2AMessage, A2AResponse, AgentCard, MessageStatus

if TYPE_CHECKING:
    from agentmesh.registry.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0   # seconds
MAX_RETRIES     = 2


class A2AClient:
    """
    HTTP client that dispatches A2AMessages to remote agent endpoints.
    Handles retries, timeouts, and error normalisation.
    """

    def __init__(self, registry: "AgentRegistry", timeout: float = DEFAULT_TIMEOUT) -> None:
        self._registry = registry
        self._timeout  = timeout

    async def send(self, message: A2AMessage) -> A2AResponse:
        """
        Send a message to the target agent and return a normalised A2AResponse.
        Retries up to MAX_RETRIES on transient failures.
        """
        agent = self._registry.get(message.receiver_id)
        if not agent:
            return A2AResponse(
                message_id = message.message_id,
                agent_id   = message.receiver_id,
                result     = None,
                status     = MessageStatus.FAILED,
                error      = f"Agent '{message.receiver_id}' not found in registry",
            )

        t0 = time.perf_counter()

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._post(agent, message)
                latency  = (time.perf_counter() - t0) * 1000
                logger.info(
                    "A2A %s → %s completed in %.1f ms (attempt %d)",
                    message.sender_id, agent.name, latency, attempt + 1,
                )
                return A2AResponse(
                    message_id = message.message_id,
                    agent_id   = agent.agent_id,
                    result     = response,
                    status     = MessageStatus.SUCCESS,
                    latency_ms = latency,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == MAX_RETRIES:
                    latency = (time.perf_counter() - t0) * 1000
                    return A2AResponse(
                        message_id = message.message_id,
                        agent_id   = agent.agent_id,
                        result     = None,
                        status     = MessageStatus.FAILED,
                        latency_ms = latency,
                        error      = str(exc),
                    )
                wait = 0.5 * (2 ** attempt)
                logger.warning("A2A attempt %d failed, retrying in %.1fs", attempt + 1, wait)
                await asyncio.sleep(wait)

    async def _post(self, agent: AgentCard, message: A2AMessage) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                agent.endpoint,
                json    = message.model_dump(mode="json"),
                headers = {"Content-Type": "application/json", "X-AgentMesh-Version": "1.0"},
            )
            resp.raise_for_status()
            return resp.json()

    # ── Helper: build a message quickly ───────────────────────────────────────

    @staticmethod
    def make_message(
        sender_id:   str,
        receiver_id: str,
        task:        str,
        payload:     dict | None = None,
        context:     dict | None = None,
    ) -> A2AMessage:
        return A2AMessage(
            sender_id   = sender_id,
            receiver_id = receiver_id,
            task        = task,
            payload     = payload or {},
            context     = context or {},
        )
