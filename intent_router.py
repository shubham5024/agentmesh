"""
AgentMesh Intent Router
RAG-based routing: embeds the incoming task, finds the nearest agent
description via FAISS cosine similarity, returns a RouteDecision.

Design decision: embedding over agent *descriptions* (not just names) gives
semantic matching — "write a poem" correctly routes to the CreativeAgent even
without an explicit "poem" capability tag.
"""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from agentmesh.models import AgentCard, RouteDecision

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Optional heavy deps — graceful degradation to keyword fallback ─────────────
try:
    import faiss                                        # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss / sentence-transformers not installed — using keyword fallback router")


class IntentRouter:
    """
    Semantic intent router backed by a FAISS flat-IP index.

    Each call to `build_index` re-embeds all active agent descriptions.
    In production you'd trigger this on registry change events; here we
    rebuild on every route call if the registry has changed (cheap for ≤50 agents).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality

    def __init__(self) -> None:
        self._model      = None
        self._index      = None
        self._agent_ids: list[str] = []
        self._index_hash = -1       # Track registry changes

        if _FAISS_AVAILABLE:
            logger.info("Loading sentence-transformer model: %s", self.MODEL_NAME)
            self._model = SentenceTransformer(self.MODEL_NAME)

    # ── Index management ──────────────────────────────────────────────────────

    def build_index(self, agents: list[AgentCard]) -> None:
        """Embed agent descriptions and build FAISS index."""
        if not _FAISS_AVAILABLE or not self._model:
            return

        if not agents:
            self._index     = None
            self._agent_ids = []
            return

        descriptions = [f"{a.name}: {a.description}" for a in agents]
        embeddings   = self._model.encode(descriptions, normalize_embeddings=True)
        embeddings   = np.array(embeddings, dtype="float32")

        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatIP(dim)   # Inner product on L2-normalised = cosine
        self._index.add(embeddings)
        self._agent_ids = [a.agent_id for a in agents]
        logger.debug("Built FAISS index with %d agents (dim=%d)", len(agents), dim)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, task: str, agents: list[AgentCard], top_k: int = 3) -> RouteDecision:
        """
        Route a task string to the best matching agent.

        Returns a RouteDecision with confidence score and top-k alternatives.
        Falls back to keyword matching when FAISS is unavailable.
        """
        if not agents:
            raise ValueError("No active agents available for routing")

        t0 = time.perf_counter()

        if _FAISS_AVAILABLE and self._model:
            decision = self._semantic_route(task, agents, top_k)
        else:
            decision = self._keyword_route(task, agents)

        ms = (time.perf_counter() - t0) * 1000
        logger.info("Routed '%s...' → %s (%.1f ms, conf=%.3f)",
                    task[:60], decision.agent_name, ms, decision.confidence)
        return decision

    def _semantic_route(self, task: str, agents: list[AgentCard], top_k: int) -> RouteDecision:
        # Rebuild index if agent list changed
        current_hash = hash(tuple(a.agent_id for a in agents))
        if current_hash != self._index_hash or self._index is None:
            self.build_index(agents)
            self._index_hash = current_hash

        query_vec = self._model.encode([task], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        k         = min(top_k, len(agents))
        scores, idxs = self._index.search(query_vec, k)

        best_idx  = int(idxs[0][0])
        best_score = float(scores[0][0])
        best_agent = next(a for a in agents if a.agent_id == self._agent_ids[best_idx])

        alternatives = [
            {"agent_id": self._agent_ids[int(idxs[0][i])],
             "confidence": float(scores[0][i])}
            for i in range(1, k)
        ]

        return RouteDecision(
            agent_id     = best_agent.agent_id,
            agent_name   = best_agent.name,
            confidence   = best_score,
            reasoning    = (
                f"Semantic similarity {best_score:.3f} between task and "
                f"'{best_agent.name}' description"
            ),
            alternatives = alternatives,
        )

    def _keyword_route(self, task: str, agents: list[AgentCard]) -> RouteDecision:
        """Simple keyword overlap fallback — no ML dependencies needed."""
        task_lower = task.lower()
        scores: list[tuple[float, AgentCard]] = []

        for agent in agents:
            text   = f"{agent.name} {agent.description} {' '.join(agent.capabilities)}".lower()
            words  = set(task_lower.split())
            hits   = sum(1 for w in words if w in text)
            score  = hits / max(len(words), 1)
            scores.append((score, agent))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_agent = scores[0]

        return RouteDecision(
            agent_id     = best_agent.agent_id,
            agent_name   = best_agent.name,
            confidence   = best_score,
            reasoning    = f"Keyword overlap score {best_score:.2f}",
            alternatives = [
                {"agent_id": a.agent_id, "confidence": s}
                for s, a in scores[1:3]
            ],
        )
