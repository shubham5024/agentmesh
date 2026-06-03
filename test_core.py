"""
AgentMesh Test Suite
pytest tests for the registry and intent router.
Run: pytest tests/ -v
"""
import pytest
from agentmesh.models import AgentCard, AgentStatus, RouteDecision
from agentmesh.registry.agent_registry import AgentRegistry
from agentmesh.router.intent_router import IntentRouter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_cards() -> list[AgentCard]:
    return [
        AgentCard(
            agent_id     = "a1",
            name         = "ResearchAgent",
            description  = "Researches topics, gathers information, and synthesises answers",
            capabilities = ["web_search", "summarisation"],
            endpoint     = "http://localhost:8001/invoke",
        ),
        AgentCard(
            agent_id     = "a2",
            name         = "CodeAgent",
            description  = "Writes, reviews, and debugs code across multiple languages",
            capabilities = ["code_generation", "debugging", "unit_tests"],
            endpoint     = "http://localhost:8002/invoke",
        ),
        AgentCard(
            agent_id     = "a3",
            name         = "SummarizerAgent",
            description  = "Summarises long documents and extracts key points",
            capabilities = ["summarisation", "key_point_extraction"],
            endpoint     = "http://localhost:8003/invoke",
        ),
    ]


@pytest.fixture
def populated_registry(sample_cards) -> AgentRegistry:
    reg = AgentRegistry()
    for card in sample_cards:
        reg.register(card)
    return reg


# ── Registry Tests ────────────────────────────────────────────────────────────

class TestAgentRegistry:

    def test_register_and_retrieve(self, sample_cards):
        reg  = AgentRegistry()
        card = sample_cards[0]
        reg.register(card)
        assert reg.get(card.agent_id) == card

    def test_register_returns_card(self, sample_cards):
        reg    = AgentRegistry()
        result = reg.register(sample_cards[0])
        assert isinstance(result, AgentCard)

    def test_list_active_excludes_inactive(self, populated_registry, sample_cards):
        populated_registry.set_status(sample_cards[0].agent_id, AgentStatus.INACTIVE)
        active = populated_registry.list_active()
        assert len(active) == 2
        assert all(a.status == AgentStatus.ACTIVE for a in active)

    def test_deregister(self, populated_registry, sample_cards):
        agent_id = sample_cards[0].agent_id
        assert populated_registry.deregister(agent_id) is True
        assert populated_registry.get(agent_id) is None

    def test_deregister_nonexistent_returns_false(self, populated_registry):
        assert populated_registry.deregister("nonexistent-id") is False

    def test_by_capability(self, populated_registry):
        code_agents = populated_registry.by_capability("code_generation")
        assert len(code_agents) == 1
        assert code_agents[0].name == "CodeAgent"

    def test_by_capability_multiple_matches(self, populated_registry):
        summarisers = populated_registry.by_capability("summarisation")
        assert len(summarisers) == 2

    def test_stats(self, populated_registry):
        stats = populated_registry.stats()
        assert stats["total"]  == 3
        assert stats["active"] == 3

    def test_len(self, populated_registry):
        assert len(populated_registry) == 3

    def test_update_existing_agent(self, populated_registry, sample_cards):
        updated = sample_cards[0].model_copy(update={"description": "Updated description"})
        populated_registry.register(updated)
        assert populated_registry.get(updated.agent_id).description == "Updated description"
        assert len(populated_registry) == 3  # No duplicate


# ── Router Tests ──────────────────────────────────────────────────────────────

class TestIntentRouter:

    def test_router_returns_route_decision(self, sample_cards):
        router   = IntentRouter()
        decision = router.route("Find information about transformer models", sample_cards)
        assert isinstance(decision, RouteDecision)
        assert decision.agent_id in {"a1", "a2", "a3"}
        assert 0.0 <= decision.confidence <= 1.0

    def test_router_keyword_fallback_code_task(self, sample_cards):
        router   = IntentRouter()
        decision = router._keyword_route("write a python function to sort a list", sample_cards)
        assert decision.agent_id == "a2"   # CodeAgent

    def test_router_keyword_fallback_research_task(self, sample_cards):
        router   = IntentRouter()
        decision = router._keyword_route("research the latest papers on RAG systems", sample_cards)
        assert decision.agent_id == "a1"   # ResearchAgent

    def test_router_keyword_fallback_summarise_task(self, sample_cards):
        router   = IntentRouter()
        decision = router._keyword_route("summarise this document", sample_cards)
        # Both research and summarizer have summarisation; summarizer score should be highest
        assert decision.agent_id in {"a1", "a3"}

    def test_router_raises_on_empty_agents(self):
        router = IntentRouter()
        with pytest.raises(ValueError, match="No active agents"):
            router.route("some task", [])

    def test_alternatives_count(self, sample_cards):
        router   = IntentRouter()
        decision = router._keyword_route("write code to search the web", sample_cards)
        assert len(decision.alternatives) <= 2   # top_k - 1
