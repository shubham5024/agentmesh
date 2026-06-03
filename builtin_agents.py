"""
AgentMesh — Built-in Agents
Three ready-to-use agents demonstrating the framework's extensibility.

  ResearchAgent   – web search + summarisation
  CodeAgent       – code generation + review
  SummarizerAgent – document/text summarisation

Each agent is ~30 lines — the framework handles routing, retries, and A2A.
"""
from __future__ import annotations
import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agentmesh.agents.base_agent import AgentState, BaseAgent
from agentmesh.models import AgentCard

logger = logging.getLogger(__name__)

_LLM = None

def _get_llm() -> ChatOpenAI:
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model       = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature = 0.2,
            api_key     = os.getenv("OPENAI_API_KEY", ""),
        )
    return _LLM


# ── Research Agent ────────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """
    Answers research questions by decomposing them into sub-queries,
    retrieving context (mocked here; swap in a real search tool), and
    synthesising a cited response.
    """

    _CARD = AgentCard(
        agent_id     = "agent-research-001",
        name         = "ResearchAgent",
        description  = (
            "Researches topics by breaking questions into sub-queries, "
            "gathering information from multiple sources, and synthesising "
            "a structured, cited answer. Best for factual lookup, literature "
            "review, and information gathering tasks."
        ),
        capabilities = ["web_search", "summarisation", "citation", "fact_checking"],
        endpoint     = "http://localhost:8001/invoke",
    )

    @property
    def card(self) -> AgentCard:
        return self._CARD

    def _plan(self, state: AgentState) -> AgentState:
        state["plan"] = [
            "Decompose task into 2-3 focused sub-queries",
            "Retrieve relevant context for each sub-query",
            "Synthesise into a structured response with citations",
        ]
        return state

    def _execute(self, state: AgentState) -> AgentState:
        llm = _get_llm()
        messages = [
            SystemMessage(content=(
                "You are a research assistant. Break the task into sub-queries, "
                "simulate retrieving relevant information, and synthesise a clear, "
                "structured response. Include [Source: ...] citations."
            )),
            HumanMessage(content=f"Research task: {state['task']}"),
        ]
        response = llm.invoke(messages)
        state["result"] = response.content
        state["observations"].append("LLM synthesis complete")
        return state


# ── Code Agent ────────────────────────────────────────────────────────────────

class CodeAgent(BaseAgent):
    """
    Generates, reviews, and explains code. Supports multiple languages.
    """

    _CARD = AgentCard(
        agent_id     = "agent-code-002",
        name         = "CodeAgent",
        description  = (
            "Writes, reviews, debugs, and explains code across Python, "
            "JavaScript, SQL, and other languages. Ideal for code generation, "
            "bug fixing, refactoring, unit test writing, and code explanation tasks."
        ),
        capabilities = ["code_generation", "code_review", "debugging",
                        "unit_tests", "refactoring"],
        endpoint     = "http://localhost:8002/invoke",
    )

    @property
    def card(self) -> AgentCard:
        return self._CARD

    def _plan(self, state: AgentState) -> AgentState:
        state["plan"] = [
            "Identify language and intent from task",
            "Generate code with inline comments",
            "Add usage example and edge-case notes",
        ]
        return state

    def _execute(self, state: AgentState) -> AgentState:
        llm = _get_llm()
        messages = [
            SystemMessage(content=(
                "You are a senior software engineer. Write clean, well-commented code. "
                "Always include: the solution, inline comments, a usage example, "
                "and brief notes on edge cases or limitations."
            )),
            HumanMessage(content=f"Coding task: {state['task']}"),
        ]
        response = llm.invoke(messages)
        state["result"] = response.content
        state["observations"].append("Code generation complete")
        return state


# ── Summarizer Agent ──────────────────────────────────────────────────────────

class SummarizerAgent(BaseAgent):
    """
    Condenses long documents or conversations into structured summaries.
    """

    _CARD = AgentCard(
        agent_id     = "agent-summarizer-003",
        name         = "SummarizerAgent",
        description  = (
            "Summarises long documents, articles, meeting notes, and chat "
            "histories into concise, structured formats. Supports bullet-point, "
            "executive, and narrative summary styles. Best for content compression "
            "and key-point extraction tasks."
        ),
        capabilities = ["summarisation", "key_point_extraction",
                        "document_analysis", "meeting_notes"],
        endpoint     = "http://localhost:8003/invoke",
    )

    @property
    def card(self) -> AgentCard:
        return self._CARD

    def _plan(self, state: AgentState) -> AgentState:
        style = state["payload"].get("style", "bullet")
        state["plan"] = [
            f"Identify content type and select '{style}' summary format",
            "Extract key themes, decisions, and action items",
            "Format into requested summary structure",
        ]
        return state

    def _execute(self, state: AgentState) -> AgentState:
        llm   = _get_llm()
        style = state["payload"].get("style", "bullet")
        content = state["payload"].get("content", state["task"])

        messages = [
            SystemMessage(content=(
                f"You are a summarisation expert. Create a '{style}' style summary. "
                "For bullet: use clear bullet points grouped by theme. "
                "For executive: 3-5 sentence paragraph for senior stakeholders. "
                "For narrative: flowing prose preserving key context."
            )),
            HumanMessage(content=f"Summarise the following:\n\n{content}"),
        ]
        response = llm.invoke(messages)
        state["result"] = response.content
        state["observations"].append(f"Summary complete ({style} style)")
        return state


# ── Registry helper ───────────────────────────────────────────────────────────

def get_default_agents() -> list[BaseAgent]:
    """Return all built-in agents ready for registration."""
    return [ResearchAgent(), CodeAgent(), SummarizerAgent()]
