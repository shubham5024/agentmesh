# AgentMesh 🕸️

**A production-grade A2A Multi-Agent Orchestration Framework**

[![CI](https://github.com/YOUR_USERNAME/agentmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentmesh/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AgentMesh lets you register independent AI agents via URL endpoints and orchestrate them through a single runtime — with semantic intent routing, standardised A2A inter-agent messaging, and MCP tool integration.

---

## The Problem It Solves

Building multi-agent systems today means writing custom dispatcher logic, hardcoded agent-to-agent API calls, and bespoke routing rules that break every time a new agent is added.

AgentMesh solves this with three primitives:

| Primitive | What it does |
|---|---|
| **Registry** | Agents register by URL — no rewrites, any HTTP service onboards instantly |
| **Intent Router** | FAISS-backed semantic routing sends each task to the right agent in <200ms |
| **A2A Protocol** | Standardised message envelope replaces point-to-point API coupling |

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           AgentMesh Runtime          │
                        │                                      │
  User Request          │  ┌──────────┐   ┌────────────────┐  │
 ──────────────────────►│  │  FastAPI  │──►│ Intent Router  │  │
                        │  │  /invoke  │   │  (FAISS RAG)   │  │
                        │  └──────────┘   └───────┬────────┘  │
                        │                         │            │
                        │                  ┌──────▼──────┐    │
                        │                  │   Registry  │    │
                        │                  └──────┬──────┘    │
                        └─────────────────────────┼───────────┘
                                                  │ A2A Message
                              ┌───────────────────┼───────────────────┐
                              │                   │                   │
                         ┌────▼────┐        ┌─────▼────┐       ┌─────▼──────┐
                         │Research │        │   Code   │       │Summarizer  │
                         │ Agent   │        │  Agent   │       │  Agent     │
                         │(port    │        │(port     │       │(port 8003) │
                         │ 8001)   │        │ 8002)    │       │            │
                         └─────────┘        └──────────┘       └────────────┘
                              │                   │
                              └──────────┬────────┘
                                    ┌────▼─────┐
                                    │   MCP    │
                                    │  Tools   │
                                    │(search,  │
                                    │ github,  │
                                    │  calc)   │
                                    └──────────┘
```

### Request Flow

1. `POST /invoke` receives a natural language task
2. **IntentRouter** embeds the task with `all-MiniLM-L6-v2` and finds the nearest agent description via FAISS cosine similarity
3. **A2AClient** wraps the task in a standardised message envelope and POSTs to the agent's registered endpoint
4. The **agent** runs a LangGraph `plan → execute → respond` workflow and returns a result
5. Response includes routing metadata (confidence, alternatives, latency)

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/agentmesh.git
cd agentmesh
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
uvicorn agentmesh.api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 4. Invoke an agent

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain how transformer attention mechanisms work",
    "context": {"user_id": "u_123"}
  }'
```

**Response:**
```json
{
  "request_id": "f3a1...",
  "routed_to": "ResearchAgent",
  "result": "Transformer attention mechanisms work by...",
  "route": {
    "agent_id": "agent-research-001",
    "agent_name": "ResearchAgent",
    "confidence": 0.847,
    "reasoning": "Semantic similarity 0.847 between task and 'ResearchAgent' description"
  },
  "total_ms": 142.3,
  "status": "success"
}
```

### With Docker

```bash
docker compose up
```

---

## Adding a Custom Agent

Extend `BaseAgent` — the framework handles routing, retries, and A2A delivery:

```python
from agentmesh.agents.base_agent import AgentState, BaseAgent
from agentmesh.models import AgentCard

class MyAgent(BaseAgent):

    _CARD = AgentCard(
        agent_id     = "my-agent-001",
        name         = "MyAgent",
        description  = "Describe what tasks this agent handles — used for semantic routing",
        capabilities = ["my_capability"],
        endpoint     = "http://localhost:8004/invoke",
    )

    @property
    def card(self) -> AgentCard:
        return self._CARD

    def _plan(self, state: AgentState) -> AgentState:
        state["plan"] = ["Step 1", "Step 2"]
        return state

    def _execute(self, state: AgentState) -> AgentState:
        state["result"] = f"Handled: {state['task']}"
        return state
```

Then register it:

```bash
curl -X POST http://localhost:8000/agents/register \
  -H "Content-Type: application/json" \
  -d '{"name": "MyAgent", "description": "...", "capabilities": [...], "endpoint": "..."}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/invoke` | Route task to best agent |
| `GET` | `/agents` | List registered agents |
| `POST` | `/agents/register` | Register a new agent |
| `DELETE` | `/agents/{id}` | Deregister an agent |
| `PATCH` | `/agents/{id}/status` | Update agent status |
| `GET` | `/tools` | List MCP tools |
| `POST` | `/tools/{name}` | Invoke an MCP tool |
| `GET` | `/health` | Health + registry stats |

Full interactive docs at `http://localhost:8000/docs`

---

## Project Structure

```
agentmesh/
├── agentmesh/
│   ├── api/
│   │   └── main.py              # FastAPI app, all endpoints
│   ├── registry/
│   │   └── agent_registry.py    # Thread-safe agent CRUD
│   ├── router/
│   │   └── intent_router.py     # FAISS semantic routing
│   ├── a2a/
│   │   └── protocol.py          # A2A message dispatch + retries
│   ├── agents/
│   │   ├── base_agent.py        # LangGraph-powered base class
│   │   └── builtin_agents.py    # ResearchAgent, CodeAgent, SummarizerAgent
│   ├── mcp/
│   │   └── tool_server.py       # MCP tool registry + dispatcher
│   └── models.py                # All Pydantic schemas
├── tests/
│   └── test_core.py             # Registry + Router test suite
├── .github/workflows/ci.yml     # GitHub Actions CI
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Key Design Decisions

**URL-based registration** — Agents register by endpoint URL, not by importing code. This means any existing HTTP service can join the mesh without modification.

**RAG-based routing** — Embedding agent *descriptions* (not just names) gives semantic matching. A task like "write unit tests" correctly routes to CodeAgent even without an explicit "unit_tests" keyword in the task.

**A2A over direct calls** — All inter-agent communication uses a standardised message envelope. Adding authentication, logging, or rate limiting to all agent calls is a one-line middleware change.

**LangGraph for agent internals** — The `plan → execute → respond` graph gives full observability into each agent's reasoning steps and makes it easy to add branching, retry, or human-in-the-loop nodes.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Roadmap

- [ ] Persistent registry (Redis / PostgreSQL backend)
- [ ] Multi-agent chaining (DAG-based task decomposition)
- [ ] Streaming responses (SSE)
- [ ] Agent health monitoring + auto-deregistration
- [ ] LangSmith tracing integration
- [ ] Real web search MCP tool (SerpAPI / Tavily)

---

## License

MIT
