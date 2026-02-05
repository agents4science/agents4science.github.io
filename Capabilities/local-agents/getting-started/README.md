# Getting Started: Building Scientific Agents

Production scientific agents need two capabilities:

1. **Intelligence** — LLM-powered reasoning, tool calling, and decision making
2. **Distribution** — Running across machines, institutions, and federated infrastructure

No single framework does both well. That's why we use two complementary frameworks:

| Framework | What It Does | Strengths |
|-----------|--------------|-----------|
| **LangGraph** | LLM reasoning and tool orchestration | ReAct patterns, state management, tool calling |
| **Academy** | Distributed agent execution | Cross-machine messaging, federation, HPC integration |

<img src="/Capabilities/Assets/stack-diagram.svg" alt="LangGraph + Academy stack diagram" style="max-width: 500px; margin: 1.5rem 0;">

## Learning Path

<img src="/Capabilities/Assets/learning-path.svg" alt="Learning path: LLM Agents → Distributed → Production" style="max-width: 700px; margin: 1rem 0;">

This guide walks you through both frameworks, building toward production agents:

| Stage | Guide | What You Learn |
|-------|-------|----------------|
| 1 | [LLM Agents](1-llm-agents.md) | Build agents that reason and call tools (LangGraph) |
| 2 | [Distributed Agents](2-distributed-agents.md) | Run agents across machines (Academy) |
| 3 | [Production Agents](3-production-agents.md) | Combine LangGraph + Academy for real deployments |

**Time investment:** Each stage builds on the previous. Plan to work through them in order.

---

## Stage 1: LLM Agents (LangGraph)

Learn to build agents that can reason and use tools.

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [calculate])
agent.invoke({"messages": [HumanMessage(content="What is 347 * 892?")]})
```

**Key concepts:** Tools, ReAct loop, state graphs, memory

**[Start Stage 1 →](1-llm-agents.md)**

---

## Stage 2: Distributed Agents (Academy)

Learn to run agents across machines and pass messages between them.

```python
from academy.agent import Agent, action
from academy.manager import Manager

class ComputeAgent(Agent):
    @action
    async def run_simulation(self, params: dict) -> dict:
        # Could run on HPC, lab instrument, cloud...
        return {"energy": -127.5, "status": "completed"}

async with await Manager.from_exchange_factory(factory) as manager:
    compute = await manager.launch(ComputeAgent)
    result = await compute.run_simulation({"temp": 300})
```

**Key concepts:** Agent classes, @action methods, Handles, Manager

**[Start Stage 2 →](2-distributed-agents.md)**

---

## Stage 3: Production Agents (LangGraph + Academy)

Combine both: LLM reasoning inside distributed Academy agents.

```python
class ResearchAgent(Agent):
    """Academy agent with LangGraph intelligence."""

    @action
    async def research(self, task: str) -> dict:
        # LangGraph handles reasoning
        result = self._langgraph_agent.invoke({
            "messages": [HumanMessage(content=task)]
        })
        return {"findings": result["messages"][-1].content}
```

**Key patterns:**
- Academy handles distribution, messaging, federation
- LangGraph handles LLM reasoning inside each agent
- Same code runs locally or across DOE infrastructure

**[Start Stage 3 →](3-production-agents.md)**

---

## Quick Decision Guide

**"I just want to experiment with LLM agents locally"**
→ Start with [Stage 1](1-llm-agents.md) (LangGraph only)

**"I need agents on HPC/federated infrastructure but no LLM yet"**
→ Start with [Stage 2](2-distributed-agents.md) (Academy only)

**"I need LLM-powered agents on DOE infrastructure"**
→ Work through all three stages, ending with [Stage 3](3-production-agents.md)

---

## Examples by Stage

| Stage | Examples | Pattern |
|-------|----------|---------|
| 1 | [Calculator](/Capabilities/local-agents/AgentsCalculator/), [RAG](/Capabilities/local-agents/AgentsRAG/), [Database](/Capabilities/local-agents/AgentsDatabase/), [API](/Capabilities/local-agents/AgentsAPI/) | LangGraph tools |
| 1 | [Conversation](/Capabilities/local-agents/AgentsConversation/), [LangGraph Pipeline](/Capabilities/local-agents/AgentsLangGraph/) | LangGraph state |
| 2 | [AcademyBasic](/Capabilities/local-agents/AgentsAcademyBasic/), [RemoteTools](/Capabilities/local-agents/AgentsRemoteTools/) | Academy messaging |
| 2 | [Persistent](/Capabilities/local-agents/AgentsPersistent/), [Federated](/Capabilities/local-agents/AgentsFederated/) | Academy patterns |
| 3 | [Hybrid](/Capabilities/local-agents/AgentsHybrid/) | LangGraph + Academy |
| 3 | [HPC Job](/Capabilities/federated-agents/AgentsHPCJob/), [CharacterizeChemicals](/Capabilities/federated-agents/CharacterizeChemicals/) | Production deployment |

---

## Prerequisites

- Python 3.10+
- For Stage 1: `pip install langchain langchain-openai langgraph`
- For Stage 2: `pip install academy-py`
- For Stage 3: Both of the above

**LLM access:** Examples support OpenAI, Ollama, and FIRST backends. See [LLM Configuration](../llm-configuration/) for setup. All examples include mock mode for testing without API keys.
