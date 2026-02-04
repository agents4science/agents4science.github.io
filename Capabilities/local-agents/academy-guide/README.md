# Getting Started with Academy

A step-by-step guide to building distributed scientific agents with the Academy framework.

## What is Academy?

**Academy** is an agent framework designed for distributed, federated scientific computing. Unlike frameworks focused on LLM reasoning chains, Academy excels at:

- **Distributed execution**: Agents can run on different machines
- **Secure communication**: Message passing with identity and policy
- **HPC integration**: Built for DOE supercomputers and scientific infrastructure
- **Long-running workflows**: Persistent agents that span hours to months

## When to Use Academy vs LangGraph

| Use Case | Recommended Framework |
|----------|----------------------|
| LLM reasoning with tools | LangGraph |
| Multi-step AI workflows | LangGraph |
| Distributed agents across machines | Academy |
| HPC job orchestration | Academy |
| Cross-institutional collaboration | Academy |
| Both LLM reasoning AND distribution | Academy + LangGraph (Hybrid) |

## Learning Path

This guide walks you through Academy concepts in order of complexity:

```
1. AgentsAcademyBasic     → Core concepts: Agent, @action, Handle, Manager
        ↓
2. AgentsRemoteTools      → Pattern: Coordinator calling remote tools
        ↓
3. AgentsHybrid           → Combining Academy with LangGraph
        ↓
4. AgentsPersistent       → Checkpointing and resuming workflows
        ↓
5. AgentsFederated        → Cross-institutional collaboration
        ↓
6. federated-agents/      → Production deployment on HPC
```

---

## Stage 1: Core Concepts (AgentsAcademyBasic)

**Goal**: Understand the fundamental building blocks of Academy.

### Key Concepts

#### 1. Agents are Classes

```python
from academy.agent import Agent, action

class CalculatorAgent(Agent):
    """An agent that performs calculations."""

    @action
    async def calculate(self, expression: str) -> str:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
```

#### 2. Actions are Remote-Callable Methods

The `@action` decorator marks methods that can be called by other agents or the main process:

```python
@action
async def calculate(self, expression: str) -> str:
    # This method can be called remotely
    return str(eval(expression))
```

#### 3. Manager Launches Agents

The Manager is the Academy runtime that launches and manages agents:

```python
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:
    # Launch an agent - returns a Handle (proxy)
    calculator = await manager.launch(CalculatorAgent)

    # Call actions on the agent via its Handle
    result = await calculator.calculate("2 + 2")
```

#### 4. Handles Enable Communication

When you launch an agent, you get a Handle (proxy) that lets you call its actions:

```python
# Launch returns a Handle
calculator = await manager.launch(CalculatorAgent)

# Use the Handle to call actions
result = await calculator.calculate("42 * 17")
```

Handles can be passed to other agents, enabling agent-to-agent communication.

### Try It

```bash
cd Capabilities/local-agents/AgentsAcademyBasic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### What You Learned

- Agents are Python classes inheriting from `Agent`
- `@action` makes methods callable remotely
- `Manager` launches agents and returns `Handle`s
- Handles are proxies for calling agent actions

---

## Stage 2: Remote Tools Pattern (AgentsRemoteTools)

**Goal**: Learn how one agent can call tools provided by another agent.

### The Pattern

```
+---------------+                    +------------------+
|  Coordinator  |                    |  ToolProvider    |
|               |  -- call action -> |                  |
|  orchestrates |                    |  run_simulation()|
|  workflow     |  <- return result- |  analyze_data()  |
+---------------+                    +------------------+
```

### Key Concepts

#### Passing Handles Between Agents

```python
class CoordinatorAgent(Agent):
    def __init__(self):
        self._tool_provider: Handle | None = None

    @action
    async def set_tool_provider(self, provider: Handle) -> None:
        """Receive a Handle to another agent."""
        self._tool_provider = provider

    @action
    async def execute_workflow(self, task: str) -> dict:
        """Use the tool provider to do work."""
        result = await self._tool_provider.run_simulation({"temp": 300})
        return result
```

#### Wiring Agents Together

```python
# Launch both agents
tool_provider = await manager.launch(ToolProviderAgent)
coordinator = await manager.launch(CoordinatorAgent)

# Pass the tool provider's Handle to the coordinator
await coordinator.set_tool_provider(tool_provider)

# Now coordinator can call tool_provider's actions
result = await coordinator.execute_workflow("optimize catalyst")
```

### Why This Matters

In production, the ToolProvider could run on:
- An HPC login node
- A lab instrument computer
- A cloud VM with GPUs

The Coordinator doesn't need to know where—Academy handles the communication.

### Try It

```bash
cd Capabilities/local-agents/AgentsRemoteTools
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### What You Learned

- Agents can receive Handles to other agents via actions
- Agents can call actions on agents they have Handles to
- This enables the Coordinator/ToolProvider pattern
- Same code works locally or distributed

---

## Stage 3: Hybrid Academy + LangGraph (AgentsHybrid)

**Goal**: Combine Academy's distribution with LangGraph's LLM reasoning.

### When to Go Hybrid

- You need LLM-powered decision making
- AND you need distributed execution
- AND you want agents on different machines

### The Pattern

```python
from langgraph.prebuilt import create_react_agent

class ResearcherAgent(Agent):
    def __init__(self):
        self._langgraph_agent = None

    def _init_langgraph(self):
        """Initialize LangGraph for LLM reasoning."""
        llm = ChatOpenAI(model="gpt-4o-mini")
        tools = [search_literature, analyze_results]
        self._langgraph_agent = create_react_agent(llm, tools)

    @action
    async def research(self, task: str) -> dict:
        """Use LangGraph for intelligent research."""
        self._init_langgraph()

        # LangGraph handles LLM reasoning
        result = self._langgraph_agent.invoke({
            "messages": [HumanMessage(content=task)]
        })

        return {"findings": result["messages"][-1].content}
```

### Division of Labor

| Component | Responsibility |
|-----------|---------------|
| **Academy** | Agent lifecycle, messaging, distribution |
| **LangGraph** | LLM reasoning, tool calling, state |

### Try It

```bash
cd Capabilities/local-agents/AgentsHybrid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py              # Mock mode
python main.py --llm openai # With real LLM
```

### What You Learned

- Academy agents can use LangGraph internally
- Academy handles distribution; LangGraph handles intelligence
- Same agent code works with mock or real LLMs

---

## Stage 4: Persistent State (AgentsPersistent)

**Goal**: Build workflows that checkpoint progress and resume after restarts.

### Why Persistence Matters

Scientific workflows can run for hours, days, or weeks. You need:
- Checkpointing after each major step
- Resume from last checkpoint on restart
- Graceful handling of interruptions

### The Pattern

```python
class AgentState:
    """Manages persistent state for an agent."""

    def __init__(self, agent_name: str, state_dir: Path):
        self.state_file = state_dir / f"{agent_name}_state.json"
        self._load()  # Load existing state if present

    def set(self, key: str, value) -> None:
        """Set value and persist to disk."""
        self._state[key] = value
        self._save()

class WorkflowAgent(Agent):
    @action
    async def execute_workflow(self, task: str) -> dict:
        # Resume from checkpoint
        current_step = self._state.get("current_step", 0)

        for i, step in enumerate(steps):
            if i < current_step:
                continue  # Skip completed steps

            result = await step()

            # Checkpoint progress
            self._state.set("current_step", i + 1)
```

### Try It

```bash
cd Capabilities/local-agents/AgentsPersistent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run workflow
python main.py

# Interrupt with Ctrl+C, then resume
python main.py  # Continues from checkpoint!

# Start fresh
python main.py --reset
```

### What You Learned

- State can be checkpointed to disk as JSON
- Agents can resume from last checkpoint on restart
- Essential for long-running scientific workflows

---

## Stage 5: Federated Collaboration (AgentsFederated)

**Goal**: Coordinate agents across institutional boundaries.

### The Vision

```
                    +------------------+
                    |   Coordinator    |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
+------------------+ +------------------+ +------------------+
|       ANL        | |       ORNL       | |       LBNL       |
| (Aurora compute) | |  (SNS/HFIR data) | |   (ML analysis)  |
+------------------+ +------------------+ +------------------+
```

### The Pattern

```python
class ANLComputeAgent(Agent):
    """Argonne's compute capabilities."""

    @action
    async def run_simulation(self, params: dict) -> dict:
        # Run on Aurora
        return {"energy": -127.5, "status": "completed"}

class FederatedCoordinator(Agent):
    @action
    async def execute_federated_workflow(self, task: str) -> dict:
        # Query data from ORNL
        data = await self._agents["ORNL"].query_database(task)

        # Run simulation at ANL
        sim = await self._agents["ANL"].run_simulation(data)

        # Analyze at LBNL
        analysis = await self._agents["LBNL"].analyze_data(sim)

        return {"data": data, "sim": sim, "analysis": analysis}
```

### Try It

```bash
cd Capabilities/local-agents/AgentsFederated
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### What You Learned

- Agents can represent institutional capabilities
- Coordinator pattern works across institutions
- Same code works locally (demo) or distributed (production)

---

## Going to Production

After mastering these examples, you're ready for production deployment:

### Next Steps

| Stage | What to Learn |
|-------|---------------|
| [Federated Agents](/Capabilities/federated-agents/) | Deploy on DOE HPC systems |
| [Governed Tool Use](/Capabilities/governed-tool-use/) | Add policy enforcement |
| [Multi-Agent Coordination](/Capabilities/multi-agent-coordination/) | Coordinate many agents |
| [Long-Lived Agents](/Capabilities/long-lived-agents/) | Agents that run for months |

### Production Checklist

- [ ] Replace `LocalExchangeFactory` with production exchange
- [ ] Configure federated identity (Globus Auth)
- [ ] Set up policy enforcement
- [ ] Add monitoring and logging
- [ ] Implement error recovery
- [ ] Set up state persistence (database or object store)

---

## Quick Reference

### Minimal Agent

```python
from academy.agent import Agent, action
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

class MyAgent(Agent):
    @action
    async def do_something(self, input: str) -> str:
        return f"Processed: {input}"

async def main():
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        agent = await manager.launch(MyAgent)
        result = await agent.do_something("hello")
        print(result)
```

### Agent with Handle Communication

```python
class WorkerAgent(Agent):
    @action
    async def work(self, task: str) -> str:
        return f"Completed: {task}"

class BossAgent(Agent):
    def __init__(self):
        self._worker: Handle | None = None

    @action
    async def set_worker(self, worker: Handle) -> None:
        self._worker = worker

    @action
    async def delegate(self, task: str) -> str:
        return await self._worker.work(task)

# Wire them together
worker = await manager.launch(WorkerAgent)
boss = await manager.launch(BossAgent)
await boss.set_worker(worker)
result = await boss.delegate("important task")
```

### Common Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Coordinator + Tools** | Orchestrate remote compute | AgentsRemoteTools |
| **Pipeline** | Sequential processing | AgentsAcademy |
| **Hub-and-Spoke** | Central control | AgentsAcademyHubSpoke |
| **Hybrid** | LLM + distribution | AgentsHybrid |
| **Federated** | Cross-institutional | AgentsFederated |

---

## Resources

- [Academy GitHub](https://github.com/academy-agents/academy)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [All Examples Index](/Capabilities/)
