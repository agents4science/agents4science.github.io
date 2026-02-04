# Academy Pipeline Example

This example demonstrates a **true pipeline pattern** using [Academy](https://academy-agents.org), where agents forward results directly to each other via messaging. The main process only sets up the pipeline and triggers the first agent.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAcademy](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAcademy)

## The Pipeline Pattern

```
Main Process                    Agent-to-Agent Messaging
     │
     │ 1. Launch agents         Scout ──▶ Planner ──▶ Operator ──▶ Analyst ──▶ Archivist
     │ 2. Connect pipeline           │          │           │          │           │
     │ 3. Trigger Scout              └──────────┴───────────┴──────────┴───────────┘
     │                                          results collected
     ▼
   [wait for completion]
```

After setup, the main process steps back. Agents communicate directly:
- Scout processes the goal, forwards result to Planner
- Planner processes, forwards to Operator
- ...and so on until Archivist signals completion

## The Application

Five specialized agents work in sequence on a scientific goal:

| Agent | Role | Receives From | Sends To |
|-------|------|---------------|----------|
| **Scout** | Surveys problem space, detects anomalies | Main process | Planner |
| **Planner** | Designs workflows, allocates resources | Scout | Operator |
| **Operator** | Executes the planned workflow safely | Planner | Analyst |
| **Analyst** | Summarizes findings, quantifies uncertainty | Operator | Archivist |
| **Archivist** | Documents everything for reproducibility | Analyst | (end) |

No LLM is used in this example—agent logic is stubbed to focus on the messaging pattern.

## Implementation

The base agent class handles pipeline forwarding:

```python
class ScienceAgent(Agent):
    _next_agent: Handle | None = None

    @action
    async def set_next(self, next_agent: Handle) -> None:
        """Set the next agent in the pipeline."""
        self._next_agent = next_agent

    @action
    async def process(self, input_text: str) -> None:
        """Process input and forward to next agent."""
        result = await self._process(input_text)

        # Forward to next agent in pipeline
        if self._next_agent:
            await self._next_agent.process(result)
```

The main process connects agents and triggers the pipeline:

```python
async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:

    # Launch agents
    scout = await manager.launch(ScoutAgent)
    planner = await manager.launch(PlannerAgent)
    # ...

    # Connect pipeline
    await scout.set_next(planner)
    await planner.set_next(operator)
    # ...

    # Trigger - from here, agents communicate directly
    await scout.process(goal)
```

## Directory Structure

```
AgentsAcademy/
├── main.py                    # Pipeline setup and trigger
├── requirements.txt           # Dependencies
└── pipeline/
    ├── base_agent.py          # ScienceAgent with forwarding
    └── roles/                 # Agent implementations
        ├── scout.py
        ├── planner.py
        ├── operator.py
        ├── analyst.py
        └── archivist.py
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsAcademy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom goal:

```bash
python main.py --goal "Design a catalyst for ammonia synthesis"
```

## Comparison with LangGraph

| Aspect | LangGraph | Academy |
|--------|-----------|---------|
| Pattern | Graph-based orchestration | True pipeline (agent-to-agent) |
| Control flow | StateGraph with edges | Agents forward via messaging |
| State | Typed PipelineState dict | Passed between agents directly |
| LLM | Required (powers agents) | Optional (not used in this example) |
| Distribution | Single process | Supports federated execution |

Academy's pipeline pattern is particularly useful when:
- Agents need to run on different machines
- You want explicit agent-to-agent communication
- LLM integration is optional or handled separately
