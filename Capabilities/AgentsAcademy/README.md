# Simple Academy Example

This example demonstrates how to build a multi-agent pipeline for scientific discovery using [Academy](https://academy-agents.org). Five specialized agents work in sequence to tackle a research goal, with each agent contributing its expertise before passing results to the next.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademy](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademy)

## The Application

The pipeline addresses a sample scientific goal: *"Find catalysts that improve CO2 conversion at room temperature."*

The workflow proceeds through five stages:

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Scout** | Surveys the problem space, identifies anomalies | Goal | Research opportunities |
| **Planner** | Designs workflows, allocates resources | Opportunities | Workflow plan |
| **Operator** | Executes the planned workflow safely | Plan | Execution results |
| **Analyst** | Summarizes findings, quantifies uncertainty | Results | Analysis summary |
| **Archivist** | Documents everything for reproducibility | Summary | Documented provenance |

Each agent implementation is just a skeleton.

The agents use Academy's `Agent` base class and `@action` decorator. No LLM is required to run this example.

## Implementation

The code uses Academy's `Agent` class, `@action` decorator for exposed methods, and `Manager` for agent orchestration. Each agent inherits from a common `ScienceAgent` base:

```python
from academy.agent import Agent, action

class ScienceAgent(Agent):
    name: str = "ScienceAgent"
    role: str = "Base agent"

    @action
    async def act(self, input_text: str) -> dict:
        result = await self._process(input_text)
        return {"agent": self.name, "output": result}
```

Specialized agents override `_process` to implement domain-specific logic:

```python
class ScoutAgent(ScienceAgent):
    name = "Scout"
    role = "Survey problem space, detect anomalies"

    async def _process(self, goal: str) -> str:
        # Identify research opportunities
        return f"Opportunities for: {goal}"
```

The main loop uses Academy's `Manager` to launch and orchestrate agents:

```python
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:

    state = "Find catalysts that improve CO2 conversion..."

    for agent_class in [ScoutAgent, PlannerAgent, ...]:
        handle = await manager.launch(agent_class)
        result = await handle.act(state)
        state = result["output"]
```

## Directory Structure

```
AgentsAcademy/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── pipeline/
    ├── base_agent.py          # ScienceAgent base class
    ├── roles/                 # Agent implementations
    │   ├── scout.py
    │   ├── planner.py
    │   ├── operator.py
    │   ├── analyst.py
    │   └── archivist.py
    └── tools/
        └── analysis.py        # analyze_dataset tool
```

## Running the Example

```bash
cd Capabilities/AgentsAcademy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom goal:

```bash
python main.py --goal "Design a catalyst for ammonia synthesis"
```

## Comparison with LangChain Version

| Aspect | AgentsLangChain | AgentsAcademy |
|--------|-----------------|---------------|
| Framework | LangChain + LangGraph | Academy |
| Agent base | `AgentExecutor` | `Agent` class |
| Tool definition | `@tool` decorator | `@action` decorator |
| Orchestration | Manual loop | `Manager` + exchange |
| LLM required | Yes (OpenAI) | No (logic in agents) |
| Execution model | Single process | Supports distributed |

Academy provides additional capabilities for federated execution across DOE systems when configured with `HttpExchangeFactory` instead of `LocalExchangeFactory`.
