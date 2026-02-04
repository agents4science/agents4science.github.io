# Academy Hub-and-Spoke Example

This example demonstrates a **hub-and-spoke pattern** using [Academy](https://academy-agents.org), where the main process orchestrates all agents sequentially. Agents don't communicate directly with each other.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAcademyHubSpoke](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsAcademyHubSpoke)

## The Hub-and-Spoke Pattern

```
Main Process (Hub)
     │
     ├──▶ Scout ────▶ result
     │                  │
     ├──▶ Planner ◀────┘
     │       │
     │       ▼ result
     ├──▶ Operator ◀───┘
     │       │
     │       ▼ result
     ├──▶ Analyst ◀────┘
     │       │
     │       ▼ result
     └──▶ Archivist ◀──┘
             │
             ▼
         [done]
```

The main process:
- Calls each agent sequentially
- Passes results from one agent to the next
- Remains in control throughout the workflow

## The Application

Five specialized agents work in sequence on a scientific goal:

| Agent | Role | Called By | Result Used By |
|-------|------|-----------|----------------|
| **Scout** | Surveys problem space, detects anomalies | Main | Planner |
| **Planner** | Designs workflows, allocates resources | Main | Operator |
| **Operator** | Executes the planned workflow safely | Main | Analyst |
| **Analyst** | Summarizes findings, quantifies uncertainty | Main | Archivist |
| **Archivist** | Documents everything for reproducibility | Main | (end) |

No LLM is used in this example—agent logic is stubbed to focus on the messaging pattern.

## Implementation

The base agent class returns results to the caller:

```python
class ScienceAgent(Agent):
    @action
    async def process(self, input_text: str) -> str:
        """Process input and return result to main process."""
        result = await self._process(input_text)
        return result
```

The main process orchestrates all agents:

```python
async with await Manager.from_exchange_factory(
    factory=LocalExchangeFactory(),
) as manager:

    # Launch agents
    scout = await manager.launch(ScoutAgent)
    planner = await manager.launch(PlannerAgent)
    # ...

    # Main process orchestrates each step
    scout_result = await scout.process(goal)
    planner_result = await planner.process(scout_result)
    operator_result = await operator.process(planner_result)
    # ...
```

## Directory Structure

```
AgentsAcademyHubSpoke/
├── main.py                    # Workflow orchestration
├── requirements.txt           # Dependencies
└── pipeline/
    ├── base_agent.py          # ScienceAgent returning results
    └── roles/                 # Agent implementations
        ├── scout.py
        ├── planner.py
        ├── operator.py
        ├── analyst.py
        └── archivist.py
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsAcademyHubSpoke
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom goal:

```bash
python main.py --goal "Design a catalyst for ammonia synthesis"
```

## Comparison: Hub-and-Spoke vs Pipeline

| Aspect | Hub-and-Spoke (this example) | Pipeline (AgentsAcademy) |
|--------|------------------------------|--------------------------|
| Control | Main process orchestrates | Agents forward to each other |
| Communication | All through main process | Direct agent-to-agent |
| Simplicity | Simpler to understand | More setup required |
| Flexibility | Easy to add conditional logic | Fixed flow after setup |
| Distribution | All agents must be reachable from main | Agents can be distributed |

Hub-and-spoke is useful when:
- You need conditional logic between steps
- The main process needs to inspect intermediate results
- You want simpler, more explicit control flow
