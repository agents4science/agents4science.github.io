# Academy Dashboard Example

A Rich dashboard wrapper for the [AgentsAcademy](../AgentsAcademy/) multi-agent pipeline. Shows live progress as five specialized agents work through multiple scientific goals.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademyDashboard](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademyDashboard)

## Features

- Full-screen Rich dashboard with live status updates
- Top panel: Goal × Agent status matrix
- Bottom panel: Current agent output
- Configurable goals via YAML
- Headless mode for scripting (`A4S_UI=0`)
- Uses Academy's `Manager` for agent orchestration

## The Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Academy Multi-Agent Pipeline                      │
├─────────────────────────────────────┬───────┬───────┬───────┬───────┤
│ Goal                                │ Scout │Planner│Operator│...   │
├─────────────────────────────────────┼───────┼───────┼───────┼───────┤
│ Find catalysts for CO2 conversion...│  Done │Running│Pending │...   │
│ Design biodegradable polymer...     │Pending│Pending│Pending │...   │
└─────────────────────────────────────┴───────┴───────┴───────┴───────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Planner Output                                                      │
│                                                                     │
│ Workflow Plan                                                       │
│ PHASE 1: Computational Screening (2 compute nodes, 24h)...          │
└─────────────────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cd Capabilities/AgentsAcademyDashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom goals:

```bash
python main.py --goals my_goals.yaml
```

Headless mode (no UI):

```bash
A4S_UI=0 python main.py
```

## How It Works

This example imports the agents from the simple [AgentsAcademy](../AgentsAcademy/) example and wraps them with a Rich dashboard:

```python
# Import agents from the simple example
sys.path.insert(0, "../AgentsAcademy")
from pipeline.roles import ScoutAgent, PlannerAgent, ...

# Build Rich dashboard
layout = Layout()
layout.split_column(
    Layout(name="status", ratio=3),   # Goal × Agent matrix
    Layout(name="output", ratio=2),   # Current output
)

# Run agents with live updates via Academy Manager
with Live(layout, console=console) as live:
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        for goal in goals:
            for agent_class in AGENT_CLASSES:
                handle = await manager.launch(agent_class)
                result = await handle.act(state)
                live.refresh()
```

## Directory Structure

```
AgentsAcademyDashboard/
├── main.py              # Dashboard entry point
├── goals.yaml           # Scientific goals
├── requirements.txt     # Dependencies (adds rich, pyyaml)
└── README.md
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `A4S_UI` | Enable dashboard (0 to disable) | 1 |

## See Also

- [AgentsAcademy](../AgentsAcademy/) — Simple version without dashboard
- [AgentsLangChainDashboard](../AgentsLangChainDashboard/) — LangChain version with dashboard
