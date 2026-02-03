# LangChain Dashboard Example

A Rich dashboard wrapper for the [AgentsLangChain](../AgentsLangChain/) multi-agent pipeline. Shows live progress as five specialized agents work through multiple scientific goals.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChainDashboard](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChainDashboard)

## Features

- Full-screen Rich dashboard with live status updates
- Top panel: Goal × Agent status matrix
- Bottom panel: Current agent output
- Configurable goals via YAML
- Headless mode for scripting (`A4S_UI=0`)

## The Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│                  LangChain Multi-Agent Pipeline                     │
├─────────────────────────────────────┬───────┬───────┬───────┬───────┤
│ Goal                                │ Scout │Planner│Operator│...   │
├─────────────────────────────────────┼───────┼───────┼───────┼───────┤
│ Find catalysts for CO2 conversion...│  Done │Running│Pending │...   │
│ Design biodegradable polymer...     │Pending│Pending│Pending │...   │
└─────────────────────────────────────┴───────┴───────┴───────┴───────┘
┌─────────────────────────────────────────────────────────────────────┐
│ Planner Output                                                      │
│                                                                     │
│ Based on the Scout's analysis, I recommend the following workflow:  │
│ 1. Screen candidate materials using DFT calculations...             │
└─────────────────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cd Capabilities/AgentsLangChainDashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
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

This example imports the agents from the simple [AgentsLangChain](../AgentsLangChain/) example and wraps them with a Rich dashboard:

```python
# Import agents from the simple example
sys.path.insert(0, "../AgentsLangChain")
from pipeline.roles import build_roles

# Build Rich dashboard
layout = Layout()
layout.split_column(
    Layout(name="status", ratio=3),   # Goal × Agent matrix
    Layout(name="output", ratio=2),   # Current output
)

# Run agents with live updates
with Live(layout, console=console) as live:
    for goal in goals:
        for agent in agents:
            output = agent.act(state)
            live.refresh()
```

## Directory Structure

```
AgentsLangChainDashboard/
├── main.py              # Dashboard entry point
├── goals.yaml           # Scientific goals
├── requirements.txt     # Dependencies (adds rich, pyyaml)
└── README.md
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `A4S_UI` | Enable dashboard (0 to disable) | 1 |
| `OPENAI_API_KEY` | OpenAI API key | Required |

## See Also

- [AgentsLangChain](../AgentsLangChain/) — Simple version without dashboard
- [AgentsAcademyDashboard](../AgentsAcademyDashboard/) — Academy version with dashboard
