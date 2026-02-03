# AgentsExample

### Multi-Agent Dashboard Demo

A demonstration of multiple agents working on scientific goals with a full-screen Rich dashboard showing live progress.

**Source code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsExample)

## Features

- Full-screen Rich dashboard with per-agent, per-goal live updates
- Configurable scientific goals in `agents4science/workflows/goals.yaml`
- Adjustable pacing via environment variables
- Optional Argonne inference service integration

## The Application

Five specialized agents collaborate on scientific goals defined in YAML:

| Agent | Role |
|-------|------|
| **Scout** | Identifies research opportunities |
| **Planner** | Designs workflows and allocates resources |
| **Operator** | Executes workflows safely |
| **Analyst** | Summarizes results and quantifies uncertainty |
| **Archivist** | Records provenance for reproducibility |

Goals are defined in `agents4science/workflows/goals.yaml` and include examples like:
- "Find catalysts that improve CO₂ conversion at room temperature"
- "Identify polymer compositions with high thermal stability"
- "Discover metal-organic frameworks for hydrogen storage"

## Quick Start

```bash
cd Capabilities/AgentsExample
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export A4S_LATENCY=0.4
export A4S_TOOL_LATENCY=0.2
python main.py
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `A4S_LATENCY` | Delay between agent steps (seconds) | 0.4 |
| `A4S_TOOL_LATENCY` | Delay for tool execution (seconds) | 0.2 |
| `A4S_UI` | Enable dashboard UI (0 to disable) | 1 |
| `A4S_USE_INFERENCE` | Use Argonne inference service | 0 |

## Run with Argonne Inference Service

```bash
A4S_USE_INFERENCE=1 python main.py
```

## Directory Structure

```
AgentsExample/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── agents4science/
    ├── base_agent.py            # Base agent class
    ├── roles/                   # Agent implementations
    │   ├── scout.py
    │   ├── planner.py
    │   ├── operator.py
    │   ├── analyst.py
    │   └── archivist.py
    ├── tools/                   # Tool implementations
    │   ├── execution.py
    │   ├── analysis.py
    │   └── provenance.py
    └── workflows/
        └── goals.yaml           # Scientific goals
```
