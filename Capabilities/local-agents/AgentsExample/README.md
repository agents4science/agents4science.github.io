# AgentsExample

### Multi-Agent Dashboard Demo

A demonstration of multiple agents working on scientific goals with a full-screen Rich dashboard showing live progress.

**Source code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsExample)

## Features

- Full-screen Rich dashboard with per-agent, per-goal live updates
- Configurable scientific goals in `agents4science/workflows/goals.yaml`
- Adjustable pacing via environment variables
- Multiple LLM backends supported

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
- "Find catalysts that improve CO2 conversion at room temperature"
- "Identify polymer compositions with high thermal stability"
- "Discover metal-organic frameworks for hydrogen storage"

## Quick Start

```bash
cd Capabilities/local-agents/AgentsExample
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## LLM Configuration

The example supports multiple LLM modes:

| Mode | Environment Variable | Description |
|------|---------------------|-------------|
| **OpenAI** | `OPENAI_API_KEY` | Uses OpenAI (gpt-4o-mini by default) |
| **FIRST** | `FIRST_API_KEY` | Uses FIRST HPC inference service |
| **Argonne** | `A4S_USE_INFERENCE=1` | Uses Argonne Inference Service |
| **Mock** | (none) | Uses mock responses for demonstration |

```bash
# OpenAI mode
export OPENAI_API_KEY=<your_key>
python main.py

# FIRST mode (for HPC environments)
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://your-first-endpoint/v1
export FIRST_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
python main.py

# Argonne Inference Service
A4S_USE_INFERENCE=1 python main.py

# Mock mode (no API key needed)
python main.py
```

## Other Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `A4S_LATENCY` | Delay between agent steps (seconds) | 0.4 |
| `A4S_TOOL_LATENCY` | Delay for tool execution (seconds) | 0.2 |
| `A4S_UI` | Enable dashboard UI (0 to disable) | 1 |
| `A4S_MODEL` | Override model name | (depends on mode) |

## Directory Structure

```
AgentsExample/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── agents4science/
    ├── base_agent.py            # Base agent class with LLM mode detection
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
