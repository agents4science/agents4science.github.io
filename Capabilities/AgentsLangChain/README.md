# Simple LangChain Example

This example demonstrates how to build a multi-agent pipeline for scientific discovery using [LangChain](https://www.langchain.com/). Five specialized agents work in sequence to tackle a research goal, with each agent contributing its expertise before passing results to the next.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChain](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChain)

## The Application

The pipeline addresses a sample scientific goal: *"Find catalysts that improve CO₂ conversion at room temperature."*

The workflow proceeds through five stages:

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Scout** | Surveys the problem space, identifies anomalies | Goal | Research opportunities |
| **Planner** | Designs workflows, allocates resources | Opportunities | Workflow plan |
| **Operator** | Executes the planned workflow safely | Plan | Execution results |
| **Analyst** | Summarizes findings, quantifies uncertainty | Results | Analysis summary |
| **Archivist** | Documents everything for reproducibility | Summary | Documented provenance |

Each agent implementation is just a skeleton.

The agents use OpenAI models as their LLM. An OpenAI key is required to run the example.

**Requirements:** Python 3.10+, LangChain 1.0+

## Implementation

The code uses LangChain's `ChatOpenAI` for LLM access, `@tool` decorator for tool definitions, and `AgentExecutor` for agent execution. Each agent is defined with a name, role description, and optional tools:

```python
from pipeline.agent import LangAgent
from pipeline.tools.analysis import analyze_dataset

agents = [
    LangAgent("Scout", "Detect anomalies and propose opportunities..."),
    LangAgent("Planner", "Design workflows and allocate resources..."),
    LangAgent("Operator", "Execute workflows safely..."),
    LangAgent("Analyst", "Summarize results...", tools=[analyze_dataset]),
    LangAgent("Archivist", "Record provenance..."),
]
```

Each agent is a `LangAgent` wrapping LangChain's `AgentExecutor`:

```python
LangAgent(
    "Analyst",
    "Summarize results and analyze datasets; use tools when appropriate.",
    tools=[analyze_dataset]
)
```

The main loop is straightforward—each agent acts on the goal, and its output becomes input for the next:

```python
goal = "Find catalysts that improve CO₂ conversion at room temperature."
agents = build_roles()

state = {"goal": goal}
for agent in agents:
    output = agent.act(state["goal"])
    state["goal"] = output
```

## Directory Structure

```
AgentsLangChain/
├── main_langchain.py      # Entry point
├── requirements.txt       # Dependencies
└── pipeline/
    ├── agent.py           # LangAgent class wrapping LangChain
    ├── roles.py           # Defines the five agents
    └── tools/
        └── analysis.py    # analyze_dataset tool
```

## Running the Example

```bash
cd Capabilities/AgentsLangChain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
python main_langchain.py
```
