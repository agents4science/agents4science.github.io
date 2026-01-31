# Simple LangChain Example

This example demonstrates how to build a multi-agent pipeline for scientific discovery using [LangChain](https://www.langchain.com/). Five specialized agents work in sequence to tackle a research goal, with each agent contributing its expertise before passing results to the next.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChain](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChain)

## The Application

The pipeline addresses a sample scientific goal: *"Find catalysts that improve CO₂ conversion at room temperature."*

The workflow proceeds through five stages:

| Agent | Role |
|-------|------|
| **Scout** | Surveys the problem space, identifies anomalies, proposes research opportunities |
| **Planner** | Designs workflows, allocates resources, sequences the work |
| **Operator** | Executes the planned workflow safely |
| **Analyst** | Summarizes findings, quantifies uncertainty (uses `analyze_dataset` tool) |
| **Archivist** | Documents everything for reproducibility |

Each agent implementation is just a skeleton.

The agents use OpenAI models as their LLM. An OpenAI key is required to run the example.

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

The main loop passes each agent's output as input to the next:

```python
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
cd Frameworks/AgentsLangChain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
python main_langchain.py
```
