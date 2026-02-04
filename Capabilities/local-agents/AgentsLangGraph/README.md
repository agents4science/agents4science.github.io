# Simple LangGraph Example

This example demonstrates how to build a multi-agent pipeline for scientific discovery using [LangGraph](https://langchain-ai.github.io/langgraph/). Five specialized agents work in sequence to tackle a research goal, with each agent contributing its expertise before passing results to the next.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsLangGraph](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/local-agents/AgentsLangGraph)

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

The agents use OpenAI models as their LLM. An OpenAI key is required to run the example.

**Requirements:** Python 3.10+, LangGraph 1.0+, LangChain 1.0+

## Why LangGraph?

LangGraph provides several advantages over plain LangChain for multi-agent workflows:

- **Typed State**: State is defined with TypedDict, providing clear contracts between nodes
- **Graph-Based Flow**: Workflows are defined as directed graphs, making complex flows easy to visualize and modify
- **Streaming**: Built-in support for streaming execution, enabling real-time progress updates
- **Checkpointing**: Native support for persistence and resumption of workflows
- **Cycles**: Unlike simple chains, graphs can include loops and conditional branching

## Implementation

The code uses LangGraph's `StateGraph` for workflow definition. State is passed between agent nodes:

```python
from typing import TypedDict, Annotated
from operator import add

class PipelineState(TypedDict):
    goal: str
    scout_output: str
    planner_output: str
    operator_output: str
    analyst_output: str
    archivist_output: str
    messages: Annotated[list[str], add]  # Accumulates across nodes
```

Each agent is a node function that receives state and returns updates:

```python
def scout_node(state: PipelineState) -> dict:
    chain = _create_chain("You are the Scout agent...")
    output = chain.invoke({"input": state["goal"]})
    return {
        "scout_output": output,
        "messages": ["Scout: Identified opportunities"]
    }
```

The graph defines the flow between agents:

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(PipelineState)

# Add nodes
graph.add_node("scout", scout_node)
graph.add_node("planner", planner_node)
graph.add_node("operator", operator_node)
graph.add_node("analyst", analyst_node)
graph.add_node("archivist", archivist_node)

# Define edges
graph.add_edge(START, "scout")
graph.add_edge("scout", "planner")
graph.add_edge("planner", "operator")
graph.add_edge("operator", "analyst")
graph.add_edge("analyst", "archivist")
graph.add_edge("archivist", END)

# Compile to executable
app = graph.compile()
```

Running the pipeline with streaming:

```python
for event in app.stream(initial_state):
    for node_name in event:
        print(f"{node_name} completed")
```

## Directory Structure

```
AgentsLangGraph/
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── pipeline/
    ├── state.py           # PipelineState TypedDict
    ├── nodes.py           # Agent node functions
    ├── graph.py           # StateGraph definition
    └── tools/
        └── analysis.py    # analyze_dataset tool
```

## Running the Example

```bash
cd Capabilities/local-agents/AgentsLangGraph
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
python main.py
```

Custom goal:

```bash
python main.py --goal "Design a catalyst for ammonia synthesis"
```

Quiet mode (less output):

```bash
python main.py --quiet
```

## Comparison with LangChain Version

| Aspect | AgentsLangChain | AgentsLangGraph |
|--------|-----------------|-----------------|
| Workflow definition | Manual loop | StateGraph |
| State management | Dict passed between calls | Typed PipelineState |
| Execution | Sequential iteration | Graph traversal with streaming |
| Flow control | Python code | Graph edges (supports cycles) |
| Persistence | Manual | Built-in checkpointing |
| Visualization | N/A | Graph can be rendered |

LangGraph is particularly useful when workflows need:
- Conditional branching based on agent outputs
- Cycles for iterative refinement
- Persistence across sessions
- Complex multi-agent coordination

## See Also

- [AgentsLangChain](/Capabilities/local-agents/AgentsLangChain/) — Simpler LangChain version
- [AgentsAcademy](/Capabilities/local-agents/AgentsAcademy/) — Academy framework version (no LLM required)
