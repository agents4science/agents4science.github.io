# Stage 1: Local Agent Execution

**Your on-ramp to CAF. No federation required.**

## Task

Implement and run a simple, persistent, stateful agentic application on a laptop or workstation, using local or remote LLMs.

## Why This Matters

Local execution lets you develop and test agent logic before deploying to HPC. LangGraph and Academy specifications are reproducible and portable—the same agent definition runs locally or at scale.

## Details

| Aspect | Value |
|--------|-------|
| **Technologies** | LangGraph, Academy |
| **Where code runs** | Laptop, workstation, VM |
| **Scale** | Single agent / small multi-agent |
| **Status** | Mature |

## Architecture

Here we deal with agentic applications in which one or more agents operate entirely locally, each potentially calling tools and/or LLMs.


```
┌─────────────────────────────┐
│        Workstation          │
│                             │
│  ┌───────┐    ┌───────┐    │
│  │ Agent │───▶│ Tools │    │
│  └───┬───┘    └───────┘    │
│      │                      │
│      ▼                      │
│  ┌───────┐                  │
│  │  LLM  │ (API or local)   │
│  └───────┘                  │
└─────────────────────────────┘
```

## Minimal Example: Calculator Agent

The simplest possible agent—an LLM that can use a calculator tool:

| Example | Framework | Code |
|---------|-----------|------|
| [AgentsCalculator](/Capabilities/AgentsCalculator/) | LangChain + LangGraph | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsCalculator) |

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [calculate])
agent.invoke({"messages": [HumanMessage(content="What is 347 * 892?")]})
```

## Five-Agent Scientific Discovery Pipeline

We use this simple example to demonstrate multi-agent coordination for scientific workflows. Five specialized agents work in sequence, each contributing domain expertise before passing results to the next:

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Scout** | Surveys problem space, detects anomalies | Goal | Research opportunities |
| **Planner** | Designs workflows, allocates resources | Opportunities | Workflow plan |
| **Operator** | Executes the planned workflow safely | Plan | Execution results |
| **Analyst** | Summarizes findings, quantifies uncertainty | Results | Analysis summary |
| **Archivist** | Documents everything for reproducibility | Summary | Documented provenance |

<img src="/Capabilities/Assets/5agents.png" alt="5-agent pipeline diagram" style="width: 60%; margin: 1rem 0;">

### Example Implementations

We provide implementations of this simple example in LangGraph, LangChain, and Academy.
Note that these implementations are toys: they create agents that communicate, but each agent's internal logic is just a stub.

| Example | Framework | Features | Code |
|---------|-----------|----------|------|
| [AgentsLangGraph](/Capabilities/AgentsLangGraph/) | LangGraph | Graph-based, typed state, LLM-powered | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangGraph) |
| [AgentsLangChain](/Capabilities/AgentsLangChain/) | LangChain | Simple, LLM-powered | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChain) |
| [AgentsAcademy](/Capabilities/AgentsAcademy/) | Academy | Simple, no LLM required | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademy) |

The simple versions are minimal implementations ideal for learning.


### Implementations with Dashboards

The dashboard versions wrap the same agents with a full-screen Rich UI showing live progress across multiple scientific goals.

| Example | Framework | Features | Code |
|---------|-----------|----------|------|
| [AgentsLangChainDashboard](/Capabilities/AgentsLangChainDashboard/) | LangChain | Rich dashboard, multi-goal | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsLangChainDashboard) |
| [AgentsAcademyDashboard](/Capabilities/AgentsAcademyDashboard/) | Academy | Rich dashboard, multi-goal | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/AgentsAcademyDashboard) |


---
