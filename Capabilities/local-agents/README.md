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

## A Five-Agent Scientific Discovery Pipeline

We use this simple example to demonstrate multi-agent coordination for scientific workflows. Five specialized agents work in sequence, each contributing domain expertise before passing results to the next:

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Scout** | Surveys problem space, detects anomalies | Goal | Research opportunities |
| **Planner** | Designs workflows, allocates resources | Opportunities | Workflow plan |
| **Operator** | Executes the planned workflow safely | Plan | Execution results |
| **Analyst** | Summarizes findings, quantifies uncertainty | Results | Analysis summary |
| **Archivist** | Documents everything for reproducibility | Summary | Documented provenance |

<img src="/Capabilities/Assets/5agents.png" alt="5-agent pipeline diagram" style="width: 60%; margin: 1rem 0;">

### Two Example Implementations

We provide implementations of this simple example in LangGraph and Academy.
Note that these implementations are toys: they create agents that communicate, but each agent's internal logic is just a stub. 

| Example | Framework | Features | Code |
|---------|-----------|----------|------|
| [AgentsLangChain](/Frameworks/AgentsLangChain/) | LangChain | Simple, LLM-powered | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChain) |
| [AgentsAcademy](/Frameworks/AgentsAcademy/) | Academy | Simple, no LLM required | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsAcademy) |

The simple versions are minimal implementations ideal for learning.


### Implementations with Dashboards

The dashboard versions wrap the same agents with a full-screen Rich UI showing live progress across multiple scientific goals.

| Example | Framework | Features | Code |
|---------|-----------|----------|------|
| [AgentsLangChainDashboard](/Frameworks/AgentsLangChainDashboard/) | LangChain | Rich dashboard, multi-goal | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsLangChainDashboard) |
| [AgentsAcademyDashboard](/Frameworks/AgentsAcademyDashboard/) | Academy | Rich dashboard, multi-goal | [View](https://github.com/agents4science/agents4science.github.io/tree/main/Frameworks/AgentsAcademyDashboard) |


---
