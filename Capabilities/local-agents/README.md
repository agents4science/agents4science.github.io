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

## Tutorial Examples

- Simple LangChain example


## Code

- The [AgentsLangChain](/Frameworks/AgentsLangChain/) code uses LangGraph to implement a simple 5-agent pipeline example 
- [AgentsExample](/Frameworks/AgentsExample/) — Dashboard demo
- [CharacterizeChemicals](/Frameworks/CharacterizeChemicals/) — Molecular property agent implemented with Academy
