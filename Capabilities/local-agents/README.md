# Stage 1: Local Agent Execution

**Your on-ramp to CAF. No federation required.**

## Task

Implement and run a persistent, stateful agentic application on a laptop or workstation, using local or remote LLMs.

## Why This Matters

Local execution lets you develop and test agent logic before deploying to HPC. LangGraph specifications are reproducible and portable—the same agent definition runs locally or at scale.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | LangGraph |
| **Where it runs** | Laptop, workstation, VM |
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

## Code

- [AgentsLangChain](/Frameworks/AgentsLangChain/) — 5-agent pipeline example
- [AgentsExample](/Frameworks/AgentsExample/) — Dashboard demo
- [CharacterizeChemicals](/Frameworks/CharacterizeChemicals/) — Molecular property agent
