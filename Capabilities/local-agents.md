# Stage 1: Run Agents Locally

## Task

Implement and run an agentic application on a laptop or workstation, with LLM accessed via API or run locally.

## Why This Matters

Local development enables rapid prototyping and testing before deploying to HPC. You can iterate on agent logic, prompts, and tool integrations without consuming shared resources.

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
