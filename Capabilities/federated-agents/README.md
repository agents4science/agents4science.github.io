# Stage 2: Federated Agent Execution

**Cross-institutional agent execution under federated identity and policy.**

## Task

Execute agentic applications that invoke tools and workflows on DOE HPC resources under federated identity and policy.

## Why This Matters

Scientific workflows often span multiple facilities. Academy provides secure, auditable tool invocation across institutional boundaries—agents authenticate once and access resources anywhere in the federation.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | LangGraph, Academy |
| **Where it runs** | DOE HPC systems (Polaris, Aurora, Perlmutter, Frontier) |
| **Scale** | Multi-agent, multi-resource |
| **Status** | Mature |

## Architecture

```
┌────────────┐          ┌─────────────────────┐
│ Agent Host │          │    DOE HPC System   │
│            │          │                     │
│ ┌───────┐  │  secure  │ ┌───────┐ ┌──────┐ │
│ │ Agent │──┼─────────▶│ │Academy│─▶│Tools │ │
│ └───┬───┘  │ federated│ └───────┘ └──────┘ │
│     │      │ identity │                     │
│     ▼      │          │ ┌─────────────────┐ │
│ ┌───────┐  │          │ │ Compute Nodes   │ │
│ │  LLM  │  │          │ └─────────────────┘ │
│ └───────┘  │          └─────────────────────┘
└────────────┘
```

## Code

- [CharacterizeChemicals](/Capabilities/CharacterizeChemicals/) — Molecular property calculations via RDKit and xTB
