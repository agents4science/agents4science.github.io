# Stage 5: Multi-Agent Coordination

**Many agents under shared governance—within one institution or across many.**

## Task

Execute hundreds or thousands of agents simultaneously with shared budgets, quotas, and policies. Enable agents owned by different institutions to cooperate under shared protocols.

## Why This Matters

Large-scale discovery requires specialized agents working together. Coordination prevents conflicts, ensures fair resource allocation, and enables cross-institutional collaboration—a core DOE differentiator.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | Shared state, policy engine, budget ledger |
| **Where it runs** | Distributed (single institution or federated) |
| **Scale** | O(10²–10³) concurrent agents |
| **Status** | Emerging |

## Architecture

<img src="/Capabilities/Assets/multi-agent-coord.svg" alt="Multi-agent coordination: Shared State, Policy, and Budget controlling multiple agents" style="max-width: 480px; margin: 1rem 0;">

## Code

| Example | Description |
|---------|-------------|
| [AgentsCoordination](/Capabilities/multi-agent-coordination/AgentsCoordination/) | Shared budget, state, and policy across agents |
| [AgentsLangGraph](/Capabilities/local-agents/AgentsLangGraph/) | 5-agent pipeline with LangGraph orchestration |
| [AgentsAcademy](/Capabilities/local-agents/AgentsAcademy/) | 5-agent pipeline with Academy messaging |
