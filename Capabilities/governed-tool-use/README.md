# Stage 4: Governed Tool Use at Scale

**Where most real-world agent failures happen. CAF takes this seriously.**

## Task

Let agents invoke expensive, stateful, or dangerous tools under proactive policy enforcement.

## Why This Matters

Scaling inference is easy—LLM calls are stateless. Scaling tool use is hard because tools have side effects, consume scarce resources, and can cause real-world harm. This is where safety and cost control matter most.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | Academy governance |
| **Where it runs** | DOE HPC systems |
| **Scale** | O(10²–10³) concurrent tool invocations |
| **Status** | Work in progress |

## Architecture

<img src="/Capabilities/Assets/governed-tools.svg" alt="Governed tool use: Governance layer with Policy, Scheduler, Audit Log controlling tool execution" style="max-width: 480px; margin: 1rem 0;">

## Code

| Example | Description |
|---------|-------------|
| [AgentsGovernedTools](/Capabilities/governed-tool-use/AgentsGovernedTools/) | Budget limits, rate limiting, approval gates, and audit logging |

## Prerequisites

Before diving into governed tool use, ensure you're comfortable with:

- [AgentsCalculator](/Capabilities/local-agents/AgentsCalculator/) — Basic LangGraph agent with tools
- [AgentsAcademyBasic](/Capabilities/local-agents/AgentsAcademyBasic/) — Academy fundamentals

## Next Steps

After understanding governed tool use:

- [Multi-Agent Coordination](/Capabilities/multi-agent-coordination/) — Coordinate agents under shared governance
- [Long-Lived Agents](/Capabilities/long-lived-agents/) — Agents that persist for days to months

## Related Topics

- [Federated Agents](/Capabilities/federated-agents/) — Cross-institutional execution (often uses governance)
- [Scale Inference](/Capabilities/scale-inference/) — Parallel LLM calls (different scaling challenge)
