# Stage 2: Federated Agent Execution

**Cross-institutional agent execution under federated identity and policy.**

## Task

Execute agentic applications that invoke tools and workflows on DOE HPC resources under federated identity and policy.

## Why This Matters

Scientific workflows often span multiple facilities. Academy provides secure, auditable tool invocation across institutional boundaries—agents authenticate once and access resources anywhere in the federation.

## Prerequisites

Before diving into federated execution, ensure you're comfortable with:

- [AgentsCalculator](/Capabilities/local-agents/AgentsCalculator/) - Basic LangGraph agent with tools
- [AgentsAcademyBasic](/Capabilities/local-agents/AgentsAcademyBasic/) - Academy fundamentals
- [AgentsRemoteTools](/Capabilities/local-agents/AgentsRemoteTools/) - Remote tool invocation pattern
- [AgentsFederated](/Capabilities/local-agents/AgentsFederated/) - Simulated cross-institutional collaboration

## Details

| Aspect | Value |
|--------|-------|
| **Technologies** | LangGraph, Academy |
| **Where it runs** | DOE HPC systems (Polaris, Aurora, Perlmutter, Frontier) |
| **Scale** | Multi-agent, multi-resource |
| **Status** | Mature |

## Architecture

```
+---------------+              +------------------------+
|  Agent Host   |              |    DOE HPC System      |
|               |              |                        |
|  +---------+  |    secure    |  +---------+           |
|  |  Agent  |--+------------->|  | Academy |           |
|  +----+----+  |  federated   |  +----+----+           |
|       |       |   identity   |       |                |
|       v       |              |       v                |
|  +---------+  |              |  +---------+  +-----+  |
|  |   LLM   |  |              |  |  Tools  |->| HPC |  |
|  +---------+  |              |  +---------+  +-----+  |
+---------------+              +------------------------+
```

## Examples

| Example | Tech | Description | Key Pattern |
|---------|------|-------------|-------------|
| [AgentsHPCJob](/Capabilities/federated-agents/AgentsHPCJob/) | LangGraph | Submit and monitor batch jobs | Job lifecycle management |
| [CharacterizeChemicals](/Capabilities/federated-agents/CharacterizeChemicals/) | Academy | Molecular properties via RDKit + xTB | LLM-planned workflows |

## Comparison: Local vs Federated

| Aspect | Local (Stage 1) | Federated (Stage 2) |
|--------|-----------------|---------------------|
| **Where tools run** | Same machine as agent | Remote HPC systems |
| **Identity** | Local user | Federated (Globus Auth) |
| **Scale** | Single machine | Multi-facility |
| **Use case** | Development, testing | Production science |

## Next Steps

After mastering federated execution:

- [Governed Tool Use](/Capabilities/governed-tool-use/) - Add policy enforcement
- [Multi-Agent Coordination](/Capabilities/multi-agent-coordination/) - Coordinate multiple agents
