# Stage 7: Agent-Mediated Scientific Workflows

**Agents that build workflows, not just execute them.**

## Task

Allow agents to dynamically construct, adapt, and execute scientific workflows—bridging agentic AI with existing workflow systems.

## Why This Matters

Traditional workflows are static. Agent-mediated workflows adapt to results, handle exceptions intelligently, and integrate naturally with DOE's existing workflow infrastructure (Parsl, Globus Flows, etc.).

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | Workflow integration layer |
| **Where it runs** | DOE infrastructure |
| **Scale** | Varies by workflow |
| **Status** | Early |

## Architecture

<img src="/Capabilities/Assets/agent-workflows.svg" alt="Agent workflows: Planning Agent constructs workflow steps" style="max-width: 480px; margin: 1rem 0;">

## Code

| Example | Description |
|---------|-------------|
| [AgentsWorkflow](/Capabilities/agent-workflows/AgentsWorkflow/) | Dynamic workflow construction with adaptive execution |

## Prerequisites

Before building workflow agents:

- [AgentsLangGraph](/Capabilities/local-agents/AgentsLangGraph/) — StateGraph patterns for multi-step workflows
- [AgentsCheckpoint](/Capabilities/long-lived-agents/AgentsCheckpoint/) — Checkpoint/resume for long-running workflows

## Integration with Existing Systems

Agent-mediated workflows complement (not replace) existing tools:

| System | How Agents Help |
|--------|-----------------|
| **Parsl** | Agents decide what to run; Parsl handles HPC execution |
| **Globus Flows** | Agents construct flows dynamically; Globus executes reliably |
| **Prefect/Airflow** | Agents adapt workflows at runtime; orchestrators manage scheduling |

## Related Topics

- [Long-Lived Agents](/Capabilities/long-lived-agents/) — Workflows that span days or weeks
- [Governed Tool Use](/Capabilities/governed-tool-use/) — Policy enforcement for workflow steps
- [Federated Agents](/Capabilities/federated-agents/) — Cross-institutional workflow execution
