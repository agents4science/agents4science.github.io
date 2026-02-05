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
