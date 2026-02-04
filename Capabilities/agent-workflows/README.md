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

```
+-----------------------------------------+
|              Agent Layer                |
|                                         |
|         +-----------------+             |
|         |  Planning Agent |             |
|         +--------+--------+             |
|                  | constructs           |
+------------------+----------------------+
                   v
+-----------------------------------------+
|           Workflow Layer                |
|                                         |
| +------+    +------+    +------+        |
| |Step 1|--->|Step 2|--->|Step 3| ...    |
| +------+    +------+    +------+        |
|                                         |
| (Parsl, Globus Flows, etc.)             |
+-----------------------------------------+
```

## Code

| Example | Description |
|---------|-------------|
| [AgentsWorkflow](/Capabilities/agent-workflows/AgentsWorkflow/) | Dynamic workflow construction with adaptive execution |
