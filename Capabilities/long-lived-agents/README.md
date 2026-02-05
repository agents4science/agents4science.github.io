# Stage 6: Long-Lived Autonomous Agents

**Agents that persist for days to months, not minutes.**

## Task

Run agents that persist for days to months, maintaining state, memory, and goals across sessions and failures.

## Why This Matters

Scientific campaigns unfold over weeks or months. Long-lived agents accumulate knowledge, adapt strategies, and pursue open-ended discovery—fundamentally different from stateless scripts or single-session tools.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | Lifecycle management, persistent state, failure recovery |
| **Where it runs** | Any (with durable storage) |
| **Scale** | Days to months of continuous operation |
| **Status** | Emerging |

## Architecture

<img src="/Capabilities/Assets/long-lived-arch.svg" alt="Long-lived agents: Lifecycle management with Checkpoint, Recovery, Oversight" style="max-width: 480px; margin: 1rem 0;">

## Code

| Example | Description |
|---------|-------------|
| [AgentsCheckpoint](/Capabilities/long-lived-agents/AgentsCheckpoint/) | Checkpoint/resume patterns for workflows that span sessions |
