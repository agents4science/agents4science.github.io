# Stage 3: Scale Agent Inference

## Task

Fan out queries to thousands of parallel LLM instances on HPC for high-throughput inference over large datasets.

## Why This Matters

Scientific applications often require millions of LLM calls (literature mining, molecular screening, hypothesis generation). Running 2000+ concurrent LLM instances turns months of work into hours.

## Architecture

```
┌────────────┐          ┌────────────────────────────┐
│Coordinator │          │     HPC (Aurora)           │
│            │          │                            │
│ ┌───────┐  │  fan out │ ┌─────┐┌─────┐    ┌─────┐ │
│ │ Agent │──┼─────────▶│ │LLM 1││LLM 2│ ···│LLM N│ │
│ └───────┘  │          │ └──┬──┘└──┬──┘    └──┬──┘ │
│            │          │    └──────┴──────────┘    │
│ ┌───────┐  │ results  │            │              │
│ │Aggreg.│◀─┼──────────┼────────────┘              │
│ └───────┘  │          └────────────────────────────┘
└────────────┘
```

## Code

Aurora 2000-node demo — documentation coming soon
