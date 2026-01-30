# Stage 6: Autonomous Scientific Systems

## Task

Long-running agents that autonomously pursue scientific goals over weeks or months, operating continuously under governance constraints.

## Why This Matters

The end goal: agents as scientific collaborators, not just tools. Persistent agents accumulate knowledge, adapt strategies based on results, and pursue open-ended discovery—while remaining safe and aligned.

## Architecture

```
┌─────────────────────────────────────────┐
│          Governance Envelope            │
│                                         │
│  Scope ─────── What agent CAN do        │
│  Boundaries ── What agent CANNOT do     │
│  Budgets ───── Resource limits          │
│  Oversight ─── Monitoring + intervention│
│  Shutdown ──── Safe termination         │
│                                         │
│         ┌─────────────────┐             │
│         │  Autonomous     │             │
│         │  Agent          │             │
│         │  (persistent)   │             │
│         └─────────────────┘             │
└─────────────────────────────────────────┘
```

## Code

Future — design in progress
