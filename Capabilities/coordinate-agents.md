# Stage 5: Coordinate Many Agents

## Task

Multiple agents collaborate on complex scientific campaigns with shared state, coordinated policies, and resource budgets.

## Why This Matters

Large-scale discovery requires specialized agents working together—scouts find candidates, analysts evaluate them, synthesizers plan routes. Coordination prevents conflicts and ensures fair resource allocation.

## Architecture

```
┌─────────────────────────────────────────┐
│          Coordination Layer             │
│                                         │
│ ┌────────────┐ ┌────────┐ ┌──────────┐ │
│ │Shared State│ │ Policy │ │ Budget   │ │
│ └─────┬──────┘ └───┬────┘ └────┬─────┘ │
│       └────────────┴───────────┘       │
│                    │                    │
└────────────────────┼────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   ┌───▼───┐    ┌────▼────┐   ┌───▼───┐
   │Agent A│    │ Agent B │   │Agent C│
   │(Scout)│    │(Analyst)│   │(Synth)│
   └───────┘    └─────────┘   └───────┘
```

## Code

- [AgentsExample](/Frameworks/AgentsExample/) — Multi-agent demonstration
