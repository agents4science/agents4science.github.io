# Stage 5: Coordinate Many Agents

Multiple agents collaborate with shared state, coordinated policies, and resource budgets.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Enable multiple agents to work together effectively |
| **Approach** | Shared state, policy, budgets |
| **Status** | <span style="color: blue;">**Emerging**</span> |
| **Scale** | Varies |

---

## The Challenge

When multiple agents operate in the same environment, new challenges emerge:

- **Shared state**: Agents must see consistent views of evolving data
- **Resource contention**: Multiple agents competing for limited resources
- **Policy consistency**: Governance rules must apply uniformly
- **Coordination overhead**: Agents need to communicate without bottlenecks

---

## When to Use

- Large-scale campaigns with specialized agent roles
- Multi-facility workflows spanning DOE resources
- Collaborative scientific discovery with complementary expertise
- Redundant/ensemble approaches for robustness

---

## Architecture

```
+─────────────────────────────────────────────────────────────+
│                   Coordination Layer                         │
│  +──────────────+  +──────────────+  +──────────────+       │
│  │ Shared State │  │   Policy     │  │   Budget     │       │
│  │    Store     │  │   Registry   │  │   Ledger     │       │
│  +──────┬───────+  +──────┬───────+  +──────┬───────+       │
│         │                 │                 │                │
│         +─────────────────┴─────────────────+                │
│                           │                                  │
+───────────────────────────┼──────────────────────────────────+
                            │
        +───────────────────┼───────────────────+
        │                   │                   │
   +────v────+         +────v────+         +────v────+
   │ Agent A │         │ Agent B │         │ Agent C │
   │ (Scout) │         │(Analyst)│         │(Synth)  │
   +─────────+         +─────────+         +─────────+
```

---

## Key Capabilities

### Shared State Store
Agents read and write to a common knowledge base:
```python
# Agent A discovers a promising candidate
state.write("candidates/mol-123", {
    "smiles": "CCO",
    "predicted_activity": 0.92,
    "discovered_by": "scout-agent"
})

# Agent B picks it up for analysis
candidate = state.read("candidates/mol-123")
```

### Policy Registry
Consistent governance across all agents:
```yaml
global_policies:
  - all agents must log tool invocations
  - no agent may exceed 10% of total budget
  - synthesis agents require human approval for novel compounds
```

### Budget Ledger
Track resource consumption across the agent collective:
```
Total allocation: 100,000 node-hours
├── Scout agents: 10,000 (used: 3,500)
├── Analysis agents: 40,000 (used: 28,000)
├── Synthesis agents: 50,000 (used: 12,000)
└── Reserve: 0 (used: 0)
```

---

## Coordination Patterns

### Blackboard Architecture
Agents communicate through a shared workspace:
```
+─────────────────────────────────────+
│            Blackboard               │
│  ┌─────────┐  ┌─────────┐          │
│  │Hypotheses│  │ Results │          │
│  └─────────┘  └─────────┘          │
│  ┌─────────┐  ┌─────────┐          │
│  │  Tasks  │  │Decisions│          │
│  └─────────┘  └─────────┘          │
+───────┬─────────────┬───────────────+
        │             │
    +───v───+     +───v───+
    │Agent A│     │Agent B│
    +───────+     +───────+
```

### Pipeline Handoffs
Agents process work in stages:
```
Scout → Planner → Executor → Analyst → Archivist
```

### Ensemble Voting
Multiple agents propose, then vote or aggregate:
```python
proposals = [agent.propose(problem) for agent in ensemble]
decision = voting_policy.aggregate(proposals)
```

### Hierarchical Delegation
Supervisor agents coordinate specialist agents:
```
Supervisor
├── assigns tasks to specialists
├── monitors progress
├── resolves conflicts
└── reports to humans
```

---

## Example: Multi-Agent Discovery Campaign

```python
from academy import AgentCollective, SharedState, BudgetLedger

# Create shared infrastructure
state = SharedState(backend="redis")
budget = BudgetLedger(total=100_000, unit="node-hours")

# Initialize agent collective
collective = AgentCollective(
    agents=[
        ScoutAgent(role="literature-mining"),
        ScoutAgent(role="database-search"),
        AnalystAgent(role="property-prediction"),
        AnalystAgent(role="feasibility-assessment"),
        SynthesisAgent(role="route-planning"),
    ],
    shared_state=state,
    budget=budget,
    policies=["approval-for-synthesis", "budget-fairshare"]
)

# Run coordinated campaign
collective.run(
    goal="Find synthesizable molecules with target property X > 0.9",
    max_iterations=1000
)
```

---

## Getting Started

Documentation and examples coming soon. See [AgentsExample](/Frameworks/AgentsExample/) for a simpler multi-agent demonstration.

---

## Next Steps

- [**Stage 6: Autonomous Scientific Systems**](autonomous-systems.md) — Persistent, governed autonomy
