# Human-in-the-Loop Workflows

Agents that propose actions and wait for scientist approval at key decision points, ensuring human oversight of critical choices.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Agent pauses for human approval before consequential actions |
| **Approach** | LangGraph interrupts + notification/UI system |
| **Status** | <span style="color: green;">**Mature**</span> |
| **Scale** | Any |

---

## When to Use

- Actions with significant cost (compute time, reagents, beam time)
- Irreversible operations (synthesis, fabrication)
- Safety-critical decisions
- Regulatory or compliance requirements
- Building trust during agent development

---

## Architecture

```
+─────────────────────────────────────────────────────────+
│                    Agent Workflow                        │
│                                                          │
│  +──────+    +──────+    +──────+    +──────+           │
│  │Plan  │───>│Review│───>│Approve│──>│Execute│          │
│  │      │    │Point │    │/Reject│   │       │          │
│  +──────+    +──┬───+    +───┬───+   +───────+          │
│                 │            │                           │
│                 v            ^                           │
│            +────────────────────+                        │
│            │   Human Scientist  │                        │
│            │   (via UI/email)   │                        │
│            +────────────────────+                        │
+─────────────────────────────────────────────────────────+
```

---

## Interrupt Patterns

### Pre-Action Approval
Agent proposes an action and waits for explicit approval:
```python
# LangGraph interrupt before tool execution
@node
def propose_experiment(state):
    plan = generate_plan(state)
    return Command(
        goto="human_review",
        update={"proposed_plan": plan}
    )

@node
def human_review(state):
    # Workflow pauses here until human responds
    interrupt(
        message=f"Proposed experiment:\n{state['proposed_plan']}\n\nApprove?",
        options=["approve", "reject", "modify"]
    )
```

### Checkpoint Reviews
Agent runs autonomously but pauses at defined checkpoints:
```python
checkpoints = ["after_planning", "after_simulation", "before_synthesis"]
```

### Anomaly Escalation
Agent runs autonomously but escalates unexpected situations:
```python
if confidence < threshold or result.is_anomalous():
    interrupt("Unexpected result detected. Please review.")
```

---

## Notification Channels

| Channel | Use Case |
|---------|----------|
| Web UI | Interactive review with full context |
| Email | Asynchronous approval for non-urgent decisions |
| Slack/Teams | Quick approvals with team visibility |
| SMS | Urgent notifications requiring immediate attention |

---

## Example: Synthesis Workflow

```python
from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph

workflow = StateGraph(SynthesisState)

workflow.add_node("design", design_molecule)
workflow.add_node("simulate", run_simulation)
workflow.add_node("review_simulation", human_review)  # Interrupt point
workflow.add_node("synthesize", execute_synthesis)
workflow.add_node("review_synthesis", human_review)   # Interrupt point
workflow.add_node("characterize", run_characterization)

workflow.add_edge("design", "simulate")
workflow.add_edge("simulate", "review_simulation")
workflow.add_conditional_edges(
    "review_simulation",
    check_approval,
    {"approved": "synthesize", "rejected": "design"}
)
```

---

## Examples on This Site

- [**AgentsExample**](/Frameworks/AgentsExample/) — Demonstrates interrupt patterns with dashboard UI

---

## Best Practices

1. **Clear context**: Provide humans with all information needed to make decisions
2. **Reasonable defaults**: Suggest recommended actions to speed review
3. **Timeout handling**: Define behavior if human doesn't respond
4. **Audit trail**: Log all human decisions for reproducibility
5. **Graceful degradation**: Allow agents to continue with safe defaults if appropriate
