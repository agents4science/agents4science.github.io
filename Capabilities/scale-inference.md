# Stage 3: Scale Agent Inference

Fan out queries to thousands of parallel LLM instances running on HPC systems.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Agentic application performs massively parallel LLM inference |
| **Approach** | FIRST + inference orchestration |
| **Status** | <span style="color: orange;">**Prototype**</span> — Demonstrated at 2000+ nodes on Aurora |
| **Scale** | 1000+ compute nodes |

---

## When to Use

- Processing large datasets requiring LLM analysis (millions of documents, molecules, etc.)
- Ensemble methods requiring many independent LLM calls
- Hyperparameter sweeps over prompts or model configurations
- Time-critical applications requiring high throughput

---

## Architecture

```
+──────────────────+         +─────────────────────────────────────+
│   Coordinator    │         │         HPC System (Aurora)          │
│  +─────────+     │         │                                      │
│  │  Agent  │─────┼────────>│  +─────+ +─────+ +─────+    +─────+ │
│  │         │     │  fan    │  │LLM 1│ │LLM 2│ │LLM 3│... │LLM N│ │
│  +────┬────+     │  out    │  +──┬──+ +──┬──+ +──┬──+    +──┬──+ │
│       │          │         │     │       │       │          │     │
│       v          │         │     +───────┴───────┴──────────+     │
│  +─────────+     │         │              │ results               │
│  │ Results │<────┼─────────┼──────────────+                       │
│  │Aggregator     │         │                                      │
│  +─────────+     │         +─────────────────────────────────────+
+──────────────────+
```

---

## Demonstrated Performance

| Metric | Value |
|--------|-------|
| System | Aurora (ALCF) |
| Nodes | 2000+ |
| Throughput | TBD |
| Model | TBD |

---

## Key Capabilities

- **Elastic scaling**: Dynamically adjust number of LLM instances
- **Load balancing**: Distribute queries across available instances
- **Fault tolerance**: Handle node failures gracefully
- **Result aggregation**: Collect and combine outputs efficiently

---

## Use Cases

### Literature Mining at Scale
Process millions of scientific papers to extract structured information:
```
10M papers x 5 queries/paper = 50M LLM calls
At 2000 nodes: hours instead of months
```

### Molecular Property Prediction
Screen large chemical libraries with LLM-based property prediction:
```
1M molecules x 10 properties = 10M predictions
```

### Hypothesis Generation
Generate and evaluate hypotheses across a parameter space:
```
1000 conditions x 100 hypotheses = 100K evaluations
```

---

## Getting Started

Documentation and examples coming soon. Contact the CAF team for early access.

---

## Next Steps

- [**Stage 4: Scale and Govern Tool Use**](scale-tool-use.md) — Coordinate massive tool invocations with governance
- [**Stage 5: Coordinate Many Agents**](coordinate-agents.md) — Multi-agent collaboration
