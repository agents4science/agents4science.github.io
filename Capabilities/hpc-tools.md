# Agents with HPC Tools

Agentic applications that invoke simulation codes, data services, or other computational tools on DOE high-performance computing systems.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Agent invokes tools (simulations, analysis codes) running on DOE HPC |
| **Approach** | LangGraph + Academy |
| **Status** | <span style="color: green;">**Mature**</span> — Documented by examples on this site |
| **Scale** | 1-100+ compute nodes |

---

## When to Use

- Agent needs to run computationally intensive simulations
- Tools require HPC resources (GPUs, large memory, parallel filesystems)
- Workflows span multiple DOE facilities
- Secure, auditable execution is required

---

## Architecture

```
+──────────────────+         +─────────────────────────────+
│   Agent Host     │         │      DOE HPC System         │
│  +─────────+     │         │  +─────────+  +─────────+   │
│  │  Agent  │─────┼────────>│  │ Academy │─>│  Tools  │   │
│  │(LangGraph)    │  secure │  │ Runtime │  │(sims,   │   │
│  +────┬────+     │  tunnel │  +─────────+  │ analysis)   │
│       │          │         │               +─────────+   │
│       v          │         │                             │
│  +─────────+     │         │  +─────────────────────+    │
│  │   LLM   │     │         │  │   Compute Nodes     │    │
│  +─────────+     │         │  +─────────────────────+    │
+──────────────────+         +─────────────────────────────+
```

---

## Key Capabilities

- **Secure tool invocation**: Academy handles authentication, authorization, and audit
- **Resource management**: Automatic job submission and monitoring
- **Data movement**: Efficient transfer between agent and HPC storage
- **Fault tolerance**: Checkpointing and recovery for long-running tools

---

## Getting Started

1. **Install Academy client** on your agent host
2. **Register tools** with Academy on target HPC system
3. **Configure agent** to use Academy-wrapped tools

```python
from academy import RemoteTool

# Define a tool that runs on HPC
xtb_tool = RemoteTool(
    name="xtb_optimize",
    endpoint="academy://polaris.alcf.anl.gov/xtb",
    description="Optimize molecular geometry using xTB"
)

# Use in LangGraph agent
agent = create_agent(tools=[xtb_tool, ...])
```

---

## Examples on This Site

- [**CharacterizeChemicals**](/Frameworks/CharacterizeChemicals/) — Molecular property calculations using RDKit and xTB

---

## Supported HPC Systems

| System | Facility | Status |
|--------|----------|--------|
| Polaris | ALCF | Available |
| Aurora | ALCF | Available |
| Perlmutter | NERSC | Coming soon |
| Frontier | OLCF | Coming soon |

---

## Next Steps

- [**Massively Parallel Inference**](parallel-inference.md) — Scale LLM inference across HPC
- [**Simulation Steering**](simulation-steering.md) — Real-time control of running simulations
