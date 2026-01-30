# Simulation Steering

Agents that monitor running simulations and adjust parameters in real-time to optimize outcomes.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Agent observes simulation state and modifies parameters during execution |
| **Approach** | Academy + simulation codes with steering APIs |
| **Status** | <span style="color: orange;">**Prototype**</span> |
| **Scale** | Varies by simulation |

---

## When to Use

- Long-running simulations that benefit from adaptive control
- Optimization problems where early results inform later parameters
- Exploratory simulations seeking rare events
- Resource-constrained scenarios requiring early termination decisions

---

## Architecture

```
+──────────────────+         +─────────────────────────────+
│   Steering Agent │         │      HPC Simulation         │
│  +─────────+     │ observe │  +─────────────────────+    │
│  │  Agent  │<────┼─────────┼──│  Simulation State   │    │
│  │         │     │         │  +─────────────────────+    │
│  │         │─────┼─────────┼─>+─────────────────────+    │
│  +────┬────+     │  steer  │  │  Control Parameters │    │
│       │          │         │  +─────────────────────+    │
│       v          │         │                             │
│  +─────────+     │         │  +─────────────────────+    │
│  │   LLM   │     │         │  │   Compute Nodes     │    │
│  +─────────+     │         │  +─────────────────────+    │
+──────────────────+         +─────────────────────────────+
```

---

## Steering Patterns

### Adaptive Sampling
Agent identifies under-sampled regions and directs simulation to explore them.

### Early Stopping
Agent monitors convergence metrics and terminates when objectives are met.

### Parameter Optimization
Agent adjusts simulation parameters (temperature, pressure, etc.) based on intermediate results.

### Rare Event Detection
Agent watches for interesting phenomena and increases sampling resolution when detected.

---

## Example: Molecular Dynamics Steering

```python
@tool
def get_simulation_state(sim_id: str) -> dict:
    """Retrieve current state of MD simulation."""
    return academy.query(f"md://{sim_id}/state")

@tool
def adjust_temperature(sim_id: str, new_temp: float):
    """Modify simulation temperature."""
    academy.send(f"md://{sim_id}/control", {"temperature": new_temp})

# Agent decides based on observations
agent_prompt = """
Monitor the protein folding simulation.
If the RMSD plateaus for >1000 steps, increase temperature by 10K.
If a folding event is detected, decrease temperature to stabilize.
"""
```

---

## Prerequisites

- Simulation code with steering API (LAMMPS, GROMACS, custom)
- Academy runtime for secure communication
- Checkpoint/restart capability for fault tolerance

---

## Getting Started

Documentation and examples coming soon.
