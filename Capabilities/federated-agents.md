# Stage 2: Run Federated Agents on DOE Resources

## Task

Agentic applications invoke tools—simulations, analysis codes, data services—running on DOE HPC systems with secure, auditable execution.

## Why This Matters

Scientific workflows require HPC resources (GPUs, parallel filesystems, large memory). Academy provides secure tool invocation across facility boundaries with full audit trails.

## Architecture

```
┌────────────┐          ┌─────────────────────┐
│ Agent Host │          │    DOE HPC System   │
│            │          │                     │
│ ┌───────┐  │  secure  │ ┌───────┐ ┌──────┐ │
│ │ Agent │──┼─────────▶│ │Academy│─▶│Tools │ │
│ └───┬───┘  │          │ └───────┘ └──────┘ │
│     │      │          │                     │
│     ▼      │          │ ┌─────────────────┐ │
│ ┌───────┐  │          │ │ Compute Nodes   │ │
│ │  LLM  │  │          │ └─────────────────┘ │
│ └───────┘  │          └─────────────────────┘
└────────────┘
```

## Code

- [CharacterizeChemicals](/Frameworks/CharacterizeChemicals/) — Molecular property calculations via RDKit and xTB
