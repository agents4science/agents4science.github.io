# Stage 3: Massively Parallel Agent Inference

**Experimental. Fan out thousands of LLM requests in parallel on HPC.**

## Task

Build agentic applications that fan out thousands of LLM inference requests in parallel on HPC systems.

## Why This Matters

Scientific applications often require millions of LLM calls—literature mining, molecular screening, hypothesis generation. Parallel inference on HPC turns months of sequential work into hours.

## Details

| Aspect | Value |
|--------|-------|
| **CAF Components** | LangGraph, FIRST, inference orchestration |
| **Where it runs** | HPC accelerator nodes |
| **Scale** | O(10³–10⁴) concurrent inference streams |
| **Status** | Prototype (Aurora: 2000+ nodes demonstrated) |

## Architecture

<img src="/Capabilities/Assets/scale-inference.svg" alt="Scale inference: Coordinator fans out to thousands of LLM instances on HPC" style="max-width: 540px; margin: 1rem 0;">

## Code

Aurora 2000-node demo — documentation coming soon
