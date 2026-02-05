# Massively Parallel Agent Inference

**Status: Prototype** — Fan out thousands of LLM requests in parallel on HPC.

## Why This Matters

Scientific applications often require millions of LLM calls—literature mining, molecular screening, hypothesis generation. Parallel inference on HPC turns months of sequential work into hours.

| Aspect | Value |
|--------|-------|
| **CAF Components** | LangGraph, FIRST, inference orchestration |
| **Where it runs** | HPC accelerator nodes |
| **Scale** | O(10³–10⁴) concurrent inference streams |
| **Status** | Prototype (Aurora: 2000+ nodes demonstrated) |

## Architecture

<img src="/Capabilities/Assets/scale-inference.svg" alt="Scale inference: Coordinator fans out to thousands of LLM instances on HPC" style="max-width: 540px; margin: 1rem 0;">

The pattern:
1. **Coordinator** receives a batch of tasks (e.g., 10,000 molecules to analyze)
2. **Fan-out** distributes tasks across HPC nodes, each running local LLM inference
3. **FIRST** provides the inference backend, optimized for HPC accelerators
4. **Fan-in** collects results back to the coordinator

## The Approach

```python
# Conceptual pattern (not production code)
from academy.manager import Manager
from academy.exchange import GlobusComputeExchange

class InferenceWorker(Agent):
    """Worker that runs LLM inference on a single item."""

    def __init__(self):
        super().__init__()
        self._llm = None  # Lazy init on HPC node

    @action
    async def analyze(self, item: dict) -> dict:
        if self._llm is None:
            # Initialize FIRST client on HPC node
            self._llm = FIRSTClient(endpoint="aurora-inference")

        result = await self._llm.generate(
            prompt=f"Analyze this molecule: {item['smiles']}"
        )
        return {"id": item["id"], "analysis": result}

async def parallel_inference(items: list[dict], num_workers: int = 1000):
    """Fan out inference across HPC nodes."""
    async with await Manager.from_exchange_factory(
        factory=GlobusComputeExchange(endpoint_id="aurora-endpoint"),
    ) as manager:
        # Launch workers across nodes
        workers = [
            await manager.launch(InferenceWorker)
            for _ in range(num_workers)
        ]

        # Distribute work
        tasks = []
        for i, item in enumerate(items):
            worker = workers[i % len(workers)]
            tasks.append(worker.analyze(item))

        # Gather results
        results = await asyncio.gather(*tasks)
        return results
```

## Use Cases

| Application | Items | LLM Task |
|-------------|-------|----------|
| **Literature mining** | 100K+ papers | Extract methods, findings, citations |
| **Molecular screening** | 10K+ molecules | Predict properties, rank candidates |
| **Hypothesis generation** | 1K+ datasets | Propose experiments, identify patterns |
| **Data annotation** | 100K+ records | Label, classify, summarize |

## Current Status

The Aurora 2000-node demonstration showed:
- **2000+ concurrent inference streams** across Aurora nodes
- **Sub-linear scaling** up to thousands of workers
- **Integration with FIRST** inference backend

### What's Available Now

- Basic fan-out pattern with Academy
- FIRST integration for HPC inference
- Prototype orchestration code

### Coming Soon

- Production-ready orchestration framework
- Automatic load balancing and fault tolerance
- Cost tracking and budget enforcement at scale
- Detailed performance benchmarks

## Prerequisites

Before exploring scale inference:

1. **[Local Agents](/Capabilities/local-agents/)** — Understand LangGraph basics
2. **[Federated Agents](/Capabilities/federated-agents/)** — Run agents on HPC
3. **FIRST access** — DOE infrastructure credentials

## Related Topics

- [Federated Agents](/Capabilities/federated-agents/) — Running agents across DOE infrastructure
- [Multi-Agent Coordination](/Capabilities/multi-agent-coordination/) — Coordinating many agents
- [LLM Configuration](/Capabilities/local-agents/llm-configuration/) — Setting up FIRST backend
