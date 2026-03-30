#!/usr/bin/env python3
"""
Analyze what experiments agents actually run.
"""

import asyncio
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import (
    create_world, ScientistAgent, parse_observation, HypothesisMiner,
    EXPERIMENT_KEYWORDS, COLORS, SIZES, MATERIALS, SHAPES
)
from llm_providers import OpenAICompatibleLLM
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

TOKEN_FILE = "/Users/ian/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class DiagnosticScientistAgent(ScientistAgent):
    """Agent that tracks experiment details for analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_details = []  # (experiment_type, color, size, material, shape, outcome)

    async def _propose_experiment(self) -> str:
        exp = await super()._propose_experiment()
        return exp


async def run_analysis(n_agents: int, experiments_per_agent: int):
    """Run agents and analyze experiment distribution."""

    world = create_world(difficulty="hard", seed=42)

    print(f"\n{'='*60}")
    print(f"Experiment Distribution Analysis")
    print(f"{'='*60}")

    llm = OpenAICompatibleLLM(
        model=MODEL,
        base_url=BASE_URL,
        api_key=TOKEN_FILE,
        n_agents=n_agents,
    )

    start_time = time.time()

    # Collect all experiments
    all_experiments = []

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=n_agents + 4),
    ) as manager:

        handles = []
        for i in range(n_agents):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, 0.0, 42 + i, None, experiments_per_agent),
                kwargs={"comm_every": 0, "comm_n_peers": 0},  # No comm for analysis
            )
            handles.append(h)

        while True:
            await asyncio.sleep(1)
            states = await asyncio.gather(*[h.get_state() for h in handles])
            if all(s.get("done", False) for s in states):
                break
            if time.time() - start_time > 300:
                break

        for h in handles:
            await h.shutdown()

    # Analyze experiments run through the world
    # Re-run a controlled analysis
    print(f"\nRunning controlled experiments to verify coverage...")

    miner = HypothesisMiner()
    exp_type_counts = defaultdict(int)
    material_counts = defaultdict(int)
    color_counts = defaultdict(int)
    size_counts = defaultdict(int)
    shape_counts = defaultdict(int)

    # Rule-specific coverage
    rule_coverage = {
        "metal+electricity": 0,
        "metal+sunlight": 0,
        "wood+fire": 0,
        "wood+water": 0,
        "red+fire": 0,
        "small+throw": 0,
        "sphere+floor": 0,
        "pyramid+scale": 0,
    }

    # Simulate the same number of experiments
    world2 = create_world(difficulty="hard", seed=42)
    for i in range(n_agents * experiments_per_agent):
        obj = world2.generate_object()
        # Pick random experiment type
        import random
        random.seed(42 + i)
        exp_types = ["water", "floor", "electricity", "fire", "sunlight", "freezer", "throw", "scale"]
        exp_type = random.choice(exp_types)

        exp_type_counts[exp_type] += 1
        material_counts[obj["material"]] += 1
        color_counts[obj["color"]] += 1
        size_counts[obj["size"]] += 1
        shape_counts[obj["shape"]] += 1

        # Check rule coverage
        if obj["material"] == "metal" and exp_type == "electricity":
            rule_coverage["metal+electricity"] += 1
        if obj["material"] == "metal" and exp_type == "sunlight":
            rule_coverage["metal+sunlight"] += 1
        if obj["material"] == "wood" and exp_type == "fire":
            rule_coverage["wood+fire"] += 1
        if obj["material"] == "wood" and exp_type == "water":
            rule_coverage["wood+water"] += 1
        if obj["color"] == "red" and exp_type == "fire":
            rule_coverage["red+fire"] += 1
        if obj["size"] == "small" and exp_type == "throw":
            rule_coverage["small+throw"] += 1
        if obj["shape"] == "sphere" and exp_type == "floor":
            rule_coverage["sphere+floor"] += 1
        if obj["shape"] == "pyramid" and exp_type == "scale":
            rule_coverage["pyramid+scale"] += 1

    print(f"\n{'='*60}")
    print("RANDOM EXPERIMENT DISTRIBUTION (simulated)")
    print(f"{'='*60}")

    print(f"\nExperiment types (if uniformly random):")
    for exp_type, count in sorted(exp_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {exp_type:12}: {count:3} ({count/(n_agents*experiments_per_agent)*100:.1f}%)")

    print(f"\nRule coverage (observations per rule):")
    for rule, count in sorted(rule_coverage.items(), key=lambda x: -x[1]):
        status = "OK" if count >= 2 else "LOW"
        print(f"  {rule:20}: {count:3} [{status}]")

    print(f"\n{'='*60}")
    print("EXPECTED VS ACTUAL ISSUE")
    print(f"{'='*60}")
    print("""
The LLM-proposed experiments are likely NOT uniformly distributed.
Common biases:
- Fire and electricity are "exciting" - oversampled
- Water, floor, scale are "boring" - undersampled
- LLM may repeat similar experiments

The hypothesis miner requires min_evidence=2 per property+experiment.
If sunlight, water, throw, floor, scale experiments are undersampled,
those rules will never be discovered.
""")


async def main():
    init_logging("ERROR")
    await run_analysis(n_agents=8, experiments_per_agent=20)


if __name__ == "__main__":
    asyncio.run(main())
