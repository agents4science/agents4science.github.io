#!/usr/bin/env python3
"""
Diagnostic script to understand what experiments agents run
and what hypotheses should be mined on Hard difficulty.
"""

import asyncio
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import (
    create_world, ScientistAgent, parse_observation,
    HypothesisMiner, COLORS, SIZES, MATERIALS, SHAPES
)
from llm_providers import OpenAICompatibleLLM
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

TOKEN_FILE = "/Users/ian/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


async def run_diagnostic(n_agents: int, experiments_per_agent: int):
    """Run agents and collect diagnostic data."""

    world = create_world(difficulty="hard", seed=42)

    print(f"\n{'='*60}")
    print(f"HARD Difficulty Diagnostic")
    print(f"Agents: {n_agents}, Experiments/agent: {experiments_per_agent}")
    print(f"{'='*60}")

    print(f"\nTarget rules to discover:")
    for i, rule in enumerate(world.rules, 1):
        print(f"  {i}. {rule.natural_language}")
        print(f"     Condition: {rule.condition}, Experiment: {rule.experiment_type}")
    print()

    llm = OpenAICompatibleLLM(
        model=MODEL,
        base_url=BASE_URL,
        api_key=TOKEN_FILE,
        n_agents=n_agents,
    )

    start_time = time.time()

    # Collect all observations
    all_observations = []

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=n_agents + 4),
    ) as manager:

        handles = []
        for i in range(n_agents):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, 0.0, 42 + i, None, experiments_per_agent),
                kwargs={"comm_every": 2, "comm_n_peers": 1},
            )
            handles.append(h)

        # Wire full connectivity
        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(n_agents) if j != i]
            await h.set_peers(peers)

        # Wait for completion
        while True:
            await asyncio.sleep(1)
            states = await asyncio.gather(*[h.get_state() for h in handles])
            if all(s.get("done", False) for s in states):
                break
            if time.time() - start_time > 300:
                break

        # Collect observations from all agent miners
        for h in handles:
            state = await h.get_state()
            hyps = await h.get_hypotheses()
            print(f"Agent {state['agent_idx']}: {state['n_observations']} obs, {len(hyps)} hypotheses")

        for h in handles:
            await h.shutdown()

    elapsed = time.time() - start_time

    # Re-parse all experiments from world
    print(f"\nRe-running experiments to analyze coverage...")

    # Run controlled experiments to verify coverage
    miner = HypothesisMiner()
    exp_type_counts = defaultdict(int)
    rule_coverage = {rule.natural_language: [] for rule in world.rules}

    # Check what experiments would need to be run to discover each rule
    print(f"\n{'='*60}")
    print("RULE COVERAGE ANALYSIS")
    print(f"{'='*60}")

    for rule in world.rules:
        print(f"\n{rule.natural_language}")
        print(f"  Needs: {rule.condition} + {rule.experiment_type} experiment")

        # Count how many objects match the condition
        if "==" in rule.condition:
            prop, val = rule.condition.split("==")
            prop = prop.strip()
            val = val.strip()
            if prop == "material":
                prob = 1 / len(MATERIALS)
            elif prop == "color":
                prob = 1 / len(COLORS)
            elif prop == "size":
                prob = 1 / len(SIZES)
            elif prop == "shape":
                prob = 1 / len(SHAPES)
            else:
                prob = 0.25
            print(f"  Object probability: {prob:.1%} ({val})")
            print(f"  Experiment probability: ~{1/8:.1%} (1 of 8 types)")
            print(f"  Combined: ~{prob/8:.2%} per random experiment")
            print(f"  Expected hits in 160 exp: {160 * prob / 8:.1f}")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    print("""
To improve Hard difficulty coverage:

1. **More experiments**: 160 experiments across 8 types means ~20 per type.
   With 4 colors/materials/shapes and 3 sizes, we need more targeted experiments.

2. **Targeted exploration**: When an interesting outcome is found, run more
   experiments with that property to confirm the pattern.

3. **Lower thresholds temporarily**: min_evidence=2 may be too high for
   rare property+experiment combinations.

4. **Issue with current run**: Agents may be over-focusing on certain
   experiment types (fire, electricity) and under-exploring others
   (scale, floor, throw).
""")


async def main():
    init_logging("ERROR")
    await run_diagnostic(n_agents=8, experiments_per_agent=20)


if __name__ == "__main__":
    asyncio.run(main())
