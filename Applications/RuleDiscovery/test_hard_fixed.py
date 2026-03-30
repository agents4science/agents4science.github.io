#!/usr/bin/env python3
"""
Test Hard difficulty with fixed scale action.
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import create_world, ScientistAgent, evaluate_hypotheses
from llm_providers import OpenAICompatibleLLM
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

TOKEN_FILE = "/Users/ian/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
EVAL_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


async def run_test(n_agents: int, experiments_per_agent: int, difficulty: str, seed: int):
    """Run test with given configuration."""

    world = create_world(difficulty=difficulty, seed=seed)

    print(f"\n{'='*60}")
    print(f"Testing {difficulty.upper()} difficulty with {n_agents} agents")
    print(f"Experiments per agent: {experiments_per_agent}")
    print(f"Total experiments: {n_agents * experiments_per_agent}")
    print(f"{'='*60}")
    print(f"\nHidden rules to discover:")
    for i, rule in enumerate(world.rules, 1):
        print(f"  {i}. {rule.natural_language}")
    print()

    llm = OpenAICompatibleLLM(
        model=MODEL,
        base_url=BASE_URL,
        api_key=TOKEN_FILE,
        n_agents=n_agents,
    )

    eval_llm = OpenAICompatibleLLM(
        model=EVAL_MODEL,
        base_url=BASE_URL,
        api_key=TOKEN_FILE,
        n_agents=1,
    )

    start_time = time.time()

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=n_agents + 4),
    ) as manager:

        handles = []
        for i in range(n_agents):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, 0.0, seed + i, None, experiments_per_agent),
                kwargs={"comm_every": 2, "comm_n_peers": 1},  # N1_E2 strategy
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
            if time.time() - start_time > 600:
                print("Timeout!")
                break

        final_states = await asyncio.gather(*[h.get_state() for h in handles])
        all_hypotheses = await asyncio.gather(*[h.get_hypotheses() for h in handles])

        for h in handles:
            await h.shutdown()

    elapsed = time.time() - start_time

    # Combine hypotheses
    combined = []
    for hyps in all_hypotheses:
        combined.extend(hyps)
    combined = sorted(combined, key=lambda x: -x.get("confidence", 0))[:10]

    print(f"\nTop 10 combined hypotheses:")
    for i, h in enumerate(combined, 1):
        print(f"  {i}. {h['rule']} ({h['confidence']:.0%})")

    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    total_exp = sum(s["steps"] for s in final_states)
    total_obs = sum(s["n_observations"] for s in final_states)
    total_msgs = sum(s["messages_sent"] for s in final_states)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total experiments: {total_exp}")
    print(f"Total observations: {total_obs}")
    print(f"Messages sent: {total_msgs}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nScore: {eval_result['score']}/100")
    print(f"Rules found: {len(eval_result['rules_found'])}/{len(world.rules)}")

    if eval_result['rules_found']:
        print("\nFound:")
        for rule in eval_result['rules_found']:
            print(f"  + {rule}")

    if eval_result['rules_missed']:
        print("\nMissed:")
        for rule in eval_result['rules_missed']:
            print(f"  - {rule}")

    return eval_result


async def main():
    init_logging("ERROR")

    # Test with 8 agents on hard, 20 experiments each
    await run_test(n_agents=8, experiments_per_agent=20, difficulty="hard", seed=42)


if __name__ == "__main__":
    asyncio.run(main())
