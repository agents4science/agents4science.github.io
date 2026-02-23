#!/usr/bin/env python3
"""
Run multiple seeds to get distribution of results.
Tests easy difficulty with the best strategies found.
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any

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

SEEDS = [42, 123, 456, 789, 1001]
STRATEGIES = [
    ("no_comm", 0, 0),
    ("N1_E2", 2, 1),
]
EXPERIMENTS = 60
N_AGENTS = 4
DIFFICULTY = "easy"

RESULTS_FILE = Path(__file__).parent / "multi_seed_results.json"


async def run_experiment(seed: int, comm_every: int, comm_n_peers: int) -> Dict[str, Any]:
    """Run a single experiment."""

    world = create_world(difficulty=DIFFICULTY, seed=seed)

    llm = OpenAICompatibleLLM(
        model=MODEL,
        base_url=BASE_URL,
        api_key=TOKEN_FILE,
        n_agents=N_AGENTS,
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
        executors=ThreadPoolExecutor(max_workers=N_AGENTS + 4),
    ) as manager:

        handles = []
        for i in range(N_AGENTS):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, 0.0, seed + i, None, EXPERIMENTS),
                kwargs={"comm_every": comm_every, "comm_n_peers": comm_n_peers},
            )
            handles.append(h)

        # Wire connectivity
        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(N_AGENTS) if j != i]
            await h.set_peers(peers)

        # Wait for completion
        while True:
            await asyncio.sleep(1)
            states = await asyncio.gather(*[h.get_state() for h in handles])
            if all(s.get("done", False) for s in states):
                break
            if time.time() - start_time > 300:
                break

        final_states = await asyncio.gather(*[h.get_state() for h in handles])
        all_hypotheses = await asyncio.gather(*[h.get_hypotheses() for h in handles])

        for h in handles:
            await h.shutdown()

    elapsed = time.time() - start_time

    combined = []
    for hyps in all_hypotheses:
        combined.extend(hyps)
    combined = sorted(combined, key=lambda x: -x.get("confidence", 0))[:10]

    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    return {
        "seed": seed,
        "score": eval_result["score"],
        "rules_found": len(eval_result["rules_found"]),
        "rules_total": len(world.rules),
        "elapsed": elapsed,
        "messages": sum(s["messages_sent"] for s in final_states),
    }


async def main():
    init_logging("ERROR")

    print("=" * 60)
    print(f"Multi-Seed Test: {DIFFICULTY} difficulty, {len(SEEDS)} seeds")
    print(f"Model: {MODEL.split('/')[-1]}")
    print(f"Strategies: {[s[0] for s in STRATEGIES]}")
    print(f"Experiments per agent: {EXPERIMENTS}")
    print("=" * 60)

    results = {}

    for strat_name, comm_every, comm_n_peers in STRATEGIES:
        print(f"\n--- {strat_name} ---")
        scores = []

        for seed in SEEDS:
            print(f"  Seed {seed}: ", end="", flush=True)
            try:
                result = await run_experiment(seed, comm_every, comm_n_peers)
                scores.append(result["score"])
                print(f"{result['score']}/100 ({result['rules_found']}/{result['rules_total']} rules, {result['elapsed']:.0f}s)")

                key = f"{strat_name}|{seed}"
                results[key] = result

            except Exception as e:
                print(f"ERROR: {e}")
                scores.append(0)

        # Statistics
        if scores:
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            print(f"\n  {strat_name} Summary:")
            print(f"    Mean: {avg:.1f}/100")
            print(f"    Min:  {min_s}/100")
            print(f"    Max:  {max_s}/100")
            print(f"    All:  {scores}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
