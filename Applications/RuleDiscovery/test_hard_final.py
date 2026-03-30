#!/usr/bin/env python3
"""
Final test on Hard difficulty with improvements.
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


async def run_test(n_agents: int, experiments_per_agent: int, seed: int = 42):
    """Run test with given configuration."""

    world = create_world(difficulty="hard", seed=seed)

    print(f"\n{'='*60}")
    print(f"HARD: {n_agents} agents x {experiments_per_agent} exp = {n_agents * experiments_per_agent} total")
    print(f"{'='*60}")

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
                kwargs={"comm_every": 2, "comm_n_peers": 1},
            )
            handles.append(h)

        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(n_agents) if j != i]
            await h.set_peers(peers)

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

    combined = []
    for hyps in all_hypotheses:
        combined.extend(hyps)
    combined = sorted(combined, key=lambda x: -x.get("confidence", 0))[:15]

    print(f"\nTop hypotheses:")
    for i, h in enumerate(combined[:8], 1):
        print(f"  {i}. {h['rule']} ({h['confidence']:.0%})")

    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    total_msgs = sum(s["messages_sent"] for s in final_states)

    print(f"\nScore: {eval_result['score']}/100 ({len(eval_result['rules_found'])}/{len(world.rules)} rules)")
    print(f"Time: {elapsed:.0f}s, Messages: {total_msgs}")

    if eval_result['rules_found']:
        print(f"\nFound:")
        for rule in eval_result['rules_found']:
            print(f"  + {rule}")
    if eval_result['rules_missed']:
        print(f"\nMissed:")
        for rule in eval_result['rules_missed']:
            print(f"  - {rule}")

    return eval_result['score'], len(eval_result['rules_found'])


async def main():
    init_logging("ERROR")

    print("Testing Hard difficulty with improvements...")

    # Test configurations: 80 and 100 experiments per agent
    configs = [
        (8, 80),   # 640 total
        (8, 100),  # 800 total
    ]

    results = []
    for n_agents, exp_per_agent in configs:
        score, rules = await run_test(n_agents, exp_per_agent)
        results.append((n_agents * exp_per_agent, score, rules))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Total Exp':>10} {'Score':>10} {'Rules':>10}")
    print("-" * 30)
    for total, score, rules in results:
        print(f"{total:>10} {score:>9}% {rules:>9}/8")


if __name__ == "__main__":
    asyncio.run(main())
