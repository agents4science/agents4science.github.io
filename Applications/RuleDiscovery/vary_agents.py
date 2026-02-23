#!/usr/bin/env python3
"""
Test how results vary with number of agents.
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

AGENT_COUNTS = [1, 2, 4, 8, 16]
STRATEGIES = [
    ("no_comm", 0, 0),
    ("N1_E2", 2, 1),
]
EXPERIMENTS_PER_AGENT = 20  # Each agent does 20 experiments
DIFFICULTY = "medium"
SEED = 42

RESULTS_FILE = Path(__file__).parent / "vary_agents_results.json"


async def run_experiment(n_agents: int, comm_every: int, comm_n_peers: int) -> dict:
    """Run experiment with given number of agents."""

    world = create_world(difficulty=DIFFICULTY, seed=SEED)

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
                args=(i, world, llm, 0.0, SEED + i, None, EXPERIMENTS_PER_AGENT),
                kwargs={"comm_every": comm_every, "comm_n_peers": comm_n_peers},
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

    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    total_exp = sum(s["steps"] for s in final_states)
    total_obs = sum(s["n_observations"] for s in final_states)
    total_msgs = sum(s["messages_sent"] for s in final_states)

    return {
        "n_agents": n_agents,
        "total_experiments": total_exp,
        "total_observations": total_obs,
        "messages": total_msgs,
        "score": eval_result["score"],
        "rules_found": len(eval_result["rules_found"]),
        "elapsed": elapsed,
    }


async def main():
    init_logging("ERROR")

    print("=" * 60)
    print(f"Varying Agent Count Test")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Experiments per agent: {EXPERIMENTS_PER_AGENT}")
    print(f"Agent counts: {AGENT_COUNTS}")
    print("=" * 60)

    results = {}

    for strat_name, comm_every, comm_n_peers in STRATEGIES:
        print(f"\n--- {strat_name} ---")
        print(f"{'Agents':>6} {'TotalExp':>8} {'Obs':>6} {'Msgs':>6} {'Score':>6} {'Time':>6}")
        print("-" * 50)

        for n_agents in AGENT_COUNTS:
            try:
                result = await run_experiment(n_agents, comm_every, comm_n_peers)

                key = f"{strat_name}|{n_agents}"
                results[key] = result

                print(f"{n_agents:>6} {result['total_experiments']:>8} {result['total_observations']:>6} "
                      f"{result['messages']:>6} {result['score']:>5}% {result['elapsed']:>5.0f}s")

                # Save after each run
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"{n_agents:>6} ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Agents':>6} {'no_comm':>10} {'N1_E2':>10} {'Improvement':>12}")
    print("-" * 40)

    for n_agents in AGENT_COUNTS:
        no_comm = results.get(f"no_comm|{n_agents}", {}).get("score", "-")
        n1e2 = results.get(f"N1_E2|{n_agents}", {}).get("score", "-")
        if isinstance(no_comm, int) and isinstance(n1e2, int) and no_comm > 0:
            imp = f"+{((n1e2 - no_comm) / no_comm * 100):.0f}%"
        else:
            imp = "-"
        print(f"{n_agents:>6} {no_comm:>9}% {n1e2:>9}% {imp:>12}")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
