#!/usr/bin/env python3
"""
Compare communication strategies across different experiment counts.

Tests:
- Experiment counts: 20, 40, 60, 80, 100 (stop early if 100%)
- Communication strategies:
  - No communication (baseline)
  - N=1 peer every E=1,2,4 experiments
  - N=2 peers every E=1,2,4 experiments
- Only Argonne models
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import create_world, ScientistAgent, evaluate_hypotheses
from llm_providers import OpenAICompatibleLLM, _load_globus_token
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

# Configuration
TOKEN_FILE = "/Users/ian/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
EVAL_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Models to test (Argonne only)
MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "google/gemma-3-27b-it",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

# Experiment counts
EXPERIMENT_COUNTS = [20, 40, 60, 80, 100]

# Communication strategies: (name, comm_every, comm_n_peers, n_agents)
# comm_every=0 means no communication
COMM_STRATEGIES = [
    ("no_comm", 0, 0, 4),           # No communication, 4 agents
    ("N1_E1", 1, 1, 4),             # 1 peer every 1 experiment
    ("N1_E2", 2, 1, 4),             # 1 peer every 2 experiments
    ("N1_E4", 4, 1, 4),             # 1 peer every 4 experiments
    ("N2_E1", 1, 2, 4),             # 2 peers every 1 experiment
    ("N2_E2", 2, 2, 4),             # 2 peers every 2 experiments
    ("N2_E4", 4, 2, 4),             # 2 peers every 4 experiments
]

RESULTS_FILE = Path(__file__).parent / "comm_strategy_results.json"


def load_results() -> Dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: Dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


async def run_experiment(
    model: str,
    difficulty: str,
    n_agents: int,
    experiments_per_agent: int,
    comm_every: int,
    comm_n_peers: int,
    seed: int = 42,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""

    world = create_world(difficulty=difficulty, seed=seed)

    llm = OpenAICompatibleLLM(
        model=model,
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
            all_done = all(s.get("done", False) for s in states)

            if all_done:
                break
            if time.time() - start_time > timeout:
                break

        # Collect results
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

    # Evaluate
    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    return {
        "n_agents": n_agents,
        "experiments_per_agent": experiments_per_agent,
        "total_experiments": sum(s["steps"] for s in final_states),
        "total_observations": sum(s["n_observations"] for s in final_states),
        "total_messages_sent": sum(s["messages_sent"] for s in final_states),
        "total_messages_recv": sum(s["messages_received"] for s in final_states),
        "elapsed_seconds": elapsed,
        "discovery_score": eval_result["score"],
        "rules_found": eval_result["rules_found"],
        "rules_missed": eval_result["rules_missed"],
        "hypotheses": combined[:5],
    }


async def main():
    init_logging("ERROR")

    print("=" * 80)
    print("Communication Strategy Comparison")
    print("=" * 80)
    print(f"Models: {len(MODELS)}")
    print(f"Experiment counts: {EXPERIMENT_COUNTS}")
    print(f"Communication strategies: {[s[0] for s in COMM_STRATEGIES]}")
    print()

    results = load_results()
    difficulty = "medium"

    for model in MODELS:
        model_short = model.split("/")[-1]
        print(f"\n{'#'*80}")
        print(f"# Model: {model_short}")
        print("#" * 80)

        for strategy_name, comm_every, comm_n_peers, n_agents in COMM_STRATEGIES:
            print(f"\n  Strategy: {strategy_name} ({n_agents} agents)")

            best_score = 0
            for exp_count in EXPERIMENT_COUNTS:
                key = f"{difficulty}|{model}|{strategy_name}|{exp_count}"

                # Skip if already have result (and not error)
                if key in results and "error" not in results[key]:
                    cached = results[key]
                    score = cached["discovery_score"]
                    print(f"    {exp_count} exp: {score}/100 (cached)")
                    if score > best_score:
                        best_score = score
                    if score >= 100:
                        print(f"    -> Stopping early (100% reached)")
                        break
                    continue

                print(f"    {exp_count} exp: ", end="", flush=True)

                try:
                    result = await run_experiment(
                        model=model,
                        difficulty=difficulty,
                        n_agents=n_agents,
                        experiments_per_agent=exp_count,
                        comm_every=comm_every,
                        comm_n_peers=comm_n_peers,
                    )

                    result["model"] = model
                    result["difficulty"] = difficulty
                    result["strategy"] = strategy_name
                    results[key] = result
                    save_results(results)

                    score = result["discovery_score"]
                    print(f"{score}/100 ({result['elapsed_seconds']:.0f}s, {result['total_messages_sent']} msgs)")

                    if score > best_score:
                        best_score = score

                    # Stop early if 100%
                    if score >= 100:
                        print(f"    -> Stopping early (100% reached)")
                        break

                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key] = {"error": str(e), "model": model, "strategy": strategy_name}
                    save_results(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model in MODELS:
        model_short = model.split("/")[-1][:30]
        print(f"\n{model_short}:")
        for strategy_name, _, _, _ in COMM_STRATEGIES:
            best_score = 0
            best_exp = 0
            for exp_count in EXPERIMENT_COUNTS:
                key = f"{difficulty}|{model}|{strategy_name}|{exp_count}"
                if key in results and "discovery_score" in results.get(key, {}):
                    score = results[key]["discovery_score"]
                    if score > best_score:
                        best_score = score
                        best_exp = exp_count
            print(f"  {strategy_name}: {best_score}/100 @ {best_exp} exp")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
