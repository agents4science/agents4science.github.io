#!/usr/bin/env python3
"""
Compare Rule Discovery performance across different LLM models and agent counts.

Tests the Argonne ALCF inference endpoints with various Llama-3 models.
Results are cached to JSON to avoid recomputation.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import (
    create_world,
    ScientistAgent,
    evaluate_hypotheses,
    SCIENTIST_SYSTEM_PROMPT,
)
from llm_providers import create_llm_provider, OpenAICompatibleLLM

from concurrent.futures import ThreadPoolExecutor

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager


# Argonne ALCF Configuration
ARGONNE_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
ARGONNE_API_KEY = os.getenv("ARGONNE_API_KEY", "")

# Models to test
# Note: Llama-3.3-70B-Instruct is often unavailable (503 errors) at Argonne
MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    # "meta-llama/Llama-3.3-70B-Instruct",  # Commented out - often unavailable
]

# Agent counts to test
AGENT_COUNTS = [2, 4, 8]

# Difficulties to test
DIFFICULTIES = ["easy", "medium", "hard"]

# Results file
RESULTS_FILE = Path(__file__).parent / "results.json"


def load_results() -> Dict:
    """Load cached results from JSON file."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: Dict):
    """Save results to JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def get_result_key(difficulty: str, model: str, n_agents: int) -> str:
    """Generate a unique key for a result."""
    return f"{difficulty}|{model}|{n_agents}"


async def run_experiment(
    n_agents: int,
    llm,
    world,
    steps: int = 30,
    comm_prob: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single experiment and return results."""

    start_time = time.time()

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=n_agents + 4),
    ) as manager:

        handles: List[Handle[ScientistAgent]] = []
        for i in range(n_agents):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llm, comm_prob, seed + i),
            )
            handles.append(h)

        # Wire full connectivity
        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(n_agents) if j != i]
            await h.set_peers(peers)

        # Run simulation
        for step in range(1, steps + 1):
            await asyncio.sleep(1.0)

            if step % 10 == 0:
                states = await asyncio.gather(*[h.get_state() for h in handles])
                total_obs = sum(s["n_observations"] for s in states)
                total_hyp = sum(s["n_hypotheses"] for s in states)
                print(f"    Step {step}: {total_obs} observations, {total_hyp} hypotheses")

        # Collect final results
        final_states = await asyncio.gather(*[h.get_state() for h in handles])
        all_hypotheses = await asyncio.gather(*[h.get_hypotheses() for h in handles])

        # Shutdown
        for h in handles:
            await h.shutdown()

    elapsed = time.time() - start_time

    # Combine hypotheses
    combined = []
    for hyps in all_hypotheses:
        combined.extend(hyps)
    combined = sorted(combined, key=lambda x: -x.get("confidence", 0))[:10]

    # Evaluate
    eval_result = evaluate_hypotheses(combined, world.rules, llm)

    return {
        "n_agents": n_agents,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "total_observations": sum(s["n_observations"] for s in final_states),
        "total_messages_sent": sum(s["messages_sent"] for s in final_states),
        "total_messages_recv": sum(s["messages_received"] for s in final_states),
        "discovery_score": eval_result["score"],
        "hypotheses": combined[:5],
        "evaluation": eval_result["evaluation"],
    }


async def main():
    init_logging("WARNING")

    # Check API key
    api_key = ARGONNE_API_KEY
    if not api_key:
        print("ERROR: ARGONNE_API_KEY environment variable not set")
        print("Usage: ARGONNE_API_KEY=your-key python compare_models.py")
        sys.exit(1)

    print("=" * 80)
    print("Rule Discovery: Comparing LLM Models and Agent Counts")
    print("=" * 80)
    print(f"\nArgonne Endpoint: {ARGONNE_BASE_URL}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Agent counts: {AGENT_COUNTS}")
    print(f"Difficulties: {DIFFICULTIES}")

    SEED = 42
    STEPS = 30

    # Load existing results
    all_results = load_results()
    print(f"\nLoaded {len(all_results)} cached results from {RESULTS_FILE}")

    # Count total runs needed
    total_runs = len(DIFFICULTIES) * len(MODELS) * len(AGENT_COUNTS)
    skipped = 0
    run_count = 0

    for difficulty in DIFFICULTIES:
        world = create_world(difficulty=difficulty, seed=SEED)
        print(f"\n{'#'*80}")
        print(f"# DIFFICULTY: {difficulty.upper()} ({len(world.rules)} rules)")
        print("#" * 80)
        print(world.get_rules_description())

        for model in MODELS:
            print(f"\n{'='*80}")
            print(f"Model: {model}")
            print("=" * 80)

            # Create LLM provider for this model
            llm = OpenAICompatibleLLM(
                model=model,
                base_url=ARGONNE_BASE_URL,
                api_key=api_key,
            )

            for n_agents in AGENT_COUNTS:
                run_count += 1
                key = get_result_key(difficulty, model, n_agents)

                # Check if we already have this result (and it's not an error)
                if key in all_results and "error" not in all_results[key]:
                    cached = all_results[key]
                    print(f"\n  [{run_count}/{total_runs}] {n_agents} agents... CACHED")
                    print(f"    Score: {cached['discovery_score']}/100")
                    skipped += 1
                    continue

                print(f"\n  [{run_count}/{total_runs}] {n_agents} agents...")

                try:
                    result = await run_experiment(
                        n_agents=n_agents,
                        llm=llm,
                        world=world,
                        steps=STEPS,
                        seed=SEED,
                    )
                    result["difficulty"] = difficulty
                    result["model"] = model
                    all_results[key] = result

                    print(f"    Score: {result['discovery_score']}/100")
                    print(f"    Time: {result['elapsed_seconds']:.1f}s")
                    print(f"    Observations: {result['total_observations']}")

                    # Save after each successful run
                    save_results(all_results)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results[key] = {"error": str(e), "difficulty": difficulty, "model": model, "n_agents": n_agents}
                    save_results(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skipped {skipped} cached results, ran {run_count - skipped} new experiments")

    for difficulty in DIFFICULTIES:
        print(f"\n--- {difficulty.upper()} ---")
        print(f"{'Model':<45} {'Agents':>8} {'Score':>8} {'Time':>10} {'Obs':>8}")
        print("-" * 85)

        for model in MODELS:
            for n_agents in AGENT_COUNTS:
                key = get_result_key(difficulty, model, n_agents)
                result = all_results.get(key, {})
                if "error" in result:
                    print(f"{model:<45} {n_agents:>8} {'ERROR':>8} {'-':>10} {'-':>8}")
                elif result:
                    score = result.get("discovery_score", 0)
                    elapsed = result.get("elapsed_seconds", 0)
                    obs = result.get("total_observations", 0)
                    print(f"{model:<45} {n_agents:>8} {score:>8} {elapsed:>9.1f}s {obs:>8}")
                else:
                    print(f"{model:<45} {n_agents:>8} {'MISSING':>8} {'-':>10} {'-':>8}")

    # Best results per difficulty
    print("\n" + "=" * 80)
    print("BEST SCORES")
    print("=" * 80)
    for difficulty in DIFFICULTIES:
        print(f"\n{difficulty.upper()}:")
        for model in MODELS:
            best_score = 0
            best_agents = 0
            for n_agents in AGENT_COUNTS:
                key = get_result_key(difficulty, model, n_agents)
                result = all_results.get(key, {})
                score = result.get("discovery_score", 0)
                if score > best_score:
                    best_score = score
                    best_agents = n_agents
            print(f"  {model}: {best_score}/100 with {best_agents} agents")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
