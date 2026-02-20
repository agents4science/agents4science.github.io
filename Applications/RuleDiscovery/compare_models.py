#!/usr/bin/env python3
"""
Compare Rule Discovery performance across different LLM models and agent counts.

Tests the Argonne ALCF inference endpoints with various models.
Results are cached to JSON to avoid recomputation.
Per-run logs are saved to logs/ directory.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import (
    create_world,
    ScientistAgent,
    evaluate_hypotheses,
    SCIENTIST_SYSTEM_PROMPT,
    set_agent_logging,
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
ARGONNE_API_KEY = os.getenv("ARGONNE_API_KEY", "") or os.getenv("LLM_API_KEY", "")

# Models to test (good mix of sizes and architectures)
MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",      # 8B baseline
    "openai/gpt-oss-20b",                          # 20B
    "google/gemma-3-27b-it",                       # 27B Gemma
    "meta-llama/Meta-Llama-3.1-70B-Instruct",     # 70B
    "openai/gpt-oss-120b",                         # 120B
    "mistralai/Mixtral-8x22B-Instruct-v0.1",      # MoE ~176B
    "meta-llama/Meta-Llama-3.1-405B-Instruct",    # 405B
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",  # Llama 4
]

# Agent counts to test
AGENT_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

# Difficulties to test
DIFFICULTIES = ["easy", "medium", "hard"]

# Results and logs
RESULTS_FILE = Path(__file__).parent / "results.json"
LOGS_DIR = Path(__file__).parent / "logs"

# Use a reliable model for evaluation (not the model being tested)
EVAL_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


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


def get_log_path(difficulty: str, model: str, n_agents: int) -> Path:
    """Generate log file path for a run."""
    # Sanitize model name for filename
    model_safe = model.replace("/", "_").replace(":", "_")
    log_dir = LOGS_DIR / difficulty
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{model_safe}_agents{n_agents}.log"


async def run_experiment(
    n_agents: int,
    llm,
    world,
    log_file,
    eval_llm,  # Separate LLM for evaluation (use a reliable model)
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
                args=(i, world, llm, comm_prob, seed + i, log_file),
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
                msg = f"    Step {step}: {total_obs} observations, {total_hyp} hypotheses"
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

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

    # Evaluate using separate eval LLM (reliable model)
    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    return {
        "n_agents": n_agents,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "total_observations": sum(s["n_observations"] for s in final_states),
        "total_messages_sent": sum(s["messages_sent"] for s in final_states),
        "total_messages_recv": sum(s["messages_received"] for s in final_states),
        "discovery_score": eval_result["score"],
        "hypotheses": combined[:5],
        "rules_found": eval_result["rules_found"],
        "rules_missed": eval_result["rules_missed"],
    }


async def main():
    init_logging("ERROR")

    # Check API key
    api_key = ARGONNE_API_KEY
    if not api_key:
        print("ERROR: ARGONNE_API_KEY environment variable not set")
        print("Usage: ARGONNE_API_KEY=your-key python compare_models.py")
        sys.exit(1)

    # Create logs directory
    LOGS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("Rule Discovery: Comparing LLM Models and Agent Counts")
    print("=" * 80)
    print(f"\nArgonne Endpoint: {ARGONNE_BASE_URL}")
    print(f"Models ({len(MODELS)}): {', '.join(m.split('/')[-1] for m in MODELS)}")
    print(f"Agent counts: {AGENT_COUNTS}")
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"Logs directory: {LOGS_DIR}")

    SEED = 42
    STEPS = 30

    # Create a reliable LLM for evaluation (separate from models being tested)
    eval_llm = OpenAICompatibleLLM(
        model=EVAL_MODEL,
        base_url=ARGONNE_BASE_URL,
        api_key=api_key,
        n_agents=1,
    )
    print(f"Evaluation model: {EVAL_MODEL}")

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

        for n_agents in AGENT_COUNTS:
            print(f"\n{'='*80}")
            print(f"{n_agents} Agent(s)")
            print("=" * 80)

            for model in MODELS:
                run_count += 1
                key = get_result_key(difficulty, model, n_agents)

                # Check if we already have this result (and it's not an error)
                if key in all_results and "error" not in all_results[key]:
                    cached = all_results[key]
                    print(f"\n  [{run_count}/{total_runs}] {model}... CACHED")
                    print(f"    Score: {cached['discovery_score']}/100")
                    skipped += 1
                    continue

                print(f"\n  [{run_count}/{total_runs}] {model}...")

                # Create LLM provider for this model
                llm = OpenAICompatibleLLM(
                    model=model,
                    base_url=ARGONNE_BASE_URL,
                    api_key=api_key,
                    n_agents=n_agents,
                )

                # Setup log file
                log_path = get_log_path(difficulty, model, n_agents)

                try:
                    with open(log_path, "w") as log_file:
                        # Write header
                        log_file.write(f"{'='*80}\n")
                        log_file.write(f"Run: {difficulty} | {model} | {n_agents} agents\n")
                        log_file.write(f"Started: {datetime.now().isoformat()}\n")
                        log_file.write(f"{'='*80}\n\n")
                        log_file.write(f"Rules to discover:\n{world.get_rules_description()}\n\n")
                        log_file.flush()

                        result = await run_experiment(
                            n_agents=n_agents,
                            llm=llm,
                            world=world,
                            log_file=log_file,
                            eval_llm=eval_llm,
                            steps=STEPS,
                            seed=SEED,
                        )

                        # Write results to log
                        log_file.write(f"\n{'='*80}\n")
                        log_file.write(f"RESULTS\n")
                        log_file.write(f"{'='*80}\n")
                        log_file.write(f"Score: {result['discovery_score']}/100\n")
                        log_file.write(f"Time: {result['elapsed_seconds']:.1f}s\n")
                        log_file.write(f"Observations: {result['total_observations']}\n")
                        log_file.write(f"Rules found: {result['rules_found']}\n")
                        log_file.write(f"Rules missed: {result['rules_missed']}\n")
                        log_file.write(f"\nTop hypotheses:\n")
                        for h in result['hypotheses']:
                            log_file.write(f"  - {h['rule']} ({h['confidence']:.0%})\n")

                    result["difficulty"] = difficulty
                    result["model"] = model
                    result["log_file"] = str(log_path)
                    all_results[key] = result

                    print(f"    Score: {result['discovery_score']}/100")
                    print(f"    Time: {result['elapsed_seconds']:.1f}s")
                    print(f"    Observations: {result['total_observations']}")
                    print(f"    Log: {log_path}")

                    # Save after each successful run
                    save_results(all_results)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    all_results[key] = {"error": str(e), "difficulty": difficulty, "model": model, "n_agents": n_agents}
                    save_results(all_results)

                    # Log the error
                    with open(log_path, "a") as log_file:
                        log_file.write(f"\nERROR: {e}\n")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skipped {skipped} cached results, ran {run_count - skipped} new experiments")

    for difficulty in DIFFICULTIES:
        print(f"\n--- {difficulty.upper()} ---")
        print(f"{'Model':<50} {'Agents':>6} {'Score':>6} {'Time':>8} {'Obs':>6}")
        print("-" * 80)

        for n_agents in AGENT_COUNTS:
            for model in MODELS:
                key = get_result_key(difficulty, model, n_agents)
                result = all_results.get(key, {})
                model_short = model.split("/")[-1][:40]
                if "error" in result:
                    print(f"{model_short:<50} {n_agents:>6} {'ERR':>6} {'-':>8} {'-':>6}")
                elif result:
                    score = result.get("discovery_score", 0)
                    elapsed = result.get("elapsed_seconds", 0)
                    obs = result.get("total_observations", 0)
                    print(f"{model_short:<50} {n_agents:>6} {score:>6} {elapsed:>7.0f}s {obs:>6}")

    # Best results per difficulty
    print("\n" + "=" * 80)
    print("BEST SCORES PER MODEL")
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
            model_short = model.split("/")[-1]
            print(f"  {model_short}: {best_score}/100 with {best_agents} agents")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Logs saved to: {LOGS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
