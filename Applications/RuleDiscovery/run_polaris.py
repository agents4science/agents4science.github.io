#!/usr/bin/env python3
"""
Run Rule Discovery agents on Polaris with embedded LLMs.

Each agent gets its own LLM instance running on a dedicated GPU.
Designed for Polaris (4 A100 GPUs per node) and Aurora (6 Intel GPUs per node).

Usage on Polaris:
    # In PBS job script:
    module load conda
    conda activate llm_env  # Your env with vLLM/transformers

    # Run with 4 agents, one per GPU
    python run_polaris.py --agents 4 --backend vllm

    # Or with specific model
    python run_polaris.py --agents 4 --model meta-llama/Llama-3.1-8B-Instruct

Usage on Aurora:
    python run_polaris.py --agents 6 --backend transformers --device xpu --gpus-per-node 6

Prerequisites:
    - PBS job with GPU allocation
    - vLLM or transformers installed
    - Model cached in ~/.cache/huggingface/
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hidden_rule_discovery import create_world, ScientistAgent, evaluate_hypotheses
from llm_providers import EmbeddedLLM, create_embedded_llms
from academy.exchange import LocalExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager


def check_environment():
    """Check HPC environment and available resources."""
    checks = {}

    # Hostname
    hostname = os.uname().nodename
    checks["hostname"] = hostname
    checks["is_polaris"] = hostname.startswith(("polaris", "x3"))
    checks["is_aurora"] = hostname.startswith(("aurora", "uan"))

    # PBS job
    checks["pbs_jobid"] = os.environ.get("PBS_JOBID")

    # CUDA/GPU
    checks["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Check for vLLM
    try:
        import vllm
        checks["vllm_available"] = True
        checks["vllm_version"] = vllm.__version__
    except ImportError:
        checks["vllm_available"] = False

    # Check for transformers
    try:
        import transformers
        checks["transformers_available"] = True
        checks["transformers_version"] = transformers.__version__
    except ImportError:
        checks["transformers_available"] = False

    # Check for torch
    try:
        import torch
        checks["torch_available"] = True
        checks["torch_version"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if checks["cuda_available"]:
            checks["cuda_device_count"] = torch.cuda.device_count()
            checks["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        checks["torch_available"] = False

    return checks


async def run_discovery(
    n_agents: int,
    model: str,
    backend: str,
    base_device: str,
    gpus_per_node: int,
    difficulty: str,
    experiments_per_agent: int,
    comm_every: int,
    comm_n_peers: int,
    seed: int,
):
    """Run Rule Discovery with embedded LLMs."""

    print("=" * 60)
    print("Rule Discovery on HPC with Embedded LLMs")
    print("=" * 60)

    # Check environment
    env = check_environment()
    print(f"\nEnvironment:")
    print(f"  Hostname: {env['hostname']}")
    print(f"  PBS Job: {env.get('pbs_jobid', 'N/A')}")
    print(f"  Backend: {backend}")
    if env.get("cuda_available"):
        print(f"  GPUs: {env.get('cuda_device_count', 0)} x {env.get('cuda_devices', ['Unknown'])[0]}")
    print(f"  vLLM: {'Yes' if env.get('vllm_available') else 'No'}")
    print(f"  Transformers: {'Yes' if env.get('transformers_available') else 'No'}")

    # Validate backend
    if backend == "vllm" and not env.get("vllm_available"):
        print("\nWARNING: vLLM not available, falling back to transformers")
        backend = "transformers"

    if backend == "transformers" and not env.get("transformers_available"):
        raise RuntimeError("Neither vLLM nor transformers available!")

    # Create world
    world = create_world(difficulty=difficulty, seed=seed)
    print(f"\nTask: {difficulty} difficulty ({len(world.rules)} rules to discover)")
    print(f"Agents: {n_agents}")
    print(f"Experiments per agent: {experiments_per_agent}")
    print(f"Model: {model}")

    # Create embedded LLMs - one per agent
    print(f"\nInitializing {n_agents} embedded LLM instances...")
    llms = create_embedded_llms(
        model=model,
        n_agents=n_agents,
        backend=backend,
        base_device=base_device,
        gpus_per_node=gpus_per_node,
    )

    for i, llm in enumerate(llms):
        print(f"  Agent {i}: {llm.name}")

    # Create a separate evaluation LLM (can reuse one of the agent LLMs)
    eval_llm = llms[0]

    start_time = time.time()

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=n_agents + 4),
    ) as manager:

        # Launch agents, each with its own LLM
        print(f"\nLaunching {n_agents} agents...")
        handles = []
        for i in range(n_agents):
            h = await manager.launch(
                ScientistAgent,
                args=(i, world, llms[i], 0.0, seed + i, None, experiments_per_agent),
                kwargs={"comm_every": comm_every, "comm_n_peers": comm_n_peers},
            )
            handles.append(h)
            print(f"  Agent {i} launched on {llms[i].device}")

        # Wire full connectivity
        for i, h in enumerate(handles):
            peers = [handles[j] for j in range(n_agents) if j != i]
            await h.set_peers(peers)

        print("\nAgents running...")

        # Wait for completion
        while True:
            await asyncio.sleep(2)
            states = await asyncio.gather(*[h.get_state() for h in handles])

            # Progress update
            total_exp = sum(s["steps"] for s in states)
            done_count = sum(1 for s in states if s.get("done", False))
            print(f"  Progress: {total_exp}/{n_agents * experiments_per_agent} experiments, "
                  f"{done_count}/{n_agents} agents done")

            if all(s.get("done", False) for s in states):
                break
            if time.time() - start_time > 3600:  # 1 hour timeout
                print("Timeout!")
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
    combined = sorted(combined, key=lambda x: -x.get("confidence", 0))[:15]

    print(f"\nTop hypotheses discovered:")
    for i, h in enumerate(combined[:8], 1):
        print(f"  {i}. {h['rule']} ({h['confidence']:.0%})")

    # Evaluate
    print("\nEvaluating against ground truth...")
    eval_result = evaluate_hypotheses(combined, world.rules, eval_llm)

    total_exp = sum(s["steps"] for s in final_states)
    total_msgs = sum(s["messages_sent"] for s in final_states)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Score: {eval_result['score']}/100")
    print(f"Rules found: {len(eval_result['rules_found'])}/{len(world.rules)}")
    print(f"Total experiments: {total_exp}")
    print(f"Messages exchanged: {total_msgs}")
    print(f"Time: {elapsed:.1f}s")

    if eval_result['rules_found']:
        print(f"\nDiscovered:")
        for rule in eval_result['rules_found']:
            print(f"  + {rule}")

    if eval_result['rules_missed']:
        print(f"\nMissed:")
        for rule in eval_result['rules_missed']:
            print(f"  - {rule}")

    # Save results
    results = {
        "score": eval_result["score"],
        "rules_found": len(eval_result["rules_found"]),
        "rules_total": len(world.rules),
        "total_experiments": total_exp,
        "messages": total_msgs,
        "elapsed": elapsed,
        "n_agents": n_agents,
        "model": model,
        "backend": backend,
        "difficulty": difficulty,
        "hostname": env["hostname"],
        "pbs_jobid": env.get("pbs_jobid"),
    }

    results_file = Path("polaris_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Rule Discovery with embedded LLMs on Polaris/Aurora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Polaris with 4 A100 GPUs
    python run_polaris.py --agents 4 --backend vllm

    # Aurora with 6 Intel GPUs
    python run_polaris.py --agents 6 --backend transformers --device xpu --gpus-per-node 6

    # Test on single GPU
    python run_polaris.py --agents 1 --backend transformers

    # With specific model
    python run_polaris.py --agents 4 --model meta-llama/Llama-3.1-8B-Instruct
        """,
    )

    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=4,
        help="Number of agents (default: 4 for Polaris)",
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["vllm", "transformers"],
        help="LLM backend (default: vllm)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Base device type (cuda for NVIDIA, xpu for Intel)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        help="GPUs per node (4 for Polaris, 6 for Aurora)",
    )
    parser.add_argument(
        "--difficulty", "-d",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Problem difficulty",
    )
    parser.add_argument(
        "--experiments", "-e",
        type=int,
        default=40,
        help="Experiments per agent",
    )
    parser.add_argument(
        "--comm-every",
        type=int,
        default=2,
        help="Share with peers every N experiments (0 = no communication)",
    )
    parser.add_argument(
        "--comm-n-peers",
        type=int,
        default=1,
        help="Number of peers to share with",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging("DEBUG" if args.verbose else "ERROR")

    # Environment check mode
    if args.check_env:
        env = check_environment()
        print("\nEnvironment Check")
        print("=" * 40)
        for key, val in env.items():
            print(f"  {key}: {val}")
        print("=" * 40)
        return

    # Run discovery
    asyncio.run(run_discovery(
        n_agents=args.agents,
        model=args.model,
        backend=args.backend,
        base_device=args.device,
        gpus_per_node=args.gpus_per_node,
        difficulty=args.difficulty,
        experiments_per_agent=args.experiments,
        comm_every=args.comm_every,
        comm_n_peers=args.comm_n_peers,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
