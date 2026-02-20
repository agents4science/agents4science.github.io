#!/usr/bin/env python3
"""
Compare performance across different agent counts.
Tests scaling: 1, 2, 4, 8, 16, 32, 64 agents
"""

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager

from independent_scientists_academy import (
    IndependentScientist,
    HiddenWorld,
    make_world,
    mse,
    build_topology,
    avg_pairwise_l2,
)


async def run_experiment(
    n_agents: int,
    comm_prob: float,
    world: HiddenWorld,
    steps: int = 300,
    eval_every: int = 20,
    agent_seed: int = 123,
    topology: str = "random",
) -> Dict[str, Any]:
    """Run a single experiment and return history."""

    eval_grid = [(-1.0 + 2.0 * i / 199.0) for i in range(200)]

    history: Dict[str, List] = {
        "steps": [],
        "best_mse": [],
        "median_mse": [],
        "worst_mse": [],
        "diversity": [],
    }

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=max(32, n_agents + 4)),
    ) as manager:
        # Launch agents
        handles: List[Handle[IndependentScientist]] = []
        for i in range(n_agents):
            h = await manager.launch(
                IndependentScientist,
                args=(
                    i,
                    world,
                    0.07,  # noise_std
                    comm_prob,
                    "summary",  # bandwidth
                    "blend",  # integration
                    8,  # max_shared_samples
                    agent_seed * 1000 + i,
                ),
            )
            handles.append(h)

        # Wire topology (for n_agents=1, no peers)
        if n_agents > 1:
            rnd = random.Random(agent_seed + 999)
            peer_lists = build_topology(handles, topology, rnd, k=min(3, n_agents-1))
            for i, h in enumerate(handles):
                await h.set_peers(peer_lists[i])

        # Run & observe
        for step in range(1, steps + 1):
            await asyncio.sleep(0.05)

            if step % eval_every == 0:
                models = await asyncio.gather(*[h.get_model() for h in handles])

                # Compute true MSEs
                mses_list: List[float] = []
                for m in models:
                    theta = [float(x) for x in m["theta"]]
                    err = mse(theta, world, eval_grid)
                    mses_list.append(err)

                mses_sorted = sorted(mses_list)
                best_err = mses_sorted[0]
                med_err = mses_sorted[len(mses_sorted) // 2]
                worst_err = mses_sorted[-1]

                thetas = [[float(x) for x in m["theta"]] for m in models]
                div = avg_pairwise_l2(thetas) if n_agents > 1 else 0.0

                history["steps"].append(step)
                history["best_mse"].append(best_err)
                history["median_mse"].append(med_err)
                history["worst_mse"].append(worst_err)
                history["diversity"].append(div)

        # Get comm stats
        comm_stats = await asyncio.gather(*[h.get_comm_stats() for h in handles])
        total_sent = sum(s["messages_sent"] for s in comm_stats)

        # Shutdown
        for h in handles:
            await h.shutdown()

    return {
        "history": history,
        "final_best": history["best_mse"][-1],
        "final_median": history["median_mse"][-1],
        "final_worst": history["worst_mse"][-1],
        "final_diversity": history["diversity"][-1],
        "total_messages": total_sent,
        "n_agents": n_agents,
    }


def plot_scaling(results_comm: Dict[int, Dict], results_no_comm: Dict[int, Dict],
                 agent_counts: List[int], save_path: str = "agent_scaling.png") -> None:
    """Plot how performance scales with agent count."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    c_comm = "blue"
    c_no_comm = "red"

    # 1. Final Best MSE vs Agent Count
    ax = axes[0, 0]
    best_comm = [results_comm[n]["final_best"] for n in agent_counts]
    best_no_comm = [results_no_comm[n]["final_best"] for n in agent_counts]
    ax.semilogy(agent_counts, best_no_comm, 'o-', color=c_no_comm, linewidth=2, markersize=8, label='No Communication')
    ax.semilogy(agent_counts, best_comm, 's-', color=c_comm, linewidth=2, markersize=8, label='Random Topology (12%)')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Final Best MSE (log)')
    ax.set_title('Best Agent Performance vs Agent Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agent_counts)

    # 2. Final Median MSE vs Agent Count
    ax = axes[0, 1]
    med_comm = [results_comm[n]["final_median"] for n in agent_counts]
    med_no_comm = [results_no_comm[n]["final_median"] for n in agent_counts]
    ax.semilogy(agent_counts, med_no_comm, 'o-', color=c_no_comm, linewidth=2, markersize=8, label='No Communication')
    ax.semilogy(agent_counts, med_comm, 's-', color=c_comm, linewidth=2, markersize=8, label='Random Topology (12%)')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Final Median MSE (log)')
    ax.set_title('Median Agent Performance vs Agent Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agent_counts)

    # 3. Final Worst MSE vs Agent Count
    ax = axes[1, 0]
    worst_comm = [results_comm[n]["final_worst"] for n in agent_counts]
    worst_no_comm = [results_no_comm[n]["final_worst"] for n in agent_counts]
    ax.semilogy(agent_counts, worst_no_comm, 'o-', color=c_no_comm, linewidth=2, markersize=8, label='No Communication')
    ax.semilogy(agent_counts, worst_comm, 's-', color=c_comm, linewidth=2, markersize=8, label='Random Topology (12%)')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Final Worst MSE (log)')
    ax.set_title('Worst Agent Performance vs Agent Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agent_counts)

    # 4. Total Messages vs Agent Count
    ax = axes[1, 1]
    msgs_comm = [results_comm[n]["total_messages"] for n in agent_counts]
    ax.plot(agent_counts, msgs_comm, 's-', color=c_comm, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Total Messages')
    ax.set_title('Communication Volume vs Agent Count')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agent_counts)

    # Add msgs per agent annotation
    for i, n in enumerate(agent_counts):
        if n > 0 and msgs_comm[i] > 0:
            per_agent = msgs_comm[i] / n
            ax.annotate(f'{per_agent:.0f}/agent', (n, msgs_comm[i]),
                       textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.suptitle('Scaling: How Performance Changes with Number of Async Agents\n(MSE measured against hidden ground-truth function)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_by_count(results_comm: Dict[int, Dict], agent_counts: List[int],
                               save_path: str = "convergence_by_count.png") -> None:
    """Plot convergence curves for different agent counts."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(agent_counts)))

    # 1. Median MSE convergence
    ax = axes[0]
    for i, n in enumerate(agent_counts):
        hist = results_comm[n]["history"]
        ax.semilogy(hist["steps"], hist["median_mse"], '-', color=cmap[i],
                   linewidth=2, label=f'{n} agents')
    ax.set_xlabel('Step')
    ax.set_ylabel('Median MSE (log)')
    ax.set_title('Median MSE Convergence by Agent Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Best MSE convergence
    ax = axes[1]
    for i, n in enumerate(agent_counts):
        hist = results_comm[n]["history"]
        ax.semilogy(hist["steps"], hist["best_mse"], '-', color=cmap[i],
                   linewidth=2, label=f'{n} agents')
    ax.set_xlabel('Step')
    ax.set_ylabel('Best MSE (log)')
    ax.set_title('Best MSE Convergence by Agent Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Convergence Speed vs Number of Async Agents (Random Topology, 12% comm)\nMSE = error vs hidden ground-truth',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_speedup(results_comm: Dict[int, Dict], results_no_comm: Dict[int, Dict],
                 agent_counts: List[int], save_path: str = "speedup.png") -> None:
    """Plot speedup/improvement ratios."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Communication benefit ratio (no_comm / comm)
    ax = axes[0]
    med_ratio = [results_no_comm[n]["final_median"] / results_comm[n]["final_median"]
                 for n in agent_counts if n > 1]  # Skip n=1
    counts_filtered = [n for n in agent_counts if n > 1]
    ax.bar(range(len(counts_filtered)), med_ratio, color='green', alpha=0.7)
    ax.set_xticks(range(len(counts_filtered)))
    ax.set_xticklabels([str(n) for n in counts_filtered])
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Improvement Ratio')
    ax.set_title('Communication Benefit\n(No-Comm MSE / With-Comm MSE)')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(med_ratio):
        ax.text(i, v + 0.05, f'{v:.2f}x', ha='center', fontsize=9)

    # 2. Single agent baseline comparison
    ax = axes[1]
    single_agent_mse = results_no_comm[1]["final_median"]
    improvement_vs_single = [single_agent_mse / results_comm[n]["final_median"]
                             for n in agent_counts]
    ax.plot(agent_counts, improvement_vs_single, 's-', color='blue', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Improvement vs Single Agent')
    ax.set_title('Scaling Efficiency\n(Single Agent MSE / N-Agent MSE)')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(agent_counts)

    plt.suptitle('Communication Benefits and Scaling Analysis',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


async def main():
    init_logging("WARNING")

    WORLD_SEED = 7
    AGENT_SEED = 123
    STEPS = 300

    AGENT_COUNTS = [1, 2, 4, 8, 16, 32, 64]

    world = make_world(WORLD_SEED)

    print("=" * 70)
    print(f"Scaling Analysis: {AGENT_COUNTS} agents")
    print(f"Problem: f(x) = {world.a:.3f}x³ + {world.b:.3f}x² + {world.c:.3f}x + {world.d:.3f}")
    print("=" * 70)

    results_comm: Dict[int, Dict] = {}
    results_no_comm: Dict[int, Dict] = {}

    total_runs = len(AGENT_COUNTS) * 2
    run_count = 0

    for n_agents in AGENT_COUNTS:
        print(f"\n--- {n_agents} agents ---")

        # With communication
        run_count += 1
        print(f"  [{run_count}/{total_runs}] With communication...", end=" ", flush=True)
        result = await run_experiment(
            n_agents=n_agents,
            comm_prob=0.12 if n_agents > 1 else 0.0,
            world=world,
            steps=STEPS,
            agent_seed=AGENT_SEED,
            topology="random",
        )
        results_comm[n_agents] = result
        print(f"median={result['final_median']:.6f}, msgs={result['total_messages']}")

        # Without communication
        run_count += 1
        print(f"  [{run_count}/{total_runs}] No communication...", end=" ", flush=True)
        result = await run_experiment(
            n_agents=n_agents,
            comm_prob=0.0,
            world=world,
            steps=STEPS,
            agent_seed=AGENT_SEED,
            topology="random",
        )
        results_no_comm[n_agents] = result
        print(f"median={result['final_median']:.6f}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    plot_scaling(results_comm, results_no_comm, AGENT_COUNTS, "agent_scaling.png")
    plot_convergence_by_count(results_comm, AGENT_COUNTS, "convergence_by_count.png")
    plot_speedup(results_comm, results_no_comm, AGENT_COUNTS, "speedup.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Agents':>8} | {'With Comm':>12} | {'No Comm':>12} | {'Improvement':>12} | {'Messages':>10}")
    print("-" * 65)
    for n in AGENT_COUNTS:
        comm_mse = results_comm[n]["final_median"]
        no_comm_mse = results_no_comm[n]["final_median"]
        improvement = no_comm_mse / comm_mse if comm_mse > 0 else 0
        msgs = results_comm[n]["total_messages"]
        print(f"{n:>8} | {comm_mse:>12.6f} | {no_comm_mse:>12.6f} | {improvement:>11.2f}x | {msgs:>10}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
