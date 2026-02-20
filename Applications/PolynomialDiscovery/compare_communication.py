#!/usr/bin/env python3
"""
Compare convergence rates with and without communication.
"""

import asyncio
import math
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
    topology: str = "ring",
) -> Dict[str, List]:
    """Run a single experiment and return history."""

    eval_grid = [(-1.0 + 2.0 * i / 199.0) for i in range(200)]

    history: Dict[str, List] = {
        "steps": [],
        "best_mse": [],
        "median_mse": [],
        "worst_mse": [],
        "diversity": [],
        "agent_mses": [],
    }

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=32),
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

        # Wire topology
        rnd = random.Random(agent_seed + 999)
        peer_lists = build_topology(handles, topology, rnd, k=3)
        for i, h in enumerate(handles):
            await h.set_peers(peer_lists[i])

        # Run & observe
        t0 = time.time()
        last_report_step = 0

        for step in range(1, steps + 1):
            await asyncio.sleep(0.05)

            if step - last_report_step >= eval_every:
                last_report_step = step

                models = await asyncio.gather(*[h.get_model() for h in handles])

                # Compute true MSEs
                mses_list: List[Tuple[float, int]] = []
                agent_mse_list: List[float] = []
                for m in models:
                    theta = [float(x) for x in m["theta"]]
                    err = mse(theta, world, eval_grid)
                    mses_list.append((err, int(m["agent_id"])))
                    agent_mse_list.append(err)

                mses_sorted = sorted(mses_list, key=lambda x: x[0])
                best_err, best_id = mses_sorted[0]
                med_err = mses_sorted[len(mses_sorted) // 2][0]
                worst_err = mses_sorted[-1][0]

                thetas = [[float(x) for x in m["theta"]] for m in models]
                div = avg_pairwise_l2(thetas)

                history["steps"].append(step)
                history["best_mse"].append(best_err)
                history["median_mse"].append(med_err)
                history["worst_mse"].append(worst_err)
                history["diversity"].append(div)
                history["agent_mses"].append(agent_mse_list)

                print(f"  [comm={comm_prob:.2f}] step={step:4d} best={best_err:.6f} median={med_err:.6f}")

        # Get comm stats
        comm_stats = await asyncio.gather(*[h.get_comm_stats() for h in handles])
        total_sent = sum(s["messages_sent"] for s in comm_stats)
        history["total_messages"] = total_sent

        # Shutdown
        for h in handles:
            await h.shutdown()

    return history


def plot_comparison(histories: Dict[str, Dict], save_path: str = "convergence_comparison.png") -> None:
    """Plot comparison of convergence rates."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "No Communication": "red",
        "Ring Topology (12%)": "green",
        "Random Topology (12%)": "blue",
    }

    # 1. Best MSE comparison (top-left)
    ax = axes[0, 0]
    for label, hist in histories.items():
        ax.semilogy(hist["steps"], hist["best_mse"], '-', linewidth=2,
                   color=colors[label], label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Best MSE (log scale)')
    ax.set_title('Best Agent Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Median MSE comparison (top-right)
    ax = axes[0, 1]
    for label, hist in histories.items():
        ax.semilogy(hist["steps"], hist["median_mse"], '-', linewidth=2,
                   color=colors[label], label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Median MSE (log scale)')
    ax.set_title('Median Agent Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Worst MSE comparison (bottom-left)
    ax = axes[1, 0]
    for label, hist in histories.items():
        ax.semilogy(hist["steps"], hist["worst_mse"], '-', linewidth=2,
                   color=colors[label], label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Worst MSE (log scale)')
    ax.set_title('Worst Agent Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Diversity comparison (bottom-right)
    ax = axes[1, 1]
    for label, hist in histories.items():
        ax.plot(hist["steps"], hist["diversity"], '-', linewidth=2,
               color=colors[label], label=label, marker='o', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Diversity (Avg L2)')
    ax.set_title('Model Diversity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Convergence: With vs Without Communication (16 agents)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_agents_comparison(histories: Dict[str, Dict], n_agents: int,
                                save_path: str = "agent_comparison.png") -> None:
    """Plot individual agent trajectories side by side."""
    n_scenarios = len(histories)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 6))
    if n_scenarios == 1:
        axes = [axes]

    cmap = plt.cm.viridis(np.linspace(0, 1, n_agents))

    for idx, (label, hist) in enumerate(histories.items()):
        ax = axes[idx]
        steps = hist["steps"]

        for agent_idx in range(n_agents):
            agent_mses = [snapshot[agent_idx] for snapshot in hist["agent_mses"]]
            ax.semilogy(steps, agent_mses, '-', color=cmap[agent_idx],
                       linewidth=1.2, alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('MSE (log scale)')
        ax.set_title(f'{label}\n(Total msgs: {hist.get("total_messages", 0)})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1e-6, 10])

    plt.suptitle('Individual Agent Trajectories by Communication Strategy',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_comparison(histories: Dict[str, Dict],
                            save_path: str = "summary_comparison.png") -> None:
    """Single plot comparing median MSE with shaded regions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    styles = {
        "No Communication": {"color": "red", "linestyle": "-"},
        "Ring Topology (12%)": {"color": "green", "linestyle": "-"},
        "Random Topology (12%)": {"color": "blue", "linestyle": "-"},
    }

    for label, hist in histories.items():
        steps = hist["steps"]
        style = styles[label]

        # Plot median with best/worst shading
        ax.semilogy(steps, hist["median_mse"], style["linestyle"],
                   linewidth=2.5, color=style["color"], label=f'{label} (median)')
        ax.fill_between(steps, hist["best_mse"], hist["worst_mse"],
                       alpha=0.2, color=style["color"])

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Convergence Rate by Communication Strategy\n(16 agents, shaded = best to worst)',
                fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


async def main():
    init_logging("WARNING")  # Reduce noise

    N_AGENTS = 16
    WORLD_SEED = 7
    AGENT_SEED = 123
    STEPS = 300

    world = make_world(WORLD_SEED)

    print("=" * 60)
    print("Running comparison: Communication Topologies")
    print("=" * 60)

    histories = {}

    # Run WITHOUT communication
    print("\n[1/3] Running WITHOUT communication (COMM_PROB=0.0)...")
    hist_no_comm = await run_experiment(
        n_agents=N_AGENTS,
        comm_prob=0.0,
        world=world,
        steps=STEPS,
        agent_seed=AGENT_SEED,
        topology="ring",
    )
    histories["No Communication"] = hist_no_comm

    # Run WITH communication - RING topology
    print("\n[2/3] Running with RING topology (COMM_PROB=0.12)...")
    hist_ring = await run_experiment(
        n_agents=N_AGENTS,
        comm_prob=0.12,
        world=world,
        steps=STEPS,
        agent_seed=AGENT_SEED,
        topology="ring",
    )
    histories["Ring Topology (12%)"] = hist_ring

    # Run WITH communication - RANDOM topology
    print("\n[3/3] Running with RANDOM topology (COMM_PROB=0.12)...")
    hist_random = await run_experiment(
        n_agents=N_AGENTS,
        comm_prob=0.12,
        world=world,
        steps=STEPS,
        agent_seed=AGENT_SEED,
        topology="random",
    )
    histories["Random Topology (12%)"] = hist_random

    # Generate comparison plots
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    plot_comparison(histories, "convergence_comparison.png")
    plot_all_agents_comparison(histories, N_AGENTS, "agent_comparison.png")
    plot_summary_comparison(histories, "summary_comparison.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for label, hist in histories.items():
        final_best = hist["best_mse"][-1]
        final_median = hist["median_mse"][-1]
        final_worst = hist["worst_mse"][-1]
        final_div = hist["diversity"][-1]
        total_msgs = hist.get("total_messages", 0)

        print(f"\n{label}:")
        print(f"  Final Best MSE:   {final_best:.6f}")
        print(f"  Final Median MSE: {final_median:.6f}")
        print(f"  Final Worst MSE:  {final_worst:.6f}")
        print(f"  Final Diversity:  {final_div:.4f}")
        print(f"  Total Messages:   {total_msgs}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
