#!/usr/bin/env python3
"""
Compare convergence across different problems (world seeds) and communication strategies.
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
    quiet: bool = False,
) -> Dict[str, Any]:
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

        # Get comm stats
        comm_stats = await asyncio.gather(*[h.get_comm_stats() for h in handles])
        total_sent = sum(s["messages_sent"] for s in comm_stats)
        history["total_messages"] = total_sent

        # Shutdown
        for h in handles:
            await h.shutdown()

    # Return summary stats
    return {
        "history": history,
        "final_best": history["best_mse"][-1],
        "final_median": history["median_mse"][-1],
        "final_worst": history["worst_mse"][-1],
        "final_diversity": history["diversity"][-1],
        "total_messages": total_sent,
    }


def plot_problem_comparison(all_results: Dict[str, Dict[int, Dict]],
                            world_seeds: List[int],
                            n_agents: int,
                            save_path: str = "problem_comparison.png") -> None:
    """Plot comparison across problems for each strategy."""

    strategies = list(all_results.keys())
    n_problems = len(world_seeds)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(n_problems)
    width = 0.25
    colors = {"No Communication": "red", "Ring (12%)": "green", "Random (12%)": "blue"}

    # 1. Final Best MSE by problem
    ax = axes[0, 0]
    for i, strategy in enumerate(strategies):
        vals = [all_results[strategy][seed]["final_best"] for seed in world_seeds]
        ax.bar(x + i*width, vals, width, label=strategy, color=colors[strategy], alpha=0.8)
    ax.set_xlabel('Problem (World Seed)')
    ax.set_ylabel('Final Best MSE')
    ax.set_title('Best Agent Performance by Problem')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in world_seeds])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Final Median MSE by problem
    ax = axes[0, 1]
    for i, strategy in enumerate(strategies):
        vals = [all_results[strategy][seed]["final_median"] for seed in world_seeds]
        ax.bar(x + i*width, vals, width, label=strategy, color=colors[strategy], alpha=0.8)
    ax.set_xlabel('Problem (World Seed)')
    ax.set_ylabel('Final Median MSE')
    ax.set_title('Median Agent Performance by Problem')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in world_seeds])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Final Worst MSE by problem
    ax = axes[1, 0]
    for i, strategy in enumerate(strategies):
        vals = [all_results[strategy][seed]["final_worst"] for seed in world_seeds]
        ax.bar(x + i*width, vals, width, label=strategy, color=colors[strategy], alpha=0.8)
    ax.set_xlabel('Problem (World Seed)')
    ax.set_ylabel('Final Worst MSE')
    ax.set_title('Worst Agent Performance by Problem')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in world_seeds])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Final Diversity by problem
    ax = axes[1, 1]
    for i, strategy in enumerate(strategies):
        vals = [all_results[strategy][seed]["final_diversity"] for seed in world_seeds]
        ax.bar(x + i*width, vals, width, label=strategy, color=colors[strategy], alpha=0.8)
    ax.set_xlabel('Problem (World Seed)')
    ax.set_ylabel('Final Diversity (L2)')
    ax.set_title('Model Diversity by Problem')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in world_seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Performance Across Different Problems\\n({n_agents} async agents, 300 driver steps, ~{n_agents*80} msgs/run with comm)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_aggregate_stats(all_results: Dict[str, Dict[int, Dict]],
                         world_seeds: List[int],
                         n_agents: int,
                         save_path: str = "aggregate_stats.png") -> None:
    """Plot aggregate statistics across all problems."""

    strategies = list(all_results.keys())
    colors = {"No Communication": "red", "Ring (12%)": "green", "Random (12%)": "blue"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Collect data for box plots
    best_data = {s: [all_results[s][seed]["final_best"] for seed in world_seeds] for s in strategies}
    median_data = {s: [all_results[s][seed]["final_median"] for seed in world_seeds] for s in strategies}
    worst_data = {s: [all_results[s][seed]["final_worst"] for seed in world_seeds] for s in strategies}

    # 1. Best MSE distribution
    ax = axes[0]
    bp = ax.boxplot([best_data[s] for s in strategies], labels=strategies, patch_artist=True)
    for patch, strategy in zip(bp['boxes'], strategies):
        patch.set_facecolor(colors[strategy])
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Best MSE')
    ax.set_title('Best Agent MSE Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    # 2. Median MSE distribution
    ax = axes[1]
    bp = ax.boxplot([median_data[s] for s in strategies], labels=strategies, patch_artist=True)
    for patch, strategy in zip(bp['boxes'], strategies):
        patch.set_facecolor(colors[strategy])
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Median MSE')
    ax.set_title('Median Agent MSE Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    # 3. Worst MSE distribution
    ax = axes[2]
    bp = ax.boxplot([worst_data[s] for s in strategies], labels=strategies, patch_artist=True)
    for patch, strategy in zip(bp['boxes'], strategies):
        patch.set_facecolor(colors[strategy])
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Worst MSE')
    ax.set_title('Worst Agent MSE Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)

    plt.suptitle(f'Aggregate Performance: {n_agents} Async Agents Across {len(world_seeds)} Problems\\n(MSE measured against hidden ground truth)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_curves(all_results: Dict[str, Dict[int, Dict]],
                            world_seeds: List[int],
                            n_agents: int,
                            save_path: str = "convergence_curves.png") -> None:
    """Plot convergence curves averaged across problems with std bands."""

    strategies = list(all_results.keys())
    colors = {"No Communication": "red", "Ring (12%)": "green", "Random (12%)": "blue"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get steps (assume all same)
    steps = all_results[strategies[0]][world_seeds[0]]["history"]["steps"]

    # 1. Median MSE convergence
    ax = axes[0]
    for strategy in strategies:
        # Collect median MSE curves for all problems
        curves = np.array([all_results[strategy][seed]["history"]["median_mse"]
                          for seed in world_seeds])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        ax.semilogy(steps, mean_curve, '-', linewidth=2, color=colors[strategy], label=strategy)
        ax.fill_between(steps,
                       np.maximum(mean_curve - std_curve, 1e-7),
                       mean_curve + std_curve,
                       color=colors[strategy], alpha=0.2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Median MSE (log scale)')
    ax.set_title('Median MSE Convergence\n(mean ± std across problems)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Best MSE convergence
    ax = axes[1]
    for strategy in strategies:
        curves = np.array([all_results[strategy][seed]["history"]["best_mse"]
                          for seed in world_seeds])
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        ax.semilogy(steps, mean_curve, '-', linewidth=2, color=colors[strategy], label=strategy)
        ax.fill_between(steps,
                       np.maximum(mean_curve - std_curve, 1e-7),
                       mean_curve + std_curve,
                       color=colors[strategy], alpha=0.2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Best MSE (log scale)')
    ax.set_title('Best MSE Convergence\n(mean ± std across problems)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Convergence of {n_agents} Async Agents (Averaged Over {len(world_seeds)} Problems)\\nMSE = error vs hidden ground-truth function',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_worlds(world_seeds: List[int], save_path: str = "worlds.png") -> None:
    """Visualize the different hidden functions (problems)."""

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-1, 1, 200)

    cmap = plt.cm.tab10(np.linspace(0, 1, len(world_seeds)))

    for i, seed in enumerate(world_seeds):
        world = make_world(seed)
        y = np.array([world.f(xi) for xi in x])
        ax.plot(x, y, '-', linewidth=2, color=cmap[i],
               label=f'Seed {seed}: {world.a:.2f}x³ + {world.b:.2f}x² + {world.c:.2f}x + {world.d:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Hidden Ground-Truth Functions (Agents See Only Noisy Samples)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


async def main():
    init_logging("WARNING")  # Reduce noise

    N_AGENTS = 16
    AGENT_SEED = 123
    STEPS = 300

    # Different problems (world seeds)
    WORLD_SEEDS = [7, 42, 99, 123, 256, 314]

    print("=" * 70)
    print(f"Running comparison across {len(WORLD_SEEDS)} different problems")
    print("=" * 70)

    # First, visualize the different problems
    print("\nVisualizing hidden functions...")
    plot_worlds(WORLD_SEEDS, "worlds.png")

    strategies = [
        ("No Communication", 0.0, "ring"),
        ("Ring (12%)", 0.12, "ring"),
        ("Random (12%)", 0.12, "random"),
    ]

    all_results: Dict[str, Dict[int, Dict]] = {s[0]: {} for s in strategies}

    total_runs = len(WORLD_SEEDS) * len(strategies)
    run_count = 0

    for seed in WORLD_SEEDS:
        world = make_world(seed)
        print(f"\n{'='*70}")
        print(f"Problem: Seed {seed} -> f(x) = {world.a:.3f}x³ + {world.b:.3f}x² + {world.c:.3f}x + {world.d:.3f}")
        print(f"{'='*70}")

        for strategy_name, comm_prob, topology in strategies:
            run_count += 1
            print(f"\n  [{run_count}/{total_runs}] {strategy_name}...", end=" ", flush=True)

            result = await run_experiment(
                n_agents=N_AGENTS,
                comm_prob=comm_prob,
                world=world,
                steps=STEPS,
                agent_seed=AGENT_SEED,
                topology=topology,
                quiet=True,
            )

            all_results[strategy_name][seed] = result
            print(f"median={result['final_median']:.6f}, best={result['final_best']:.6f}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating comparison plots...")
    print("=" * 70)

    plot_problem_comparison(all_results, WORLD_SEEDS, N_AGENTS, "problem_comparison.png")
    plot_aggregate_stats(all_results, WORLD_SEEDS, N_AGENTS, "aggregate_stats.png")
    plot_convergence_curves(all_results, WORLD_SEEDS, N_AGENTS, "convergence_curves.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (across all problems)")
    print("=" * 70)

    print(f"\n{'Strategy':<20} {'Metric':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 83)

    for strategy_name, _, _ in strategies:
        results = all_results[strategy_name]

        for metric, label in [("final_best", "Best MSE"),
                              ("final_median", "Median MSE"),
                              ("final_worst", "Worst MSE")]:
            vals = [results[seed][metric] for seed in WORLD_SEEDS]
            print(f"{strategy_name:<20} {label:<15} {np.mean(vals):>12.6f} {np.std(vals):>12.6f} "
                  f"{np.min(vals):>12.6f} {np.max(vals):>12.6f}")
        print()

    # Win rate comparison
    print("\n" + "=" * 70)
    print("WIN RATES (which strategy had best median MSE per problem)")
    print("=" * 70)

    wins = {s[0]: 0 for s in strategies}
    for seed in WORLD_SEEDS:
        best_strategy = min(strategies, key=lambda s: all_results[s[0]][seed]["final_median"])
        wins[best_strategy[0]] += 1

    for strategy_name, _, _ in strategies:
        pct = 100 * wins[strategy_name] / len(WORLD_SEEDS)
        print(f"  {strategy_name}: {wins[strategy_name]}/{len(WORLD_SEEDS)} ({pct:.0f}%)")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
