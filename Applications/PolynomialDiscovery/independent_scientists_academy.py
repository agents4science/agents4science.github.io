#!/usr/bin/env python3
"""
independent_scientists_academy.py

Toy demo: 8 independent "scientist" agents (no coordinator) learning a hidden function.
Agents explore independently (local noisy samples) and may exchange information with peers.

Key knobs:
- TOPOLOGY: all | ring | random
- COMM_PROB: probability of sending a message each step
- BANDWIDTH: full | summary
- INTEGRATION: replace | blend (how to incorporate peer hypotheses)
- NOISE_STD: observation noise
- STEPS / RUNTIME

There is NO central agent controlling state. The driver only *observes* by polling models.

Academy notes:
- Uses LocalExchangeFactory + ThreadPoolExecutor for a single-process demo.
- Each agent runs a persistent @loop.
- Agents communicate peer-to-peer by calling each other's @action "receive_message".

Install:
  pip install academy-py

Run:
  python independent_scientists_academy.py

Env overrides (optional):
  export TOPOLOGY=ring
  export COMM_PROB=0.15
  export BANDWIDTH=full
  export INTEGRATION=blend
  export NOISE_STD=0.05
  export STEPS=400
  export EVAL_EVERY=20
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from academy.agent import Agent, action, loop
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.logging import init_logging
from academy.manager import Manager


# ----------------------------
# Hidden "world" (oracle)
# ----------------------------

@dataclass(frozen=True)
class HiddenWorld:
    # Hidden coefficients for a cubic: f(x) = a x^3 + b x^2 + c x + d
    a: float
    b: float
    c: float
    d: float

    def f(self, x: float) -> float:
        return ((self.a * x + self.b) * x + self.c) * x + self.d


def make_world(seed: int = 7) -> HiddenWorld:
    rnd = random.Random(seed)
    # Keep coefficients moderate so values aren't huge
    a = rnd.uniform(-1.2, 1.2)
    b = rnd.uniform(-1.0, 1.0)
    c = rnd.uniform(-0.8, 0.8)
    d = rnd.uniform(-0.5, 0.5)
    return HiddenWorld(a=a, b=b, c=c, d=d)


# ----------------------------
# Small linear algebra: least squares for cubic
# Fit params theta = [a,b,c,d] minimizing sum (y - Phi(x)*theta)^2
# Phi(x) = [x^3, x^2, x, 1]
# Solve normal equations: (X^T X) theta = X^T y
# We'll do Gaussian elimination on 4x4 (safe for toy).
# ----------------------------

def solve_4x4(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    # Gaussian elimination with partial pivoting
    n = 4
    M = [row[:] + [b_i] for row, b_i in zip(A, b)]  # augmented 4x5

    for col in range(n):
        # pivot
        pivot = col
        max_abs = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > max_abs:
                max_abs = v
                pivot = r
        if max_abs < 1e-12:
            return None  # singular / ill-conditioned

        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        # eliminate
        piv = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / piv
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # back-substitution
    x = [0.0] * n
    for r in reversed(range(n)):
        s = M[r][n]
        for c in range(r + 1, n):
            s -= M[r][c] * x[c]
        denom = M[r][r]
        if abs(denom) < 1e-12:
            return None
        x[r] = s / denom
    return x


def fit_cubic_least_squares(xs: List[float], ys: List[float]) -> Optional[List[float]]:
    if len(xs) < 4:
        return None

    # Build normal equations: A = X^T X (4x4), b = X^T y (4)
    A = [[0.0] * 4 for _ in range(4)]
    B = [0.0] * 4
    for x, y in zip(xs, ys):
        phi = [x**3, x**2, x, 1.0]
        for i in range(4):
            B[i] += phi[i] * y
            for j in range(4):
                A[i][j] += phi[i] * phi[j]

    return solve_4x4(A, B)


def predict(theta: List[float], x: float) -> float:
    a, b, c, d = theta
    return ((a * x + b) * x + c) * x + d


def mse(theta: List[float], world: HiddenWorld, grid: List[float]) -> float:
    s = 0.0
    for x in grid:
        y = world.f(x)
        yhat = predict(theta, x)
        e = yhat - y
        s += e * e
    return s / max(1, len(grid))


# ----------------------------
# Messaging protocol
# ----------------------------

# BANDWIDTH:
# - "summary": share only (theta, claimed_mse, n_samples)
# - "full": additionally share some sample points (xs, ys) (bounded)
#
# INTEGRATION:
# - "replace": adopt peer theta if peer seems better on your local validation
# - "blend": convex blend with your theta, weighted by relative performance


# ----------------------------
# Independent scientist agent
# ----------------------------

class IndependentScientist(Agent):
    def __init__(
        self,
        agent_id: int,
        world: HiddenWorld,
        noise_std: float,
        comm_prob: float,
        bandwidth: str,
        integration: str,
        max_shared_samples: int,
        seed: int,
    ):
        super().__init__()
        self.agent_idx = agent_id
        self.world = world
        self.noise_std = noise_std
        self.comm_prob = comm_prob
        self.bandwidth = bandwidth
        self.integration = integration
        self.max_shared_samples = max_shared_samples

        self.rnd = random.Random(seed)
        self.peers: List[Handle[IndependentScientist]] = []

        # Local dataset
        self.xs: List[float] = []
        self.ys: List[float] = []

        # Local hypothesis (theta)
        self.theta: List[float] = self._random_theta()

        # Inbox of peer messages
        self.inbox: List[Dict[str, Any]] = []

        # Track progress
        self.steps = 0
        self.last_fit_step = -999
        self.last_comm_step = -999
        self.start_time = time.time()

        # Communication stats
        self.messages_sent = 0
        self.messages_received = 0

        # Agent-specific validation grid (private): encourages diversity
        self.val_grid = [self.rnd.uniform(-1.0, 1.0) for _ in range(32)]

    def _random_theta(self) -> List[float]:
        # small random initial model
        return [
            self.rnd.uniform(-1.0, 1.0),
            self.rnd.uniform(-1.0, 1.0),
            self.rnd.uniform(-1.0, 1.0),
            self.rnd.uniform(-0.5, 0.5),
        ]

    def _sample(self) -> Tuple[float, float]:
        x = self.rnd.uniform(-1.0, 1.0)
        y = self.world.f(x) + self.rnd.gauss(0.0, self.noise_std)
        return x, y

    def _local_mse(self, theta: Optional[List[float]] = None) -> float:
        th = theta if theta is not None else self.theta
        return mse(th, self.world, self.val_grid)

    def _refit(self) -> None:
        if len(self.xs) < 4:
            return
        theta_hat = fit_cubic_least_squares(self.xs, self.ys)
        if theta_hat is None:
            return

        # Optional exploration: sometimes keep a slightly "noisy" version
        if self.rnd.random() < 0.10:
            theta_hat = [t + self.rnd.gauss(0.0, 0.02) for t in theta_hat]

        self.theta = theta_hat

    def _maybe_restart(self) -> None:
        # Rare random restart to maintain exploration diversity
        if self.rnd.random() < 0.01:
            self.theta = self._random_theta()
            # Keep data (scientists don't forget), but model jumps

    @action
    async def set_peers(self, peers: List[Handle["IndependentScientist"]]) -> None:
        # Set peer handles (topology determined by driver)
        self.peers = peers

    @action
    async def receive_message(self, msg: Dict[str, Any]) -> None:
        # Peer-to-peer messaging endpoint
        self.messages_received += 1
        # Keep inbox bounded
        if len(self.inbox) > 200:
            self.inbox = self.inbox[-100:]
        self.inbox.append(msg)

    @action
    async def get_comm_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_idx,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
        }

    @action
    async def get_model(self) -> Dict[str, Any]:
        # For external observation (driver)
        return {
            "agent_id": self.agent_idx,
            "theta": self.theta,
            "n_samples": len(self.xs),
            "local_mse": self._local_mse(),
            "steps": self.steps,
            "uptime_s": time.time() - self.start_time,
        }

    def _integrate_peer(self, msg: Dict[str, Any]) -> None:
        peer_theta = msg.get("theta")
        if not (isinstance(peer_theta, list) and len(peer_theta) == 4):
            return
        peer_theta = [float(x) for x in peer_theta]

        # If full bandwidth, optionally ingest some samples (bounded)
        if self.bandwidth == "full":
            xs = msg.get("xs")
            ys = msg.get("ys")
            if isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys) and len(xs) > 0:
                # add a small random subset up to max_shared_samples
                idxs = list(range(len(xs)))
                self.rnd.shuffle(idxs)
                take = min(self.max_shared_samples, len(idxs))
                for i in idxs[:take]:
                    try:
                        x = float(xs[i])
                        y = float(ys[i])
                    except Exception:
                        continue
                    self.xs.append(x)
                    self.ys.append(y)

        # Decide whether/how to incorporate peer theta
        my_err = self._local_mse(self.theta)
        peer_err = self._local_mse(peer_theta)

        if self.integration == "replace":
            if peer_err + 1e-12 < my_err:
                self.theta = peer_theta
        elif self.integration == "blend":
            # Blend more strongly toward the better hypothesis
            if peer_err < my_err:
                # weight based on relative improvement
                improvement = (my_err - peer_err) / max(1e-9, my_err)
                alpha = min(0.8, max(0.05, 0.2 + 0.6 * improvement))
                self.theta = [(1 - alpha) * t + alpha * pt for t, pt in zip(self.theta, peer_theta)]
        else:
            # Unknown integration mode: ignore
            return

    async def _maybe_send(self) -> None:
        if not self.peers:
            return
        if self.rnd.random() > self.comm_prob:
            return

        # Pick a random peer
        peer = self.rnd.choice(self.peers)

        # Compose message
        msg: Dict[str, Any] = {
            "from": self.agent_idx,
            "theta": self.theta,
            "claimed_mse": self._local_mse(),
            "n_samples": len(self.xs),
            "step": self.steps,
            "ts": time.time(),
        }

        if self.bandwidth == "full":
            # Share a bounded slice of samples
            n = len(self.xs)
            if n > 0:
                idxs = list(range(n))
                self.rnd.shuffle(idxs)
                take = min(self.max_shared_samples, len(idxs))
                msg["xs"] = [self.xs[i] for i in idxs[:take]]
                msg["ys"] = [self.ys[i] for i in idxs[:take]]

        await peer.receive_message(msg)
        self.messages_sent += 1

    @loop
    async def run(self, shutdown: asyncio.Event) -> None:
        # Persistent “not lazy” loop:
        # sample -> refit periodically -> process inbox -> maybe communicate -> occasional restart
        while not shutdown.is_set():
            self.steps += 1

            # 1) sample a new datapoint
            x, y = self._sample()
            self.xs.append(x)
            self.ys.append(y)

            # 2) refit every few steps
            if self.steps - self.last_fit_step >= 3:
                self._refit()
                self.last_fit_step = self.steps

            # 3) process some inbox messages (bounded per step)
            if self.inbox:
                # process up to k messages each step
                k = 3
                for _ in range(min(k, len(self.inbox))):
                    msg = self.inbox.pop(0)
                    self._integrate_peer(msg)

            # 4) maintain exploration diversity
            self._maybe_restart()

            # 5) maybe communicate
            await self._maybe_send()

            # 6) throttle so this is easy to watch
            await asyncio.sleep(0.02)


# ----------------------------
# Topology helper
# ----------------------------

def build_topology(handles: List[Handle[IndependentScientist]], topology: str, rnd: random.Random, k: int = 2) -> List[List[Handle[IndependentScientist]]]:
    n = len(handles)
    peers: List[List[Handle[IndependentScientist]]] = [[] for _ in range(n)]

    if topology == "all":
        for i in range(n):
            peers[i] = [h for j, h in enumerate(handles) if j != i]
        return peers

    if topology == "ring":
        for i in range(n):
            peers[i] = [handles[(i - 1) % n], handles[(i + 1) % n]]
        return peers

    if topology == "random":
        # random k peers per agent
        for i in range(n):
            choices = [h for j, h in enumerate(handles) if j != i]
            rnd.shuffle(choices)
            peers[i] = choices[: max(1, min(k, n - 1))]
        return peers

    raise ValueError(f"Unknown TOPOLOGY={topology} (use all|ring|random)")


# ----------------------------
# Driver (observer only)
# ----------------------------

def plot_mse_over_time(history: Dict[str, List], save_path: str = "mse_over_time.png") -> None:
    """Plot best, median, and worst MSE over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = history["steps"]
    ax.semilogy(steps, history["best_mse"], 'g-', linewidth=2, label='Best MSE')
    ax.semilogy(steps, history["median_mse"], 'b-', linewidth=2, label='Median MSE')
    ax.semilogy(steps, history["worst_mse"], 'r-', linewidth=2, label='Worst MSE')

    ax.fill_between(steps, history["best_mse"], history["worst_mse"], alpha=0.2, color='blue')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Agent Performance Over Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_agent_trajectories(history: Dict[str, List], n_agents: int, save_path: str = "agent_trajectories.png") -> None:
    """Plot individual agent MSE trajectories over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = history["steps"]
    cmap = plt.cm.viridis(np.linspace(0, 1, n_agents))

    for agent_idx in range(n_agents):
        agent_mses = [snapshot[agent_idx] for snapshot in history["agent_mses"]]
        ax.semilogy(steps, agent_mses, '-', color=cmap[agent_idx], linewidth=1.5,
                   alpha=0.7, label=f'Agent {agent_idx}')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Individual Agent MSE Trajectories', fontsize=14)

    if n_agents <= 8:
        ax.legend(loc='upper right', fontsize=9, ncol=2)
    else:
        ax.legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.9)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_diversity_over_time(history: Dict[str, List], save_path: str = "diversity_over_time.png") -> None:
    """Plot model diversity (avg pairwise L2) over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = history["steps"]
    ax.plot(steps, history["diversity"], 'purple', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Diversity (Avg Pairwise L2)', fontsize=12)
    ax.set_title('Model Diversity Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_models(world: 'HiddenWorld', final_thetas: List[List[float]],
                      save_path: str = "final_models.png") -> None:
    """Plot all agent final models vs the true function."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(-1, 1, 200)

    # True function
    y_true = np.array([world.f(xi) for xi in x])
    ax.plot(x, y_true, 'k-', linewidth=3, label='True Function', zorder=10)

    # Agent models
    n_agents = len(final_thetas)
    cmap = plt.cm.tab20(np.linspace(0, 1, n_agents))

    for i, theta in enumerate(final_thetas):
        a, b, c, d = theta
        y_pred = a * x**3 + b * x**2 + c * x + d
        ax.plot(x, y_pred, '-', color=cmap[i], linewidth=1.2, alpha=0.7, label=f'Agent {i}')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Final Agent Models vs True Function', fontsize=14)

    if n_agents <= 8:
        ax.legend(loc='best', fontsize=9, ncol=2)
    else:
        ax.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.9, bbox_to_anchor=(1.02, 1))
        fig.subplots_adjust(right=0.78)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_dashboard(history: Dict[str, List], world: 'HiddenWorld',
                           final_thetas: List[List[float]], n_agents: int,
                           save_path: str = "summary_dashboard.png") -> None:
    """Create a 2x2 summary dashboard with all key plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = history["steps"]

    # 1. MSE over time (top-left)
    ax = axes[0, 0]
    ax.semilogy(steps, history["best_mse"], 'g-', linewidth=2, label='Best')
    ax.semilogy(steps, history["median_mse"], 'b-', linewidth=2, label='Median')
    ax.semilogy(steps, history["worst_mse"], 'r-', linewidth=2, label='Worst')
    ax.fill_between(steps, history["best_mse"], history["worst_mse"], alpha=0.15, color='blue')
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Performance Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Agent trajectories (top-right)
    ax = axes[0, 1]
    cmap = plt.cm.viridis(np.linspace(0, 1, n_agents))
    for agent_idx in range(n_agents):
        agent_mses = [snapshot[agent_idx] for snapshot in history["agent_mses"]]
        ax.semilogy(steps, agent_mses, '-', color=cmap[agent_idx], linewidth=1, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title(f'Individual Agent Trajectories (n={n_agents})')
    ax.grid(True, alpha=0.3)

    # 3. Diversity over time (bottom-left)
    ax = axes[1, 0]
    ax.plot(steps, history["diversity"], 'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Diversity (Avg L2)')
    ax.set_title('Model Diversity Over Time')
    ax.grid(True, alpha=0.3)

    # 4. Final models (bottom-right)
    ax = axes[1, 1]
    x = np.linspace(-1, 1, 200)
    y_true = np.array([world.f(xi) for xi in x])
    ax.plot(x, y_true, 'k-', linewidth=3, label='True', zorder=10)

    for i, theta in enumerate(final_thetas):
        a, b, c, d = theta
        y_pred = a * x**3 + b * x**2 + c * x + d
        ax.plot(x, y_pred, '-', color=cmap[i], linewidth=1, alpha=0.6)

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Final Models vs True Function')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Independent Scientists Simulation ({n_agents} agents)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


async def main() -> None:
    init_logging("INFO")

    # Experiment parameters
    N_AGENTS = 16
    WORLD_SEED = int(os.getenv("WORLD_SEED", "7"))
    AGENT_SEED = int(os.getenv("AGENT_SEED", "123"))
    TOPOLOGY = os.getenv("TOPOLOGY", "ring").strip().lower()          # all | ring | random
    COMM_PROB = float(os.getenv("COMM_PROB", "0.12"))                # send probability per step
    BANDWIDTH = os.getenv("BANDWIDTH", "summary").strip().lower()    # summary | full
    INTEGRATION = os.getenv("INTEGRATION", "blend").strip().lower()  # replace | blend
    NOISE_STD = float(os.getenv("NOISE_STD", "0.07"))
    MAX_SHARED_SAMPLES = int(os.getenv("MAX_SHARED_SAMPLES", "8"))   # only used for BANDWIDTH=full

    STEPS = int(os.getenv("STEPS", "300"))
    EVAL_EVERY = int(os.getenv("EVAL_EVERY", "20"))                  # driver polling interval in steps
    RANDOM_K = int(os.getenv("RANDOM_K", "3"))                        # topology=random

    world = make_world(WORLD_SEED)

    # External evaluation grid (driver-only)
    eval_grid = [(-1.0 + 2.0 * i / 199.0) for i in range(200)]

    print("\n=== Independent Agents Toy (no coordinator) ===")
    print(f"TOPOLOGY={TOPOLOGY} COMM_PROB={COMM_PROB} BANDWIDTH={BANDWIDTH} INTEGRATION={INTEGRATION} NOISE_STD={NOISE_STD}")
    print(f"STEPS={STEPS} EVAL_EVERY={EVAL_EVERY}")
    print("World is hidden to agents in principle; in this toy it is embedded equally in each agent as the physics.\n")

    # Local exchange + executor
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=32),
    ) as manager:
        # Launch agents
        handles: List[Handle[IndependentScientist]] = []
        for i in range(N_AGENTS):
            h = await manager.launch(
                IndependentScientist,
                args=(
                    i,
                    world,
                    NOISE_STD,
                    COMM_PROB,
                    BANDWIDTH,
                    INTEGRATION,
                    MAX_SHARED_SAMPLES,
                    AGENT_SEED * 1000 + i,
                ),
            )
            handles.append(h)

        # Wire topology
        rnd = random.Random(AGENT_SEED + 999)
        peer_lists = build_topology(handles, TOPOLOGY, rnd, k=RANDOM_K)
        for i, h in enumerate(handles):
            await h.set_peers(peer_lists[i])

        # Run & observe
        # We'll poll each agent model periodically and compute *true* MSE on eval_grid (driver-only).
        t0 = time.time()
        last_report_step = 0

        # History for plotting
        history: Dict[str, List] = {
            "steps": [],
            "best_mse": [],
            "median_mse": [],
            "worst_mse": [],
            "diversity": [],
            "agent_mses": [],  # List of lists: each entry is [mse_agent0, mse_agent1, ...]
        }
        final_thetas: List[List[float]] = []

        for step in range(1, STEPS + 1):
            await asyncio.sleep(0.05)  # pace the demo; agents are running internally

            if step - last_report_step >= EVAL_EVERY:
                last_report_step = step

                models = await asyncio.gather(*[h.get_model() for h in handles])

                # Compute true MSEs (external evaluation)
                mses: List[Tuple[float, int]] = []
                agent_mse_list: List[float] = []
                for m in models:
                    theta = [float(x) for x in m["theta"]]
                    err = mse(theta, world, eval_grid)
                    mses.append((err, int(m["agent_id"])))
                    agent_mse_list.append(err)

                mses_sorted = sorted(mses, key=lambda x: x[0])
                best_err, best_id = mses_sorted[0]
                med_err = mses_sorted[len(mses_sorted) // 2][0]
                worst_err = mses_sorted[-1][0]

                # Diversity proxy: average pairwise L2 distance between theta vectors
                thetas = [[float(x) for x in m["theta"]] for m in models]
                div = avg_pairwise_l2(thetas)

                # Store history
                history["steps"].append(step)
                history["best_mse"].append(best_err)
                history["median_mse"].append(med_err)
                history["worst_mse"].append(worst_err)
                history["diversity"].append(div)
                history["agent_mses"].append(agent_mse_list)

                # Keep final thetas for plotting
                final_thetas = thetas

                print(
                    f"[t+{time.time()-t0:5.1f}s step={step:4d}] "
                    f"best(MSE)={best_err:.6f} (agent {best_id}) "
                    f"median={med_err:.6f} worst={worst_err:.6f} "
                    f"div(L2)={div:.4f}"
                )

        # Collect communication stats before shutdown
        comm_stats = await asyncio.gather(*[h.get_comm_stats() for h in handles])
        total_sent = sum(s["messages_sent"] for s in comm_stats)
        total_received = sum(s["messages_received"] for s in comm_stats)

        print("\n--- Communication Statistics ---")
        print(f"Total messages sent: {total_sent}")
        print(f"Total messages received: {total_received}")
        print(f"Avg messages sent per agent: {total_sent / N_AGENTS:.1f}")
        print(f"Avg messages received per agent: {total_received / N_AGENTS:.1f}")
        print("\nPer-agent breakdown:")
        for s in sorted(comm_stats, key=lambda x: x["agent_id"]):
            print(f"  Agent {s['agent_id']:2d}: sent={s['messages_sent']:3d}, received={s['messages_received']:3d}")

        # Shutdown agents
        for h in handles:
            await h.shutdown()

    print("\nDone.\n")

    # Generate plots
    print("\nGenerating plots...")
    plot_mse_over_time(history, "mse_over_time.png")
    plot_agent_trajectories(history, N_AGENTS, "agent_trajectories.png")
    plot_diversity_over_time(history, "diversity_over_time.png")
    plot_final_models(world, final_thetas, "final_models.png")
    plot_summary_dashboard(history, world, final_thetas, N_AGENTS, "summary_dashboard.png")
    print("All plots generated.")
    # If you want the actual hidden coefficients (for debugging), uncomment:
    # print(world)


def avg_pairwise_l2(thetas: List[List[float]]) -> float:
    n = len(thetas)
    if n < 2:
        return 0.0
    s = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += l2(thetas[i], thetas[j])
            cnt += 1
    return s / cnt


def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


if __name__ == "__main__":
    asyncio.run(main())
