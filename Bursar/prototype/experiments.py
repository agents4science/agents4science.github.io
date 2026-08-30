"""Parameter-sweep experiments backing the paper's policy claims.

Three figures (written to figures/), each answering a question the paper
poses but the single demo scenario cannot:

1. fig_latency_price.png — What actually buys low latency? (Secs. 7, 14)
   Sweeping warm headroom barely moves time-to-capacity (a work-conserving
   opportunistic tier refills spare GPUs anyway) but costs utilization;
   sweeping the preemption grace period moves latency one-for-one. The
   trade the paper says the pilot must price is measured here: latency is
   bought with the preemptible tier's grace period, not with idle headroom
   — headroom helps only when it exceeds the request size.

2. fig_overbooking.png — Do guarantees survive overbooking? (Sec. 10)
   Correlated adversarial storms are added until burst ceilings are 7x the
   pool. Guaranteed service stays at 1 s with zero violations at every phi;
   scarcity lands entirely on non-guaranteed burst service.

3. fig_bucket.png — Does the token bucket actually bound an adversary? (Sec. 4)
   Yes, but only with the right enforcement semantics: checking tokens only
   at grant instants leaks burst via lease-duration granularity (a leak this
   sweep found, fixed by prepaid leases: the above-R portion of a lease must
   be token-covered for its whole duration at grant time).

Latency measurements use a probe campaign: a standardized 128-GPU request
every 5 minutes, timed to *full* capacity, pooled across seeds. Probe
observations are censored at 591 s.

Usage: python3 experiments.py [--quick]      (quick: 1 seed, fewer points)
Requires matplotlib (the only part of the prototype that does).
"""

import argparse
import json
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario import run, metrics, pct

POOL = 512
FIGSIZE = (7.2, 4.4)
DPI = 150


def sweep(name, values, make_kw, seeds, probe=False):
    rows = []
    for v in values:
        per_seed = []
        for s in seeds:
            t0 = time.time()
            m = metrics(run(seed=s, probe=probe, **make_kw(v)))
            per_seed.append(m)
            print(f"  {name}={v} seed={s}  ({time.time()-t0:.1f}s)")
        rows.append((v, per_seed))
    return rows


def band(rows, key):
    """Mean with min-max band for scalar metrics."""
    mean, lo, hi = [], [], []
    for _, per_seed in rows:
        vals = [m[key] for m in per_seed]
        mu = statistics.mean(vals)
        mean.append(mu)
        lo.append(mu - min(vals))
        hi.append(max(vals) - mu)
    return mean, [lo, hi]


def pooled(rows, key, p):
    """Percentile over the pooled raw latency lists of all seeds."""
    return [pct([l for m in per_seed for l in m[key]], p) for _, per_seed in rows]


def fig_latency_price(rows_head, rows_grace, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=DPI)

    # left: headroom sweep — utilization falls, latency barely moves
    x = [h for h, _ in rows_head]
    util, util_err = band(rows_head, "utilization")
    ax1.errorbar(x, [u * 100 for u in util],
                 yerr=[[e * 100 for e in util_err[0]], [e * 100 for e in util_err[1]]],
                 marker="s", capsize=3, color="#8a8a8a", label="pool utilization (left)")
    ax1.set_ylabel("mean pool utilization (%)", color="#666")
    ax1.set_xlabel("warm headroom held free (GPUs)")
    ax1.set_ylim(0, 100)
    ax1b = ax1.twinx()
    ax1b.plot(x, pooled(rows_head, "probe_lats", 50), marker="o", color="#4e79a7",
              label="time to 128 GPUs, p50 (right)")
    ax1b.plot(x, pooled(rows_head, "probe_lats", 95), marker="o", ls="--", color="#4e79a7",
              alpha=0.5, label="p95 (right)")
    ax1b.set_ylabel("probe time-to-capacity (s)", color="#4e79a7")
    ax1b.set_ylim(bottom=0)
    ax1.set_title("headroom costs utilization,\nbarely buys latency")
    lines = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax1b.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)

    # right: grace sweep — latency tracks the preemption grace one-for-one
    xg = [g for g, _ in rows_grace]
    ax2.plot(xg, pooled(rows_grace, "probe_lats", 50), marker="o", color="#f28e2b",
             label="time to 128 GPUs, p50")
    ax2.plot([0, max(xg)], [0, max(xg)], "--", color="#888", label="y = grace period")
    ax2.set_xlabel("opportunistic preemption grace (s)")
    ax2.set_ylabel("probe time-to-capacity, p50 (s)")
    ax2.set_title("the preemption grace period\nis what prices latency")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("What buys low latency on a full machine (paper Secs. 7, 14): "
                 "a preemptible tier and its grace period, not idle headroom", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def fig_overbooking(rows, path):
    phi = [(64 + 2 * 384 + 384 * k + POOL) / POOL for k, _ in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(phi, pooled(rows, "probe_lats_in", 50), marker="o", color="#4e79a7",
            label="non-guaranteed 128-GPU probe during storm, p50")
    ax.plot(phi, pooled(rows, "probe_lats_in", 95), marker="o", ls="--", alpha=0.5,
            color="#4e79a7", label="probe p95 (censored at 591 s)")
    vic, vic_err = band(rows, "victim_p95_in")
    ax.errorbar(phi, vic, yerr=vic_err, marker="s", capsize=3, color="#59a14f",
                label="guaranteed interactive campaign, p95")
    viol, _ = band(rows, "victim_violations_in")
    total_viol = sum(viol)
    ax.annotate(f"guaranteed campaign: {total_viol:.0f} violations (>60 s) "
                f"across every \u03c6 and seed",
                (phi[0], vic[0]), textcoords="offset points", xytext=(10, 12),
                fontsize=8.5, color="#2e7d32")
    ax.set_xlabel("overbooking factor φ = Σ burst ceilings / pool")
    ax.set_ylabel("lease latency during correlated storms (s)")
    ax.set_title("Overbooked ceilings vs. isolation (paper Sec. 10):\n"
                 "guarantees hold at any φ; scarcity lands only on non-guaranteed burst service")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_bucket(rows_loud, rows_quiet, rows_leaky, path,
               storm_R=64, storm_burst=384, window=1800):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for rows, color, label in (
            (rows_leaky, "#e15759", "idle pool, grant-time-only enforcement (leaky)"),
            (rows_quiet, "#f28e2b", "idle pool, prepaid leases"),
            (rows_loud, "#9c5c1e", "contended pool (full scenario), prepaid leases")):
        x = [b / 1000 for b, _ in rows]
        excess, err = band(rows, "storm_excess_gpu_s")
        ax.errorbar(x, [e / 1000 for e in excess],
                    yerr=[[e / 1000 for e in err[0]], [e / 1000 for e in err[1]]],
                    marker="o", capsize=3, color=color, label=label)
    xs = [b / 1000 for b, _ in rows_quiet]
    ax.plot(xs, xs, "--", color="#888", label="bucket capacity B (y = x)")
    ax.plot(xs, [x + storm_R * window / 1000 for x in xs], ":", color="#aaa",
            label="B + R×window (refill-assisted bound)")
    ax.set_xlabel("storm bucket capacity B (thousands of GPU-seconds)")
    ax.set_ylabel("excess GPU-seconds extracted during storm (thousands)")
    ax.set_title("Token-bucket enforcement semantics matter (paper Secs. 4, 10):\n"
                 "grant-time-only checks leak burst via lease-duration granularity;\n"
                 "prepaid leases pin extraction to B")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 seed, fewer points")
    args = p.parse_args()
    seeds = [7] if args.quick else [7, 11, 13]
    os.makedirs("figures", exist_ok=True)
    summary = {}

    print("sweep 1a/3: warm headroom (probe latency + utilization)")
    values = [32, 128, 256] if args.quick else [16, 32, 64, 128, 192, 256]
    rows_head = sweep("headroom", values, lambda h: {"headroom": h}, seeds, probe=True)
    print("sweep 1b/3: preemption grace period (probe latency)")
    gvalues = [15, 60, 120] if args.quick else [15, 30, 60, 90, 120]
    rows_grace = sweep("grace", gvalues, lambda g: {"grace": g}, seeds, probe=True)
    fig_latency_price(rows_head, rows_grace, "figures/fig_latency_price.png")
    summary["headroom"] = [{"headroom": v,
                            "utilization": statistics.mean(m["utilization"] for m in ms),
                            "probe_p50": pct([l for m in ms for l in m["probe_lats"]], 50)}
                           for v, ms in rows_head]
    summary["grace"] = [{"grace": v,
                         "probe_p50": pct([l for m in ms for l in m["probe_lats"]], 50)}
                        for v, ms in rows_grace]

    print("sweep 2/3: overbooking factor (correlated storms vs isolation)")
    values = [1, 3, 6] if args.quick else [1, 2, 3, 4, 5, 6]
    rows = sweep("n_storms", values, lambda k: {"n_storms": k}, seeds, probe=True)
    fig_overbooking(rows, "figures/fig_overbooking.png")
    summary["overbooking"] = [{"n_storms": v,
                               "phi": (64 + 2 * 384 + 384 * v + POOL) / POOL,
                               "victim_p95_in": statistics.mean(m["victim_p95_in"] for m in ms),
                               "victim_violations": statistics.mean(m["victim_violations_in"] for m in ms),
                               "probe_p50_in": pct([l for m in ms for l in m["probe_lats_in"]], 50)}
                              for v, ms in rows]

    print("sweep 3/3: storm bucket capacity B (enforcement semantics)")
    values = [50_000, 150_000, 600_000] if args.quick else \
             [25_000, 50_000, 100_000, 150_000, 300_000, 600_000]
    rows_loud = sweep("storm_B", values, lambda bb: {"storm_B": bb}, seeds)
    rows_quiet = sweep("storm_B(idle)", values,
                       lambda bb: {"storm_B": bb, "n_tasks": 0}, seeds)
    rows_leaky = sweep("storm_B(idle,leaky)", values,
                       lambda bb: {"storm_B": bb, "n_tasks": 0,
                                   "prepaid_leases": False}, seeds)
    fig_bucket(rows_loud, rows_quiet, rows_leaky, "figures/fig_bucket.png")
    summary["bucket"] = [{"storm_B": v,
                          "excess_contended": statistics.mean(m["storm_excess_gpu_s"] for m in ms),
                          "excess_idle_prepaid": statistics.mean(m["storm_excess_gpu_s"] for m in mq),
                          "excess_idle_leaky": statistics.mean(m["storm_excess_gpu_s"] for m in ml)}
                         for (v, ms), (_, mq), (_, ml) in zip(rows_loud, rows_quiet, rows_leaky)]

    with open("figures/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print(f"{'headroom':>9s} {'util%':>6s} {'probe p50':>10s}")
    for r in summary["headroom"]:
        print(f"{r['headroom']:9d} {r['utilization']*100:6.1f} {r['probe_p50']:9.0f}s")
    print(f"\n{'grace':>6s} {'probe p50':>10s}")
    for r in summary["grace"]:
        print(f"{r['grace']:5d}s {r['probe_p50']:9.0f}s")
    print(f"\n{'storms':>7s} {'phi':>5s} {'victim p95':>11s} {'violations':>11s} {'probe p50 in':>13s}")
    for r in summary["overbooking"]:
        print(f"{r['n_storms']:7d} {r['phi']:5.2f} {r['victim_p95_in']:11.1f} "
              f"{r['victim_violations']:11.1f} {r['probe_p50_in']:12.0f}s")
    print(f"\n{'B (k)':>7s} {'idle leaky (k)':>15s} {'idle prepaid (k)':>17s} {'contended (k)':>14s}")
    for r in summary["bucket"]:
        print(f"{r['storm_B']/1000:7.0f} {r['excess_idle_leaky']/1000:15.1f} "
              f"{r['excess_idle_prepaid']/1000:17.1f} {r['excess_contended']/1000:14.1f}")
    print("\nwrote figures/fig_latency_price.png, fig_overbooking.png, fig_bucket.png, summary.json")


if __name__ == "__main__":
    main()
