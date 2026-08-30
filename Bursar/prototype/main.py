"""Bursar prototype — simulation driver.

Runs a 2-hour simulated scenario on a 512-GPU agent-burst pool with five
campaigns, prints a report mapped to the paper's Sec. 13 success criteria,
and writes journal.jsonl + report.html (animated replay).

Usage: python3 main.py && open report.html
"""

import json

from broker import Broker, Envelope, Dim
from agents import (make_tasks, NaiveAgent, AdaptiveAgent,
                    StormAgent, VictimAgent, BackfillAgent)
from viz import write_report

SIM = 7200          # simulated seconds (2 h)
SAMPLE = 5          # sample the timeseries every N ticks
STORM = (3600, 5400)
POOL = 512

COLORS = {"victim": "#59a14f", "adaptive-sci": "#4e79a7", "naive-sci": "#e15759",
          "storm": "#f28e2b", "backfill": "#8a8a8a"}


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    # facility flow capacities: 28 GB/s filesystem, 5000 tok/s inference service
    b = Broker(POOL, io_capacity=28.0, inf_capacity=5000.0)
    SCI_IO = Dim(rate=6.0, bucket=3_600, ceiling=16.0)          # GB/s
    SCI_INF = Dim(rate=500.0, bucket=100_000, ceiling=2_000.0)  # tokens/s
    b.register(Envelope("victim",       5_000,  32,  32,  64,  60_000, "interactive"))
    b.register(Envelope("adaptive-sci", 20_000, 64, 128, 384, 300_000, "agent-burst",
                        io=SCI_IO, inference=SCI_INF))
    b.register(Envelope("naive-sci",    20_000, 64, 128, 384, 300_000, "agent-burst",
                        io=SCI_IO, inference=SCI_INF))
    b.register(Envelope("storm",        8_000,  16,  64, 384, 150_000, "agent-burst",
                        inference=Dim(rate=50.0, bucket=5_000, ceiling=500.0)))
    b.register(Envelope("backfill",     1e9,     0, 512, 512, 1e9,     "opportunistic"))

    # exercise the Sec. 10 invariant: guarantees are never overbooked
    try:
        b.register(Envelope("overbooked", 1, 400, 400, 400, 1))
        print("ERROR: overbooked guarantee was accepted")
    except ValueError as e:
        print(f"[invariant ok] rejected overbooked campaign: {e}")

    agents = [
        VictimAgent("victim", b),
        AdaptiveAgent("adaptive-sci", b, make_tasks(3000, seed=7)),
        NaiveAgent("naive-sci", b, make_tasks(3000, seed=7)),   # identical task set
        StormAgent("storm", b, STORM),
        BackfillAgent("backfill", b),
    ]
    adaptive, naive = agents[1], agents[2]

    names = list(COLORS)
    series = {"t": [], "mult": [],
              "held": {n: [] for n in names},
              "tokens": {n: [] for n in names},
              "credits": {n: [] for n in names},
              "io": {n: [] for n in names},
              "inf": {n: [] for n in names}}

    from broker import congestion_multiplier
    for t in range(1, SIM + 1):
        for a in agents:
            a.step(t)
        b.tick()
        if t % SAMPLE == 0:
            series["t"].append(t)
            series["mult"].append(congestion_multiplier(b.congestion_index()))
            for n in names:
                series["held"][n].append(b.held(n))
                series["tokens"][n].append(round(b.tokens[n] / b.envelopes[n].bucket_cap, 4))
                series["credits"][n].append(round(b.spent(n), 1))
                series["io"][n].append(round(b.dim_last["io"][n], 2))
                series["inf"][n].append(round(b.dim_last["inference"][n], 1))

    # ------------------------------------------------------------------
    # report (paper Sec. 13)
    # ------------------------------------------------------------------
    print(f"\n=== Bursar — 512-GPU pool, {SIM}s simulated ===\n")

    grants = b.grant_log
    for path in ("warm", "reclaim", "cold"):
        lats = [g["latency"] for g in grants if g["path"] == path]
        if lats:
            print(f"grants via {path:7s}: n={len(lats):4d}  "
                  f"p50={pct(lats, 50):4d}s  p95={pct(lats, 95):4d}s  max={max(lats):4d}s")

    vic = [g for g in grants if g["campaign"] == "victim"]
    in_storm = [g["latency"] for g in vic if STORM[0] <= g["t"] < STORM[1] + 60]
    outside = [g["latency"] for g in vic if not (STORM[0] <= g["t"] < STORM[1] + 60)]
    print(f"\n[isolation] victim grant latency outside storm: "
          f"p50={pct(outside, 50)}s p95={pct(outside, 95)}s (n={len(outside)})")
    print(f"[isolation] victim grant latency during  storm: "
          f"p50={pct(in_storm, 50)}s p95={pct(in_storm, 95)}s (n={len(in_storm)})")

    throttles = [j for j in b.journal if j["event"] == "throttle" and j["campaign"] == "storm"]
    storm_held = [h for i, h in enumerate(series["held"]["storm"])
                  if STORM[0] <= series["t"][i] < STORM[1]]
    print(f"\n[back pressure] storm throttle events: {len(throttles)}; "
          f"storm held during window: peak={max(storm_held)} GPUs, "
          f"mean={sum(storm_held)/len(storm_held):.0f} GPUs "
          f"(ceiling 384, sustained R=64 — bucket forces reversion)")

    print(f"\n[adaptivity]  {'campaign':13s} {'science':>9s} {'tasks':>6s} "
          f"{'credits':>9s} {'credits/science':>16s}")
    for ag in (naive, adaptive):
        spent = b.spent(ag.name)
        cps = spent / ag.science if ag.science else float("nan")
        print(f"              {ag.name:13s} {ag.science:9.1f} {ag.completed:6d} "
              f"{spent:9.1f} {cps:16.2f}")

    print(f"\n[envelope dims] {'campaign':13s} {'mean GB/s':>10s} {'io-throttled':>13s} "
          f"{'mean tok/s':>11s} {'inf-throttled':>14s}")
    for n in ("naive-sci", "adaptive-sci", "storm"):
        io_s = [series["io"][n][i] for i in range(len(series["t"]))]
        inf_s = [series["inf"][n][i] for i in range(len(series["t"]))]
        print(f"                {n:13s} {sum(io_s)/len(io_s):10.1f} "
              f"{b.dim_throttled['io'][n]:12d}s {sum(inf_s)/len(inf_s):11.0f} "
              f"{b.dim_throttled['inference'][n]:13d}s")

    util = sum(b.util_hist) / len(b.util_hist)
    print(f"\n[utilization] mean pool utilization: {util*100:.1f}% "
          f"(warm headroom + preemption grace are the cost of latency)")

    audit = [j for j in b.journal if j["event"] == "grant" and j["campaign"] == "victim"][:1] + \
            [j for j in b.journal if j["event"] == "throttle"][:1] + \
            [j for j in b.journal if j["event"] == "preempt-notice"][:1]
    print("\n[governance] sample journal records (full log in journal.jsonl):")
    for j in audit:
        print("  " + json.dumps(j))

    b.dump_journal("journal.jsonl")
    write_report("report.html", {
        "pool": POOL, "dt": SAMPLE, "storm": list(STORM),
        "campaigns": names, "colors": COLORS,
        "t": series["t"], "mult": series["mult"],
        "held": series["held"], "tokens": series["tokens"], "credits": series["credits"],
        "io": series["io"], "inf": series["inf"],
        "io_cap": 28.0,
        "grants": [{"t": g["t"], "c": g["campaign"], "lat": g["latency"],
                    "path": g["path"], "gpus": g["gpus"]} for g in grants],
    })
    print(f"\nwrote journal.jsonl ({len(b.journal)} records) and report.html — "
          f"open report.html for the animated replay")


if __name__ == "__main__":
    main()
