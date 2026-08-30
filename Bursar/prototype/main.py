"""Bursar prototype — simulation driver.

Runs the default five-campaign scenario (scenario.py) on a 512-GPU pool,
prints a report mapped to the paper's Sec. 13 success criteria, and writes
journal.jsonl + report.html (animated replay).

Usage: python3 main.py && open report.html
"""

import json

from broker import Broker, Envelope
from scenario import run, metrics, pct, COLORS
from viz import write_report

STORM = (3600, 5400)


def main():
    # exercise the Sec. 10 invariant: guarantees are never overbooked
    probe = Broker(64)
    probe.register(Envelope("a", 1, 40, 40, 40, 1))
    try:
        probe.register(Envelope("overbooked", 1, 40, 40, 40, 1))
        print("ERROR: overbooked guarantee was accepted")
    except ValueError as e:
        print(f"[invariant ok] rejected overbooked campaign: {e}")

    result = run(storm_window=STORM)
    b, series, m = result["broker"], result["series"], metrics(result)

    print(f"\n=== Bursar — {result['pool']}-GPU pool, {series['t'][-1]}s simulated ===\n")

    for path in ("warm", "reclaim", "cold"):
        lats = [g["latency"] for g in b.grant_log if g["path"] == path]
        if lats:
            print(f"grants via {path:7s}: n={len(lats):4d}  "
                  f"p50={pct(lats, 50):4d}s  p95={pct(lats, 95):4d}s  max={max(lats):4d}s")

    print(f"\n[isolation] victim grant latency outside storm: "
          f"p50={m['victim_p50_out']}s p95={m['victim_p95_out']}s")
    print(f"[isolation] victim grant latency during  storm: "
          f"p50={m['victim_p50_in']}s p95={m['victim_p95_in']}s "
          f"(violations >60s: {m['victim_violations_in']})")

    throttles = [j for j in b.journal if j["event"] == "throttle" and j["campaign"] == "storm"]
    print(f"\n[back pressure] storm throttle events: {len(throttles)}; "
          f"storm held during window: peak={m['storm_peak_held']:.0f} GPUs, "
          f"mean={m['storm_mean_held']:.0f} GPUs "
          f"(ceiling 384, sustained R=64 — bucket forces reversion)")

    print(f"\n[adaptivity]  {'campaign':13s} {'science':>9s} {'tasks':>6s} "
          f"{'credits':>9s} {'credits/science':>16s}")
    for name in ("naive-sci", "adaptive-sci"):
        ag = result["agents"][name]
        spent = b.spent(name)
        print(f"              {name:13s} {ag.science:9.1f} {ag.completed:6d} "
              f"{spent:9.1f} {spent / ag.science if ag.science else float('nan'):16.2f}")

    print(f"\n[envelope dims] {'campaign':13s} {'mean GB/s':>10s} {'io-throttled':>13s} "
          f"{'mean tok/s':>11s} {'inf-throttled':>14s}")
    for n in ("naive-sci", "adaptive-sci", "storm"):
        io_s, inf_s = series["io"][n], series["inf"][n]
        print(f"                {n:13s} {sum(io_s)/len(io_s):10.1f} "
              f"{b.dim_throttled['io'][n]:12d}s {sum(inf_s)/len(inf_s):11.0f} "
              f"{b.dim_throttled['inference'][n]:13d}s")

    print(f"\n[utilization] mean pool utilization: {m['utilization']*100:.1f}% "
          f"(warm headroom + preemption grace are the cost of latency)")

    audit = [j for j in b.journal if j["event"] == "grant" and j["campaign"] == "victim"][:1] + \
            [j for j in b.journal if j["event"] == "throttle"][:1] + \
            [j for j in b.journal if j["event"] == "preempt-notice"][:1]
    print("\n[governance] sample journal records (full log in journal.jsonl):")
    for j in audit:
        print("  " + json.dumps(j))

    b.dump_journal("journal.jsonl")
    write_report("report.html", {
        "pool": result["pool"], "dt": result["sample"], "storm": list(STORM),
        "campaigns": result["names"], "colors": COLORS,
        "t": series["t"], "mult": series["mult"],
        "held": series["held"], "tokens": series["tokens"], "credits": series["credits"],
        "io": series["io"], "inf": series["inf"], "io_cap": 28.0,
        "grants": [{"t": g["t"], "c": g["campaign"], "lat": g["latency"],
                    "path": g["path"], "gpus": g["gpus"]} for g in b.grant_log],
    })
    print(f"\nwrote journal.jsonl ({len(b.journal)} records) and report.html — "
          f"open report.html for the animated replay")


if __name__ == "__main__":
    main()
