"""Parameterized Bursar scenario: the five-campaign demo as a callable.

`run(**knobs)` builds and executes the simulation and returns broker, agents,
and sampled timeseries; `metrics(result)` reduces a run to the paper's Sec. 13
measures. main.py uses the defaults; experiments.py sweeps the knobs.
"""

from broker import Broker, Envelope, Dim, congestion_multiplier
from agents import (make_tasks, NaiveAgent, AdaptiveAgent,
                    StormAgent, VictimAgent, BackfillAgent, ProbeAgent)

COLORS = {"victim": "#59a14f", "adaptive-sci": "#4e79a7", "naive-sci": "#e15759",
          "storm": "#f28e2b", "backfill": "#8a8a8a"}


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def run(pool=512, io_capacity=28.0, inf_capacity=5000.0,
        headroom=160, seed=7, n_tasks=3000, prepaid_leases=True, grace=60,
        sci_R=128, sci_B=300_000, sci_burst=384,
        n_storms=1, storm_R=64, storm_B=150_000, storm_burst=384,
        storm_window=(3600, 5400), probe=False,
        sim=7200, sample=5):
    b = Broker(pool, io_capacity=io_capacity, inf_capacity=inf_capacity,
               prepaid_leases=prepaid_leases, preempt_grace=grace)
    sci_io = Dim(rate=6.0, bucket=3_600, ceiling=16.0)           # GB/s
    sci_inf = Dim(rate=500.0, bucket=100_000, ceiling=2_000.0)   # tokens/s

    b.register(Envelope("victim",       5_000,  32,  32,  64,  60_000, "interactive"))
    b.register(Envelope("adaptive-sci", 20_000, 64, sci_R, sci_burst, sci_B, "agent-burst",
                        io=sci_io, inference=sci_inf))
    b.register(Envelope("naive-sci",    20_000, 64, sci_R, sci_burst, sci_B, "agent-burst",
                        io=sci_io, inference=sci_inf))
    storm_names = ["storm"] + [f"storm{i}" for i in range(2, n_storms + 1)]
    for name in storm_names:
        b.register(Envelope(name, 8_000, 16, storm_R, storm_burst, storm_B, "agent-burst",
                            inference=Dim(rate=50.0, bucket=5_000, ceiling=500.0)))
    if probe:
        b.register(Envelope("probe", 1e9, 0, 128, 128, 1e9, "agent-burst"))
    b.register(Envelope("backfill", 1e9, 0, pool, pool, 1e9, "opportunistic"))

    agents = ([VictimAgent("victim", b),
               AdaptiveAgent("adaptive-sci", b, make_tasks(n_tasks, seed)),
               NaiveAgent("naive-sci", b, make_tasks(n_tasks, seed))]
              + [StormAgent(name, b, storm_window) for name in storm_names]
              + ([ProbeAgent("probe", b)] if probe else [])
              + [BackfillAgent("backfill", b, headroom=headroom)])

    # sampled series cover the display campaigns; extra storms fold into "storm"
    names = list(COLORS)
    series = {"t": [], "mult": [],
              "held": {n: [] for n in names},
              "tokens": {n: [] for n in names},
              "credits": {n: [] for n in names},
              "io": {n: [] for n in names},
              "inf": {n: [] for n in names}}

    def held_of(n):
        if n == "storm":
            return sum(b.held(s) for s in storm_names)
        return b.held(n)

    for t in range(1, sim + 1):
        for a in agents:
            a.step(t)
        b.tick()
        if t % sample == 0:
            series["t"].append(t)
            series["mult"].append(congestion_multiplier(b.congestion_index()))
            for n in names:
                series["held"][n].append(held_of(n))
                series["tokens"][n].append(round(b.tokens[n] / b.envelopes[n].bucket_cap, 4))
                series["credits"][n].append(round(b.spent(n), 1))
                series["io"][n].append(round(b.dim_last["io"][n], 2))
                series["inf"][n].append(round(b.dim_last["inference"][n], 1))

    return {"broker": b, "agents": {a.name: a for a in agents}, "series": series,
            "names": names, "storm_names": storm_names, "window": storm_window,
            "pool": pool, "sample": sample, "params": {
                "headroom": headroom, "n_storms": n_storms, "storm_B": storm_B,
                "storm_R": storm_R, "storm_burst": storm_burst, "seed": seed}}


def metrics(result) -> dict:
    """Reduce a run to the Sec. 13 measures used by main.py and experiments.py."""
    b = result["broker"]
    lo, hi = result["window"]
    series, sample = result["series"], result["sample"]
    grants = b.grant_log

    def lat(campaigns, cls=None, inside=None):
        out = []
        for g in grants:
            if g["campaign"] not in campaigns:
                continue
            if cls and g["cls"] != cls:
                continue
            in_win = lo <= g["t"] < hi + 60
            if inside is not None and in_win != inside:
                continue
            out.append(g["latency"])
        return out

    vic_in, vic_out = lat({"victim"}, inside=True), lat({"victim"}, inside=False)
    probe = result["agents"].get("probe")
    probe_out = [l for l, t in probe.lats if not lo <= t < hi + 60] if probe else []
    probe_in = [l for l, t in probe.lats if lo <= t < hi + 60] if probe else []
    sci = {"adaptive-sci", "naive-sci"}
    sci_burst_all = lat(sci, cls="agent-burst")
    sci_burst_in = lat(sci, cls="agent-burst", inside=True)

    idx = [i for i, t in enumerate(series["t"]) if lo <= t < hi]
    storm_held = [series["held"]["storm"][i] for i in idx]
    storm_R = result["params"]["storm_R"] * result["params"]["n_storms"]
    excess = sum(max(0, h - storm_R) for h in storm_held) * sample   # GPU-seconds

    util = sum(b.util_hist) / len(b.util_hist)
    prod = [sum(series["held"][n][i] for n in ("victim", "adaptive-sci", "naive-sci", "storm"))
            for i in range(len(series["t"]))]
    naive, adaptive = result["agents"]["naive-sci"], result["agents"]["adaptive-sci"]

    return {
        "victim_p50_in": pct(vic_in, 50), "victim_p95_in": pct(vic_in, 95),
        "victim_p50_out": pct(vic_out, 50), "victim_p95_out": pct(vic_out, 95),
        "victim_violations_in": sum(1 for l in vic_in if l > 60),
        "sci_burst_p50": pct(sci_burst_all, 50), "sci_burst_p95": pct(sci_burst_all, 95),
        "sci_burst_p95_in": pct(sci_burst_in, 95),
        "probe_lats": probe_out, "probe_lats_in": probe_in,
        "warm": sum(1 for g in grants if g["path"] == "warm"),
        "reclaim": sum(1 for g in grants if g["path"] == "reclaim"),
        "cold": sum(1 for g in grants if g["path"] == "cold"),
        "utilization": util,
        "productive_utilization": sum(prod) / len(prod) / result["pool"],
        "storm_mean_held": sum(storm_held) / len(storm_held),
        "storm_peak_held": max(storm_held),
        "storm_excess_gpu_s": excess,
        "naive_science": naive.science, "naive_spent": b.spent("naive-sci"),
        "adaptive_science": adaptive.science, "adaptive_spent": b.spent("adaptive-sci"),
    }
