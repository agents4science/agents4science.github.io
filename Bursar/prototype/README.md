# Bursar — an Agent Resource Broker for agent-native HPC (simulation prototype)

A runnable, zero-dependency prototype of the **Agent-Native HPC** proposal
(`../../Agent_Native_HPC_Proposal_v2.md`): campaign-level resource envelopes,
token-bucket back pressure on resource *acquisition* (never on task creation),
short-lived leases served through warm/reclaim/cold paths, and a deterministic,
journaled admission rule.

This corresponds to the paper's **Phase 1** (broker + synthetic load): the
cluster is simulated, the policy engine is real.

## Run it

```
python3 main.py && open report.html      # offline simulation + animated replay
python3 api.py --speed 20                # REST facade: the broker as a live service
python3 demo_client.py                   # an agent negotiating with it over HTTP
python3 demo_client.py --selftest        # self-contained end-to-end test
python3 parsl_demo.py                    # a real Parsl workflow on Bursar leases (needs parsl)
```

Pure Python stdlib — still zero dependencies. The simulation outputs:

- a console report mapped to the paper's Sec. 13 success criteria
- `journal.jsonl` — the full audit journal (every grant/throttle/preemption with the rule inputs)
- `report.html` — self-contained animated replay (press Play): pool occupancy
  stacked by campaign, token-bucket levels, congestion multiplier, and every
  lease grant plotted by latency and path

## The scenario

512-GPU agent-burst pool, 2 simulated hours, five campaigns:

| Campaign | Behavior | Envelope (guar / R / burst / B) |
|---|---|---|
| `victim` | small periodic interactive requests; the isolation probe | 32 / 32 / 64 / 60k |
| `adaptive-sci` | reads the supply API; runs work whose value/GPU-h clears the congestion price, conserves its bucket, mops up cheap opportunistic capacity | 64 / 128 / 384 / 300k |
| `naive-sci` | same task set, but demands 256 GPUs always and ignores prices | 64 / 128 / 384 / 300k |
| `storm` | adversarial: hammers burst-ceiling requests for 30 min | 16 / 64 / 384 / 150k |
| `backfill` | facility-side opportunistic filler, fully preemptible | 0 / 512 / 512 / ∞ |

## What it demonstrates

1. **Back pressure works** — the storm asks for 384 GPUs continuously but its
   token bucket forces reversion toward its sustained rate (mean ≈ 215 GPUs,
   then throttled); it never has to be trusted, only metered.
2. **Isolation holds** — victim grant latency is p95 ≈ 1 s *during* the storm,
   because consumption beyond any campaign's guarantee may never squeeze
   another campaign's unmet guarantee (broker `_grantable`).
3. **Adaptivity pays** — with identical task sets, the adaptive agent produces
   ~55% more science at ~2.6× lower credit cost than the naive agent, purely by
   reading supply signals and deferring low-value work to cheap capacity.
4. **Utilization survives** — opportunistic backfill recovers most of the warm
   headroom (~89% mean pool utilization).
5. **Envelopes are multi-dimensional** — filesystem bandwidth and model
   inference are metered with the same token-bucket semantics as GPUs
   (throttle-at-source): the I/O-blind naive agent loses ~40% of its science to
   I/O throttling, and the storm's inference spam (200 tok/s demanded) is pinned
   to its 50 tok/s sustained rate. Even the adversary's *thinking* is metered.
6. **Governance is auditable** — every admission decision is journaled with the
   rule inputs (class rank, fair-share deficit, token state).
7. **Joining a pilot is a config change, not a rewrite** — `parsl_provider.py`
   is a Parsl execution provider that maps each Parsl block to a Bursar lease:
   scale-out negotiates a lease (sized to live supply signals), scale-in
   releases it, and a revoked or expired lease tears its block down. The
   science code in `parsl_demo.py` is ordinary Parsl; the entire integration
   surface is the provider block in the Parsl config.

## Parameter-sweep experiments

`python3 experiments.py` (requires matplotlib — the only part that does) runs
three sweeps over the scenario and writes paper-ready figures to `figures/`:

1. **What buys low latency** (`fig_latency_price.png`, §7/§14) — sweeping warm
   headroom 16→256 GPUs costs 13 points of utilization but leaves
   time-to-128-GPUs pinned at ~62 s, because a work-conserving preemptible tier
   refills spare capacity anyway; headroom helps only once it exceeds the
   request size. Sweeping the preemption grace period 15→120 s moves latency
   one-for-one (p50 ≈ grace + 2 s). **The grace period, not idle headroom, is
   what prices latency.**
2. **Overbooking vs. isolation** (`fig_overbooking.png`, §10) — with correlated
   adversarial storms pushing Σ burst ceilings to 7× the pool, the guaranteed
   campaign stays at p95 = 1 s with zero violations at every φ; scarcity lands
   entirely on non-guaranteed burst service (probe p50 41 s → 591 s censored).
3. **Enforcement semantics** (`fig_bucket.png`, §4) — checking tokens only at
   grant instants leaks burst through lease-duration granularity (an adversary
   with a 25k GPU-s bucket extracted 167k GPU-s). The sweep found this leak;
   **prepaid leases** (the above-R portion of a lease must be token-covered for
   its whole duration, now the broker default) pin extraction to B almost
   exactly (26k at B=25k).

Latency is measured with a probe campaign — a standardized 128-GPU request
every 5 minutes, timed to *full* capacity — pooled across three task seeds.

## Code map (→ paper sections)

- `broker.py` — envelopes, token buckets, credits (§4–5); I/O + inference flow
  dimensions with throttle-at-source `consume()` (§4); supply/intent API (§6);
  warm/reclaim/cold grant paths with preemption grace (§7); guarantee-honoring,
  deterministic, journaled admission (§10)
- `api.py` — REST facade: the broker as an HTTP service with campaign-scoped
  bearer tokens (§6, §11) and an OpenAPI description at `/openapi.json`
- `demo_client.py` — a scripted agent negotiating over HTTP; `--selftest` runs
  the full loop against an in-process server
- `parsl_provider.py` — framework adapter: Parsl blocks as Bursar leases (§12);
  `parsl_demo.py` runs a real Parsl workflow on top (the only file needing parsl)
- `agents.py` — synthetic campaigns (§12 step 4), including the latency probe
- `scenario.py` — the five-campaign scenario as a parameterized callable + metrics
- `main.py` — default scenario + Sec. 13 report
- `experiments.py` — parameter sweeps + figures (§4, §7, §10, §14)
- `viz.py` — animated HTML replay generator

## Non-goals (next steps)

- Parsl is the only framework adapter so far; Globus Compute and Ray launchers
  would follow the same shape
- No real scheduler binding — a PBS Professional / Slurm binding would replace
  the simulated pool with real allocations behind the same broker interface
- Single-node, single-process; no broker HA / journal replay (paper §11)
- I/O enforcement is cooperative (throttle-at-source): the runtime is trusted
  to obey `consume()` allowances; a facility would back this with filesystem
  QOS where available
