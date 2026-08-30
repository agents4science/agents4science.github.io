"""End-to-end demo: a real Parsl workflow drawing capacity from Bursar.

Starts an in-process Bursar service, registers a campaign, then runs a
Parsl HighThroughputExecutor whose *provider* is BursarProvider — every
Parsl block is a Bursar lease, negotiated over HTTP. The science code
(the @python_app) knows nothing about any of it.

Requires parsl (pip install parsl). Run: python3 parsl_demo.py
"""

import threading
import time

import parsl
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl import python_app

from api import serve
from parsl_provider import BursarProvider

PORT = 8663


@python_app
def simulate(i):
    """Stand-in science task; knows nothing about Bursar."""
    import time as t
    t.sleep(1.5)
    return i * i


def main():
    # 1. a live Bursar service (accelerated clock: 10 broker-seconds / second)
    httpd, service = serve(port=PORT, pool=512, speed=10.0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Bursar up on :{PORT} (512-GPU simulated pool, 10x clock)")

    # 2. Parsl config whose provider negotiates with Bursar.
    #    This block is the entire integration surface for an application team.
    provider = BursarProvider(
        campaign="parsl-demo",
        base_url=f"http://127.0.0.1:{PORT}",
        envelope=dict(budget_credits=10_000, guaranteed=32, sustained=128,
                      burst_ceiling=256, bucket_cap=120_000),
        gpus_per_block=64, lease_duration=1200, adaptive=True,
        init_blocks=1, min_blocks=0, max_blocks=3,
    )
    config = Config(
        executors=[HighThroughputExecutor(
            label="bursar_htex", provider=provider,
            max_workers_per_node=4, encrypted=False)],
        run_dir="/tmp/parsl_bursar_demo",
    )
    parsl.load(config)

    # 3. ordinary Parsl science code
    t0 = time.time()
    futures = [simulate(i) for i in range(24)]
    results = [f.result() for f in futures]
    elapsed = time.time() - t0
    assert results == [i * i for i in range(24)]
    print(f"\n24 Parsl tasks completed in {elapsed:.1f}s on Bursar-leased capacity")

    # 4. show the negotiation that happened underneath
    client = provider.client
    grants = client._call("GET", f"/grants?campaign=parsl-demo")[1]
    print("\nleases negotiated for Parsl blocks:")
    for g in grants:
        print(f"  t={g['t']:>4} {g['gpus']:>3} GPUs  {g['cls']:<11} "
              f"latency={g['latency']}s path={g['path']}")
    s = client.supply()
    print(f"\ncampaign state after run: {s['credits_remaining']:.1f} credits left, "
          f"tokens={s['tokens']:.0f}/{s['bucket_cap']:.0f}")

    parsl.dfk().cleanup()
    active = client.leases()["active"]
    print(f"leases still held after Parsl cleanup: {len(active)} (blocks released on scale-in)")
    service.stop()
    httpd.shutdown()
    print("\nPARSL DEMO PASS" if not active else "\nWARNING: leases leaked")


if __name__ == "__main__":
    main()
