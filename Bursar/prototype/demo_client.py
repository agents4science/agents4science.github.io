"""A scripted agent negotiating with Bursar over HTTP.

Demonstrates the full loop from the paper's Sec. 6: register a campaign,
read supply signals, request a lease, report I/O + inference consumption
(throttle-at-source), and release. Python stdlib only.

Run against a live server:   python3 api.py --speed 20   (in another terminal)
                             python3 demo_client.py
Self-contained test:         python3 demo_client.py --selftest
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8642"
TOKEN = None


def call(method: str, path: str, body: dict | None = None, auth: bool = True):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    if auth and TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def say(label, obj):
    print(f"  {label:<28} {json.dumps(obj)[:150]}")


def run_demo(assertions: bool = False):
    global TOKEN

    print("\n1. Who is there?")
    code, info = call("GET", "/", auth=False)
    say("GET /", {k: info[k] for k in ("service", "now", "pool_gpus")})

    print("\n2. Register a campaign (allocation-committee role) -> bearer token")
    code, reg = call("POST", "/campaigns", {
        "campaign": "demo-sci", "budget_credits": 10_000,
        "guaranteed": 32, "sustained": 64, "burst_ceiling": 256, "bucket_cap": 120_000,
        "io": {"rate": 4.0, "bucket": 2_000, "ceiling": 10.0},
        "inference": {"rate": 200.0, "bucket": 20_000, "ceiling": 800.0}})
    say("POST /campaigns", reg)
    assert code == 201, reg
    TOKEN = reg["token"]

    print("\n3. Mutations without the token are refused (campaign-scoped credentials)")
    saved, TOKEN = TOKEN, None
    code, err = call("POST", "/leases", {"gpus": 8, "duration": 60})
    say("POST /leases (no auth)", {"status": code, **err})
    if assertions:
        assert code == 401, (code, err)
    TOKEN = saved

    print("\n4. Read supply signals before asking")
    code, s = call("GET", "/supply?campaign=demo-sci", auth=False)
    say("GET /supply", {k: s[k] for k in ("free_gpus", "congestion_multiplier",
                                          "tokens", "expected_path")})
    say("  io dimension", s["io"])
    say("  inference dimension", s["inference"])
    if assertions:
        assert "io" in s and "inference" in s and s["expected_path"] == "warm"

    print("\n5. Request a lease sized to supply, then poll for the grant")
    want = min(64, s["free_gpus"])
    code, r = call("POST", "/leases", {"gpus": want, "duration": 120, "cls": "agent-burst"})
    say(f"POST /leases gpus={want}", r)
    assert code == 202, r
    granted = None
    for _ in range(50):
        code, leases = call("GET", "/leases?campaign=demo-sci", auth=False)
        if leases["active"]:
            granted = leases["active"][0]
            break
        time.sleep(0.2)
    say("GET /leases (granted)", granted or leases)
    if assertions:
        assert granted and granted["gpus"] == want, leases

    print("\n6. Consume inside the lease: report I/O + inference, obey the allowance")
    code, c1 = call("POST", "/consume", {"io": 8.0, "inference": 500})
    say("POST /consume (burst)", c1)
    code, c2 = call("POST", "/consume", {"io": 20.0})   # above our 10 GB/s ceiling
    say("POST /consume (io=20)", c2)
    if assertions:
        assert c1["allowed"]["io"] <= 8.0 + 1e-9
        assert c2["allowed"]["io"] <= 10.0, c2          # ceiling enforced

    print("\n7. Release, and check the audit trail")
    code, rel = call("DELETE", "/leases")
    say("DELETE /leases", rel)
    code, grants = call("GET", "/grants?campaign=demo-sci", auth=False)
    say("GET /grants", grants[-1] if grants else grants)
    if assertions:
        assert rel["released"] and grants and grants[-1]["path"] == "warm"

    print("\ndemo complete: registered, negotiated, leased, consumed, released — "
          "all metered and journaled.")


def selftest():
    """Spin an in-process server at high speed and run the demo with assertions."""
    global BASE
    from api import serve
    import threading
    httpd, service = serve(port=8643, speed=50.0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    BASE = "http://127.0.0.1:8643"
    try:
        run_demo(assertions=True)
        print("\nSELFTEST PASS")
    finally:
        service.stop()
        httpd.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=BASE)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        BASE = args.url
        run_demo()
