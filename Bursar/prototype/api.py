"""Bursar REST facade — the broker API over HTTP (paper Sec. 6).

Turns the in-process broker into a service any agent can talk to, in any
language, from any machine. Python stdlib only: ThreadingHTTPServer plus a
real-time ticker thread that advances the broker clock.

Security model (prototype-grade, paper Sec. 11): registering a campaign
returns a campaign-scoped bearer token; all mutations (leases, consumption)
require it, so a campaign can act only as itself. In a facility deployment
registration would be an allocation-committee operation, not an open endpoint.

Run:    python3 api.py [--port 8642] [--speed 1] [--pool 512]
Try:    curl localhost:8642/            (service info + endpoint list)
        python3 demo_client.py          (a scripted agent negotiating over HTTP)
"""

import argparse
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from broker import Broker, Envelope, Dim


class BrokerService:
    """Wraps a Broker with a lock and a wall-clock ticker (1 tick = 1 simulated
    second; --speed N runs N ticks per real second)."""

    def __init__(self, broker: Broker, speed: float = 1.0):
        self.broker = broker
        self.lock = threading.Lock()
        self.auth: dict[str, str] = {}       # bearer token -> campaign
        self.speed = speed
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            with self.lock:
                self.broker.tick()
            time.sleep(1.0 / self.speed)

    def register(self, env: Envelope) -> str:
        with self.lock:
            self.broker.register(env)
        token = secrets.token_hex(16)
        self.auth[token] = env.campaign
        return token


def _dim(payload) -> Dim | None:
    return Dim(rate=payload["rate"], bucket=payload["bucket"],
               ceiling=payload["ceiling"]) if payload else None


def make_handler(service: BrokerService):
    b = service.broker

    class Handler(BaseHTTPRequestHandler):
        # ---- plumbing ------------------------------------------------
        def log_message(self, *a):          # quiet by default
            pass

        def _send(self, code: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _campaign(self) -> str | None:
            """Campaign identity from the bearer token (Sec. 11:
            campaign-scoped credentials, not user credentials)."""
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                return service.auth.get(auth[7:])
            return None

        def _q(self, name, default=None):
            return parse_qs(urlparse(self.path).query).get(name, [default])[0]

        # ---- GET -----------------------------------------------------
        def do_GET(self):
            path = urlparse(self.path).path
            with service.lock:
                if path == "/":
                    return self._send(200, {
                        "service": "bursar", "now": b.now, "pool_gpus": b.pool,
                        "io_capacity": b.io_capacity, "inf_capacity": b.inf_capacity,
                        "campaigns": sorted(b.envelopes),
                        "endpoints": ["GET /supply?campaign=", "GET /campaigns",
                                      "POST /campaigns", "POST /leases", "GET /leases?campaign=",
                                      "DELETE /leases", "POST /consume", "GET /grants?campaign=",
                                      "GET /journal?n=", "GET /openapi.json", "GET /healthz"]})
                if path == "/healthz":
                    return self._send(200, {"ok": True, "now": b.now})
                if path == "/openapi.json":
                    return self._send(200, OPENAPI)
                if path == "/supply":
                    c = self._q("campaign")
                    if c not in b.envelopes:
                        return self._send(404, {"error": f"unknown campaign {c!r}"})
                    return self._send(200, {"now": b.now, **b.supply(c)})
                if path == "/campaigns":
                    return self._send(200, [{
                        "campaign": c, "held_gpus": b.held(c),
                        "credits_remaining": round(b.credits[c], 1),
                        "gpu_tokens": round(b.tokens[c], 1),
                        "guaranteed": e.guaranteed, "sustained": e.sustained,
                        "burst_ceiling": e.burst_ceiling, "service_class": e.service_class,
                    } for c, e in sorted(b.envelopes.items())])
                if path == "/leases":
                    c = self._q("campaign")
                    if c not in b.envelopes:
                        return self._send(404, {"error": f"unknown campaign {c!r}"})
                    return self._send(200, {
                        "active": [{"id": l.id, "gpus": l.gpus, "cls": l.cls,
                                    "granted": l.granted, "expires": l.expires,
                                    "preempt_at": l.preempt_at}
                                   for l in b.leases if l.campaign == c],
                        "pending": [{"id": r.id, "gpus": r.gpus, "cls": r.cls,
                                     "submitted": r.submitted}
                                    for r in b.pending if r.campaign == c]})
                if path == "/grants":
                    c = self._q("campaign")
                    return self._send(200, [g for g in b.grant_log
                                            if c is None or g["campaign"] == c][-200:])
                if path == "/journal":
                    n = int(self._q("n", "100"))
                    return self._send(200, b.journal[-n:])
            self._send(404, {"error": "not found"})

        # ---- POST ----------------------------------------------------
        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._body()
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})

            if path == "/campaigns":
                try:
                    env = Envelope(
                        campaign=body["campaign"],
                        budget_credits=float(body["budget_credits"]),
                        guaranteed=int(body["guaranteed"]),
                        sustained=int(body["sustained"]),
                        burst_ceiling=int(body["burst_ceiling"]),
                        bucket_cap=float(body["bucket_cap"]),
                        service_class=body.get("service_class", "agent-burst"),
                        io=_dim(body.get("io")),
                        inference=_dim(body.get("inference")))
                    token = service.register(env)
                except (KeyError, TypeError) as e:
                    return self._send(400, {"error": f"bad envelope: {e}"})
                except ValueError as e:      # e.g. overbooked guarantees (Sec. 10)
                    return self._send(409, {"error": str(e)})
                return self._send(201, {"campaign": env.campaign, "token": token,
                                        "note": "send as 'Authorization: Bearer <token>'"})

            campaign = self._campaign()
            if campaign is None:
                return self._send(401, {"error": "campaign bearer token required"})

            if path == "/leases":
                with service.lock:
                    rid = b.request_lease(campaign, int(body["gpus"]),
                                          int(body["duration"]), body.get("cls"))
                if rid < 0:
                    return self._send(402, {"error": "budget exhausted: opportunistic only"})
                return self._send(202, {"request_id": rid, "campaign": campaign,
                                        "note": "poll GET /leases for grant"})

            if path == "/consume":
                with service.lock:
                    allowed = b.consume(campaign, io=float(body.get("io", 0)),
                                        inference=float(body.get("inference", 0)))
                return self._send(200, {"allowed": allowed})

            self._send(404, {"error": "not found"})

        # ---- DELETE --------------------------------------------------
        def do_DELETE(self):
            if urlparse(self.path).path == "/leases":
                campaign = self._campaign()
                if campaign is None:
                    return self._send(401, {"error": "campaign bearer token required"})
                with service.lock:
                    b.release_all(campaign, self._q("cls"))
                return self._send(200, {"released": True})
            self._send(404, {"error": "not found"})

    return Handler


OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "Bursar — Agent Resource Broker API", "version": "0.1",
             "description": "Campaign-level resource leases with token-bucket "
                            "back pressure. Register a campaign to obtain a "
                            "campaign-scoped bearer token; mutations require it."},
    "paths": {
        "/supply": {"get": {"summary": "Supply signals: free/reclaimable GPUs, congestion "
                                       "index+multiplier, token-bucket state (GPU, io, "
                                       "inference), credits, expected grant path",
                            "parameters": [{"name": "campaign", "in": "query", "required": True,
                                            "schema": {"type": "string"}}]}},
        "/campaigns": {
            "get": {"summary": "List campaigns and their envelope/consumption state"},
            "post": {"summary": "Register a campaign envelope; returns bearer token",
                     "requestBody": {"content": {"application/json": {"schema": {
                         "type": "object",
                         "required": ["campaign", "budget_credits", "guaranteed",
                                      "sustained", "burst_ceiling", "bucket_cap"],
                         "properties": {
                             "campaign": {"type": "string"},
                             "budget_credits": {"type": "number"},
                             "guaranteed": {"type": "integer"},
                             "sustained": {"type": "integer", "description": "R, GPUs"},
                             "burst_ceiling": {"type": "integer"},
                             "bucket_cap": {"type": "number", "description": "B, GPU-seconds"},
                             "service_class": {"type": "string",
                                               "enum": ["deadline", "interactive",
                                                        "agent-burst", "opportunistic"]},
                             "io": {"$ref": "#/components/schemas/Dim"},
                             "inference": {"$ref": "#/components/schemas/Dim"}}}}}}}},
        "/leases": {
            "get": {"summary": "Active + pending leases for a campaign",
                    "parameters": [{"name": "campaign", "in": "query", "required": True,
                                    "schema": {"type": "string"}}]},
            "post": {"summary": "Request a lease (auth required); grant is asynchronous — "
                                "poll GET /leases",
                     "requestBody": {"content": {"application/json": {"schema": {
                         "type": "object", "required": ["gpus", "duration"],
                         "properties": {"gpus": {"type": "integer"},
                                        "duration": {"type": "integer"},
                                        "cls": {"type": "string"}}}}}}},
            "delete": {"summary": "Release all leases (auth required); optional ?cls= filter"}},
        "/consume": {"post": {"summary": "Report intended io (GB/s) / inference (tok/s) "
                                         "consumption; returns the allowed amounts "
                                         "(throttle-at-source)",
                              "requestBody": {"content": {"application/json": {"schema": {
                                  "type": "object",
                                  "properties": {"io": {"type": "number"},
                                                 "inference": {"type": "number"}}}}}}}},
        "/grants": {"get": {"summary": "Grant log (latency + path per lease)"}},
        "/journal": {"get": {"summary": "Tail of the audit journal"}},
    },
    "components": {"schemas": {"Dim": {
        "type": "object", "required": ["rate", "bucket", "ceiling"],
        "properties": {"rate": {"type": "number", "description": "sustained units/s"},
                       "bucket": {"type": "number", "description": "burst headroom, units"},
                       "ceiling": {"type": "number", "description": "max instantaneous units/s"}}}}},
}


def serve(port: int = 8642, pool: int = 512, io_cap: float = 28.0,
          inf_cap: float = 5000.0, speed: float = 1.0):
    service = BrokerService(Broker(pool, io_capacity=io_cap, inf_capacity=inf_cap), speed)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    service.start()
    return httpd, service


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bursar REST facade")
    p.add_argument("--port", type=int, default=8642)
    p.add_argument("--pool", type=int, default=512)
    p.add_argument("--io-cap", type=float, default=28.0)
    p.add_argument("--inf-cap", type=float, default=5000.0)
    p.add_argument("--speed", type=float, default=1.0,
                   help="simulated seconds per real second")
    args = p.parse_args()
    httpd, service = serve(args.port, args.pool, args.io_cap, args.inf_cap, args.speed)
    print(f"Bursar listening on http://127.0.0.1:{args.port}  "
          f"(pool={args.pool} GPUs, speed={args.speed}x) — Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        service.stop()
