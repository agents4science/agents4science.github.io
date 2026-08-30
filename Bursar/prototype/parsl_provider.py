"""BursarProvider — a Parsl execution provider that acquires capacity from
Bursar leases instead of submitting scheduler jobs.

This is the framework-adapter argument made concrete (paper Sec. 12): an
application team that already runs on Parsl joins a Bursar pilot by changing
a config block, not their science code. Each Parsl *block* becomes a Bursar
*lease*: scale-out requests a lease through the intent API (optionally sized
to live supply signals), scale-in releases it, and if Bursar revokes or
expires a lease the provider tears the block down — the broker's word is
enforced, not advisory.

In this prototype the leased "GPUs" are simulated, so granted blocks launch
workers locally (via LocalProvider) as the stand-in data plane. On a real
facility the same negotiation logic would sit in front of the scheduler
binding; nothing in the Parsl-facing surface changes.

Usage sketch:

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl_provider import BursarProvider

    provider = BursarProvider(
        campaign="my-campaign",
        base_url="http://127.0.0.1:8642",
        envelope=dict(budget_credits=10_000, guaranteed=32, sustained=64,
                      burst_ceiling=256, bucket_cap=120_000),   # or token=...
        gpus_per_block=64, lease_duration=600, adaptive=True,
        init_blocks=1, min_blocks=0, max_blocks=3)

    config = Config(executors=[HighThroughputExecutor(provider=provider, ...)])
"""

import json
import logging
import time
import urllib.error
import urllib.request

from parsl.providers import LocalProvider
from parsl.jobs.states import JobState, JobStatus

logger = logging.getLogger("parsl.providers.bursar")


class BursarClient:
    """Minimal stdlib client for the Bursar REST API."""

    def __init__(self, base_url: str, campaign: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.campaign = campaign
        self.token = token

    def _call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base_url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def register(self, envelope: dict) -> str:
        code, resp = self._call("POST", "/campaigns",
                                {"campaign": self.campaign, **envelope})
        if code == 409 and "exceeds pool" in resp.get("error", ""):
            raise ValueError(resp["error"])
        if code != 201:
            raise RuntimeError(f"campaign registration failed: {resp}")
        self.token = resp["token"]
        return self.token

    def supply(self) -> dict:
        code, resp = self._call("GET", f"/supply?campaign={self.campaign}")
        if code != 200:
            raise RuntimeError(f"supply query failed: {resp}")
        return resp

    def request_lease(self, gpus: int, duration: int, cls: str) -> int:
        code, resp = self._call("POST", "/leases",
                                {"gpus": gpus, "duration": duration, "cls": cls})
        if code != 202:
            raise RuntimeError(f"lease request failed: {resp}")
        return resp["request_id"]

    def leases(self) -> dict:
        code, resp = self._call("GET", f"/leases?campaign={self.campaign}")
        return resp if code == 200 else {"active": [], "pending": []}

    def release_lease(self, lease_id: int) -> bool:
        code, resp = self._call("DELETE", f"/leases/{lease_id}")
        return code == 200 and resp.get("released", False)


class BursarProvider(LocalProvider):
    """Parsl provider that maps blocks to Bursar leases.

    Bursar-specific parameters (everything else is LocalProvider/Parsl):
      campaign        campaign name
      base_url        Bursar REST endpoint
      envelope        dict for auto-registration (prototype convenience), or
      token           campaign bearer token from prior registration
      gpus_per_block  lease size requested per Parsl block
      lease_duration  lease length in broker seconds (renewed by rescale)
      service_class   Bursar service class for the leases
      adaptive        size requests to live supply signals instead of
                      demanding gpus_per_block regardless
      min_grant       smallest acceptable adaptive request
      grant_timeout   wall-clock seconds to wait for a grant before giving up
    """

    def __init__(self, campaign: str, base_url: str = "http://127.0.0.1:8642",
                 envelope: dict | None = None, token: str | None = None,
                 gpus_per_block: int = 64, lease_duration: int = 600,
                 service_class: str = "agent-burst", adaptive: bool = True,
                 min_grant: int = 8, grant_timeout: float = 120.0, **local_kwargs):
        super().__init__(**local_kwargs)
        self.client = BursarClient(base_url, campaign, token)
        if token is None:
            if envelope is None:
                raise ValueError("provide either token= or envelope= for registration")
            self.client.register(envelope)
        self.gpus_per_block = gpus_per_block
        self.lease_duration = lease_duration
        self.service_class = service_class
        self.adaptive = adaptive
        self.min_grant = min_grant
        self.grant_timeout = grant_timeout
        self._lease_of: dict[object, int] = {}    # local job id -> lease id
        self._gpus_of: dict[object, int] = {}

    @property
    def label(self):
        return "bursar"

    # ------------------------------------------------------------------
    def _size_request(self) -> int:
        """Adaptive sizing (paper Sec. 6): ask for what supply can actually
        grant rather than demanding the configured block size blindly."""
        want = self.gpus_per_block
        if not self.adaptive:
            return want
        try:
            s = self.client.supply()
        except RuntimeError:
            return want
        grantable = s["free_gpus"] + s["reclaimable_gpus"]
        if s["tokens"] <= 0:                      # bucket empty: confined to R
            grantable = min(grantable, int(s["replenish_rate"]))
        return max(self.min_grant, min(want, grantable))

    def submit(self, command, tasks_per_node, job_name="parsl.bursar"):
        gpus = self._size_request()
        req_id = self.client.request_lease(gpus, self.lease_duration, self.service_class)
        logger.info("bursar: requested lease %s (%d GPUs, %ds, %s)",
                    req_id, gpus, self.lease_duration, self.service_class)

        deadline = time.time() + self.grant_timeout
        lease = None
        while time.time() < deadline:
            active = {l["id"]: l for l in self.client.leases()["active"]}
            if req_id in active:
                lease = active[req_id]
                break
            time.sleep(0.5)
        if lease is None:
            self.client.release_lease(req_id)     # withdraw the pending request
            logger.warning("bursar: lease %s not granted within %.0fs; "
                           "reporting at-capacity to Parsl", req_id, self.grant_timeout)
            return None                           # Parsl treats None as at-capacity

        logger.info("bursar: lease %s granted (%d GPUs) — launching block",
                    lease["id"], lease["gpus"])
        job_id = super().submit(command, tasks_per_node, job_name)
        if job_id is None:
            self.client.release_lease(lease["id"])
            return None
        self._lease_of[job_id] = lease["id"]
        self._gpus_of[job_id] = lease["gpus"]
        return job_id

    def status(self, job_ids):
        """Local process status, overridden by lease state: if Bursar has
        revoked or expired a block's lease, the block is torn down and
        reported COMPLETED — enforcement, not advice."""
        statuses = super().status(job_ids)
        try:
            active = {l["id"] for l in self.client.leases()["active"]}
        except Exception:
            return statuses
        for i, jid in enumerate(job_ids):
            lease_id = self._lease_of.get(jid)
            if lease_id is not None and lease_id not in active \
                    and statuses[i].state == JobState.RUNNING:
                logger.info("bursar: lease %s gone (expired/preempted); "
                            "tearing down block %s", lease_id, jid)
                super().cancel([jid])
                self._lease_of.pop(jid, None)
                statuses[i] = JobStatus(JobState.COMPLETED)
        return statuses

    def cancel(self, job_ids):
        for jid in job_ids:
            lease_id = self._lease_of.pop(jid, None)
            if lease_id is not None:
                self.client.release_lease(lease_id)
                logger.info("bursar: released lease %s with block %s", lease_id, jid)
        return super().cancel(job_ids)
