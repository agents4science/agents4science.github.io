"""Bursar — an Agent Resource Broker (simulation prototype).

Implements the core semantics of the Agent-Native HPC proposal (v2):
  - token-bucket back pressure on resource acquisition   (Sec. 4)
  - service classes, credits, congestion index           (Sec. 5)
  - supply / intent API                                  (Sec. 6)
  - warm / reclaim / cold lease-grant paths              (Sec. 7)
  - deterministic, journaled admission; no overbooked
    guarantees; fair-share arbitration of bursts         (Sec. 10)

Time is simulated in 1-second ticks. The "cluster" is the agent-burst
warm pool only; the conventional batch side of the machine is out of
scope, exactly as in the pilot design.
"""

from dataclasses import dataclass, field
from collections import deque
import json

CLASS_RANK = {"deadline": 0, "interactive": 1, "agent-burst": 2, "opportunistic": 3}
BASE_MULT = {"deadline": 4.0, "interactive": 2.0, "agent-burst": 3.0, "opportunistic": 0.25}
PREEMPT_GRACE = 60        # s: contractual checkpoint window for opportunistic leases
CONGESTION_WINDOW = 60    # s: trailing window for the congestion index
FAIRSHARE_WINDOW = 600    # s: trailing window for weighted-fair-share deficit
REQUEST_TTL = 900         # s: pending requests older than this are dropped


def congestion_multiplier(index: float) -> float:
    """Fixed, published lookup table (Sec. 5). Bounded 1x-3x."""
    if index < 0.5:
        return 1.0
    if index < 1.0:
        return 1.5
    if index < 2.0:
        return 2.25
    return 3.0


@dataclass
class Dim:
    """A flow-controlled envelope dimension beyond GPUs (Sec. 4: 'the envelope
    is not compute-only'). Same token-bucket semantics as compute: tokens
    replenish at `rate`, accumulate up to `bucket`, and drain on consumption;
    an empty bucket confines the campaign to `rate`; `ceiling` bounds
    instantaneous use. Enforcement is throttle-at-source: consume() returns
    the allowed amount and the campaign runtime is expected to obey."""
    rate: float               # sustained consumption (units/s); token replenishment
    bucket: float             # token capacity (units of burst headroom)
    ceiling: float            # max instantaneous consumption (units/s)


UNMETERED = Dim(rate=1e12, bucket=1e12, ceiling=1e12)
INFERENCE_CREDITS_PER_TOKEN = 0.02 / 1000   # inference metered against the budget (Sec. 4)


@dataclass
class Envelope:
    campaign: str
    budget_credits: float     # class-weighted GPU-hour credits (Sec. 5)
    guaranteed: int           # GPUs available on demand; never overbooked
    sustained: int            # R: token replenishment rate, GPUs
    burst_ceiling: int        # GPUs
    bucket_cap: float         # B: GPU-seconds
    service_class: str = "agent-burst"
    io: Dim = None            # filesystem bandwidth, GB/s
    inference: Dim = None     # model inference, tokens/s

    def __post_init__(self):
        self.io = self.io or UNMETERED
        self.inference = self.inference or UNMETERED


@dataclass
class Request:
    id: int
    campaign: str
    gpus: int
    duration: int
    cls: str
    submitted: int
    throttle_logged: bool = False


@dataclass
class Lease:
    id: int
    campaign: str
    gpus: int
    cls: str
    submitted: int
    granted: int
    expires: int
    preempt_at: int = -1      # tick at which a preempted lease is reclaimed


class Broker:
    def __init__(self, pool_gpus: int, io_capacity: float = 1e12, inf_capacity: float = 1e12,
                 prepaid_leases: bool = True, preempt_grace: int = PREEMPT_GRACE):
        # preempt_grace: the contractual checkpoint window opportunistic leases
        # get before reclamation — and therefore the price of the reclaim path.
        self.preempt_grace = preempt_grace
        # prepaid_leases: a lease's above-sustained-rate portion must be covered
        # by tokens for its whole duration at grant time. Without this, enforcement
        # happens only at grant instants, and each sliver of refilled tokens buys a
        # full lease-duration burst at the ceiling — a leak the parameter sweeps
        # exposed (extraction exceeded B + R x window on an idle pool).
        self.prepaid_leases = prepaid_leases
        self.pool = pool_gpus
        self.io_capacity = io_capacity      # facility filesystem bandwidth, GB/s
        self.inf_capacity = inf_capacity    # facility inference service, tokens/s
        self.now = 0
        self.envelopes: dict[str, Envelope] = {}
        self.tokens: dict[str, float] = {}
        self.credits: dict[str, float] = {}
        self.leases: list[Lease] = []
        self.pending: list[Request] = []
        self.journal: list[dict] = []
        self.grant_log: list[dict] = []
        self.util_hist: list[float] = []
        self._demand_hist = deque(maxlen=CONGESTION_WINDOW)
        self._usage_hist: dict[str, deque] = {}
        self._next_id = 1
        # per-dimension state: token buckets, this-tick facility headroom,
        # this-tick per-campaign usage, last-tick usage (for sampling/stats)
        self.dim_tokens: dict[str, dict[str, float]] = {"io": {}, "inference": {}}
        self._dim_left = {"io": io_capacity, "inference": inf_capacity}
        self._dim_used: dict[str, dict[str, float]] = {"io": {}, "inference": {}}
        self.dim_last: dict[str, dict[str, float]] = {"io": {}, "inference": {}}
        self.dim_throttled: dict[str, dict[str, int]] = {"io": {}, "inference": {}}
        self._dim_log_at: dict[tuple, int] = {}

    # ------------------------------------------------------------------
    # registration (allocation-committee role)
    # ------------------------------------------------------------------
    def register(self, env: Envelope):
        total_guaranteed = sum(e.guaranteed for e in self.envelopes.values()) + env.guaranteed
        if total_guaranteed > self.pool:
            raise ValueError("sum of guarantees exceeds pool: guarantees are never overbooked (Sec. 10)")
        self.envelopes[env.campaign] = env
        self.tokens[env.campaign] = env.bucket_cap   # buckets start full
        self.credits[env.campaign] = env.budget_credits
        self._usage_hist[env.campaign] = deque(maxlen=FAIRSHARE_WINDOW)
        for d in ("io", "inference"):
            self.dim_tokens[d][env.campaign] = getattr(env, d).bucket
            self._dim_used[d][env.campaign] = 0.0
            self.dim_last[d][env.campaign] = 0.0
            self.dim_throttled[d][env.campaign] = 0
        self._log("register", campaign=env.campaign, guaranteed=env.guaranteed,
                  R=env.sustained, B=env.bucket_cap, burst=env.burst_ceiling)

    # ------------------------------------------------------------------
    # supply API (Sec. 6)
    # ------------------------------------------------------------------
    def supply(self, campaign: str) -> dict:
        env = self.envelopes[campaign]
        free = self.free_gpus()
        reclaimable = sum(l.gpus for l in self.leases
                          if l.cls == "opportunistic" and l.preempt_at < 0)
        idx = self.congestion_index()
        return {
            "free_gpus": free,
            "reclaimable_gpus": reclaimable,
            "congestion_index": round(idx, 3),
            "congestion_multiplier": congestion_multiplier(idx),
            "tokens": self.tokens[campaign],
            "bucket_cap": env.bucket_cap,
            "replenish_rate": env.sustained,
            "credits_remaining": self.credits[campaign],
            "expected_path": "warm" if free > 0 else ("reclaim" if reclaimable > 0 else "cold"),
            "io": {"tokens": self.dim_tokens["io"][campaign], "bucket": env.io.bucket,
                   "rate": env.io.rate, "ceiling": env.io.ceiling,
                   "facility_left": round(self._dim_left["io"], 2)},
            "inference": {"tokens": self.dim_tokens["inference"][campaign],
                          "bucket": env.inference.bucket, "rate": env.inference.rate,
                          "ceiling": env.inference.ceiling,
                          "facility_left": round(self._dim_left["inference"], 1)},
        }

    # ------------------------------------------------------------------
    # consumption API (Sec. 4): flow dimensions beyond GPUs.
    # Throttle-at-source: returns what the campaign may consume this second;
    # the runtime is expected to obey (violations would be a policy event).
    # ------------------------------------------------------------------
    def consume(self, campaign: str, io: float = 0.0, inference: float = 0.0) -> dict:
        allowed = {}
        for dim, desired in (("io", io), ("inference", inference)):
            if desired <= 0:
                allowed[dim] = 0.0
                continue
            d: Dim = getattr(self.envelopes[campaign], dim)
            cap = d.ceiling if self.dim_tokens[dim][campaign] > 0 else d.rate
            room = max(0.0, cap - self._dim_used[dim][campaign])
            grant = min(desired, room, self._dim_left[dim])
            self._dim_left[dim] -= grant
            self._dim_used[dim][campaign] += grant
            allowed[dim] = grant
            if grant < desired - 1e-9:
                self.dim_throttled[dim][campaign] += 1
                key = (dim, campaign)
                if self.now - self._dim_log_at.get(key, -999) >= 60:   # rate-limit journal spam
                    self._dim_log_at[key] = self.now
                    self._log(f"throttle-{dim}", campaign=campaign,
                              desired=round(desired, 2), allowed=round(grant, 2),
                              tokens=round(self.dim_tokens[dim][campaign], 1))
        if allowed.get("inference"):
            self.credits[campaign] -= allowed["inference"] * INFERENCE_CREDITS_PER_TOKEN
        return allowed

    # ------------------------------------------------------------------
    # intent API (Sec. 6): request / release leases
    # ------------------------------------------------------------------
    def request_lease(self, campaign: str, gpus: int, duration: int, cls: str | None = None) -> int:
        env = self.envelopes[campaign]
        cls = cls or env.service_class
        req = Request(self._next_id, campaign, gpus, duration, cls, self.now)
        self._next_id += 1
        # exhausted budget confines the campaign to opportunistic (Sec. 4 rule 2)
        if self.credits[campaign] <= 0 and cls != "opportunistic":
            self._log("deny", campaign=campaign, gpus=gpus, cls=cls, reason="budget-exhausted")
            return -1
        self.pending.append(req)
        return req.id

    def release_all(self, campaign: str, cls: str | None = None):
        for l in [l for l in self.leases
                  if l.campaign == campaign and (cls is None or l.cls == cls)]:
            self._release(l, reason="released-by-agent")
        self.pending = [r for r in self.pending
                        if not (r.campaign == campaign and (cls is None or r.cls == cls))]

    # ------------------------------------------------------------------
    # bookkeeping queries
    # ------------------------------------------------------------------
    def held(self, campaign: str, cls: str | None = None) -> int:
        return sum(l.gpus for l in self.leases
                   if l.campaign == campaign and (cls is None or l.cls == cls))

    def outstanding(self, campaign: str, cls: str | None = None) -> int:
        """Held plus pending: what the campaign has asked for in total."""
        pend = sum(r.gpus for r in self.pending
                   if r.campaign == campaign and (cls is None or r.cls == cls))
        return self.held(campaign, cls) + pend

    def free_gpus(self) -> int:
        return self.pool - sum(l.gpus for l in self.leases)

    def spent(self, campaign: str) -> float:
        return self.envelopes[campaign].budget_credits - self.credits[campaign]

    def congestion_index(self) -> float:
        if not self._demand_hist:
            return 0.0
        demand = sum(d for d, _ in self._demand_hist)
        avail = sum(a for _, a in self._demand_hist)
        return min(10.0, demand / max(avail, 1))

    # ------------------------------------------------------------------
    # the clock
    # ------------------------------------------------------------------
    def tick(self):
        self.now += 1
        self._expire_leases()
        self._update_buckets()
        self._account()
        self._admit()
        # record unmet priced-class demand vs free capacity for the congestion index
        # (opportunistic filler demand is not scarcity and must not inflate the signal)
        self._demand_hist.append((sum(r.gpus for r in self.pending if r.cls != "opportunistic"),
                                  max(self.free_gpus(), 0)))
        for c in self.envelopes:
            self._usage_hist[c].append(self.held(c))
        self.util_hist.append(1.0 - self.free_gpus() / self.pool)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _expire_leases(self):
        for l in list(self.leases):
            if l.expires <= self.now:
                self._release(l, reason="expired")
            elif 0 <= l.preempt_at <= self.now:
                self._release(l, reason="preempted")

    def _release(self, lease: Lease, reason: str):
        self.leases.remove(lease)
        self._log("release", lease=lease.id, campaign=lease.campaign,
                  gpus=lease.gpus, reason=reason)

    def _update_buckets(self):
        """Tokens replenish at R and drain at *held* capacity (Sec. 4):
        holding above R is a burst that empties the bucket; holding below R refills it.
        Idle holding is inherently costly. Flow dimensions (io, inference) follow
        the same rule with consumption in place of holding."""
        for c, env in self.envelopes.items():
            self.tokens[c] = max(0.0, min(env.bucket_cap,
                                          self.tokens[c] + env.sustained - self.held(c)))
            for dim in ("io", "inference"):
                d: Dim = getattr(env, dim)
                used = self._dim_used[dim][c]
                self.dim_tokens[dim][c] = max(0.0, min(d.bucket,
                                                       self.dim_tokens[dim][c] + d.rate - used))
                self.dim_last[dim][c] = used
                self._dim_used[dim][c] = 0.0
        self._dim_left = {"io": self.io_capacity, "inference": self.inf_capacity}

    def _account(self):
        """Credits drain against held capacity at class rate x congestion multiplier."""
        cong = congestion_multiplier(self.congestion_index())
        for l in self.leases:
            mult = BASE_MULT[l.cls] * (cong if l.cls in ("agent-burst", "interactive") else 1.0)
            self.credits[l.campaign] -= l.gpus * mult / 3600.0

    def _rate_cap(self, campaign: str) -> int:
        """Empty bucket confines the campaign to R; the guarantee is honored
        regardless (Sec. 4 rules 3-4)."""
        env = self.envelopes[campaign]
        cap = env.burst_ceiling if self.tokens[campaign] > 0 else env.sustained
        return max(cap, env.guaranteed)

    def _fair_deficit(self, campaign: str) -> float:
        """Entitlement share minus realized usage share over the trailing window."""
        total_g = sum(e.guaranteed for e in self.envelopes.values()) or 1
        entitlement = self.envelopes[campaign].guaranteed / total_g
        my_use = sum(self._usage_hist[campaign]) or 0
        all_use = sum(sum(h) for h in self._usage_hist.values()) or 1
        return entitlement - my_use / all_use

    def _grantable(self, campaign: str, gpus: int) -> int:
        """Capacity available to this campaign right now, honoring guarantees:
        consumption beyond a campaign's own guarantee must leave enough free
        capacity to back every other campaign's unmet guarantee (Sec. 4 rule 4,
        Sec. 10 'guarantees are never overbooked')."""
        env = self.envelopes[campaign]
        free = self.free_gpus()
        reserve = sum(max(0, e.guaranteed - self.held(c))
                      for c, e in self.envelopes.items() if c != campaign)
        guar_part = min(gpus, max(0, env.guaranteed - self.held(campaign)), free)
        extra = max(0, free - reserve - guar_part)
        return guar_part + min(gpus - guar_part, extra)

    def _admit(self):
        """Deterministic, published arbitration (Sec. 10): class first, then
        weighted-fair-share deficit, then token wealth. Every grant is journaled."""
        self.pending.sort(key=lambda r: (CLASS_RANK[r.cls],
                                         -self._fair_deficit(r.campaign),
                                         -self.tokens[r.campaign]))
        for req in list(self.pending):
            if self.now - req.submitted > REQUEST_TTL:
                self.pending.remove(req)
                self._log("request-expired", campaign=req.campaign, gpus=req.gpus, cls=req.cls)
                continue
            # token-bucket rate cap: trim, or wait if fully throttled
            env = self.envelopes[req.campaign]
            if self.prepaid_leases:
                # above-R capacity must be affordable for the full lease duration
                afford = env.sustained + self.tokens[req.campaign] / max(req.duration, 1)
                cap = max(min(env.burst_ceiling, int(afford)), env.guaranteed)
            else:
                cap = self._rate_cap(req.campaign)
            headroom = cap - self.held(req.campaign)
            if headroom <= 0:
                if not req.throttle_logged:
                    self._log("throttle", campaign=req.campaign, gpus=req.gpus,
                              cls=req.cls, tokens=round(self.tokens[req.campaign], 1),
                              cap=cap)
                    req.throttle_logged = True
                continue
            want = min(req.gpus, headroom)
            grantable = self._grantable(req.campaign, want)

            if grantable >= want:
                self._grant(req, want)
            elif grantable >= min(want, 16):
                self._grant(req, grantable)      # partial grant; agent re-requests the rest
            elif req.cls != "opportunistic":
                self._preempt_for(want - grantable)   # reclaim path: wait for grace

    def _preempt_for(self, needed: int):
        in_flight = sum(l.gpus for l in self.leases if l.preempt_at >= 0)
        for l in sorted((l for l in self.leases
                         if l.cls == "opportunistic" and l.preempt_at < 0),
                        key=lambda l: -l.gpus):
            if in_flight >= needed:
                break
            l.preempt_at = self.now + self.preempt_grace
            in_flight += l.gpus
            self._log("preempt-notice", lease=l.id, campaign=l.campaign,
                      gpus=l.gpus, grace=self.preempt_grace)

    def _grant(self, req: Request, gpus: int):
        self.pending.remove(req)
        latency = self.now - req.submitted
        path = "warm" if latency <= 1 else ("reclaim" if latency <= self.preempt_grace + 10 else "cold")
        lease = Lease(req.id, req.campaign, gpus, req.cls, req.submitted,
                      self.now, self.now + req.duration)
        self.leases.append(lease)
        self.grant_log.append({"t": self.now, "campaign": req.campaign, "cls": req.cls,
                               "gpus": gpus, "latency": latency, "path": path})
        self._log("grant", lease=req.id, campaign=req.campaign, gpus=gpus, cls=req.cls,
                  latency=latency, path=path,
                  rule={"class_rank": CLASS_RANK[req.cls],
                        "fair_deficit": round(self._fair_deficit(req.campaign), 4),
                        "tokens": round(self.tokens[req.campaign], 1)})

    def _log(self, event: str, **kw):
        self.journal.append({"t": self.now, "event": event, **kw})

    def dump_journal(self, path: str):
        with open(path, "w") as f:
            for rec in self.journal:
                f.write(json.dumps(rec) + "\n")
