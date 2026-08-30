"""Synthetic campaign agents for the broker simulation (paper Sec. 12, Phase 1).

Two science campaigns get *identical* seeded task sets so the naive/adaptive
comparison is fair; a storm campaign attacks the pool; a victim campaign
measures isolation; a backfill campaign recovers utilization opportunistically.
"""

import random


class Task:
    __slots__ = ("value", "size", "remaining", "density", "io")

    def __init__(self, value: float, size: float, io: float):
        self.value = value                       # scientific value (arbitrary units)
        self.size = size                         # GPU-seconds of work
        self.remaining = size
        self.density = value / (size / 3600.0)   # value per GPU-hour
        self.io = io                             # GB of filesystem I/O per GPU-second


def make_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [Task(rng.lognormvariate(0.0, 1.0), rng.uniform(200, 2000),
                 rng.uniform(0.01, 0.09)) for _ in range(n)]


class ScienceAgent:
    """Shared machinery: hold leases, burn through work, accumulate science value."""

    def __init__(self, name, broker, tasks):
        self.name = name
        self.broker = broker
        self.todo = tasks
        self.science = 0.0
        self.completed = 0

    def process_work(self, order=None):
        """Burn GPU-seconds through tasks; throughput is coupled to the I/O the
        broker allows this second (throttle-at-source, paper Sec. 4). `order`
        overrides the processing order (used by the adaptive agent when
        I/O-starved)."""
        b = self.broker
        held = b.held(self.name)
        if held == 0 or not self.todo:
            return
        order = self.todo if order is None else order
        front = order[:20]
        mean_io = sum(t.io for t in front) / len(front)
        desired_io = held * mean_io
        allowed_io = b.consume(self.name, io=desired_io)["io"]
        factor = 1.0 if desired_io <= 0 else allowed_io / desired_io
        budget = held * factor                        # GPU-seconds we can actually feed
        for t in list(order):
            if budget <= 0:
                break
            take = min(budget, t.remaining)
            t.remaining -= take
            budget -= take
            if t.remaining <= 0:
                self.science += t.value
                self.completed += 1
                self.todo.remove(t)


class NaiveAgent(ScienceAgent):
    """Requests a big fixed slice in agent-burst class and runs tasks FIFO,
    ignoring price, congestion, and its own token state."""

    TARGET = 256

    def step(self, now):
        self.process_work()
        b = self.broker
        if not self.todo:
            b.release_all(self.name)
            return
        out = b.outstanding(self.name, "agent-burst")
        if out < self.TARGET:
            b.request_lease(self.name, self.TARGET - out, 600, "agent-burst")


class AdaptiveAgent(ScienceAgent):
    """Reads the supply API: runs only work whose value-per-GPU-hour clears the
    congestion-adjusted price, conserves its token bucket when it runs low, and
    mops up deferred low-value tasks with cheap opportunistic capacity when the
    pool is quiet (paper Secs. 5-6)."""

    def __init__(self, name, broker, tasks):
        super().__init__(name, broker, sorted(tasks, key=lambda t: -t.density))

    def step(self, now):
        b = self.broker
        s = b.supply(self.name)
        # planning costs inference: a base rate plus a share proportional to
        # activity. If Bursar throttles our thinking below half, keep the last
        # plan (degraded step: work continues, replanning pauses).
        inf_desired = 20 + 0.5 * b.held(self.name)
        inf_allowed = b.consume(self.name, inference=inf_desired)["inference"]
        # when I/O-starved, run the science that is cheapest to feed:
        # order the workfront by value-per-GB instead of value-per-GPU-hour
        io_starved = s["io"]["tokens"] < 0.2 * s["io"]["bucket"]
        order = None
        if io_starved and self.todo:
            order = sorted(self.todo[:200],
                           key=lambda t: -t.value / (t.io * t.size + 1e-9))
        self.process_work(order)
        if not self.todo:
            b.release_all(self.name)
            return
        if inf_allowed < 0.5 * inf_desired:
            return
        price = 3.0 * s["congestion_multiplier"]      # agent-burst credits per GPU-hour
        hot_work = 0.0
        for t in self.todo:                           # todo is sorted by density desc
            if t.density < price:
                break
            hot_work += t.remaining
        if hot_work > 0:
            want = max(64, min(int(hot_work / 600), 384))
            if s["tokens"] < 0.2 * s["bucket_cap"]:   # conserve burst capacity
                want = min(want, int(s["replenish_rate"]))
            if io_starved:
                # don't hold GPUs we cannot feed: cap at what sustained I/O supports
                front = (order or self.todo)[:20]
                mean_io = sum(t.io for t in front) / len(front)
                want = min(want, max(32, int(s["io"]["rate"] / mean_io)))
            out = b.outstanding(self.name, "agent-burst")
            if out < want:
                b.request_lease(self.name, want - out, 600, "agent-burst")
        else:
            b.release_all(self.name, "agent-burst")   # nothing worth today's price
        # opportunistic mop-up of deferred work whenever idle capacity exists:
        # opportunistic is flat-priced (0.25x, never congestion-multiplied) and
        # preemptible, so it is worthwhile at any congestion level
        if s["free_gpus"] > 96:
            out = b.outstanding(self.name, "opportunistic")
            want = min(96, s["free_gpus"] - 64)
            if out < want:
                b.request_lease(self.name, want - out, 300, "opportunistic")


class StormAgent:
    """Adversarial: hammers burst-ceiling requests throughout its window.
    Exists to demonstrate token-bucket throttling and cross-campaign isolation."""

    def __init__(self, name, broker, window=(3600, 5400)):
        self.name = name
        self.broker = broker
        self.window = window

    def step(self, now):
        lo, hi = self.window
        if lo <= now < hi:
            # the storm's agent also hammers the inference service; its
            # inference envelope throttles that too (Sec. 4)
            self.broker.consume(self.name, inference=200)
            if now % 20 == 0:
                out = self.broker.outstanding(self.name)
                if out < 384:
                    self.broker.request_lease(self.name, 384 - out, 120, "agent-burst")
        elif now == hi:
            self.broker.release_all(self.name)


class VictimAgent:
    """Small periodic interactive requests; its grant latency inside vs outside
    the storm window is the isolation measurement (paper Sec. 13)."""

    def __init__(self, name, broker):
        self.name = name
        self.broker = broker

    def step(self, now):
        if now % 120 == 0 and self.broker.outstanding(self.name) == 0:
            self.broker.request_lease(self.name, 16, 60, "interactive")


class ProbeAgent:
    """Measurement instrument, not a workload: every 5 minutes it acquires a
    full 128 GPUs (topping up through partial grants) and records the time to
    full capacity, so configurations and seeds compare apples-to-apples."""

    GIVE_UP = 590   # censor an observation at this latency

    def __init__(self, name, broker, gpus=128, period=300):
        self.name = name
        self.broker = broker
        self.gpus = gpus
        self.period = period
        self.t0 = None
        self.lats: list[tuple] = []    # (time-to-full-capacity, completion tick)

    def step(self, now):
        b = self.broker
        if self.t0 is None:
            if now % self.period == 0:
                self.t0 = now
                b.request_lease(self.name, self.gpus, 180, "agent-burst")
            return
        if b.held(self.name) >= self.gpus or now - self.t0 > self.GIVE_UP:
            self.lats.append((now - self.t0, now))
            b.release_all(self.name)   # done measuring; give it back
            self.t0 = None
            return
        out = b.outstanding(self.name)
        if out < self.gpus:            # top up through partial grants / expiries
            b.request_lease(self.name, self.gpus - out, 180, "agent-burst")


class BackfillAgent:
    """Facility-side opportunistic filler: recovers utilization from idle pool
    capacity while staying fully preemptible (paper Sec. 7). Leaves headroom so
    the warm path stays warm."""

    def __init__(self, name, broker, headroom=160):
        self.name = name
        self.broker = broker
        self.headroom = headroom

    def step(self, now):
        b = self.broker
        extra = b.free_gpus() - self.headroom
        pending = b.outstanding(self.name) - b.held(self.name)
        if extra > 16 and pending <= 0:
            b.request_lease(self.name, extra, 300, "opportunistic")
