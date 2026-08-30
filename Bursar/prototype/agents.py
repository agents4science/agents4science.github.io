"""Synthetic campaign agents for the broker simulation (paper Sec. 12, Phase 1).

Two science campaigns get *identical* seeded task sets so the naive/adaptive
comparison is fair; a storm campaign attacks the pool; a victim campaign
measures isolation; a backfill campaign recovers utilization opportunistically.
"""

import random


class Task:
    __slots__ = ("value", "size", "remaining", "density")

    def __init__(self, value: float, size: float):
        self.value = value                       # scientific value (arbitrary units)
        self.size = size                         # GPU-seconds of work
        self.remaining = size
        self.density = value / (size / 3600.0)   # value per GPU-hour


def make_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    return [Task(rng.lognormvariate(0.0, 1.0), rng.uniform(200, 2000)) for _ in range(n)]


class ScienceAgent:
    """Shared machinery: hold leases, burn through work, accumulate science value."""

    def __init__(self, name, broker, tasks):
        self.name = name
        self.broker = broker
        self.todo = tasks
        self.science = 0.0
        self.completed = 0

    def process_work(self):
        budget = float(self.broker.held(self.name))   # GPU-seconds available this tick
        while budget > 0 and self.todo:
            t = self.todo[0]
            take = min(budget, t.remaining)
            t.remaining -= take
            budget -= take
            if t.remaining <= 0:
                self.science += t.value
                self.completed += 1
                self.todo.pop(0)


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
        self.process_work()
        b = self.broker
        if not self.todo:
            b.release_all(self.name)
            return
        s = b.supply(self.name)
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


class BackfillAgent:
    """Facility-side opportunistic filler: recovers utilization from idle pool
    capacity while staying fully preemptible (paper Sec. 7). Leaves headroom so
    the warm path stays warm."""

    HEADROOM = 160

    def __init__(self, name, broker):
        self.name = name
        self.broker = broker

    def step(self, now):
        b = self.broker
        extra = b.free_gpus() - self.HEADROOM
        pending = b.outstanding(self.name) - b.held(self.name)
        if extra > 16 and pending <= 0:
            b.request_lease(self.name, extra, 300, "opportunistic")
