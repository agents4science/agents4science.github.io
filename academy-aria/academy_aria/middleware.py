"""Academy action middleware that emits ARIA tool.call events.

Plugs into academy.runtime.RuntimeConfig(action_middleware=(...,)) — the
extension point proposed in academy-agents/academy#461. Each action
invocation produces a started event and a terminal succeeded/failed event;
the invocation tag becomes the envelope correlationId so journal queries
reconstruct per-invocation causality (RFC 018 convention).

Falls back cleanly where #461 is absent: the module imports no Academy
symbols at runtime, so it also serves as the template for the interim
Runtime-subclass path described in the gap analysis.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from .events import journal_run_event, tool_call_event
from .sinks import EventSink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from academy.runtime import ActionCall


class AriaEventEmitter:
    """ActionMiddleware emitting tool.call events for every invocation.

    Args:
        sink: Where events go. Wrap in ValidatingSink for conformance.
        run_id: The ARIA run this actor's work is attributed to. (The run
            wrapper component will supply this; until then the launcher
            chooses it.)
        token_cost: Fixed per-invocation token cost; 0 for non-LLM actors.
    """

    def __init__(self, sink: EventSink, run_id: str, *, token_cost: int = 0) -> None:
        self.sink = sink
        self.run_id = run_id
        self.token_cost = token_cost

    def _correlation(self, call: 'ActionCall') -> str:
        return str(call.tag) if call.tag is not None else self.run_id

    async def __call__(
        self,
        call: 'ActionCall',
        next_handler: Callable[['ActionCall'], Awaitable[Any]],
    ) -> Any:
        source = str(call.agent_id)
        correlation = self._correlation(call)
        self.sink.append(tool_call_event(
            run_id=self.run_id, tool_name=call.action, status='started',
            latency_ms=0, token_cost=0, source=source, correlation_id=correlation))
        start = time.monotonic()
        try:
            result = await next_handler(call)
        except BaseException:
            self.sink.append(tool_call_event(
                run_id=self.run_id, tool_name=call.action, status='failed',
                latency_ms=int((time.monotonic() - start) * 1000),
                token_cost=self.token_cost, source=source, correlation_id=correlation))
            raise
        self.sink.append(tool_call_event(
            run_id=self.run_id, tool_name=call.action, status='succeeded',
            latency_ms=int((time.monotonic() - start) * 1000),
            token_cost=self.token_cost, source=source, correlation_id=correlation))
        return result

    # Run-lifecycle provenance helpers; the future run wrapper owns these
    # calls, but the demo and tests use them to produce a complete stream.
    def emit_run_started(self, source: str) -> None:
        self.sink.append(journal_run_event(
            'journal.run.started', run_id=self.run_id, source=source,
            correlation_id=self.run_id))

    def emit_run_completed(self, source: str, *, failure_class: str = 'F0_NONE',
                           details: dict | None = None) -> None:
        event_type = ('journal.run.completed' if failure_class == 'F0_NONE'
                      else 'journal.run.failed')
        self.sink.append(journal_run_event(
            event_type, run_id=self.run_id, source=source,
            failure_class=failure_class, correlation_id=self.run_id,
            details=details))
