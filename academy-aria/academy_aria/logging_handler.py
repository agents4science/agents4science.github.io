"""Fallback emitter for Academy versions without the middleware hook.

Academy's runtime already logs structured extras on every action
(academy.action, academy.action_tag, academy.action_state). This handler
converts those records into the same tool.call events the middleware emits.
Latency is reconstructed by pairing execute_start with the terminal state
for the same tag, so numbers are approximate (log-time, not dispatch-time).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .events import tool_call_event
from .sinks import EventSink

_TERMINAL = {
    'execute_success': 'succeeded',
    'execute_exception': 'failed',
    'execute_cancelled': 'failed',
}


class AriaLoggingHandler(logging.Handler):
    def __init__(self, sink: EventSink, run_id: str, *, source: str,
                 token_cost: int = 0) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink
        self.run_id = run_id
        self.source = source
        self.token_cost = token_cost
        self._starts: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        state = getattr(record, 'academy.action_state', None) or getattr(
            record, 'academy_action_state', None)
        action = getattr(record, 'academy.action', None) or getattr(
            record, 'academy_action', None)
        tag = getattr(record, 'academy.action_tag', None) or getattr(
            record, 'academy_action_tag', None)
        if not state or not action:
            return
        correlation = str(tag) if tag is not None else self.run_id
        if state == 'execute_start':
            self._starts[correlation] = time.monotonic()
            self.sink.append(tool_call_event(
                run_id=self.run_id, tool_name=str(action), status='started',
                latency_ms=0, token_cost=0, source=self.source,
                correlation_id=correlation))
        elif state in _TERMINAL:
            started = self._starts.pop(correlation, None)
            latency = int((time.monotonic() - started) * 1000) if started else 0
            self.sink.append(tool_call_event(
                run_id=self.run_id, tool_name=str(action),
                status=_TERMINAL[state], latency_ms=latency,
                token_cost=self.token_cost, source=self.source,
                correlation_id=correlation))


def make_record_extra(action: str, tag: Any, state: str) -> dict[str, Any]:
    """Build the extras dict Academy attaches to action log records
    (exposed for tests)."""
    return {
        'academy.action': action,
        'academy.action_tag': tag,
        'academy.action_state': state,
    }
