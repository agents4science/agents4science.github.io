"""Constructors for ARIA GmpEvent documents emitted by the adapter.

Shapes follow the canonical contracts in the ARIA repo
(schemas/events/tool-call-event.schema.json and
journal-envelope-event.schema.json). The tool.call payload is closed
(additionalProperties: false), so invocation identity rides the envelope
correlationId, per RFC 018's convention.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

JOURNAL_EVENT_TYPES = (
    'journal.run.submitted',
    'journal.run.started',
    'journal.run.target_failed',
    'journal.run.retry',
    'journal.run.completed',
    'journal.run.failed',
    'journal.policy.checked',
    'journal.artifact.validated',
)

TOOL_CALL_STATUSES = ('started', 'succeeded', 'failed')


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _event_id() -> str:
    return f'evt_{uuid.uuid4().hex[:20]}'


def tool_call_event(
    *,
    run_id: str,
    tool_name: str,
    status: str,
    latency_ms: int,
    source: str,
    correlation_id: str,
    token_cost: int = 0,
) -> dict[str, Any]:
    """One tool.call event; for Academy actors a tool call is an action
    invocation and tokenCost is 0 unless the action itself reports LLM use.
    """
    if status not in TOOL_CALL_STATUSES:
        raise ValueError(f'status must be one of {TOOL_CALL_STATUSES}, got {status!r}')
    return {
        'eventId': _event_id(),
        'eventType': 'tool.call',
        'occurredAt': _now(),
        'source': source,
        'correlationId': correlation_id,
        'payload': {
            'runId': run_id,
            'toolName': tool_name,
            'status': status,
            'latencyMs': int(latency_ms),
            'tokenCost': int(token_cost),
        },
    }


def journal_run_event(
    event_type: str,
    *,
    run_id: str,
    source: str,
    failure_class: str = 'F0_NONE',
    correlation_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One journal.* event (run lifecycle provenance)."""
    if event_type not in JOURNAL_EVENT_TYPES:
        raise ValueError(f'unknown journal event type {event_type!r}')
    payload: dict[str, Any] = {'runId': run_id, 'failureClass': failure_class}
    if details is not None:
        payload['details'] = details
    event: dict[str, Any] = {
        'eventId': _event_id(),
        'eventType': event_type,
        'occurredAt': _now(),
        'source': source,
        'payload': payload,
    }
    if correlation_id is not None:
        event['correlationId'] = correlation_id
    return event
