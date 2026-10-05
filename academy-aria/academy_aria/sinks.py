"""Event sinks: where emitted GmpEvents go.

CollectingSink and JsonlSink are the local/test paths; ValidatingSink wraps
any sink and rejects events that do not validate against the canonical
contracts; HttpSink posts to an ARIA control plane's appendEvent operation.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .validate import SchemaStore, validate_event


class EventSink(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...


class CollectingSink:
    """Hold events in memory (tests, demos)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def __getstate__(self):  # picklable across executor boundaries
        return {'events': list(self.events)}

    def __setstate__(self, state):
        self.events = state['events']
        self._lock = threading.Lock()


class JsonlSink:
    """Append events to a JSONL file; the SAVE-EARLY idiom, standardized."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, sort_keys=True)
        with self._lock, self.path.open('a') as f:
            f.write(line + '\n')

    def __getstate__(self):
        return {'path': str(self.path)}

    def __setstate__(self, state):
        self.path = Path(state['path'])
        self._lock = threading.Lock()


class InvalidEventError(ValueError):
    pass


class ValidatingSink:
    """Validate every event against the contracts before forwarding.

    This is the conformance guarantee: an adapter bug produces a loud local
    failure, never a malformed record in the control plane.
    """

    def __init__(self, inner: EventSink, contracts_dir: str | Path | None = None) -> None:
        self.inner = inner
        self._contracts_dir = str(contracts_dir) if contracts_dir else None
        self._store: SchemaStore | None = None

    def _ensure_store(self) -> SchemaStore:
        if self._store is None:
            self._store = SchemaStore(self._contracts_dir)
        return self._store

    def append(self, event: dict[str, Any]) -> None:
        errors = validate_event(event, self._ensure_store())
        if errors:
            raise InvalidEventError(
                f'event {event.get("eventId")} fails contract validation: {errors}')
        self.inner.append(event)

    def __getstate__(self):  # SchemaStore holds Paths; rebuild lazily
        return {'inner': self.inner, '_contracts_dir': self._contracts_dir}

    def __setstate__(self, state):
        self.inner = state['inner']
        self._contracts_dir = state['_contracts_dir']
        self._store = None


class HttpSink:
    """POST events to an ARIA appendEvent endpoint.

    The append DTO (per the RFC 005 events-append fixture) is derived from
    the GmpEvent: payload and eventType carry over; agentId is the envelope
    source; runId is lifted from the payload when present.
    """

    def __init__(self, base_url: str, *, token: str | None = None,
                 timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.timeout = timeout

    def append(self, event: dict[str, Any]) -> None:
        dto = {
            'eventType': event['eventType'],
            'payload': event['payload'],
            'agentId': event['source'],
            'metadata': {'eventId': event['eventId'], 'occurredAt': event['occurredAt']},
        }
        run_id = event.get('payload', {}).get('runId')
        if run_id:
            dto['runId'] = run_id
        if 'correlationId' in event:
            dto['correlationId'] = event['correlationId']
        req = urllib.request.Request(
            f'{self.base_url}/events',
            data=json.dumps(dto).encode(),
            headers={'Content-Type': 'application/json'}
            | ({'Authorization': f'Bearer {self.token}'} if self.token else {}),
            method='POST',
        )
        urllib.request.urlopen(req, timeout=self.timeout).read()
