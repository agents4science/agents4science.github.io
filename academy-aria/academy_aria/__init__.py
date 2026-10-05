"""academy-aria: ARIA conformance adapter for Academy.

First component: event emission. An Academy actor's action lifecycle is
emitted as ARIA tool.call / journal.* events that validate against the
canonical contracts.
"""

from .events import journal_run_event, tool_call_event
from .logging_handler import AriaLoggingHandler
from .middleware import AriaEventEmitter
from .sinks import (
    CollectingSink,
    EventSink,
    HttpSink,
    InvalidEventError,
    JsonlSink,
    ValidatingSink,
)
from .validate import SchemaStore, validate_event

__all__ = [
    'AriaEventEmitter',
    'AriaLoggingHandler',
    'CollectingSink',
    'EventSink',
    'HttpSink',
    'InvalidEventError',
    'JsonlSink',
    'SchemaStore',
    'ValidatingSink',
    'journal_run_event',
    'tool_call_event',
    'validate_event',
]
