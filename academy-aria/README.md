# academy-aria

ARIA conformance adapter for [Academy](https://github.com/academy-agents/academy).
This is the first component — **event emission** — of the adapter sketched in the
ARIA v3 ↔ Academy gap analysis §7: an Academy actor's action lifecycle becomes an
ARIA event stream that validates against the canonical contracts.

## What it does

| Academy side | ARIA side |
|---|---|
| Action dispatch (via the [#461](https://github.com/academy-agents/academy/pull/461) `action_middleware` hook) | `tool.call` events: one `started`, one `succeeded`/`failed` per invocation, with measured `latencyMs` |
| Invocation tag (`ActionCall.tag`) | Envelope `correlationId` (RFC 018 convention) |
| Actor lifetime (run wrapper, future component) | `journal.run.started` / `journal.run.completed` / `journal.run.failed` |
| Structured logging extras (`academy.action_state`) | Same `tool.call` events, via `AriaLoggingHandler`, for Academy versions without the middleware hook |

Every sink can be wrapped in `ValidatingSink`, which checks each event against the
contract schemas (`event-envelope.schema.json` and its closed `oneOf`) before
forwarding — an adapter bug fails loudly and locally, never as a malformed record
in a control plane.

## Use

```python
from academy.runtime import Runtime, RuntimeConfig
from academy_aria import AriaEventEmitter, JsonlSink, ValidatingSink

emitter = AriaEventEmitter(ValidatingSink(JsonlSink('events.jsonl')), run_id='run_...')
config = RuntimeConfig(action_middleware=(emitter,))
# launch the actor with this config; its actions now emit ARIA events
```

Sinks: `CollectingSink` (memory), `JsonlSink` (file), `HttpSink` (POSTs the
`appendEvent` DTO to a control plane), `ValidatingSink` (wraps any of them).

## Contracts

Schema validation needs the ARIA contracts: set `ARIA_CONTRACTS=/path/to/ARIA/contracts`
or keep an `aria-spec` checkout next to this repo. Validation is a ~150-line stdlib
mini-validator covering the subset of JSON Schema the contracts use; the ARIA repo's
own validator remains the authority.

## Tests and demo

```
pytest tests/          # includes live tests against a real Academy Runtime
python demo.py         # actor -> validated event stream -> terminal-state replay
```

## Scope and next components

In scope here: event emission only. The remaining adapter components from the gap
analysis §7, in intended order: session/token client (Globus Auth ↔ ARIA capability
tokens), capability publisher (`agent_describe()` → `registerCapability`), run wrapper
(`submitRun` → `Manager.launch`, idempotency, failure-class mapping, ExecutionAttempt
records), action interceptor (token scope / policy / budget via the same middleware
chain), and the JSON-only federation serialization profile.
