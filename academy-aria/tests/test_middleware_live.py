"""Live test: a real Academy Runtime with the emitter installed as action
middleware (academy-agents/academy#461). Every emitted event passes contract
validation or the sink raises and the test fails.
"""

import uuid

import pytest
from academy.exchange import LocalExchangeTransport, UserExchangeClient
from academy.runtime import Runtime, RuntimeConfig

from academy_aria import AriaEventEmitter, CollectingSink, ValidatingSink

from conftest import CounterAgent, ErrorAgent

RUN_ID = 'run-emitter-test'


@pytest.mark.asyncio
async def test_action_lifecycle_emits_validated_events(
    exchange_client: UserExchangeClient[LocalExchangeTransport],
) -> None:
    registration = await exchange_client.register_agent(CounterAgent)
    collected = CollectingSink()
    emitter = AriaEventEmitter(ValidatingSink(collected), RUN_ID)

    tag = uuid.uuid4()
    async with Runtime(
        CounterAgent(),
        config=RuntimeConfig(action_middleware=(emitter,)),
        exchange_factory=exchange_client.factory(),
        registration=registration,
    ) as runtime:
        emitter.emit_run_started(str(runtime.agent_id))
        await runtime.action('add', exchange_client.client_id,
                             args=(5,), kwargs={}, tag=tag)
        result = await runtime.action('count', exchange_client.client_id,
                                      args=(), kwargs={})
        emitter.emit_run_completed(str(runtime.agent_id),
                                   details={'result': result})
        runtime.signal_shutdown()

    assert result == 5
    kinds = [(e['eventType'], e['payload'].get('toolName'),
              e['payload'].get('status')) for e in collected.events]
    assert kinds == [
        ('journal.run.started', None, None),
        ('tool.call', 'add', 'started'),
        ('tool.call', 'add', 'succeeded'),
        ('tool.call', 'count', 'started'),
        ('tool.call', 'count', 'succeeded'),
        ('journal.run.completed', None, None),
    ]
    # the invocation tag is the correlation id (RFC 018 convention)
    add_events = [e for e in collected.events
                  if e['payload'].get('toolName') == 'add']
    assert {e['correlationId'] for e in add_events} == {str(tag)}
    # started and terminal events of one invocation share the correlation
    assert all(e['payload']['runId'] == RUN_ID for e in collected.events)


@pytest.mark.asyncio
async def test_failed_action_emits_failed_event(
    exchange_client: UserExchangeClient[LocalExchangeTransport],
) -> None:
    registration = await exchange_client.register_agent(ErrorAgent)
    collected = CollectingSink()
    emitter = AriaEventEmitter(ValidatingSink(collected), RUN_ID)

    async with Runtime(
        ErrorAgent(),
        config=RuntimeConfig(action_middleware=(emitter,)),
        exchange_factory=exchange_client.factory(),
        registration=registration,
    ) as runtime:
        with pytest.raises(RuntimeError):
            await runtime.action('fails', exchange_client.client_id,
                                 args=(), kwargs={})
        runtime.signal_shutdown()

    statuses = [e['payload']['status'] for e in collected.events]
    assert statuses == ['started', 'failed']
