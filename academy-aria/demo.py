"""Demo: an Academy actor whose action history becomes an ARIA event stream.

Runs a small actor, drives three actions (one fails), and prints the
contract-validated events plus a terminal-state reconstruction from the
stream alone — the replay property the C1 gate asks for.

    /tmp/academy-venv/bin/python demo.py
"""

import asyncio
import json
import uuid

from academy.agent import Agent, action
from academy.exchange.local import LocalExchangeFactory
from academy.runtime import Runtime, RuntimeConfig

from academy_aria import AriaEventEmitter, CollectingSink, JsonlSink, ValidatingSink


class ScreenAgent(Agent):
    """Stand-in for a science actor: screens accessions, one of which fails."""

    @action
    async def screen(self, accession: str) -> str:
        if accession.endswith('X'):
            raise RuntimeError(f'no assembly found for {accession}')
        return f'{accession}: 3 AMR genes'


async def main() -> None:
    run_id = f'run_{uuid.uuid4().hex[:12]}'
    collected = CollectingSink()
    sink = ValidatingSink(collected)          # every event must pass the contracts
    emitter = AriaEventEmitter(sink, run_id)

    factory = LocalExchangeFactory()
    async with await factory.create_user_client() as client:
        registration = await client.register_agent(ScreenAgent)
        async with Runtime(
            ScreenAgent(),
            config=RuntimeConfig(action_middleware=(emitter,)),
            exchange_factory=factory,
            registration=registration,
        ) as runtime:
            source = str(runtime.agent_id)
            emitter.emit_run_started(source)
            for accession in ('GCF_001', 'GCF_002', 'GCF_00X'):
                try:
                    await runtime.action('screen', client.client_id,
                                         args=(accession,), kwargs={},
                                         tag=uuid.uuid4())
                except RuntimeError:
                    pass
            emitter.emit_run_completed(
                source, failure_class='F0_NONE',
                details={'screened': 2, 'failed': 1})
            runtime.signal_shutdown()

    events = collected.events
    JsonlSink('demo-events.jsonl').path.unlink(missing_ok=True)
    out = JsonlSink('demo-events.jsonl')
    for e in events:
        out.append(e)

    print(f'{len(events)} events emitted, all contract-valid '
          f'(run {run_id}):\n')
    for e in events:
        p = e['payload']
        label = p.get('toolName') or e['eventType']
        status = p.get('status') or p.get('failureClass')
        print(f"  {e['occurredAt']}  {e['eventType']:<22} {label:<22} "
              f"{status:<10} corr={e.get('correlationId', '')[:13]}")

    # Replay: reconstruct terminal state from the stream alone.
    invocations: dict[str, str] = {}
    run_state = 'unknown'
    for e in events:
        if e['eventType'] == 'tool.call':
            invocations[e['correlationId']] = e['payload']['status']
        elif e['eventType'] == 'journal.run.completed':
            run_state = 'completed'
        elif e['eventType'] == 'journal.run.failed':
            run_state = 'failed'
    terminal = {s: list(invocations.values()).count(s)
                for s in ('succeeded', 'failed')}
    print(f'\nreplayed terminal state: run={run_state}, '
          f'invocations={json.dumps(terminal)}')
    print('events written to demo-events.jsonl')


if __name__ == '__main__':
    asyncio.run(main())
