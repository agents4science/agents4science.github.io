import pytest

from academy_aria import (
    SchemaStore,
    journal_run_event,
    tool_call_event,
    validate_event,
)


@pytest.fixture(scope='module')
def store() -> SchemaStore:
    return SchemaStore()


def test_tool_call_event_validates(store):
    event = tool_call_event(
        run_id='run-001', tool_name='add', status='succeeded',
        latency_ms=12, source='agent-abc', correlation_id='tag-1')
    assert validate_event(event, store) == []


def test_journal_events_validate(store):
    for kind in ('journal.run.started', 'journal.run.completed', 'journal.run.failed'):
        event = journal_run_event(
            kind, run_id='run-001', source='agent-abc',
            failure_class='F0_NONE', details={'note': 'x'})
        assert validate_event(event, store) == []


def test_bad_status_rejected():
    with pytest.raises(ValueError):
        tool_call_event(run_id='r', tool_name='t', status='exploded',
                        latency_ms=0, source='s', correlation_id='c')


def test_broken_event_fails_contract(store):
    event = tool_call_event(
        run_id='run-001', tool_name='add', status='succeeded',
        latency_ms=1, source='agent-abc', correlation_id='tag-1')
    event['payload']['extra'] = 'nope'   # closed payload
    assert validate_event(event, store)
    del event['payload']['extra']
    del event['payload']['tokenCost']    # required field
    assert validate_event(event, store)
