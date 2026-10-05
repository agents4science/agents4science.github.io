import logging
import uuid

from academy_aria import AriaLoggingHandler, CollectingSink, ValidatingSink
from academy_aria.logging_handler import make_record_extra


def test_logging_handler_pairs_start_and_terminal():
    collected = CollectingSink()
    handler = AriaLoggingHandler(
        ValidatingSink(collected), 'run-log-test', source='agent-log')
    logger = logging.getLogger('academy_aria.test.handler')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        tag = uuid.uuid4()
        logger.debug('start', extra=make_record_extra('add', tag, 'execute_start'))
        logger.debug('done', extra=make_record_extra('add', tag, 'execute_success'))
        tag2 = uuid.uuid4()
        logger.debug('start', extra=make_record_extra('boom', tag2, 'execute_start'))
        logger.debug('err', extra=make_record_extra('boom', tag2, 'execute_exception'))
        logger.debug('noise without extras')
    finally:
        logger.removeHandler(handler)

    kinds = [(e['payload']['toolName'], e['payload']['status'])
             for e in collected.events]
    assert kinds == [('add', 'started'), ('add', 'succeeded'),
                     ('boom', 'started'), ('boom', 'failed')]
    assert collected.events[1]['correlationId'] == collected.events[0]['correlationId']
