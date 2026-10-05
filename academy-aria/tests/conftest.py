"""Self-contained fixtures: a local exchange client and two tiny agents,
mirroring Academy's own test helpers so the live tests exercise the real
Runtime without depending on Academy's internal testing package.
"""

from collections.abc import AsyncGenerator

import pytest
from academy.agent import Agent, action
from academy.exchange import LocalExchangeTransport, UserExchangeClient
from academy.exchange.local import LocalExchangeFactory


class CounterAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self._count = 0

    @action
    async def add(self, value: int) -> None:
        self._count += value

    @action
    async def count(self) -> int:
        return self._count


class ErrorAgent(Agent):
    @action
    async def fails(self) -> None:
        raise RuntimeError('This action always fails.')


@pytest.fixture
async def exchange_client() -> AsyncGenerator[
    UserExchangeClient[LocalExchangeTransport]
]:
    factory = LocalExchangeFactory()
    async with await factory.create_user_client() as client:
        yield client
