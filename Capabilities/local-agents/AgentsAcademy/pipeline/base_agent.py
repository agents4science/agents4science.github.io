"""
Base agent class for Academy-based scientific discovery pipeline.

Implements a true pipeline pattern where agents forward results
directly to the next agent in the chain.
"""

import logging
from typing import Any
from academy.agent import Agent, action
from academy.handle import Handle


class ScienceAgent(Agent):
    """
    Base class for scientific discovery pipeline agents.

    Each agent processes input and forwards results to the next
    agent in the pipeline via direct messaging.
    """

    name: str = "ScienceAgent"
    role: str = "Base agent for scientific discovery"

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"a4s.{self.name}")
        self._next_agent: Handle | None = None
        self._results_collector: Handle | None = None
        self.logger.info("Initialized %s: %s", self.name, self.role)

    @action
    async def set_next(self, next_agent: Handle) -> None:
        """Set the next agent in the pipeline."""
        self._next_agent = next_agent
        self.logger.info("%s -> next agent set", self.name)

    @action
    async def set_results_collector(self, collector: Handle) -> None:
        """Set the results collector to report completions."""
        self._results_collector = collector

    @action
    async def process(self, input_text: str) -> None:
        """
        Process input and forward to next agent.

        This is the main pipeline action. Each agent:
        1. Processes the input using _process()
        2. Reports result to the collector
        3. Forwards result to the next agent (if any)
        """
        self.logger.info("[%s] Processing...", self.name)

        # Process the input
        result = await self._process(input_text)

        # Report to collector
        if self._results_collector:
            await self._results_collector.collect(self.name, result)

        # Forward to next agent in pipeline
        if self._next_agent:
            self.logger.info("[%s] Forwarding to next agent", self.name)
            await self._next_agent.process(result)
        else:
            # End of pipeline - signal completion
            self.logger.info("[%s] Pipeline complete", self.name)
            if self._results_collector:
                await self._results_collector.pipeline_complete()

    async def _process(self, input_text: str) -> str:
        """Override in subclass to implement agent-specific logic."""
        raise NotImplementedError("Subclasses must implement _process")
