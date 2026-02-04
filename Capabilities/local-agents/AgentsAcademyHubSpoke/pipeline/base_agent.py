"""
Base agent class for Academy-based scientific discovery (hub-and-spoke pattern).

In this pattern, the main process orchestrates all agents.
Agents don't communicate directly with each other.
"""

import logging
from academy.agent import Agent, action


class ScienceAgent(Agent):
    """
    Base class for scientific discovery agents.

    Each agent processes input and returns results to the main process.
    The main process handles all orchestration.
    """

    name: str = "ScienceAgent"
    role: str = "Base agent for scientific discovery"

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"a4s.{self.name}")
        self.logger.info("Initialized %s: %s", self.name, self.role)

    @action
    async def process(self, input_text: str) -> str:
        """
        Process input and return result to main process.

        Unlike the pipeline pattern, results are returned directly
        to the caller rather than forwarded to another agent.
        """
        self.logger.info("[%s] Processing...", self.name)
        result = await self._process(input_text)
        self.logger.info("[%s] Done", self.name)
        return result

    async def _process(self, input_text: str) -> str:
        """Override in subclass to implement agent-specific logic."""
        raise NotImplementedError("Subclasses must implement _process")
