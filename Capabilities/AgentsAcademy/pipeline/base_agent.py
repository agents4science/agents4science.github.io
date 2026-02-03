"""
Base agent class for Academy-based scientific discovery pipeline.

Each specialized agent inherits from ScienceAgent, which wraps Academy's Agent.
"""

import logging
from typing import Dict, Any
from academy.agent import Agent, action


class ScienceAgent(Agent):
    """
    Base class for scientific discovery agents.

    Provides common functionality for all pipeline agents:
    - Logging
    - State management
    - Action interface
    """

    name: str = "ScienceAgent"
    role: str = "Base agent for scientific discovery"

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"a4s.{self.name}")
        self.logger.info("Initialized %s: %s", self.name, self.role)

    @action
    async def act(self, input_text: str) -> Dict[str, Any]:
        """
        Process input and return structured output.

        This is the main action that downstream agents call.
        Each specialized agent overrides _process to implement
        domain-specific logic.
        """
        self.logger.info("[%s] Processing input: %s...", self.name, input_text[:100])

        result = await self._process(input_text)

        self.logger.info("[%s] Completed processing", self.name)
        return {
            "agent": self.name,
            "role": self.role,
            "input": input_text,
            "output": result,
        }

    async def _process(self, input_text: str) -> str:
        """
        Override in subclass to implement agent-specific logic.
        """
        raise NotImplementedError("Subclasses must implement _process")
