#!/usr/bin/env python3
"""
Minimal Academy Example

The simplest possible Academy agent setup: two agents that communicate.
- Requester: Sends a calculation request to Calculator
- Calculator: Performs the calculation and returns the result

This demonstrates Academy basics before moving to complex pipelines.

Usage:
    python main.py
    python main.py --expression "2 ** 10"
"""

import asyncio
import argparse
import logging

from academy.agent import Agent, action
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("academy.basic")


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class CalculatorAgent(Agent):
    """
    A simple agent that performs calculations.

    This agent receives calculation requests via the `calculate` action
    and returns the result directly to the caller.
    """

    @action
    async def calculate(self, expression: str) -> str:
        """
        Perform a calculation and return the result.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            The calculation result as a string
        """
        logger.info(f"Calculator received: {expression}")

        try:
            # Safe evaluation (only math operations)
            result = eval(expression, {"__builtins__": {}}, {})
            response = f"{expression} = {result}"
            logger.info(f"Calculator computed: {response}")
        except Exception as e:
            response = f"Error: {e}"
            logger.error(f"Calculator error: {e}")

        return response


class RequesterAgent(Agent):
    """
    An agent that requests calculations from another agent.

    This demonstrates how agents can receive Handle objects
    to communicate with other agents.
    """

    def __init__(self) -> None:
        super().__init__()
        self._calculator: Handle | None = None

    @action
    async def set_calculator(self, calculator: Handle) -> None:
        """
        Set the calculator agent to use for calculations.

        Args:
            calculator: Handle to the Calculator agent
        """
        self._calculator = calculator
        logger.info("Requester: Calculator handle received")

    @action
    async def request_calculation(self, expression: str) -> str:
        """
        Send a calculation request to the Calculator agent.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            The calculation result from the Calculator
        """
        if not self._calculator:
            error_msg = "Error: No calculator set!"
            logger.error(error_msg)
            return error_msg

        logger.info(f"Requester sending: {expression}")

        # Call the calculator's action and await the result
        result = await self._calculator.calculate(expression)

        logger.info(f"Requester received: {result}")
        return result


# ============================================================================
# MAIN
# ============================================================================

async def main(expression: str) -> None:
    """
    Run the minimal Academy example.

    Steps:
    1. Create a Manager (the Academy runtime)
    2. Launch two agents: Calculator and Requester
    3. Connect the Requester to the Calculator
    4. Send a calculation request and get the result
    """

    print("=" * 50)
    print("MINIMAL ACADEMY EXAMPLE")
    print("=" * 50)
    print(f"Expression: {expression}")
    print("-" * 50)

    # Create the Academy runtime with local message exchange
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        # Launch agents
        # manager.launch() returns a Handle (proxy) for the agent
        calculator = await manager.launch(CalculatorAgent)
        requester = await manager.launch(RequesterAgent)

        logger.info(f"Launched Calculator: {calculator.identifier}")
        logger.info(f"Launched Requester: {requester.identifier}")

        # Connect the requester to the calculator
        # This passes the Calculator's Handle to the Requester
        await requester.set_calculator(calculator)

        # Send calculation request and get result
        # The requester calls calculator.calculate() internally
        result = await requester.request_calculation(expression=expression)

        print("-" * 50)
        print(f"RESULT: {result}")
        print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Academy example with two communicating agents"
    )
    parser.add_argument(
        "--expression", "-e",
        type=str,
        default="42 * 17",
        help="Mathematical expression to calculate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.expression))
