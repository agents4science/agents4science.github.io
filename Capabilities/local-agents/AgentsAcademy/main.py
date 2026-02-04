"""
AgentsAcademy - Multi-agent pipeline using Academy framework.

This example demonstrates how to build a multi-agent pipeline for scientific
discovery using Academy. Five specialized agents work in sequence to tackle
a research goal, with each agent contributing its expertise before passing
results to the next.

Usage:
    python main.py
    python main.py --goal "Design a catalyst for ammonia synthesis"
"""

import os
import asyncio
import logging
import argparse

from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

from pipeline.roles import (
    ScoutAgent,
    PlannerAgent,
    OperatorAgent,
    AnalystAgent,
    ArchivistAgent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("a4s.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-agent scientific discovery pipeline with Academy."
    )
    parser.add_argument(
        "--goal",
        "-g",
        type=str,
        default="Find catalysts that improve CO2 conversion at room temperature.",
        help="Scientific goal for the pipeline to address",
    )
    return parser.parse_args()


async def run_pipeline(manager: Manager, goal: str) -> dict:
    """
    Execute the multi-agent pipeline.

    Each agent is launched via the Manager, and its output becomes
    the input for the next agent in the sequence.
    """
    logger.info("=" * 60)
    logger.info("Starting pipeline with goal: %s", goal)
    logger.info("=" * 60)

    # Define the agent sequence
    agent_classes = [
        ScoutAgent,
        PlannerAgent,
        OperatorAgent,
        AnalystAgent,
        ArchivistAgent,
    ]

    state = goal
    results = {}

    for agent_class in agent_classes:
        # Launch the agent
        handle = await manager.launch(agent_class)

        # Call the agent's act method
        logger.info("-" * 40)
        logger.info("Running %s...", agent_class.name)
        result = await handle.act(state)

        # Store result and update state for next agent
        results[agent_class.name] = result
        state = result["output"]

        logger.info("%s completed.", agent_class.name)

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

    return results


async def main(args: argparse.Namespace) -> None:
    """
    Main entry point using Academy's Manager with local execution.
    """
    # Use local exchange for single-machine execution
    # For federated execution, use HttpExchangeFactory instead
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        results = await run_pipeline(manager, args.goal)

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)

        for agent_name, result in results.items():
            print(f"\n### {agent_name} ###")
            print(result["output"][:500])
            if len(result["output"]) > 500:
                print("...")


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
