"""
AgentsAcademyHubSpoke - Multi-agent workflow using Academy framework.

This example demonstrates a HUB-AND-SPOKE pattern where the main
process orchestrates all agents sequentially. Agents don't communicate
directly with each other.

Usage:
    python main.py
    python main.py --goal "Design a catalyst for ammonia synthesis"
"""

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
        description="Run multi-agent scientific discovery workflow with Academy."
    )
    parser.add_argument(
        "--goal",
        "-g",
        type=str,
        default="Find catalysts that improve CO2 conversion at room temperature.",
        help="Scientific goal for the workflow to address",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    """
    Run the hub-and-spoke workflow.

    The main process:
    1. Launches all agents
    2. Calls each agent sequentially, passing results forward
    3. Collects all results

    The main process remains in control throughout.
    """
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        logger.info("=" * 60)
        logger.info("Starting workflow for goal:")
        logger.info("  %s", args.goal)
        logger.info("=" * 60)

        # Launch all agents
        scout = await manager.launch(ScoutAgent)
        planner = await manager.launch(PlannerAgent)
        operator = await manager.launch(OperatorAgent)
        analyst = await manager.launch(AnalystAgent)
        archivist = await manager.launch(ArchivistAgent)

        # Run workflow: main process orchestrates each step
        results: dict[str, str] = {}

        logger.info("-" * 60)
        logger.info("Step 1: Scout surveys the problem space")
        results["Scout"] = await scout.process(args.goal)

        logger.info("-" * 60)
        logger.info("Step 2: Planner designs workflow")
        results["Planner"] = await planner.process(results["Scout"])

        logger.info("-" * 60)
        logger.info("Step 3: Operator executes the plan")
        results["Operator"] = await operator.process(results["Planner"])

        logger.info("-" * 60)
        logger.info("Step 4: Analyst summarizes findings")
        results["Analyst"] = await analyst.process(results["Operator"])

        logger.info("-" * 60)
        logger.info("Step 5: Archivist documents everything")
        results["Archivist"] = await archivist.process(results["Analyst"])

        # Print results
        print("\n" + "=" * 60)
        print("WORKFLOW RESULTS")
        print("=" * 60)

        for agent_name, result in results.items():
            print(f"\n### {agent_name} ###")
            print(result[:500])
            if len(result) > 500:
                print("...")


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
