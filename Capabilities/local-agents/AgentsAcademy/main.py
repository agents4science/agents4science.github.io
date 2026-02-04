"""
AgentsAcademy - Multi-agent pipeline using Academy framework.

This example demonstrates a TRUE PIPELINE pattern where agents
forward results directly to each other via messaging. The main
process only sets up the pipeline and triggers the first agent.

Usage:
    python main.py
    python main.py --goal "Design a catalyst for ammonia synthesis"
"""

import asyncio
import logging
import argparse

from academy.agent import Agent, action
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


class ResultsCollector(Agent):
    """
    Collects results from pipeline agents and signals completion.

    This agent receives results as they flow through the pipeline,
    allowing monitoring without interrupting agent-to-agent messaging.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: dict[str, str] = {}
        self._complete = asyncio.Event()

    @action
    async def collect(self, agent_name: str, result: str) -> None:
        """Receive a result from a pipeline agent."""
        self.results[agent_name] = result
        logger.info("Collected result from %s", agent_name)

    @action
    async def pipeline_complete(self) -> None:
        """Signal that the pipeline has finished."""
        self._complete.set()

    @action
    async def get_results(self) -> dict[str, str]:
        """Wait for pipeline to complete and return all results."""
        await self._complete.wait()
        return self.results


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


async def main(args: argparse.Namespace) -> None:
    """
    Set up and run the pipeline.

    The main process:
    1. Launches all agents
    2. Connects them into a pipeline (Scout -> Planner -> ... -> Archivist)
    3. Triggers the first agent
    4. Waits for completion

    After setup, agents communicate directly with each other.
    """
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        logger.info("=" * 60)
        logger.info("Setting up pipeline for goal:")
        logger.info("  %s", args.goal)
        logger.info("=" * 60)

        # Launch all agents
        collector = await manager.launch(ResultsCollector)
        scout = await manager.launch(ScoutAgent)
        planner = await manager.launch(PlannerAgent)
        operator = await manager.launch(OperatorAgent)
        analyst = await manager.launch(AnalystAgent)
        archivist = await manager.launch(ArchivistAgent)

        # Connect pipeline: Scout -> Planner -> Operator -> Analyst -> Archivist
        pipeline = [scout, planner, operator, analyst, archivist]
        for i, agent in enumerate(pipeline):
            await agent.set_results_collector(collector)
            if i < len(pipeline) - 1:
                await agent.set_next(pipeline[i + 1])

        logger.info("Pipeline connected: Scout -> Planner -> Operator -> Analyst -> Archivist")
        logger.info("-" * 60)

        # Trigger the pipeline by sending goal to Scout
        # From here, agents communicate directly with each other
        logger.info("Triggering pipeline...")
        await scout.process(args.goal)

        # Wait for pipeline to complete and get results
        results = await collector.get_results()

        # Print results
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)

        for agent_name, result in results.items():
            print(f"\n### {agent_name} ###")
            print(result[:500])
            if len(result) > 500:
                print("...")


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
