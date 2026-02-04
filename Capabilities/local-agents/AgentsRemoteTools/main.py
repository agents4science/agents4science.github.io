#!/usr/bin/env python3
"""
Academy Remote Tools Example

Demonstrates a pattern where one agent provides tools that another agent
can call remotely. This is a key pattern for distributed computing scenarios
where expensive tools (simulations, data analysis, HPC jobs) run on remote
systems.

Pattern:
- ToolProviderAgent: Exposes computation tools as @actions
- CoordinatorAgent: Calls tools on the ToolProvider, orchestrates workflow

This pattern scales naturally: the ToolProvider could run on an HPC node,
a cloud VM, or a lab instrument computer, while the Coordinator runs
anywhere with network access.

Usage:
    python main.py
    python main.py --task "analyze protein folding data"
"""

import asyncio
import argparse
import logging
import random
from datetime import datetime

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
logger = logging.getLogger("academy.remote")


# ============================================================================
# TOOL PROVIDER AGENT
# ============================================================================

class ToolProviderAgent(Agent):
    """
    An agent that provides computational tools to other agents.

    In a real scenario, this agent would run on an HPC system, instrument
    computer, or specialized compute node. Its @action methods become
    remotely-callable tools.

    This demonstrates how Academy enables secure, federated tool access
    across institutional boundaries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.job_counter = 0

    @action
    async def run_simulation(
        self,
        parameters: dict[str, float],
        duration_seconds: float = 0.5,
    ) -> dict:
        """
        Run a computational simulation with given parameters.

        Args:
            parameters: Simulation parameters (e.g., temperature, pressure)
            duration_seconds: Simulated compute time

        Returns:
            Simulation results including computed values and metadata
        """
        self.job_counter += 1
        job_id = f"SIM-{self.job_counter:04d}"
        logger.info(f"[ToolProvider] Starting simulation {job_id}")
        logger.info(f"[ToolProvider] Parameters: {parameters}")

        # Simulate computation time
        await asyncio.sleep(duration_seconds)

        # Generate mock results (in real scenario: actual simulation)
        results = {
            "job_id": job_id,
            "status": "completed",
            "parameters": parameters,
            "results": {
                "energy": -127.5 + random.uniform(-5, 5),
                "convergence": random.uniform(0.98, 0.999),
                "iterations": random.randint(50, 200),
            },
            "compute_time_ms": int(duration_seconds * 1000),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[ToolProvider] Completed {job_id}: energy={results['results']['energy']:.2f}")
        return results

    @action
    async def analyze_data(self, dataset_id: str, analysis_type: str) -> dict:
        """
        Analyze a dataset with specified analysis type.

        Args:
            dataset_id: Identifier for the dataset to analyze
            analysis_type: Type of analysis (e.g., "statistical", "clustering")

        Returns:
            Analysis results
        """
        logger.info(f"[ToolProvider] Analyzing {dataset_id} with {analysis_type}")

        # Simulate analysis
        await asyncio.sleep(0.3)

        results = {
            "dataset_id": dataset_id,
            "analysis_type": analysis_type,
            "status": "completed",
            "summary": {
                "records": random.randint(1000, 10000),
                "features": random.randint(10, 50),
                "outliers_detected": random.randint(0, 20),
                "clusters_found": random.randint(2, 8) if analysis_type == "clustering" else None,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[ToolProvider] Analysis complete: {results['summary']['records']} records")
        return results

    @action
    async def get_status(self) -> dict:
        """Get current status of the tool provider."""
        return {
            "status": "ready",
            "jobs_completed": self.job_counter,
            "available_tools": ["run_simulation", "analyze_data"],
        }


# ============================================================================
# COORDINATOR AGENT
# ============================================================================

class CoordinatorAgent(Agent):
    """
    An agent that coordinates work by calling tools on remote agents.

    This demonstrates the "coordinator" pattern where one agent orchestrates
    workflows by invoking tools provided by specialized agents. The coordinator
    focuses on decision-making while delegating compute to tool providers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tool_provider: Handle | None = None

    @action
    async def set_tool_provider(self, provider: Handle) -> None:
        """Configure the tool provider to use."""
        self._tool_provider = provider
        logger.info("[Coordinator] Tool provider configured")

    @action
    async def execute_workflow(self, task: str) -> dict:
        """
        Execute a multi-step workflow for the given task.

        This demonstrates how a coordinator can:
        1. Parse a task into steps
        2. Call appropriate tools for each step
        3. Aggregate and return results

        Args:
            task: Description of the task to execute

        Returns:
            Workflow results including all tool outputs
        """
        if not self._tool_provider:
            return {"error": "No tool provider configured"}

        logger.info(f"[Coordinator] Starting workflow for: {task}")

        # Step 1: Check tool provider status
        status = await self._tool_provider.get_status()
        logger.info(f"[Coordinator] Tool provider status: {status['status']}")

        workflow_results = {
            "task": task,
            "steps": [],
            "status": "running",
        }

        # Step 2: Run initial simulation
        logger.info("[Coordinator] Step 1: Running initial simulation")
        sim_result = await self._tool_provider.run_simulation(
            parameters={"temperature": 300.0, "pressure": 1.0},
            duration_seconds=0.3,
        )
        workflow_results["steps"].append({
            "step": 1,
            "action": "run_simulation",
            "result": sim_result,
        })

        # Step 3: Analyze results
        logger.info("[Coordinator] Step 2: Analyzing data")
        analysis_result = await self._tool_provider.analyze_data(
            dataset_id=sim_result["job_id"],
            analysis_type="statistical",
        )
        workflow_results["steps"].append({
            "step": 2,
            "action": "analyze_data",
            "result": analysis_result,
        })

        # Step 4: Run optimized simulation based on analysis
        logger.info("[Coordinator] Step 3: Running optimized simulation")
        optimized_result = await self._tool_provider.run_simulation(
            parameters={
                "temperature": 310.0,  # Adjusted based on "analysis"
                "pressure": 1.2,
            },
            duration_seconds=0.3,
        )
        workflow_results["steps"].append({
            "step": 3,
            "action": "run_simulation",
            "result": optimized_result,
        })

        # Finalize
        workflow_results["status"] = "completed"
        workflow_results["summary"] = {
            "initial_energy": workflow_results["steps"][0]["result"]["results"]["energy"],
            "optimized_energy": workflow_results["steps"][2]["result"]["results"]["energy"],
            "improvement": (
                workflow_results["steps"][0]["result"]["results"]["energy"] -
                workflow_results["steps"][2]["result"]["results"]["energy"]
            ),
        }

        logger.info(f"[Coordinator] Workflow complete: improvement={workflow_results['summary']['improvement']:.2f}")
        return workflow_results


# ============================================================================
# MAIN
# ============================================================================

async def main(task: str) -> None:
    """
    Run the remote tools example.

    This demonstrates:
    1. Launching a ToolProvider agent (simulating remote compute)
    2. Launching a Coordinator agent
    3. Connecting them via Handle passing
    4. Executing a multi-step workflow
    """

    print("=" * 60)
    print("ACADEMY REMOTE TOOLS EXAMPLE")
    print("=" * 60)
    print(f"Task: {task}")
    print("-" * 60)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        # Launch agents
        # In production: ToolProvider could run on HPC, Coordinator elsewhere
        tool_provider = await manager.launch(ToolProviderAgent)
        coordinator = await manager.launch(CoordinatorAgent)

        logger.info(f"Launched ToolProvider: {tool_provider.identifier}")
        logger.info(f"Launched Coordinator: {coordinator.identifier}")

        # Connect coordinator to tool provider
        await coordinator.set_tool_provider(tool_provider)

        # Execute workflow
        results = await coordinator.execute_workflow(task)

        # Display results
        print("-" * 60)
        print("WORKFLOW RESULTS")
        print("-" * 60)
        print(f"Status: {results['status']}")
        print(f"Steps completed: {len(results['steps'])}")
        print()
        print("Summary:")
        print(f"  Initial energy:   {results['summary']['initial_energy']:.2f}")
        print(f"  Optimized energy: {results['summary']['optimized_energy']:.2f}")
        print(f"  Improvement:      {results['summary']['improvement']:.2f}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academy example demonstrating remote tool invocation"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="optimize molecular structure for catalysis",
        help="Task description for the workflow",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.task))
