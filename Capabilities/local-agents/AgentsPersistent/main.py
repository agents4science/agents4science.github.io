#!/usr/bin/env python3
"""
Academy Persistent State Example

Demonstrates how Academy agents can persist state and resume after restarts.
This is essential for long-running scientific workflows that may span hours,
days, or even weeks.

Features:
- State checkpointing to disk
- Resume from last checkpoint on restart
- Progress tracking across sessions
- Graceful handling of interruptions

Usage:
    python main.py                    # Start fresh or resume
    python main.py --reset            # Start fresh, ignore checkpoints
    python main.py --state-dir ./my-state  # Custom state directory

    # Simulate interruption:
    # Run once, Ctrl+C to interrupt, run again to resume
"""

import asyncio
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

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
logger = logging.getLogger("academy.persist")


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class AgentState:
    """
    Manages persistent state for an Academy agent.

    State is saved as JSON to disk, allowing agents to resume
    after restarts, crashes, or interruptions.
    """

    def __init__(self, agent_name: str, state_dir: Path):
        self.agent_name = agent_name
        self.state_dir = state_dir
        self.state_file = state_dir / f"{agent_name}_state.json"
        self._state: dict = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk if it exists."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                self._state = json.load(f)
            logger.info(f"[{self.agent_name}] Loaded state from {self.state_file}")
        else:
            self._state = {"created_at": datetime.now().isoformat()}
            logger.info(f"[{self.agent_name}] Starting fresh (no checkpoint found)")

    def save(self) -> None:
        """Save current state to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)
        logger.debug(f"[{self.agent_name}] State saved")

    def get(self, key: str, default=None):
        """Get a state value."""
        return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        """Set a state value and save."""
        self._state[key] = value
        self.save()

    def clear(self) -> None:
        """Clear all state."""
        self._state = {"created_at": datetime.now().isoformat()}
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info(f"[{self.agent_name}] State cleared")


# ============================================================================
# PERSISTENT WORKFLOW AGENT
# ============================================================================

class WorkflowAgent(Agent):
    """
    An Academy agent with persistent state.

    This agent can:
    - Track progress through a multi-step workflow
    - Checkpoint state after each step
    - Resume from last checkpoint on restart
    """

    def __init__(self) -> None:
        super().__init__()
        self._state: AgentState | None = None
        self._complete = asyncio.Event()
        self._results: dict | None = None

    @action
    async def configure(self, state_dir: str) -> None:
        """Configure the agent with a state directory."""
        self._state = AgentState("workflow", Path(state_dir))
        logger.info(f"[Workflow] Configured with state dir: {state_dir}")

    @action
    async def execute_workflow(self, task: str) -> dict:
        """
        Execute a multi-step workflow with checkpointing.

        Each step is checkpointed. On resume, the agent skips
        completed steps and continues from where it left off.
        """
        if not self._state:
            raise RuntimeError("Agent not configured with state directory")

        logger.info(f"[Workflow] Starting task: {task}")

        # Initialize or resume workflow state
        if not self._state.get("task"):
            self._state.set("task", task)
            self._state.set("current_step", 0)
            self._state.set("step_results", [])
            logger.info("[Workflow] Starting fresh workflow")
        else:
            logger.info(f"[Workflow] Resuming from step {self._state.get('current_step')}")

        # Define workflow steps
        steps = [
            ("literature_review", self._literature_review),
            ("data_collection", self._data_collection),
            ("computation", self._computation),
            ("analysis", self._analysis),
            ("documentation", self._documentation),
        ]

        # Execute steps, resuming from checkpoint
        current_step = self._state.get("current_step", 0)
        step_results = self._state.get("step_results", [])

        for i, (step_name, step_func) in enumerate(steps):
            if i < current_step:
                logger.info(f"[Workflow] Skipping completed step {i+1}: {step_name}")
                continue

            logger.info(f"[Workflow] Executing step {i+1}/{len(steps)}: {step_name}")

            # Execute step
            result = await step_func(task)

            # Checkpoint progress
            step_results.append({
                "step": i + 1,
                "name": step_name,
                "result": result,
                "completed_at": datetime.now().isoformat(),
            })
            self._state.set("current_step", i + 1)
            self._state.set("step_results", step_results)

            logger.info(f"[Workflow] Step {i+1} complete, checkpoint saved")

        # Workflow complete
        final_results = {
            "task": task,
            "status": "completed",
            "steps": step_results,
            "completed_at": datetime.now().isoformat(),
        }
        self._state.set("status", "completed")
        self._state.set("completed_at", final_results["completed_at"])

        logger.info("[Workflow] All steps completed")
        self._results = final_results
        self._complete.set()
        return final_results

    async def _literature_review(self, task: str) -> str:
        """Step 1: Literature review."""
        await asyncio.sleep(0.5)  # Simulate work
        return f"Reviewed 47 papers related to '{task}'"

    async def _data_collection(self, task: str) -> str:
        """Step 2: Data collection."""
        await asyncio.sleep(0.5)  # Simulate work
        return "Collected 1,247 data points from 3 sources"

    async def _computation(self, task: str) -> str:
        """Step 3: Computation."""
        await asyncio.sleep(0.5)  # Simulate work
        return "Completed 10 DFT calculations, 5 MD simulations"

    async def _analysis(self, task: str) -> str:
        """Step 4: Analysis."""
        await asyncio.sleep(0.5)  # Simulate work
        return "Identified 3 promising candidates with >90% selectivity"

    async def _documentation(self, task: str) -> str:
        """Step 5: Documentation."""
        await asyncio.sleep(0.5)  # Simulate work
        return "Generated final report with reproducibility artifacts"

    @action
    async def get_status(self) -> dict:
        """Get current workflow status."""
        if not self._state:
            return {"status": "not_configured"}

        return {
            "task": self._state.get("task", "none"),
            "current_step": self._state.get("current_step", 0),
            "status": self._state.get("status", "in_progress"),
            "total_steps": 5,
        }

    @action
    async def get_results(self) -> dict:
        """Wait for workflow to complete and return results."""
        await self._complete.wait()
        return self._results


# ============================================================================
# MAIN
# ============================================================================

async def main(state_dir: Path, reset: bool, task: str) -> None:
    """
    Run the persistent state example.

    Demonstrates:
    1. State checkpointing and resumption
    2. Progress tracking across sessions
    3. Graceful handling of interruptions
    """

    print("=" * 60)
    print("ACADEMY PERSISTENT STATE EXAMPLE")
    print("=" * 60)
    print(f"State directory: {state_dir}")
    print(f"Reset: {reset}")
    print(f"Task: {task}")
    print("-" * 60)

    # Handle reset
    if reset:
        state_file = state_dir / "workflow_state.json"
        if state_file.exists():
            state_file.unlink()
            print("Cleared previous state")
        print("-" * 60)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        # Launch workflow agent
        workflow = await manager.launch(WorkflowAgent)
        logger.info(f"Launched Workflow agent: {workflow.identifier}")

        # Configure with state directory
        await workflow.configure(str(state_dir))

        # Check initial status
        status = await workflow.get_status()
        if status["current_step"] > 0:
            print(f"Resuming from step {status['current_step']}/{status['total_steps']}")
        else:
            print("Starting fresh workflow")
        print("-" * 60)

        # Execute workflow (will resume if previously interrupted)
        try:
            results = await workflow.execute_workflow(task)

            # Display results
            print("-" * 60)
            print("WORKFLOW COMPLETED")
            print("-" * 60)
            print(f"Status: {results['status']}")
            print(f"Steps completed: {len(results['steps'])}")
            print()
            for step in results["steps"]:
                print(f"  Step {step['step']}: {step['name']}")
                print(f"    Result: {step['result']}")
            print("=" * 60)

        except asyncio.CancelledError:
            # Handle Ctrl+C gracefully
            status = await workflow.get_status()
            print("\n" + "-" * 60)
            print("WORKFLOW INTERRUPTED")
            print("-" * 60)
            print(f"Progress saved at step {status['current_step']}/{status['total_steps']}")
            print("Run again to resume from this point")
            print("=" * 60)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academy example demonstrating persistent state"
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".workflow_state"),
        help="Directory to store state checkpoints",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear any existing state and start fresh",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="discover catalysts for sustainable ammonia synthesis",
        help="Task description for the workflow",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(args.state_dir, args.reset, args.task))
    except KeyboardInterrupt:
        pass  # Handled in main
