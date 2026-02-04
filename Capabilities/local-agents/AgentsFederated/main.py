#!/usr/bin/env python3
"""
Academy Federated Collaboration Example

Demonstrates how Academy agents at different institutions can collaborate
on scientific tasks. This example simulates a cross-institutional workflow
between DOE national labs.

Pattern:
- Each institution runs agents providing specialized capabilities
- A coordinator orchestrates work across institutional boundaries
- Academy handles secure identity and messaging

This example runs locally to demonstrate the pattern. In production,
each institution would run its own agents with appropriate authentication.

Usage:
    python main.py
    python main.py --task "characterize novel superconducting materials"
"""

import asyncio
import argparse
import logging
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
logger = logging.getLogger("academy.federated")


# ============================================================================
# INSTITUTION-SPECIFIC AGENTS
# ============================================================================

class ANLComputeAgent(Agent):
    """
    Agent representing Argonne National Lab's compute capabilities.

    In production, this agent would run at ANL with access to:
    - Aurora exascale supercomputer
    - Theta/Polaris systems
    - AI accelerators for ML workloads
    """

    institution = "ANL"

    @action
    async def run_simulation(self, parameters: dict) -> dict:
        """Run a large-scale simulation on ANL resources."""
        logger.info(f"[ANL] Starting simulation with {parameters}")

        # Simulate compute work
        await asyncio.sleep(0.5)

        result = {
            "institution": self.institution,
            "resource": "Aurora",
            "parameters": parameters,
            "result": {
                "energy": -127.5,
                "converged": True,
                "wall_time_hours": 4.2,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[ANL] Simulation complete: energy={result['result']['energy']}")
        return result

    @action
    async def get_capabilities(self) -> dict:
        """Report available capabilities."""
        return {
            "institution": self.institution,
            "capabilities": ["DFT", "MD", "ML_inference"],
            "resources": ["Aurora", "Polaris", "AI_cluster"],
            "status": "available",
        }


class ORNLDataAgent(Agent):
    """
    Agent representing Oak Ridge National Lab's data capabilities.

    In production, this agent would provide access to:
    - Summit and Frontier supercomputers
    - Neutron scattering data (SNS, HFIR)
    - Data management infrastructure
    """

    institution = "ORNL"

    @action
    async def query_database(self, query: str) -> dict:
        """Query ORNL's scientific databases."""
        logger.info(f"[ORNL] Querying database: {query}")

        await asyncio.sleep(0.3)

        result = {
            "institution": self.institution,
            "query": query,
            "records_found": 1247,
            "sources": ["SNS", "HFIR", "materials_db"],
            "sample_records": [
                {"id": "ORNL-001", "type": "neutron_scattering"},
                {"id": "ORNL-002", "type": "crystallography"},
            ],
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[ORNL] Found {result['records_found']} records")
        return result

    @action
    async def get_capabilities(self) -> dict:
        """Report available capabilities."""
        return {
            "institution": self.institution,
            "capabilities": ["neutron_data", "materials_db", "data_curation"],
            "resources": ["Frontier", "SNS", "HFIR"],
            "status": "available",
        }


class LBNLAnalysisAgent(Agent):
    """
    Agent representing Lawrence Berkeley Lab's analysis capabilities.

    In production, this agent would provide access to:
    - NERSC compute (Perlmutter)
    - ALS beamline data
    - ML/AI analysis pipelines
    """

    institution = "LBNL"

    @action
    async def analyze_data(self, data: dict, method: str) -> dict:
        """Perform advanced analysis on scientific data."""
        logger.info(f"[LBNL] Analyzing data with method: {method}")

        await asyncio.sleep(0.4)

        result = {
            "institution": self.institution,
            "method": method,
            "input_records": data.get("records_found", 0),
            "analysis_result": {
                "clusters_found": 5,
                "outliers": 23,
                "confidence": 0.94,
                "key_features": ["temperature_dependence", "pressure_sensitivity"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[LBNL] Analysis complete: confidence={result['analysis_result']['confidence']}")
        return result

    @action
    async def get_capabilities(self) -> dict:
        """Report available capabilities."""
        return {
            "institution": self.institution,
            "capabilities": ["ML_analysis", "beamline_data", "visualization"],
            "resources": ["Perlmutter", "ALS", "ML_cluster"],
            "status": "available",
        }


# ============================================================================
# COORDINATOR AGENT
# ============================================================================

class FederatedCoordinator(Agent):
    """
    Coordinates work across multiple institutions.

    In production, this agent would:
    - Authenticate with each institution's identity provider
    - Respect institutional policies and access controls
    - Handle data transfer and provenance tracking
    """

    def __init__(self) -> None:
        super().__init__()
        self._agents: dict[str, Handle] = {}
        self._results: dict | None = None
        self._complete = asyncio.Event()

    @action
    async def register_agent(self, institution: str, agent: Handle) -> None:
        """Register an institutional agent."""
        self._agents[institution] = agent
        logger.info(f"[Coordinator] Registered agent from {institution}")

    @action
    async def execute_federated_workflow(self, task: str) -> dict:
        """
        Execute a workflow that spans multiple institutions.

        This demonstrates how a single scientific task can leverage
        capabilities distributed across the DOE complex.
        """
        logger.info(f"[Coordinator] Starting federated workflow: {task}")

        workflow_results = {
            "task": task,
            "steps": [],
            "institutions_involved": list(self._agents.keys()),
        }

        # Step 1: Query capabilities
        logger.info("[Coordinator] Step 1: Querying institutional capabilities")
        capabilities = {}
        for institution, agent in self._agents.items():
            cap = await agent.get_capabilities()
            capabilities[institution] = cap
            logger.info(f"[Coordinator] {institution}: {cap['capabilities']}")

        workflow_results["steps"].append({
            "step": 1,
            "action": "capability_discovery",
            "result": capabilities,
        })

        # Step 2: Query data from ORNL
        logger.info("[Coordinator] Step 2: Querying data from ORNL")
        ornl_data = await self._agents["ORNL"].query_database(
            f"materials related to '{task}'"
        )
        workflow_results["steps"].append({
            "step": 2,
            "action": "data_query",
            "institution": "ORNL",
            "result": ornl_data,
        })

        # Step 3: Run simulation at ANL
        logger.info("[Coordinator] Step 3: Running simulation at ANL")
        anl_sim = await self._agents["ANL"].run_simulation({
            "task": task,
            "input_data": ornl_data["records_found"],
            "method": "DFT",
        })
        workflow_results["steps"].append({
            "step": 3,
            "action": "simulation",
            "institution": "ANL",
            "result": anl_sim,
        })

        # Step 4: Analyze results at LBNL
        logger.info("[Coordinator] Step 4: Analyzing results at LBNL")
        lbnl_analysis = await self._agents["LBNL"].analyze_data(
            data=ornl_data,
            method="ML_clustering",
        )
        workflow_results["steps"].append({
            "step": 4,
            "action": "analysis",
            "institution": "LBNL",
            "result": lbnl_analysis,
        })

        # Compile final results
        workflow_results["status"] = "completed"
        workflow_results["summary"] = {
            "data_records_used": ornl_data["records_found"],
            "simulation_energy": anl_sim["result"]["energy"],
            "analysis_confidence": lbnl_analysis["analysis_result"]["confidence"],
        }
        workflow_results["timestamp"] = datetime.now().isoformat()

        logger.info("[Coordinator] Federated workflow complete")
        self._results = workflow_results
        self._complete.set()
        return workflow_results

    @action
    async def get_results(self) -> dict:
        """Wait for workflow to complete and return results."""
        await self._complete.wait()
        return self._results


# ============================================================================
# MAIN
# ============================================================================

async def main(task: str) -> None:
    """
    Run the federated collaboration example.

    Demonstrates:
    1. Agents representing different DOE institutions
    2. A coordinator orchestrating cross-institutional work
    3. Data and compute flowing across organizational boundaries
    """

    print("=" * 70)
    print("ACADEMY FEDERATED COLLABORATION EXAMPLE")
    print("=" * 70)
    print(f"Task: {task}")
    print("-" * 70)
    print("Simulating agents at:")
    print("  - ANL (Argonne): Compute capabilities (Aurora)")
    print("  - ORNL (Oak Ridge): Data capabilities (SNS, HFIR)")
    print("  - LBNL (Berkeley): Analysis capabilities (Perlmutter)")
    print("-" * 70)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        # Launch institutional agents
        # In production, each would run at its respective institution
        anl = await manager.launch(ANLComputeAgent)
        ornl = await manager.launch(ORNLDataAgent)
        lbnl = await manager.launch(LBNLAnalysisAgent)
        coordinator = await manager.launch(FederatedCoordinator)

        logger.info("Launched all institutional agents")

        # Register agents with coordinator
        await coordinator.register_agent("ANL", anl)
        await coordinator.register_agent("ORNL", ornl)
        await coordinator.register_agent("LBNL", lbnl)

        # Execute federated workflow
        results = await coordinator.execute_federated_workflow(task)

        # Display results
        print("-" * 70)
        print("FEDERATED WORKFLOW RESULTS")
        print("-" * 70)
        print(f"Status: {results['status']}")
        print(f"Institutions: {', '.join(results['institutions_involved'])}")
        print()
        print("Summary:")
        print(f"  Data records used (ORNL):    {results['summary']['data_records_used']}")
        print(f"  Simulation energy (ANL):     {results['summary']['simulation_energy']}")
        print(f"  Analysis confidence (LBNL):  {results['summary']['analysis_confidence']}")
        print()
        print("Workflow steps:")
        for step in results["steps"]:
            inst = step.get("institution", "all")
            print(f"  {step['step']}. {step['action']} ({inst})")
        print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academy example demonstrating federated collaboration"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="characterize novel superconducting materials",
        help="Scientific task for the federated workflow",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.task))
