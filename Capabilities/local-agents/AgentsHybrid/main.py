#!/usr/bin/env python3
"""
Academy + LangGraph Hybrid Example

Demonstrates combining Academy for distributed agent coordination with
LangGraph for LLM-based reasoning within agents.

Pattern:
- Academy handles: Agent lifecycle, messaging, distributed execution
- LangGraph handles: LLM reasoning, tool calling, state management

This hybrid approach gives you:
- Distributed, federated agent coordination (Academy)
- Intelligent, LLM-powered decision making (LangGraph)

Usage:
    python main.py                          # Uses mock LLM (no API key needed)
    python main.py --llm openai             # Uses OpenAI
    python main.py --llm ollama             # Uses local Ollama
    python main.py --task "analyze catalysts for CO2 conversion"
"""

import asyncio
import argparse
import logging
import os
from typing import Annotated

from academy.agent import Agent, action
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory
from academy.handle import Handle

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("academy.hybrid")


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

def get_llm(llm_type: str):
    """Get the appropriate LLM based on configuration."""
    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif llm_type == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}. Use 'openai' or 'ollama'.")


class MockAgent:
    """
    Mock agent for testing without API keys.

    Simulates LangGraph agent behavior without requiring an actual LLM.
    """

    def invoke(self, state: dict) -> dict:
        """Return a mock response based on message content."""
        messages = state.get("messages", [])
        last_msg = messages[-1].content if messages else ""

        if "research" in last_msg.lower() or "analyze" in last_msg.lower():
            response = """Based on my analysis:

1. **Key Finding**: The data shows promising catalyst candidates
2. **Recommendation**: Proceed with computational screening
3. **Next Steps**: Run DFT calculations on top 5 candidates

I'll now prepare the research plan."""
        elif "plan" in last_msg.lower():
            response = """Research Plan:

Phase 1: Literature Review (identify gaps)
Phase 2: Computational Screening (1000+ candidates)
Phase 3: Experimental Validation (top 10)
Phase 4: Optimization and Scale-up

Ready to execute this plan."""
        else:
            response = f"Understood. Processing: {last_msg[:50]}..."

        return {"messages": messages + [AIMessage(content=response)]}


# ============================================================================
# TOOLS FOR LANGGRAPH AGENTS
# ============================================================================

@tool
def search_literature(query: str) -> str:
    """Search scientific literature for relevant papers."""
    # Mock implementation
    return f"Found 47 papers related to '{query}'. Top hits: catalyst efficiency, reaction mechanisms, scalability studies."


@tool
def run_calculation(structure: str, method: str = "DFT") -> str:
    """Run a computational chemistry calculation."""
    # Mock implementation
    return f"Calculation complete for {structure} using {method}. Energy: -127.5 eV, Convergence: 0.999"


@tool
def analyze_results(data: str) -> str:
    """Analyze computational results and extract insights."""
    # Mock implementation
    return f"Analysis of '{data}': Identified 3 promising candidates with >90% selectivity."


# ============================================================================
# ACADEMY AGENTS WITH LANGGRAPH REASONING
# ============================================================================

class ResearcherAgent(Agent):
    """
    An Academy agent that uses LangGraph for intelligent research tasks.

    This demonstrates the hybrid pattern:
    - Academy: Handles agent identity, messaging, distributed coordination
    - LangGraph: Handles LLM reasoning, tool calling for research tasks
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_type = "mock"
        self._llm = None
        self._langgraph_agent = None
        self._planner: Handle | None = None

    def _init_langgraph(self):
        """Initialize LangGraph agent (lazy initialization)."""
        if self._langgraph_agent is None:
            if self.llm_type == "mock":
                # Use mock agent for testing
                self._langgraph_agent = MockAgent()
            else:
                # Use real LangGraph with actual LLM
                self._llm = get_llm(self.llm_type)
                tools = [search_literature, analyze_results]
                self._langgraph_agent = create_react_agent(self._llm, tools)

    @action
    async def configure(self, llm_type: str) -> None:
        """Configure the LLM type to use."""
        self.llm_type = llm_type
        self._langgraph_agent = None  # Reset so it reinitializes
        logger.info(f"[Researcher] Configured with LLM: {llm_type}")

    @action
    async def set_planner(self, planner: Handle) -> None:
        """Set the planner agent to forward research findings to."""
        self._planner = planner
        logger.info("[Researcher] Connected to Planner")

    @action
    async def research(self, task: str) -> dict:
        """
        Conduct research using LangGraph-powered LLM reasoning.

        The LLM decides which tools to use and synthesizes findings.
        """
        self._init_langgraph()
        logger.info(f"[Researcher] Starting research: {task}")

        # Use LangGraph for LLM reasoning with tools
        messages = [
            SystemMessage(content="You are a scientific researcher. Analyze the task and use available tools to gather information."),
            HumanMessage(content=f"Research task: {task}")
        ]

        try:
            # Invoke LangGraph agent
            result = self._langgraph_agent.invoke({"messages": messages})
            final_message = result["messages"][-1].content
        except Exception as e:
            logger.warning(f"[Researcher] LangGraph error: {e}, using fallback")
            final_message = f"Research findings for '{task}': Identified promising directions in computational screening and experimental validation."

        findings = {
            "task": task,
            "findings": final_message,
            "status": "completed",
            "source": "langgraph_researcher",
        }

        logger.info(f"[Researcher] Research complete")

        # Forward to planner if connected
        if self._planner:
            logger.info("[Researcher] Forwarding findings to Planner")
            await self._planner.plan(findings)

        return findings


class PlannerAgent(Agent):
    """
    An Academy agent that uses LangGraph for planning research workflows.

    Takes research findings and creates actionable plans using LLM reasoning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_type = "mock"
        self._llm = None
        self._langgraph_agent = None
        self._results: dict | None = None
        self._complete = asyncio.Event()

    def _init_langgraph(self):
        """Initialize LangGraph agent."""
        if self._langgraph_agent is None:
            if self.llm_type == "mock":
                # Use mock agent for testing
                self._langgraph_agent = MockAgent()
            else:
                # Use real LangGraph with actual LLM
                self._llm = get_llm(self.llm_type)
                tools = [run_calculation]
                self._langgraph_agent = create_react_agent(self._llm, tools)

    @action
    async def configure(self, llm_type: str) -> None:
        """Configure the LLM type to use."""
        self.llm_type = llm_type
        self._langgraph_agent = None  # Reset so it reinitializes
        logger.info(f"[Planner] Configured with LLM: {llm_type}")

    @action
    async def plan(self, findings: dict) -> dict:
        """
        Create a research plan based on findings.

        Uses LangGraph for intelligent plan generation.
        """
        self._init_langgraph()
        logger.info(f"[Planner] Creating plan from findings")

        messages = [
            SystemMessage(content="You are a research planner. Create an actionable plan based on the findings."),
            HumanMessage(content=f"Create a research plan based on: {findings['findings']}")
        ]

        try:
            result = self._langgraph_agent.invoke({"messages": messages})
            plan_content = result["messages"][-1].content
        except Exception as e:
            logger.warning(f"[Planner] LangGraph error: {e}, using fallback")
            plan_content = """Research Plan:
1. Screen candidate materials computationally
2. Validate top candidates experimentally
3. Optimize and characterize best performers"""

        plan = {
            "findings": findings,
            "plan": plan_content,
            "status": "ready",
            "source": "langgraph_planner",
        }

        logger.info(f"[Planner] Plan created")
        self._results = plan
        self._complete.set()
        return plan

    @action
    async def get_results(self) -> dict:
        """Wait for planning to complete and return results."""
        await self._complete.wait()
        return self._results


# ============================================================================
# MAIN
# ============================================================================

async def main(task: str, llm_type: str) -> None:
    """
    Run the hybrid Academy + LangGraph example.

    Demonstrates:
    1. Academy agents for distributed coordination
    2. LangGraph inside agents for LLM-powered reasoning
    3. Agent-to-agent communication with intelligent processing
    """

    print("=" * 60)
    print("ACADEMY + LANGGRAPH HYBRID EXAMPLE")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"LLM: {llm_type}")
    print("-" * 60)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:

        # Launch agents
        # In production, these could run on different machines
        researcher = await manager.launch(ResearcherAgent)
        planner = await manager.launch(PlannerAgent)

        logger.info(f"Launched Researcher: {researcher.identifier}")
        logger.info(f"Launched Planner: {planner.identifier}")

        # Configure agents with LLM type
        await researcher.configure(llm_type)
        await planner.configure(llm_type)

        # Connect agents into a pipeline
        await researcher.set_planner(planner)

        # Start the workflow
        # Researcher uses LangGraph to research, then forwards to Planner
        # Planner uses LangGraph to create a plan
        logger.info("Starting hybrid workflow...")
        await researcher.research(task)

        # Get final results
        results = await planner.get_results()

        # Display results
        print("-" * 60)
        print("WORKFLOW RESULTS")
        print("-" * 60)
        print("\n### Research Findings ###")
        print(results["findings"]["findings"][:500])
        print("\n### Research Plan ###")
        print(results["plan"][:500])
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Academy + LangGraph example"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="analyze catalysts for CO2 conversion at room temperature",
        help="Research task to execute",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["mock", "openai", "ollama"],
        default="mock",
        help="LLM backend to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.task, args.llm))
