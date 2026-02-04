#!/usr/bin/env python3
"""
Multi-Agent Coordination with Shared Resources

Demonstrates coordination patterns for multiple agents:
- Shared budget: All agents draw from the same resource pool
- Shared state (blackboard): Agents share findings and avoid duplicate work
- Policy enforcement: Per-agent quotas and rate limits
- Conflict resolution: Handle resource contention gracefully

Run with default 3 agents:
    python main.py

Run with custom agent count:
    python main.py --agents 5 --budget 200
"""

import os
import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

LLM_MODE = None

if os.environ.get("OPENAI_API_KEY"):
    LLM_MODE = "openai"
elif os.environ.get("FIRST_API_KEY"):
    LLM_MODE = "first"
elif os.environ.get("OLLAMA_MODEL"):
    LLM_MODE = "ollama"
else:
    LLM_MODE = "mock"


def get_llm():
    """Get configured LLM based on available credentials."""
    if LLM_MODE == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini")
    elif LLM_MODE == "first":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            base_url="https://api.first.argonne.gov/v1",
            api_key=os.environ["FIRST_API_KEY"]
        )
    elif LLM_MODE == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"))
    else:
        return None


# ============================================================================
# SHARED COORDINATION STATE
# ============================================================================

@dataclass
class SharedBudget:
    """Thread-safe shared budget across all agents."""
    total: float
    spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request(self, amount: float, agent_id: str) -> tuple[bool, str]:
        """Request budget allocation. Returns (success, message)."""
        with self._lock:
            if self.spent + amount > self.total:
                remaining = self.total - self.spent
                return False, f"Insufficient budget. Requested: ${amount:.2f}, Available: ${remaining:.2f}"
            self.spent += amount
            remaining = self.total - self.spent
            return True, f"Allocated ${amount:.2f} to {agent_id}. Remaining: ${remaining:.2f}"

    def get_status(self) -> dict:
        """Get current budget status."""
        with self._lock:
            return {
                "total": self.total,
                "spent": self.spent,
                "remaining": self.total - self.spent,
                "utilization": (self.spent / self.total * 100) if self.total > 0 else 0
            }


@dataclass
class SharedBlackboard:
    """Thread-safe shared state for agent coordination."""
    _data: dict = field(default_factory=dict)
    _claims: dict = field(default_factory=dict)  # Track who's working on what
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def post(self, key: str, value: any, agent_id: str) -> str:
        """Post a finding to the shared blackboard."""
        with self._lock:
            self._data[key] = {
                "value": value,
                "posted_by": agent_id,
                "timestamp": datetime.now().isoformat()
            }
            return f"Posted '{key}' to blackboard"

    def read(self, key: str) -> Optional[dict]:
        """Read a value from the blackboard."""
        with self._lock:
            return self._data.get(key)

    def read_all(self) -> dict:
        """Read all blackboard entries."""
        with self._lock:
            return dict(self._data)

    def claim_task(self, task_id: str, agent_id: str) -> tuple[bool, str]:
        """Claim a task to prevent duplicate work."""
        with self._lock:
            if task_id in self._claims:
                owner = self._claims[task_id]["agent_id"]
                return False, f"Task '{task_id}' already claimed by {owner}"
            self._claims[task_id] = {
                "agent_id": agent_id,
                "claimed_at": datetime.now().isoformat()
            }
            return True, f"Task '{task_id}' claimed by {agent_id}"

    def release_task(self, task_id: str, agent_id: str) -> str:
        """Release a claimed task."""
        with self._lock:
            if task_id in self._claims and self._claims[task_id]["agent_id"] == agent_id:
                del self._claims[task_id]
                return f"Task '{task_id}' released"
            return f"Cannot release task '{task_id}' (not owned by {agent_id})"

    def get_claimed_tasks(self) -> dict:
        """Get all claimed tasks."""
        with self._lock:
            return dict(self._claims)


@dataclass
class AgentQuota:
    """Per-agent resource quotas."""
    max_operations: int = 10
    operations_used: int = 0
    max_budget_per_op: float = 20.0

    def can_operate(self) -> bool:
        return self.operations_used < self.max_operations

    def use_operation(self):
        self.operations_used += 1


@dataclass
class CoordinationHub:
    """Central coordination hub for all agents."""
    budget: SharedBudget
    blackboard: SharedBlackboard
    agent_quotas: dict = field(default_factory=dict)
    event_log: list = field(default_factory=list)
    _log_lock: threading.Lock = field(default_factory=threading.Lock)

    def register_agent(self, agent_id: str, quota: AgentQuota = None):
        """Register an agent with the coordination hub."""
        self.agent_quotas[agent_id] = quota or AgentQuota()
        self.log_event(agent_id, "registered", "Agent registered with coordination hub")

    def log_event(self, agent_id: str, event_type: str, message: str):
        """Log a coordination event."""
        with self._log_lock:
            self.event_log.append({
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
                "event_type": event_type,
                "message": message
            })

    def get_recent_events(self, n: int = 10) -> list:
        """Get recent coordination events."""
        with self._log_lock:
            return self.event_log[-n:]


# Global coordination hub
HUB: Optional[CoordinationHub] = None


# ============================================================================
# COORDINATION TOOLS
# ============================================================================

def make_tools(agent_id: str):
    """Create tools bound to a specific agent."""

    @tool
    def request_budget(amount: float) -> str:
        """Request budget allocation for an operation.

        Args:
            amount: Amount of budget to request (in dollars)
        """
        quota = HUB.agent_quotas.get(agent_id)
        if quota and amount > quota.max_budget_per_op:
            return f"Request exceeds per-operation limit (${quota.max_budget_per_op:.2f})"

        if quota and not quota.can_operate():
            return f"Agent {agent_id} has exhausted operation quota"

        success, message = HUB.budget.request(amount, agent_id)
        HUB.log_event(agent_id, "budget_request", message)

        if success and quota:
            quota.use_operation()

        return message

    @tool
    def post_finding(key: str, value: str) -> str:
        """Share a finding with other agents via the blackboard.

        Args:
            key: Identifier for the finding
            value: The finding to share
        """
        result = HUB.blackboard.post(key, value, agent_id)
        HUB.log_event(agent_id, "post_finding", f"Posted '{key}'")
        return result

    @tool
    def read_findings() -> str:
        """Read all findings from the shared blackboard."""
        findings = HUB.blackboard.read_all()
        if not findings:
            return "No findings on blackboard yet"
        result = "Shared findings:\n"
        for key, data in findings.items():
            result += f"  - {key}: {data['value']} (by {data['posted_by']})\n"
        return result

    @tool
    def claim_task(task_id: str) -> str:
        """Claim a task to prevent other agents from duplicating work.

        Args:
            task_id: Identifier of the task to claim
        """
        success, message = HUB.blackboard.claim_task(task_id, agent_id)
        HUB.log_event(agent_id, "claim_task", message)
        return message

    @tool
    def release_task(task_id: str) -> str:
        """Release a previously claimed task.

        Args:
            task_id: Identifier of the task to release
        """
        result = HUB.blackboard.release_task(task_id, agent_id)
        HUB.log_event(agent_id, "release_task", result)
        return result

    @tool
    def check_budget_status() -> str:
        """Check the shared budget status."""
        status = HUB.budget.get_status()
        return f"Budget: ${status['spent']:.2f} / ${status['total']:.2f} ({status['utilization']:.1f}% used)"

    @tool
    def get_my_quota() -> str:
        """Check this agent's remaining quota."""
        quota = HUB.agent_quotas.get(agent_id)
        if quota:
            remaining = quota.max_operations - quota.operations_used
            return f"Operations: {quota.operations_used}/{quota.max_operations} used, {remaining} remaining"
        return "No quota set for this agent"

    @tool
    def analyze_compound(compound_id: str) -> str:
        """Analyze a chemical compound (simulated)."""
        time.sleep(0.3)  # Simulate work
        return f"Analysis of {compound_id}: LogP=2.3, TPSA=45.2, MW=284.3"

    @tool
    def run_calculation(calc_type: str, parameters: str) -> str:
        """Run a calculation (simulated)."""
        time.sleep(0.3)  # Simulate work
        return f"{calc_type} calculation complete: Energy=-1523.4 eV"

    # Return tools with proper names
    request_budget.__name__ = "request_budget"
    post_finding.__name__ = "post_finding"
    read_findings.__name__ = "read_findings"
    claim_task.__name__ = "claim_task"
    release_task.__name__ = "release_task"
    check_budget_status.__name__ = "check_budget_status"
    get_my_quota.__name__ = "get_my_quota"
    analyze_compound.__name__ = "analyze_compound"
    run_calculation.__name__ = "run_calculation"

    return [
        request_budget,
        post_finding,
        read_findings,
        claim_task,
        release_task,
        check_budget_status,
        get_my_quota,
        analyze_compound,
        run_calculation,
    ]


# ============================================================================
# AGENT EXECUTION
# ============================================================================

def run_agent_task(agent_id: str, task: str, llm):
    """Run a single agent on a task."""
    tools = make_tools(agent_id)

    if llm is None:
        # Mock mode
        return _run_mock_agent(agent_id, task)

    agent = create_react_agent(llm, tools)

    result = agent.invoke({
        "messages": [("user", f"""You are Agent {agent_id} in a multi-agent system.

COORDINATION RULES:
1. Always request_budget BEFORE expensive operations
2. Use claim_task to prevent duplicate work with other agents
3. Post important findings to share with other agents
4. Check read_findings to see what others have discovered
5. Respect your quota limits

YOUR TASK: {task}

Work efficiently and coordinate with other agents.""")]
    })

    return result["messages"][-1].content


def _run_mock_agent(agent_id: str, task: str) -> str:
    """Simulated agent execution for mock mode."""
    print(f"\n[{agent_id}] Starting task: {task}")

    # Simulate coordination
    tools = make_tools(agent_id)

    # Claim task
    claim_result = tools[3].invoke({"task_id": task.split()[0]})
    print(f"[{agent_id}] {claim_result}")

    # Request budget
    budget_result = tools[0].invoke({"amount": 10.0})
    print(f"[{agent_id}] {budget_result}")

    if "Insufficient" in budget_result or "exhausted" in budget_result:
        return f"Agent {agent_id}: Could not complete task - {budget_result}"

    # Do work
    time.sleep(0.5)

    # Post finding
    finding = f"Result from {agent_id}: analysis complete"
    post_result = tools[1].invoke({"key": f"{agent_id}_result", "value": finding})
    print(f"[{agent_id}] {post_result}")

    # Release task
    release_result = tools[4].invoke({"task_id": task.split()[0]})
    print(f"[{agent_id}] {release_result}")

    return f"Agent {agent_id} completed: {task}"


def run_coordinated_agents(num_agents: int, total_budget: float, tasks: list):
    """Run multiple agents with coordination."""
    global HUB

    # Initialize coordination hub
    HUB = CoordinationHub(
        budget=SharedBudget(total=total_budget),
        blackboard=SharedBlackboard()
    )

    # Register agents with quotas
    for i in range(num_agents):
        agent_id = f"Agent_{i+1}"
        HUB.register_agent(agent_id, AgentQuota(
            max_operations=5,
            max_budget_per_op=total_budget / num_agents
        ))

    llm = get_llm()

    print(f"\n{'='*60}")
    print("MULTI-AGENT COORDINATION")
    print(f"{'='*60}")
    print(f"Agents: {num_agents}")
    print(f"Total budget: ${total_budget:.2f}")
    print(f"Tasks: {len(tasks)}")
    print(f"Mode: {LLM_MODE.upper()}")
    print(f"{'='*60}\n")

    results = []

    if LLM_MODE == "mock":
        # Run agents concurrently in mock mode
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                agent_id = f"Agent_{(i % num_agents) + 1}"
                future = executor.submit(_run_mock_agent, agent_id, task)
                futures[future] = (agent_id, task)

            for future in as_completed(futures):
                agent_id, task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"{agent_id} failed: {e}")
    else:
        # Run with real LLM (sequential to avoid rate limits)
        for i, task in enumerate(tasks):
            agent_id = f"Agent_{(i % num_agents) + 1}"
            print(f"\n--- {agent_id} working on: {task} ---")
            result = run_agent_task(agent_id, task, llm)
            results.append(result)
            print(f"\n{agent_id}: {result}")

    # Summary
    print(f"\n{'='*60}")
    print("COORDINATION SUMMARY")
    print(f"{'='*60}")

    # Budget status
    status = HUB.budget.get_status()
    print(f"\nBudget: ${status['spent']:.2f} / ${status['total']:.2f} ({status['utilization']:.1f}% used)")

    # Shared findings
    findings = HUB.blackboard.read_all()
    if findings:
        print(f"\nShared findings ({len(findings)}):")
        for key, data in findings.items():
            print(f"  - {key}: {data['value'][:50]}...")

    # Recent events
    events = HUB.get_recent_events(10)
    if events:
        print(f"\nRecent coordination events:")
        for event in events[-5:]:
            print(f"  [{event['agent_id']}] {event['event_type']}: {event['message'][:50]}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Coordination Demo")
    parser.add_argument("--agents", "-a", type=int, default=3,
                        help="Number of agents")
    parser.add_argument("--budget", "-b", type=float, default=100.0,
                        help="Total shared budget")
    parser.add_argument("--tasks", "-t", type=int, default=6,
                        help="Number of tasks to process")

    args = parser.parse_args()

    # Generate sample tasks
    sample_tasks = [
        "Analyze compound_A for binding affinity",
        "Analyze compound_B for toxicity",
        "Run DFT calculation on catalyst_1",
        "Run MD simulation of protein_X",
        "Analyze compound_C for solubility",
        "Run energy calculation on complex_Y",
        "Analyze compound_D for stability",
        "Run optimization of structure_Z",
    ]

    tasks = sample_tasks[:args.tasks]

    run_coordinated_agents(args.agents, args.budget, tasks)
