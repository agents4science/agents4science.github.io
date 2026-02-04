#!/usr/bin/env python3
"""
Governed Tool Use Agent

Demonstrates policy enforcement for agentic tool use:
- Budget limits: Stop agent if compute cost exceeds budget
- Rate limiting: Prevent too many operations per time window
- Approval gates: Require explicit approval for dangerous actions
- Audit logging: Track all policy decisions

This is a key CAF differentiator: safe, auditable tool use at scale.
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from collections import deque

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
# GOVERNANCE LAYER
# ============================================================================

@dataclass
class Policy:
    """Policy configuration for tool governance."""
    budget_limit: float = 100.0  # Maximum compute cost in dollars
    rate_limit_calls: int = 10  # Max calls per time window
    rate_limit_window_seconds: int = 60  # Time window for rate limiting
    require_approval_for: list = field(default_factory=lambda: ["delete_data", "submit_production_job"])
    log_all_actions: bool = True


@dataclass
class GovernanceState:
    """Tracks governance state across tool invocations."""
    total_cost: float = 0.0
    call_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    pending_approvals: dict = field(default_factory=dict)
    audit_log: list = field(default_factory=list)
    approved_actions: set = field(default_factory=set)


# Global governance state
POLICY = Policy()
STATE = GovernanceState()


def audit_log(action: str, tool_name: str, decision: str, details: dict = None):
    """Log a governance decision for audit purposes."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "tool": tool_name,
        "decision": decision,
        "details": details or {}
    }
    STATE.audit_log.append(entry)
    if POLICY.log_all_actions:
        print(f"[AUDIT] {entry['timestamp']} | {tool_name} | {decision} | {details}")


def check_budget(cost: float, tool_name: str) -> tuple[bool, str]:
    """Check if operation is within budget."""
    if STATE.total_cost + cost > POLICY.budget_limit:
        audit_log("budget_check", tool_name, "DENIED",
                  {"current": STATE.total_cost, "requested": cost, "limit": POLICY.budget_limit})
        return False, f"Budget exceeded. Current: ${STATE.total_cost:.2f}, Requested: ${cost:.2f}, Limit: ${POLICY.budget_limit:.2f}"
    return True, "OK"


def check_rate_limit(tool_name: str) -> tuple[bool, str]:
    """Check if operation is within rate limit."""
    now = datetime.now()
    window_start = now - timedelta(seconds=POLICY.rate_limit_window_seconds)

    # Count calls in window
    recent_calls = sum(1 for ts in STATE.call_timestamps if ts > window_start)

    if recent_calls >= POLICY.rate_limit_calls:
        audit_log("rate_limit_check", tool_name, "DENIED",
                  {"calls_in_window": recent_calls, "limit": POLICY.rate_limit_calls})
        return False, f"Rate limit exceeded. {recent_calls} calls in last {POLICY.rate_limit_window_seconds}s (limit: {POLICY.rate_limit_calls})"

    STATE.call_timestamps.append(now)
    return True, "OK"


def check_approval(tool_name: str, action_id: str) -> tuple[bool, str]:
    """Check if action requiring approval has been approved."""
    if tool_name not in POLICY.require_approval_for:
        return True, "OK"

    if action_id in STATE.approved_actions:
        return True, "Previously approved"

    # In a real system, this would trigger an approval workflow
    audit_log("approval_check", tool_name, "PENDING",
              {"action_id": action_id, "message": "Requires human approval"})
    STATE.pending_approvals[action_id] = {
        "tool": tool_name,
        "requested_at": datetime.now().isoformat(),
        "status": "pending"
    }
    return False, f"Action '{tool_name}' requires approval. Approval ID: {action_id}"


def approve_action(action_id: str) -> str:
    """Approve a pending action (would be called by human operator)."""
    if action_id in STATE.pending_approvals:
        STATE.approved_actions.add(action_id)
        STATE.pending_approvals[action_id]["status"] = "approved"
        audit_log("approval_granted", STATE.pending_approvals[action_id]["tool"], "APPROVED",
                  {"action_id": action_id})
        return f"Action {action_id} approved"
    return f"No pending approval found for {action_id}"


def governed_tool(cost: float = 0.0, requires_approval: bool = False):
    """Decorator that wraps a tool with governance checks."""
    from functools import wraps

    def decorator(func):
        @wraps(func)  # Preserves function signature for @tool decorator
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            action_id = f"{tool_name}_{int(time.time())}"

            # Check rate limit
            allowed, msg = check_rate_limit(tool_name)
            if not allowed:
                return f"POLICY VIOLATION: {msg}"

            # Check budget
            allowed, msg = check_budget(cost, tool_name)
            if not allowed:
                return f"POLICY VIOLATION: {msg}"

            # Check approval if required
            if requires_approval or tool_name in POLICY.require_approval_for:
                allowed, msg = check_approval(tool_name, action_id)
                if not allowed:
                    return f"APPROVAL REQUIRED: {msg}"

            # Execute tool
            audit_log("tool_execution", tool_name, "ALLOWED", {"cost": cost})
            STATE.total_cost += cost

            result = func(*args, **kwargs)

            audit_log("tool_completed", tool_name, "SUCCESS", {"result_preview": str(result)[:100]})
            return result

        return wrapper
    return decorator


# ============================================================================
# GOVERNED TOOLS
# ============================================================================

# Tool cost estimates (in dollars)
COSTS = {
    "quick_analysis": 0.01,
    "cpu_simulation": 1.00,
    "gpu_simulation": 10.00,
    "hpc_job": 50.00,
    "delete_data": 0.00,  # Free but requires approval
}


@tool
@governed_tool(cost=COSTS["quick_analysis"])
def run_quick_analysis(data_id: str) -> str:
    """Run a quick analysis on a dataset. Low cost, no approval needed."""
    return f"Quick analysis complete for {data_id}. Found 3 anomalies, mean=42.5, std=3.2"


@tool
@governed_tool(cost=COSTS["cpu_simulation"])
def run_cpu_simulation(parameters: str) -> str:
    """Run a CPU-based simulation. Moderate cost."""
    return f"CPU simulation complete with parameters: {parameters}. Energy: -1523.4 eV"


@tool
@governed_tool(cost=COSTS["gpu_simulation"])
def run_gpu_simulation(model_name: str, iterations: int = 1000) -> str:
    """Run a GPU-accelerated simulation. Higher cost."""
    return f"GPU simulation of {model_name} complete after {iterations} iterations. Accuracy: 94.2%"


@tool
@governed_tool(cost=COSTS["hpc_job"])
def submit_hpc_job(job_script: str, nodes: int = 1) -> str:
    """Submit a job to HPC cluster. High cost."""
    return f"HPC job submitted: {job_script} on {nodes} nodes. Job ID: HPC-12345"


@tool
@governed_tool(cost=COSTS["delete_data"], requires_approval=True)
def delete_data(data_id: str) -> str:
    """Delete a dataset. Requires explicit approval due to irreversibility."""
    return f"Dataset {data_id} deleted permanently."


@tool
def check_budget_status() -> str:
    """Check current budget usage and remaining budget."""
    remaining = POLICY.budget_limit - STATE.total_cost
    return f"Budget used: ${STATE.total_cost:.2f} / ${POLICY.budget_limit:.2f} (${remaining:.2f} remaining)"


@tool
def get_pending_approvals() -> str:
    """List all actions waiting for approval."""
    if not STATE.pending_approvals:
        return "No pending approvals"
    pending = [f"- {aid}: {info['tool']} (requested {info['requested_at']})"
               for aid, info in STATE.pending_approvals.items()
               if info['status'] == 'pending']
    return "Pending approvals:\n" + "\n".join(pending) if pending else "No pending approvals"


@tool
def grant_approval(action_id: str) -> str:
    """Grant approval for a pending action. (Simulates human approval)"""
    return approve_action(action_id)


# ============================================================================
# AGENT SETUP
# ============================================================================

TOOLS = [
    run_quick_analysis,
    run_cpu_simulation,
    run_gpu_simulation,
    submit_hpc_job,
    delete_data,
    check_budget_status,
    get_pending_approvals,
    grant_approval,
]


def run_agent(task: str):
    """Run the governed agent on a task."""
    llm = get_llm()

    if llm is None:
        print(f"\n{'='*60}")
        print("MOCK MODE - Simulating governed tool execution")
        print(f"{'='*60}")
        print(f"\nTask: {task}")
        print("\n--- Simulated Agent Execution ---\n")

        # Simulate a workflow that demonstrates governance
        print("Agent: I'll run several analyses to complete this task.\n")

        # Quick analyses (should all pass)
        for i in range(3):
            result = run_quick_analysis.invoke({"data_id": f"dataset_{i}"})
            print(f"Tool call: run_quick_analysis(dataset_{i})")
            print(f"Result: {result}\n")

        # CPU simulation
        result = run_cpu_simulation.invoke({"parameters": "temp=300K,pressure=1atm"})
        print(f"Tool call: run_cpu_simulation(temp=300K,pressure=1atm)")
        print(f"Result: {result}\n")

        # GPU simulation
        result = run_gpu_simulation.invoke({"model_name": "catalyst_model", "iterations": 5000})
        print(f"Tool call: run_gpu_simulation(catalyst_model, 5000)")
        print(f"Result: {result}\n")

        # Check budget
        result = check_budget_status.invoke({})
        print(f"Tool call: check_budget_status()")
        print(f"Result: {result}\n")

        # Try to delete data (requires approval)
        result = delete_data.invoke({"data_id": "old_results"})
        print(f"Tool call: delete_data(old_results)")
        print(f"Result: {result}\n")

        # Show pending approvals
        result = get_pending_approvals.invoke({})
        print(f"Tool call: get_pending_approvals()")
        print(f"Result: {result}\n")

        # Simulate hitting budget limit
        print("--- Demonstrating budget enforcement ---\n")
        POLICY.budget_limit = 15.0  # Lower budget for demo
        STATE.total_cost = 12.0  # Already spent most of budget

        result = run_gpu_simulation.invoke({"model_name": "expensive_model", "iterations": 10000})
        print(f"Tool call: run_gpu_simulation(expensive_model, 10000)")
        print(f"Result: {result}\n")

        # Simulate hitting rate limit
        print("--- Demonstrating rate limit enforcement ---\n")
        POLICY.rate_limit_calls = 2
        POLICY.rate_limit_window_seconds = 60
        STATE.call_timestamps.clear()

        for i in range(4):
            result = run_quick_analysis.invoke({"data_id": f"rapid_test_{i}"})
            print(f"Tool call: run_quick_analysis(rapid_test_{i})")
            print(f"Result: {result}\n")

        print("--- Final Audit Log ---\n")
        for entry in STATE.audit_log[-10:]:
            print(f"{entry['timestamp']} | {entry['tool']:25} | {entry['decision']:10} | {entry['details']}")

        return {"status": "mock_completed", "audit_log": STATE.audit_log}

    # Real LLM mode
    print(f"\n{'='*60}")
    print(f"GOVERNED TOOL USE AGENT ({LLM_MODE.upper()} mode)")
    print(f"{'='*60}")
    print(f"Budget limit: ${POLICY.budget_limit:.2f}")
    print(f"Rate limit: {POLICY.rate_limit_calls} calls per {POLICY.rate_limit_window_seconds}s")
    print(f"Requires approval: {POLICY.require_approval_for}")
    print(f"{'='*60}\n")

    agent = create_react_agent(llm, TOOLS)

    result = agent.invoke({
        "messages": [("user", f"""You are a research assistant with access to computational tools.
Your tools are governed by policies:
- Budget limit: ${POLICY.budget_limit:.2f} (check with check_budget_status)
- Some actions require approval (check with get_pending_approvals)
- Rate limits apply to prevent overuse

Task: {task}

Start with quick analyses, then move to more expensive simulations if needed.
Always check your budget status before expensive operations.""")]
    })

    print("\n--- Final Response ---")
    print(result["messages"][-1].content)

    print("\n--- Audit Log Summary ---")
    for entry in STATE.audit_log[-10:]:
        print(f"{entry['timestamp']} | {entry['tool']:25} | {entry['decision']:10}")

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Governed Tool Use Agent")
    parser.add_argument("--task", "-t", default="Analyze datasets 1-3, run a simulation, and report results",
                        help="Task for the agent")
    parser.add_argument("--budget", "-b", type=float, default=100.0,
                        help="Budget limit in dollars")
    parser.add_argument("--rate-limit", "-r", type=int, default=10,
                        help="Max calls per minute")

    args = parser.parse_args()

    POLICY.budget_limit = args.budget
    POLICY.rate_limit_calls = args.rate_limit

    result = run_agent(args.task)
