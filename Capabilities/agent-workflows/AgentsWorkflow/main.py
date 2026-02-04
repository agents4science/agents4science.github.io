#!/usr/bin/env python3
"""
Agent-Mediated Scientific Workflows

Demonstrates agents that dynamically construct and adapt workflows:
- Dynamic planning: Agent creates workflow steps based on goal
- Adaptive execution: Modify plan based on intermediate results
- Failure recovery: Retry with different parameters on failure
- Workflow patterns: Sequential, parallel, conditional branching

This bridges agentic AI with traditional workflow systems (Parsl, Globus Flows).

Run with a goal:
    python main.py --goal "Optimize catalyst for CO2 reduction"

Run with predefined workflow:
    python main.py --workflow screening
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

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
# WORKFLOW ENGINE
# ============================================================================

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    id: str
    name: str
    task_type: str  # e.g., "compute", "analyze", "decision"
    parameters: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retries": self.retries
        }


@dataclass
class Workflow:
    """Dynamic workflow that can be modified during execution."""
    id: str
    goal: str
    steps: list = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""
    execution_log: list = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.modified_at = self.created_at

    def add_step(self, step: WorkflowStep) -> str:
        """Add a step to the workflow."""
        self.steps.append(step)
        self.modified_at = datetime.now().isoformat()
        self.log(f"Added step: {step.name}")
        return f"Added step '{step.name}' (id: {step.id})"

    def insert_step_after(self, after_id: str, step: WorkflowStep) -> str:
        """Insert a step after an existing step."""
        for i, s in enumerate(self.steps):
            if s.id == after_id:
                self.steps.insert(i + 1, step)
                self.modified_at = datetime.now().isoformat()
                self.log(f"Inserted step '{step.name}' after '{after_id}'")
                return f"Inserted '{step.name}' after '{after_id}'"
        return f"Step '{after_id}' not found"

    def remove_step(self, step_id: str) -> str:
        """Remove a step from the workflow."""
        for i, s in enumerate(self.steps):
            if s.id == step_id:
                removed = self.steps.pop(i)
                self.modified_at = datetime.now().isoformat()
                self.log(f"Removed step: {removed.name}")
                return f"Removed step '{removed.name}'"
        return f"Step '{step_id}' not found"

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list:
        """Get steps that are ready to execute (dependencies satisfied)."""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if all(dep in completed_ids for dep in step.depends_on):
                    ready.append(step)
        return ready

    def get_status(self) -> dict:
        """Get workflow execution status."""
        status_counts = {}
        for step in self.steps:
            status_counts[step.status.value] = status_counts.get(step.status.value, 0) + 1
        return {
            "total_steps": len(self.steps),
            "status": status_counts,
            "is_complete": all(s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED] for s in self.steps),
            "has_failures": any(s.status == StepStatus.FAILED for s in self.steps)
        }

    def log(self, message: str):
        """Add to execution log."""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "execution_log": self.execution_log
        }


# Global workflow
WORKFLOW: Optional[Workflow] = None


# ============================================================================
# TASK EXECUTORS (Simulated Parsl-like tasks)
# ============================================================================

def execute_compute_task(step: WorkflowStep) -> str:
    """Execute a compute task (DFT, MD, etc.)."""
    time.sleep(0.5)  # Simulate computation
    calc_type = step.parameters.get("calc_type", "energy")
    system = step.parameters.get("system", "unknown")

    # Simulate occasional failures for retry demonstration
    if step.retries == 0 and "unstable" in system.lower():
        raise Exception("Calculation did not converge")

    return f"{calc_type} calculation complete for {system}: Energy=-1523.4 eV, Converged=True"


def execute_analyze_task(step: WorkflowStep) -> str:
    """Execute an analysis task."""
    time.sleep(0.3)
    analysis_type = step.parameters.get("analysis_type", "basic")
    data_source = step.parameters.get("data_source", "previous_step")
    return f"{analysis_type} analysis complete: Found 3 candidates, top score=0.95"


def execute_decision_task(step: WorkflowStep) -> str:
    """Execute a decision/branching task."""
    condition = step.parameters.get("condition", "default")
    threshold = step.parameters.get("threshold", 0.5)
    # Simulate decision based on "previous results"
    score = 0.72  # Mock score from previous step
    if score > threshold:
        return f"Decision: PASS (score {score} > threshold {threshold}). Continue to next phase."
    else:
        return f"Decision: FAIL (score {score} <= threshold {threshold}). Need refinement."


def execute_data_task(step: WorkflowStep) -> str:
    """Execute a data retrieval/preparation task."""
    time.sleep(0.2)
    source = step.parameters.get("source", "database")
    query = step.parameters.get("query", "")
    return f"Retrieved 150 records from {source} matching '{query}'"


TASK_EXECUTORS = {
    "compute": execute_compute_task,
    "analyze": execute_analyze_task,
    "decision": execute_decision_task,
    "data": execute_data_task,
}


def execute_step(step: WorkflowStep) -> tuple[bool, str]:
    """Execute a workflow step."""
    executor = TASK_EXECUTORS.get(step.task_type)
    if not executor:
        return False, f"Unknown task type: {step.task_type}"

    try:
        step.status = StepStatus.RUNNING
        result = executor(step)
        step.status = StepStatus.COMPLETED
        step.result = result
        return True, result
    except Exception as e:
        step.error = str(e)
        if step.retries < step.max_retries:
            step.retries += 1
            step.status = StepStatus.PENDING
            return False, f"Failed (attempt {step.retries}/{step.max_retries}): {e}. Will retry."
        else:
            step.status = StepStatus.FAILED
            return False, f"Failed after {step.max_retries} retries: {e}"


# ============================================================================
# WORKFLOW TOOLS
# ============================================================================

@tool
def create_workflow_step(
    step_id: str,
    name: str,
    task_type: str,
    parameters: str = "{}",
    depends_on: str = "[]"
) -> str:
    """Create and add a new step to the workflow.

    Args:
        step_id: Unique identifier for the step
        name: Human-readable name
        task_type: Type of task (compute, analyze, decision, data)
        parameters: JSON string of parameters for the task
        depends_on: JSON array of step IDs this depends on
    """
    try:
        params = json.loads(parameters) if parameters else {}
        deps = json.loads(depends_on) if depends_on else []
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    step = WorkflowStep(
        id=step_id,
        name=name,
        task_type=task_type,
        parameters=params,
        depends_on=deps
    )
    return WORKFLOW.add_step(step)


@tool
def insert_step_after(after_step_id: str, step_id: str, name: str, task_type: str, parameters: str = "{}") -> str:
    """Insert a new step after an existing step (for adaptive workflows).

    Args:
        after_step_id: ID of the step to insert after
        step_id: ID for the new step
        name: Name of the new step
        task_type: Type of task
        parameters: JSON parameters
    """
    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    step = WorkflowStep(
        id=step_id,
        name=name,
        task_type=task_type,
        parameters=params,
        depends_on=[after_step_id]
    )
    return WORKFLOW.insert_step_after(after_step_id, step)


@tool
def remove_step(step_id: str) -> str:
    """Remove a step from the workflow.

    Args:
        step_id: ID of the step to remove
    """
    return WORKFLOW.remove_step(step_id)


@tool
def execute_next_step() -> str:
    """Execute the next ready step in the workflow."""
    ready = WORKFLOW.get_ready_steps()
    if not ready:
        status = WORKFLOW.get_status()
        if status["is_complete"]:
            return "Workflow complete! All steps finished."
        elif status["has_failures"]:
            return "Workflow blocked: Some steps failed."
        else:
            return "No steps ready (waiting on dependencies)"

    step = ready[0]
    WORKFLOW.log(f"Executing: {step.name}")
    success, result = execute_step(step)
    WORKFLOW.log(f"Result: {result}")
    return f"[{step.name}] {result}"


@tool
def execute_all_ready_steps() -> str:
    """Execute all steps that are ready (parallel execution)."""
    ready = WORKFLOW.get_ready_steps()
    if not ready:
        return "No steps ready to execute"

    results = []
    for step in ready:
        WORKFLOW.log(f"Executing: {step.name}")
        success, result = execute_step(step)
        WORKFLOW.log(f"Result: {result}")
        results.append(f"[{step.name}] {result}")

    return "\n".join(results)


@tool
def get_workflow_status() -> str:
    """Get current workflow status and progress."""
    status = WORKFLOW.get_status()
    steps_summary = []
    for step in WORKFLOW.steps:
        icon = {"pending": "○", "running": "◐", "completed": "●", "failed": "✗", "skipped": "○"}
        steps_summary.append(f"  {icon.get(step.status.value, '?')} {step.id}: {step.name} [{step.status.value}]")

    return f"""Workflow: {WORKFLOW.goal}
Steps: {status['total_steps']}
Status: {json.dumps(status['status'])}
Complete: {status['is_complete']}

Steps:
""" + "\n".join(steps_summary)


@tool
def get_step_result(step_id: str) -> str:
    """Get the result of a completed step.

    Args:
        step_id: ID of the step
    """
    step = WORKFLOW.get_step(step_id)
    if not step:
        return f"Step '{step_id}' not found"
    if step.status != StepStatus.COMPLETED:
        return f"Step '{step_id}' not completed (status: {step.status.value})"
    return f"[{step.name}] {step.result}"


@tool
def modify_step_parameters(step_id: str, new_parameters: str) -> str:
    """Modify parameters of a pending step (for adaptive execution).

    Args:
        step_id: ID of the step to modify
        new_parameters: JSON string of new parameters
    """
    step = WORKFLOW.get_step(step_id)
    if not step:
        return f"Step '{step_id}' not found"
    if step.status != StepStatus.PENDING:
        return f"Cannot modify step '{step_id}' (status: {step.status.value})"

    try:
        params = json.loads(new_parameters)
        step.parameters.update(params)
        WORKFLOW.log(f"Modified parameters for {step.name}")
        return f"Updated parameters for '{step.name}'"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


# ============================================================================
# PREDEFINED WORKFLOW TEMPLATES
# ============================================================================

def create_screening_workflow() -> Workflow:
    """Create a compound screening workflow."""
    wf = Workflow(id="screening_001", goal="Screen compounds for catalyst activity")

    wf.add_step(WorkflowStep(
        id="data_fetch", name="Fetch compound library",
        task_type="data",
        parameters={"source": "pubchem", "query": "copper complexes"}
    ))

    wf.add_step(WorkflowStep(
        id="filter", name="Filter by properties",
        task_type="analyze",
        parameters={"analysis_type": "property_filter", "criteria": "MW<500,LogP<3"},
        depends_on=["data_fetch"]
    ))

    wf.add_step(WorkflowStep(
        id="compute_batch", name="Compute binding energies",
        task_type="compute",
        parameters={"calc_type": "binding_energy", "system": "Cu-complex-batch"},
        depends_on=["filter"]
    ))

    wf.add_step(WorkflowStep(
        id="rank", name="Rank candidates",
        task_type="analyze",
        parameters={"analysis_type": "ranking", "metric": "binding_energy"},
        depends_on=["compute_batch"]
    ))

    wf.add_step(WorkflowStep(
        id="decide", name="Select top candidates",
        task_type="decision",
        parameters={"condition": "top_n", "threshold": 0.8},
        depends_on=["rank"]
    ))

    return wf


def create_optimization_workflow() -> Workflow:
    """Create a structure optimization workflow."""
    wf = Workflow(id="optimize_001", goal="Optimize catalyst structure")

    wf.add_step(WorkflowStep(
        id="init_struct", name="Generate initial structure",
        task_type="data",
        parameters={"source": "builder", "template": "Cu-slab"}
    ))

    wf.add_step(WorkflowStep(
        id="opt_geom", name="Optimize geometry",
        task_type="compute",
        parameters={"calc_type": "geometry_optimization", "system": "Cu-slab"},
        depends_on=["init_struct"]
    ))

    wf.add_step(WorkflowStep(
        id="calc_energy", name="Calculate final energy",
        task_type="compute",
        parameters={"calc_type": "single_point_energy", "system": "Cu-slab-optimized"},
        depends_on=["opt_geom"]
    ))

    wf.add_step(WorkflowStep(
        id="analyze", name="Analyze results",
        task_type="analyze",
        parameters={"analysis_type": "convergence_check"},
        depends_on=["calc_energy"]
    ))

    return wf


WORKFLOW_TEMPLATES = {
    "screening": create_screening_workflow,
    "optimization": create_optimization_workflow,
}


# ============================================================================
# AGENT EXECUTION
# ============================================================================

TOOLS = [
    create_workflow_step,
    insert_step_after,
    remove_step,
    execute_next_step,
    execute_all_ready_steps,
    get_workflow_status,
    get_step_result,
    modify_step_parameters,
]


def run_workflow_agent(goal: str, template: str = None):
    """Run the workflow agent."""
    global WORKFLOW

    if template and template in WORKFLOW_TEMPLATES:
        WORKFLOW = WORKFLOW_TEMPLATES[template]()
        print(f"Loaded template workflow: {template}")
    else:
        WORKFLOW = Workflow(
            id=f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            goal=goal
        )

    llm = get_llm()

    print(f"\n{'='*60}")
    print("AGENT-MEDIATED WORKFLOW")
    print(f"{'='*60}")
    print(f"Goal: {WORKFLOW.goal}")
    print(f"Mode: {LLM_MODE.upper()}")
    print(f"{'='*60}\n")

    if llm is None:
        print("Running in MOCK mode\n")
        _run_mock_workflow()
        return

    agent = create_react_agent(llm, TOOLS)

    if template:
        # Execute predefined workflow
        prompt = f"""You are executing a predefined scientific workflow.

Goal: {WORKFLOW.goal}

The workflow has been pre-configured with steps. Your job is to:
1. Check the workflow status
2. Execute steps in order (respecting dependencies)
3. If a step fails, decide whether to retry, modify parameters, or skip
4. Report final results

Start by checking the workflow status, then execute steps."""
    else:
        # Build workflow dynamically
        prompt = f"""You are a scientific workflow planner and executor.

Goal: {goal}

Your job is to:
1. Design a workflow to achieve this goal
2. Create steps using create_workflow_step (types: compute, analyze, decision, data)
3. Execute steps and adapt based on results
4. If intermediate results suggest changes, modify the workflow

Available task types:
- data: Fetch/prepare data (parameters: source, query)
- compute: Run calculations (parameters: calc_type, system)
- analyze: Analyze results (parameters: analysis_type)
- decision: Make decisions (parameters: condition, threshold)

Start by creating 3-5 workflow steps, then execute them."""

    result = agent.invoke({"messages": [("user", prompt)]})
    print(f"\n{result['messages'][-1].content}")

    # Print final workflow state
    print(f"\n{'='*60}")
    print("FINAL WORKFLOW STATE")
    print(f"{'='*60}")
    print(get_workflow_status.invoke({}))

    if WORKFLOW.execution_log:
        print(f"\nExecution log ({len(WORKFLOW.execution_log)} entries):")
        for entry in WORKFLOW.execution_log[-10:]:
            print(f"  {entry['timestamp']}: {entry['message']}")


def _run_mock_workflow():
    """Run workflow in mock mode."""
    # If no steps, create a sample workflow
    if not WORKFLOW.steps:
        print("Creating sample workflow...\n")
        WORKFLOW.add_step(WorkflowStep(
            id="step1", name="Fetch catalyst data",
            task_type="data",
            parameters={"source": "database", "query": "Cu catalysts"}
        ))
        WORKFLOW.add_step(WorkflowStep(
            id="step2", name="Run DFT calculations",
            task_type="compute",
            parameters={"calc_type": "DFT", "system": "Cu-surface"},
            depends_on=["step1"]
        ))
        WORKFLOW.add_step(WorkflowStep(
            id="step3", name="Analyze binding energies",
            task_type="analyze",
            parameters={"analysis_type": "binding_analysis"},
            depends_on=["step2"]
        ))
        WORKFLOW.add_step(WorkflowStep(
            id="step4", name="Select best candidates",
            task_type="decision",
            parameters={"condition": "energy_threshold", "threshold": -1.5},
            depends_on=["step3"]
        ))

    print("Workflow steps:")
    print(get_workflow_status.invoke({}))

    print("\n--- Executing Workflow ---\n")

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1
        status = WORKFLOW.get_status()

        if status["is_complete"]:
            print("\nWorkflow complete!")
            break

        ready = WORKFLOW.get_ready_steps()
        if not ready:
            if status["has_failures"]:
                print("\nWorkflow blocked due to failures")
                break
            print("Waiting for dependencies...")
            time.sleep(0.5)
            continue

        for step in ready:
            print(f"Executing: {step.name}...")
            success, result = execute_step(step)
            print(f"  Result: {result}")

            # Demonstrate adaptive workflow: if step failed, insert retry with different params
            if not success and "retry" in result.lower():
                print(f"  Will retry with modified parameters...")

    # Final status
    print(f"\n{'='*60}")
    print("FINAL WORKFLOW STATE")
    print(f"{'='*60}")
    print(get_workflow_status.invoke({}))

    # Show results
    print("\nStep results:")
    for step in WORKFLOW.steps:
        if step.result:
            print(f"  [{step.name}] {step.result}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent-Mediated Scientific Workflows")
    parser.add_argument("--goal", "-g", type=str, default=None,
                        help="Goal for dynamic workflow creation")
    parser.add_argument("--workflow", "-w", type=str, choices=["screening", "optimization"],
                        help="Use predefined workflow template")
    parser.add_argument("--list-templates", action="store_true",
                        help="List available workflow templates")

    args = parser.parse_args()

    if args.list_templates:
        print("Available workflow templates:")
        for name in WORKFLOW_TEMPLATES:
            wf = WORKFLOW_TEMPLATES[name]()
            print(f"  {name}: {wf.goal} ({len(wf.steps)} steps)")
        exit(0)

    if args.workflow:
        run_workflow_agent(goal="", template=args.workflow)
    elif args.goal:
        run_workflow_agent(goal=args.goal)
    else:
        run_workflow_agent(goal="Optimize copper catalyst for CO2 reduction")
