#!/usr/bin/env python3
"""
Long-Lived Agent with Checkpoint/Resume

Demonstrates patterns for agents that persist across sessions:
- Automatic checkpointing of agent state
- Resume from checkpoint after restart or crash
- Multi-step workflow recovery
- Goal and progress persistence

Run a new workflow:
    python main.py --goal "Analyze catalyst performance"

Resume from checkpoint:
    python main.py --resume

List available checkpoints:
    python main.py --list
"""

import os
import json
import time
import signal
import atexit
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
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
# CHECKPOINT SYSTEM
# ============================================================================

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A single task in the workflow."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            description=d["description"],
            status=TaskStatus(d["status"]),
            result=d.get("result"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            attempts=d.get("attempts", 0)
        )


@dataclass
class AgentState:
    """Complete agent state for checkpointing."""
    # Identity
    agent_id: str = ""
    created_at: str = ""

    # Goals
    primary_goal: str = ""
    sub_goals: list = field(default_factory=list)

    # Task tracking
    tasks: list = field(default_factory=list)
    current_task_id: Optional[str] = None

    # Progress
    steps_completed: int = 0
    total_steps_estimated: int = 0

    # Memory (key findings, intermediate results)
    memory: dict = field(default_factory=dict)

    # Execution log
    execution_log: list = field(default_factory=list)

    # Checkpoint metadata
    last_checkpoint: str = ""
    checkpoint_count: int = 0

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def add_task(self, task_id: str, description: str) -> Task:
        """Add a new task to the workflow."""
        task = Task(id=task_id, description=description)
        self.tasks.append(task)
        self.total_steps_estimated = len(self.tasks)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_next_task(self) -> Optional[Task]:
        """Get the next pending or in-progress task."""
        # First, check for in-progress tasks (resume interrupted work)
        for task in self.tasks:
            if task.status == TaskStatus.IN_PROGRESS:
                return task
        # Then, get next pending task
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def start_task(self, task_id: str):
        """Mark a task as started."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()
            task.attempts += 1
            self.current_task_id = task_id
            self.log(f"Started task: {task.description}")

    def complete_task(self, task_id: str, result: str):
        """Mark a task as completed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()
            self.steps_completed += 1
            self.current_task_id = None
            self.log(f"Completed task: {task.description}")

    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.result = f"FAILED: {error}"
            self.current_task_id = None
            self.log(f"Failed task: {task.description} - {error}")

    def log(self, message: str):
        """Add to execution log."""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })

    def remember(self, key: str, value: str):
        """Store a finding in memory."""
        self.memory[key] = {
            "value": value,
            "recorded_at": datetime.now().isoformat()
        }
        self.log(f"Remembered: {key}")

    def recall(self, key: str) -> Optional[str]:
        """Recall a finding from memory."""
        if key in self.memory:
            return self.memory[key]["value"]
        return None

    def to_dict(self):
        """Serialize state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "primary_goal": self.primary_goal,
            "sub_goals": self.sub_goals,
            "tasks": [t.to_dict() if isinstance(t, Task) else t for t in self.tasks],
            "current_task_id": self.current_task_id,
            "steps_completed": self.steps_completed,
            "total_steps_estimated": self.total_steps_estimated,
            "memory": self.memory,
            "execution_log": self.execution_log,
            "last_checkpoint": self.last_checkpoint,
            "checkpoint_count": self.checkpoint_count
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize state from dictionary."""
        state = cls(
            agent_id=d["agent_id"],
            created_at=d["created_at"],
            primary_goal=d["primary_goal"],
            sub_goals=d.get("sub_goals", []),
            current_task_id=d.get("current_task_id"),
            steps_completed=d.get("steps_completed", 0),
            total_steps_estimated=d.get("total_steps_estimated", 0),
            memory=d.get("memory", {}),
            execution_log=d.get("execution_log", []),
            last_checkpoint=d.get("last_checkpoint", ""),
            checkpoint_count=d.get("checkpoint_count", 0)
        )
        state.tasks = [Task.from_dict(t) for t in d.get("tasks", [])]
        return state

    def save_checkpoint(self, reason: str = "periodic"):
        """Save current state to checkpoint file."""
        self.last_checkpoint = datetime.now().isoformat()
        self.checkpoint_count += 1
        self.log(f"Checkpoint saved ({reason})")

        checkpoint_file = CHECKPOINT_DIR / f"{self.agent_id}.json"
        checkpoint_file.write_text(json.dumps(self.to_dict(), indent=2))

        # Also save to timestamped backup
        backup_file = CHECKPOINT_DIR / f"{self.agent_id}_{self.checkpoint_count:04d}.json"
        backup_file.write_text(json.dumps(self.to_dict(), indent=2))

        return checkpoint_file

    @classmethod
    def load_checkpoint(cls, agent_id: str) -> Optional["AgentState"]:
        """Load state from checkpoint file."""
        checkpoint_file = CHECKPOINT_DIR / f"{agent_id}.json"
        if not checkpoint_file.exists():
            return None
        data = json.loads(checkpoint_file.read_text())
        return cls.from_dict(data)

    @classmethod
    def get_latest_checkpoint(cls) -> Optional["AgentState"]:
        """Load the most recent checkpoint."""
        checkpoints = list(CHECKPOINT_DIR.glob("agent_*.json"))
        # Filter out numbered backups
        checkpoints = [c for c in checkpoints if not c.stem.split("_")[-1].isdigit()]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text())
        return cls.from_dict(data)

    @classmethod
    def list_checkpoints(cls) -> list:
        """List all available checkpoints."""
        checkpoints = list(CHECKPOINT_DIR.glob("agent_*.json"))
        checkpoints = [c for c in checkpoints if not c.stem.split("_")[-1].isdigit()]
        result = []
        for cp in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(cp.read_text())
            result.append({
                "agent_id": data["agent_id"],
                "goal": data["primary_goal"],
                "progress": f"{data['steps_completed']}/{data['total_steps_estimated']}",
                "last_checkpoint": data.get("last_checkpoint", "unknown"),
                "file": str(cp)
            })
        return result


# Global state
STATE: Optional[AgentState] = None


# ============================================================================
# CHECKPOINT TOOLS
# ============================================================================

@tool
def save_checkpoint(reason: str = "manual") -> str:
    """Save current agent state to checkpoint.

    Args:
        reason: Why checkpoint is being saved (e.g., "before_risky_operation")
    """
    if STATE:
        filepath = STATE.save_checkpoint(reason)
        return f"Checkpoint saved to {filepath}"
    return "No active state to checkpoint"


@tool
def remember_finding(key: str, value: str) -> str:
    """Store an important finding or intermediate result.

    Args:
        key: Identifier for the finding
        value: The finding to remember
    """
    if STATE:
        STATE.remember(key, value)
        return f"Stored: {key} = {value}"
    return "No active state"


@tool
def recall_finding(key: str) -> str:
    """Recall a previously stored finding.

    Args:
        key: Identifier of the finding to recall
    """
    if STATE:
        value = STATE.recall(key)
        if value:
            return f"{key}: {value}"
        return f"No finding stored for '{key}'"
    return "No active state"


@tool
def get_progress() -> str:
    """Get current workflow progress."""
    if STATE:
        completed = [t for t in STATE.tasks if t.status == TaskStatus.COMPLETED]
        pending = [t for t in STATE.tasks if t.status == TaskStatus.PENDING]
        in_progress = [t for t in STATE.tasks if t.status == TaskStatus.IN_PROGRESS]
        failed = [t for t in STATE.tasks if t.status == TaskStatus.FAILED]

        status = f"""Progress: {STATE.steps_completed}/{STATE.total_steps_estimated} tasks
- Completed: {len(completed)}
- In Progress: {len(in_progress)}
- Pending: {len(pending)}
- Failed: {len(failed)}

Goal: {STATE.primary_goal}
Checkpoints saved: {STATE.checkpoint_count}"""
        return status
    return "No active state"


@tool
def mark_task_complete(task_id: str, result: str) -> str:
    """Mark a task as completed with its result.

    Args:
        task_id: ID of the task to complete
        result: Result or output of the task
    """
    if STATE:
        STATE.complete_task(task_id, result)
        STATE.save_checkpoint("task_completed")
        return f"Task {task_id} marked complete"
    return "No active state"


# ============================================================================
# DOMAIN TOOLS (for demonstration)
# ============================================================================

@tool
def analyze_data(dataset_id: str) -> str:
    """Analyze a dataset and return findings."""
    time.sleep(0.5)  # Simulate work
    return f"Analysis of {dataset_id}: Mean=42.3, Std=5.7, Outliers=3"


@tool
def run_simulation(parameters: str) -> str:
    """Run a simulation with given parameters."""
    time.sleep(0.5)  # Simulate work
    return f"Simulation complete: Energy=-1523.4 eV, Converged=True"


@tool
def generate_report(title: str) -> str:
    """Generate a summary report."""
    time.sleep(0.5)  # Simulate work
    return f"Report '{title}' generated with 5 sections, 12 figures"


# ============================================================================
# AGENT EXECUTION
# ============================================================================

TOOLS = [
    save_checkpoint,
    remember_finding,
    recall_finding,
    get_progress,
    mark_task_complete,
    analyze_data,
    run_simulation,
    generate_report,
]


def setup_signal_handlers():
    """Setup handlers for graceful shutdown."""
    def handle_signal(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}, saving checkpoint...")
        if STATE:
            STATE.save_checkpoint("signal_interrupt")
        print("[SIGNAL] Checkpoint saved. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Also save on normal exit
    def save_on_exit():
        if STATE and STATE.get_next_task():
            STATE.save_checkpoint("exit")
            print(f"[EXIT] Checkpoint saved. Resume with: python main.py --resume")

    atexit.register(save_on_exit)


def create_workflow(goal: str) -> AgentState:
    """Create a new workflow with tasks for the given goal."""
    state = AgentState(primary_goal=goal)

    # Create a sample multi-step workflow
    state.add_task("t1_analyze", "Analyze existing data")
    state.add_task("t2_simulate", "Run computational simulation")
    state.add_task("t3_validate", "Validate simulation results")
    state.add_task("t4_report", "Generate final report")

    state.log(f"Created workflow for goal: {goal}")
    state.save_checkpoint("workflow_created")

    return state


def run_agent(state: AgentState):
    """Run the agent workflow with checkpointing."""
    global STATE
    STATE = state

    setup_signal_handlers()

    llm = get_llm()

    print(f"\n{'='*60}")
    print("LONG-LIVED AGENT WITH CHECKPOINT/RESUME")
    print(f"{'='*60}")
    print(f"Agent ID: {state.agent_id}")
    print(f"Goal: {state.primary_goal}")
    print(f"Progress: {state.steps_completed}/{state.total_steps_estimated}")
    print(f"Mode: {LLM_MODE.upper()}")
    print(f"{'='*60}\n")

    if llm is None:
        # Mock mode
        print("Running in MOCK mode\n")
        _run_mock_workflow(state)
        return

    agent = create_react_agent(llm, TOOLS)

    while True:
        task = state.get_next_task()
        if not task:
            print("\n[COMPLETE] All tasks finished!")
            state.log("Workflow completed")
            state.save_checkpoint("workflow_complete")
            break

        print(f"\n--- Task: {task.id} ---")
        print(f"Description: {task.description}")
        print(f"Attempt: {task.attempts + 1}")

        state.start_task(task.id)
        state.save_checkpoint("task_started")

        try:
            result = agent.invoke({
                "messages": [("user", f"""You are working on a long-running scientific workflow.

Goal: {state.primary_goal}
Current task: {task.description}
Task ID: {task.id}

Complete this task using the available tools. When done, use mark_task_complete
with the task_id and a summary of what you accomplished.

If you discover important findings, use remember_finding to store them.
Use save_checkpoint before any risky operations.""")]
            })

            response = result["messages"][-1].content
            print(f"\nAgent: {response}")

        except Exception as e:
            print(f"\n[ERROR] Task failed: {e}")
            state.fail_task(task.id, str(e))
            state.save_checkpoint("task_failed")

    # Final summary
    print("\n--- Workflow Summary ---")
    print(get_progress.invoke({}))

    if state.memory:
        print("\n--- Findings ---")
        for key, data in state.memory.items():
            print(f"  {key}: {data['value']}")


def _run_mock_workflow(state: AgentState):
    """Simulated workflow for mock mode."""
    while True:
        task = state.get_next_task()
        if not task:
            print("\n[COMPLETE] All tasks finished!")
            break

        print(f"\n--- Task: {task.id} ---")
        print(f"Description: {task.description}")

        state.start_task(task.id)
        state.save_checkpoint("task_started")

        # Simulate work
        time.sleep(1)

        # Mock results based on task
        if "analyze" in task.description.lower():
            result = analyze_data.invoke({"dataset_id": "catalyst_data"})
            state.remember("analysis_result", result)
        elif "simulat" in task.description.lower():
            result = run_simulation.invoke({"parameters": "temp=300K"})
            state.remember("simulation_result", result)
        elif "validat" in task.description.lower():
            result = "Validation passed: simulation matches experimental data within 5%"
        elif "report" in task.description.lower():
            result = generate_report.invoke({"title": state.primary_goal})
        else:
            result = f"Completed: {task.description}"

        print(f"Result: {result}")
        state.complete_task(task.id, result)
        state.save_checkpoint("task_completed")

    # Final summary
    print("\n--- Workflow Summary ---")
    print(f"Goal: {state.primary_goal}")
    print(f"Tasks completed: {state.steps_completed}/{state.total_steps_estimated}")
    print(f"Checkpoints saved: {state.checkpoint_count}")

    if state.memory:
        print("\n--- Findings ---")
        for key, data in state.memory.items():
            print(f"  {key}: {data['value']}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Long-Lived Agent with Checkpoint/Resume")
    parser.add_argument("--goal", "-g", type=str, default=None,
                        help="Start new workflow with this goal")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--resume-id", type=str, default=None,
                        help="Resume specific agent by ID")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available checkpoints")

    args = parser.parse_args()

    if args.list:
        checkpoints = AgentState.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
        else:
            print(f"\nAvailable checkpoints ({len(checkpoints)}):\n")
            for cp in checkpoints:
                print(f"  {cp['agent_id']}")
                print(f"    Goal: {cp['goal']}")
                print(f"    Progress: {cp['progress']}")
                print(f"    Last checkpoint: {cp['last_checkpoint']}")
                print()
        exit(0)

    if args.resume or args.resume_id:
        if args.resume_id:
            state = AgentState.load_checkpoint(args.resume_id)
        else:
            state = AgentState.get_latest_checkpoint()

        if state:
            print(f"Resuming agent {state.agent_id}...")
            state.log("Resumed from checkpoint")
            run_agent(state)
        else:
            print("No checkpoint found to resume from.")
            print("Start a new workflow with: python main.py --goal 'Your goal'")
        exit(0)

    if args.goal:
        state = create_workflow(args.goal)
        run_agent(state)
    else:
        # Default demo
        state = create_workflow("Analyze catalyst performance for CO2 reduction")
        run_agent(state)
