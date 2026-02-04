"""
HPC Job Submission Agent: Submit and monitor batch jobs on HPC systems.

This example demonstrates federated agent execution where an agent
submits computational jobs to remote HPC resources and monitors their progress.

Supports four LLM modes:
1. OPENAI_API_KEY set -> uses OpenAI
2. FIRST_API_KEY set -> uses FIRST (HPC inference service)
3. OLLAMA_MODEL set -> uses Ollama (local LLM)
4. None of the above -> uses mock responses to demonstrate the pattern

The HPC interaction is simulated for demonstration purposes.
In production, these tools would connect to actual HPC schedulers
(SLURM, PBS, etc.) via SSH or REST APIs.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Data directory
DATA_DIR = Path(__file__).parent / "data"

# Simulated job database (in-memory for demo)
_jobs = {}
_job_counter = 1000


def get_llm():
    """Get the appropriate LLM based on available credentials."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return (
            ChatOpenAI(model="gpt-4o-mini"),
            "OpenAI",
            "OPENAI_API_KEY found in environment",
        )

    if os.environ.get("FIRST_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("FIRST_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        base_url = os.environ.get("FIRST_API_BASE", "https://api.first.example.com/v1")
        return (
            ChatOpenAI(model=model, api_key=os.environ["FIRST_API_KEY"], base_url=base_url),
            "FIRST",
            f"FIRST_API_KEY found in environment (model: {model})",
        )

    if os.environ.get("OLLAMA_MODEL"):
        from langchain_openai import ChatOpenAI
        model = os.environ["OLLAMA_MODEL"]
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        base_url = f"{host}/v1"
        return (
            ChatOpenAI(model=model, api_key="ollama", base_url=base_url),
            "Ollama",
            f"OLLAMA_MODEL found in environment (model: {model})",
        )

    return (
        None,
        "Mock",
        "No API key or OLLAMA_MODEL found; using hardcoded responses",
    )


def print_mode_info(mode: str, reason: str):
    """Print information about the selected LLM mode."""
    print("=" * 60)
    print(f"LLM Mode: {mode}")
    print(f"  Reason: {reason}")
    print("=" * 60)


@tool
def submit_job(script_name: str, nodes: int = 1, walltime_hours: int = 1) -> str:
    """
    Submit a batch job to the HPC cluster.

    Args:
        script_name: Name of the job script (from data/ directory)
        nodes: Number of compute nodes requested
        walltime_hours: Maximum runtime in hours

    Returns:
        Job submission confirmation with job ID
    """
    global _job_counter

    # Check if script exists
    script_path = DATA_DIR / script_name
    if not script_path.exists():
        available = [f.name for f in DATA_DIR.glob("*.sh")]
        return f"Error: Script '{script_name}' not found. Available scripts: {available}"

    # Create job entry
    job_id = _job_counter
    _job_counter += 1

    _jobs[job_id] = {
        "id": job_id,
        "script": script_name,
        "nodes": nodes,
        "walltime_hours": walltime_hours,
        "status": "QUEUED",
        "submit_time": datetime.now().isoformat(),
        "start_time": None,
        "end_time": None,
        "output": None,
    }

    return f"""Job submitted successfully!
- Job ID: {job_id}
- Script: {script_name}
- Nodes: {nodes}
- Walltime: {walltime_hours} hour(s)
- Status: QUEUED
- Queue position: {random.randint(1, 5)}

Use check_job_status({job_id}) to monitor progress."""


@tool
def check_job_status(job_id: int) -> str:
    """
    Check the status of a submitted job.

    Args:
        job_id: The job ID returned from submit_job

    Returns:
        Current job status and details
    """
    if job_id not in _jobs:
        return f"Error: Job {job_id} not found. Use list_jobs() to see your jobs."

    job = _jobs[job_id]

    # Simulate job progression
    if job["status"] == "QUEUED":
        # 70% chance to start running
        if random.random() < 0.7:
            job["status"] = "RUNNING"
            job["start_time"] = datetime.now().isoformat()
    elif job["status"] == "RUNNING":
        # 60% chance to complete
        if random.random() < 0.6:
            job["status"] = "COMPLETED"
            job["end_time"] = datetime.now().isoformat()
            job["output"] = _generate_mock_output(job["script"])

    status_str = f"""Job {job_id} Status:
- Script: {job['script']}
- Status: {job['status']}
- Nodes: {job['nodes']}
- Submitted: {job['submit_time']}"""

    if job["start_time"]:
        status_str += f"\n- Started: {job['start_time']}"
    if job["end_time"]:
        status_str += f"\n- Completed: {job['end_time']}"

    if job["status"] == "QUEUED":
        status_str += f"\n- Estimated wait: {random.randint(1, 10)} minutes"
    elif job["status"] == "RUNNING":
        status_str += f"\n- Progress: ~{random.randint(20, 80)}%"

    return status_str


@tool
def get_job_output(job_id: int) -> str:
    """
    Retrieve the output from a completed job.

    Args:
        job_id: The job ID to get output for

    Returns:
        Job output or error if job not complete
    """
    if job_id not in _jobs:
        return f"Error: Job {job_id} not found."

    job = _jobs[job_id]

    if job["status"] != "COMPLETED":
        return f"Error: Job {job_id} is not complete (status: {job['status']}). Cannot retrieve output yet."

    return f"""Output for Job {job_id}:
================================================================================
{job['output']}
================================================================================
Job completed successfully."""


@tool
def list_jobs() -> str:
    """
    List all jobs submitted in this session.

    Returns:
        Summary of all submitted jobs
    """
    if not _jobs:
        return "No jobs have been submitted yet."

    lines = ["Your submitted jobs:", "-" * 50]
    for job_id, job in _jobs.items():
        lines.append(f"  {job_id}: {job['script']} - {job['status']}")

    return "\n".join(lines)


@tool
def cancel_job(job_id: int) -> str:
    """
    Cancel a queued or running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Confirmation of cancellation
    """
    if job_id not in _jobs:
        return f"Error: Job {job_id} not found."

    job = _jobs[job_id]

    if job["status"] == "COMPLETED":
        return f"Error: Job {job_id} has already completed."

    job["status"] = "CANCELLED"
    job["end_time"] = datetime.now().isoformat()

    return f"Job {job_id} has been cancelled."


def _generate_mock_output(script_name: str) -> str:
    """Generate realistic mock output based on the script type."""
    if "dft" in script_name.lower() or "catalyst" in script_name.lower():
        return """DFT Calculation Results for Cu-catalyst surface
================================================
Method: PBE-D3/def2-SVP
System: Cu(111) surface with CO2 adsorbate

Optimization converged in 45 iterations.

Final Energy: -1847.234521 Hartree
Binding Energy (CO2): -0.82 eV

Electronic Properties:
  HOMO-LUMO gap: 1.23 eV
  Fermi level: -4.56 eV

Mulliken Charges:
  Cu (surface): +0.12
  C (CO2): +0.45
  O (CO2): -0.28

Vibrational Analysis:
  CO2 stretch: 2341 cm^-1 (cf. gas phase: 2349 cm^-1)
  Cu-CO2 bend: 312 cm^-1

Calculation completed in 2h 34m on 4 nodes."""

    elif "md" in script_name.lower() or "simulation" in script_name.lower():
        return """Molecular Dynamics Simulation Complete
======================================
System: 10,000 atoms (Cu nanoparticle in water)
Ensemble: NPT (300K, 1 atm)
Duration: 10 ns

Performance: 45 ns/day on 2 nodes

Final Statistics:
  Temperature: 299.8 ± 2.1 K
  Pressure: 1.01 ± 0.05 atm
  Density: 1.42 g/cm³

RDF Analysis:
  Cu-Cu first peak: 2.56 Å
  Cu-O first peak: 2.21 Å

Trajectory saved to: trajectory.dcd
Energy file: energy.csv"""

    else:
        return """Job Output
==========
Calculation completed successfully.

Results written to: output.dat
Log file: job.log

Total runtime: 45 minutes
Resources used: 1 node, 32 cores
Memory peak: 12.4 GB"""


def run_with_llm(llm, task: str):
    """Run the HPC job agent with a real LLM."""
    from langgraph.prebuilt import create_react_agent

    print(f"\nTask: {task}")
    print("-" * 60)

    # Show available scripts
    scripts = list(DATA_DIR.glob("*.sh"))
    if scripts:
        print("Available job scripts:")
        for s in scripts:
            print(f"  - {s.name}")
        print()

    agent = create_react_agent(
        llm,
        [submit_job, check_job_status, get_job_output, list_jobs, cancel_job]
    )

    for step in agent.stream({"messages": [HumanMessage(content=task)]}):
        if "agent" in step:
            msg = step["agent"]["messages"][0]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"Agent calls: {tc['name']}({tc['args']})")
            if msg.content:
                print(f"\nAgent: {msg.content}")
        elif "tools" in step:
            content = step["tools"]["messages"][0].content
            print(f"Result:\n{content}\n")


def run_mock(task: str):
    """Demonstrate the HPC job pattern with mock responses."""
    print("\nDemonstrating HPC job submission pattern with mock responses.")
    print("Set OPENAI_API_KEY to use a real LLM.\n")

    print(f"Task: {task}")
    print("-" * 60)

    # Show available scripts
    scripts = list(DATA_DIR.glob("*.sh"))
    if scripts:
        print("Available job scripts:")
        for s in scripts:
            print(f"  - {s.name}")
        print()

    # Simulate agent workflow
    print("Agent calls: submit_job({'script_name': 'dft_catalyst.sh', 'nodes': 4, 'walltime_hours': 4})")
    result = submit_job.invoke({"script_name": "dft_catalyst.sh", "nodes": 4, "walltime_hours": 4})
    print(f"Result:\n{result}\n")

    # Get the job ID from the result
    job_id = list(_jobs.keys())[-1] if _jobs else 1000

    print(f"Agent calls: check_job_status({{'job_id': {job_id}}})")
    # Force progression for demo
    if job_id in _jobs:
        _jobs[job_id]["status"] = "RUNNING"
        _jobs[job_id]["start_time"] = datetime.now().isoformat()
    result = check_job_status.invoke({"job_id": job_id})
    print(f"Result:\n{result}\n")

    print(f"Agent calls: check_job_status({{'job_id': {job_id}}})")
    # Force completion for demo
    if job_id in _jobs:
        _jobs[job_id]["status"] = "COMPLETED"
        _jobs[job_id]["end_time"] = datetime.now().isoformat()
        _jobs[job_id]["output"] = _generate_mock_output("dft_catalyst.sh")
    result = check_job_status.invoke({"job_id": job_id})
    print(f"Result:\n{result}\n")

    print(f"Agent calls: get_job_output({{'job_id': {job_id}}})")
    result = get_job_output.invoke({"job_id": job_id})
    print(f"Result:\n{result}\n")

    answer = """The DFT calculation for the Cu-catalyst surface has completed successfully.

Key findings:
- **Binding Energy**: CO2 binds to the Cu(111) surface with -0.82 eV
- **Electronic Gap**: HOMO-LUMO gap of 1.23 eV
- **Vibrational Shift**: CO2 stretch mode shifts from 2349 to 2341 cm⁻¹ upon adsorption

The calculation used 4 nodes and completed in about 2.5 hours. Results suggest
moderate CO2 activation on this copper surface, consistent with experimental observations."""

    print(f"Agent: {answer}")


def main():
    parser = argparse.ArgumentParser(description="HPC Job Submission Agent")
    parser.add_argument(
        "--task",
        default="Submit a DFT calculation for a copper catalyst and monitor it until completion. Report the results.",
        help="Task for the agent to perform",
    )
    args = parser.parse_args()

    llm, mode, reason = get_llm()
    print_mode_info(mode, reason)

    if llm:
        run_with_llm(llm, args.task)
    else:
        run_mock(args.task)


if __name__ == "__main__":
    main()
