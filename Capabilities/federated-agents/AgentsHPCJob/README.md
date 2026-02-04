# HPC Job Submission Agent

An agent that submits and monitors batch jobs on HPC systems.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/federated-agents/AgentsHPCJob](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/federated-agents/AgentsHPCJob)

## What It Does

1. User requests a computation (e.g., "run a DFT calculation")
2. Agent submits a batch job to the HPC scheduler
3. Agent monitors job status until completion
4. Agent retrieves and interprets the results

This demonstrates **federated agent execution** — an agent running locally (or on one system) that orchestrates computations on remote HPC resources.

## The Code

```python
@tool
def submit_job(script_name: str, nodes: int = 1, walltime_hours: int = 1) -> str:
    """Submit a batch job to the HPC cluster."""
    # In production: SSH to cluster or call scheduler API
    return f"Job {job_id} submitted to queue"

@tool
def check_job_status(job_id: int) -> str:
    """Check the status of a submitted job."""
    # In production: Query SLURM/PBS scheduler
    return f"Job {job_id}: RUNNING"

@tool
def get_job_output(job_id: int) -> str:
    """Retrieve output from a completed job."""
    # In production: Fetch stdout/stderr from cluster
    return "DFT calculation results..."

agent = create_react_agent(llm, [submit_job, check_job_status, get_job_output])
```

## Running the Example

```bash
cd Capabilities/federated-agents/AgentsHPCJob
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom task:

```bash
python main.py --task "Run an MD simulation and report the results"
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details on configuring LLM backends, including Argonne's FIRST service.

## Sample Job Scripts

The `data/` directory contains example SLURM job scripts:

```
data/
├── dft_catalyst.sh    # DFT calculation for Cu-catalyst CO2 adsorption
└── md_simulation.sh   # MD simulation of Cu nanoparticle in water
```

## Tools

| Tool | Description |
|------|-------------|
| `submit_job` | Submit a batch job to the cluster |
| `check_job_status` | Check if job is QUEUED, RUNNING, or COMPLETED |
| `get_job_output` | Retrieve stdout/results from completed job |
| `list_jobs` | List all submitted jobs |
| `cancel_job` | Cancel a queued or running job |

## Production Integration

This example simulates HPC interaction. In production, the tools would connect to real schedulers:

**SLURM:**
```python
def submit_job(script):
    result = subprocess.run(["sbatch", script], capture_output=True)
    job_id = parse_job_id(result.stdout)
    return job_id

def check_status(job_id):
    result = subprocess.run(["squeue", "-j", str(job_id)], capture_output=True)
    return parse_status(result.stdout)
```

**REST API (e.g., via Globus Compute):**
```python
def submit_job(script):
    response = requests.post(f"{HPC_API}/jobs", json={"script": script})
    return response.json()["job_id"]
```

## Key Points

- **Job lifecycle**: Submit → Queue → Run → Complete → Retrieve
- **Asynchronous**: Jobs run independently; agent polls for status
- **Federated**: Agent and HPC system are separate; connected via scheduler

## Requirements

- Python 3.10+
- LangGraph 1.0+
- OpenAI API key, FIRST token, Ollama, or run in mock mode
