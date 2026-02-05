# Governed Tool Use Agent

An agent demonstrating policy enforcement for safe, auditable tool invocation.

**Code:** [github.com/agents4science/agents4science.github.io/tree/main/Capabilities/governed-tool-use/AgentsGovernedTools](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/governed-tool-use/AgentsGovernedTools)

## What It Does

Demonstrates three key governance patterns:

1. **Budget limits**: Stop agent if compute cost exceeds budget
2. **Rate limiting**: Prevent too many operations per time window
3. **Approval gates**: Require explicit approval for dangerous actions
4. **Audit logging**: Track all policy decisions

This is a key CAF differentiator: scaling tool use is hard because tools have side effects, consume scarce resources, and can cause real-world harm.

## The Code

```python
@tool
@governed_tool(cost=10.00)  # $10 per invocation
def run_gpu_simulation(model_name: str, iterations: int = 1000) -> str:
    """Run a GPU-accelerated simulation. Higher cost."""
    return f"GPU simulation of {model_name} complete"

@tool
@governed_tool(cost=0.00, requires_approval=True)  # Requires human approval
def delete_data(data_id: str) -> str:
    """Delete a dataset. Requires explicit approval due to irreversibility."""
    return f"Dataset {data_id} deleted permanently."
```

The `@governed_tool` decorator wraps tools with policy checks:

```python
def governed_tool(cost: float = 0.0, requires_approval: bool = False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check rate limit
            allowed, msg = check_rate_limit(tool_name)
            if not allowed:
                return f"POLICY VIOLATION: {msg}"

            # Check budget
            allowed, msg = check_budget(cost, tool_name)
            if not allowed:
                return f"POLICY VIOLATION: {msg}"

            # Check approval
            if requires_approval:
                allowed, msg = check_approval(tool_name, action_id)
                if not allowed:
                    return f"APPROVAL REQUIRED: {msg}"

            # Execute tool
            audit_log("tool_execution", tool_name, "ALLOWED", {"cost": cost})
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Running the Example

```bash
cd Capabilities/governed-tool-use/AgentsGovernedTools
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Custom budget:

```bash
python main.py --budget 50.0
```

Custom rate limit:

```bash
python main.py --rate-limit 5
```

## LLM Configuration

Supports OpenAI, FIRST (HPC inference), Ollama (local), or mock mode.

See [LLM Configuration](/Capabilities/local-agents/llm-configuration/) for details.

## Governance Features

### Budget Enforcement

```
[AUDIT] 2024-01-15T10:30:45 | run_gpu_simulation | DENIED |
  {'current': 92.0, 'requested': 10.0, 'limit': 100.0}

POLICY VIOLATION: Budget exceeded. Current: $92.00, Requested: $10.00, Limit: $100.00
```

### Rate Limiting

```
[AUDIT] 2024-01-15T10:30:46 | run_quick_analysis | DENIED |
  {'calls_in_window': 10, 'limit': 10}

POLICY VIOLATION: Rate limit exceeded. 10 calls in last 60s (limit: 10)
```

### Approval Gates

```
[AUDIT] 2024-01-15T10:30:47 | delete_data | PENDING |
  {'action_id': 'delete_data_1705312247', 'message': 'Requires human approval'}

APPROVAL REQUIRED: Action 'delete_data' requires approval. Approval ID: delete_data_1705312247
```

### Audit Trail

Every policy decision is logged:

```
2024-01-15T10:30:42 | run_quick_analysis       | ALLOWED   | {'cost': 0.01}
2024-01-15T10:30:43 | run_cpu_simulation       | ALLOWED   | {'cost': 1.0}
2024-01-15T10:30:44 | run_gpu_simulation       | ALLOWED   | {'cost': 10.0}
2024-01-15T10:30:45 | run_gpu_simulation       | DENIED    | budget exceeded
2024-01-15T10:30:46 | delete_data              | PENDING   | requires approval
```

## Tools

| Tool | Cost | Approval | Description |
|------|------|----------|-------------|
| `run_quick_analysis` | $0.01 | No | Fast dataset analysis |
| `run_cpu_simulation` | $1.00 | No | CPU-based simulation |
| `run_gpu_simulation` | $10.00 | No | GPU-accelerated simulation |
| `submit_hpc_job` | $50.00 | No | Submit HPC batch job |
| `delete_data` | $0.00 | **Yes** | Delete dataset (irreversible) |
| `check_budget_status` | $0.00 | No | Check remaining budget |
| `get_pending_approvals` | $0.00 | No | List pending approvals |
| `grant_approval` | $0.00 | No | Approve pending action |

## Policy Configuration

Policies can be configured in `data/policies.yaml`:

```yaml
budget:
  limit: 100.00
  alert_threshold: 0.80

rate_limiting:
  max_calls: 10
  window_seconds: 60

approval:
  required_for:
    - delete_data
    - submit_production_job

audit:
  log_all_actions: true
```

## Architecture

<img src="/Capabilities/Assets/governed-tools-example.svg" alt="Governance layer with budget check, rate limit, approval gate, and audit log over tool execution layer" style="max-width: 400px; margin: 1rem 0;">

## Production Considerations

This example demonstrates the patterns. Production systems should add:

- **Persistent audit logs**: Write to database or log aggregation service
- **Approval workflows**: Integration with ticketing systems (ServiceNow, Jira)
- **Budget alerts**: Notify operators before limit is reached
- **Per-user limits**: Track budget and rate limits per user/project
- **Cost estimation**: Predict cost before execution, not just track after
- **Rollback support**: For approval-gated actions that fail

## Requirements

- Python 3.10+
- LangGraph 1.0+
- OpenAI API key, FIRST token, Ollama, or run in mock mode
