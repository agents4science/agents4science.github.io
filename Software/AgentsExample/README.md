
# Agents4Science Roles — Dashboard v2 (Color Columns Only)

- Full-screen Rich dashboard with **only colored columns** (no progress bars)
- Per-agent, per-goal live updates; smooth rendering
- Goals in **agents4science/workflows/goals.yaml**
- Adjustable pacing via env vars: A4S_LATENCY, A4S_TOOL_LATENCY
- Enable Argonne inference service LLM access via env var: A4S_USE_INFERENCE=1
- Disable UI via env var: A4S_UI=0

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export A4S_LATENCY=0.4
export A4S_TOOL_LATENCY=0.2
python main.py
```
