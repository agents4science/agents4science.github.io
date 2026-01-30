# Local Agentic Applications

Run agentic applications on a laptop or workstation, with LLM access via API or a locally-hosted model.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Implement and run an agentic application locally |
| **Approach** | LangGraph (or similar framework) with OpenAI API, Anthropic API, or local LLM (Ollama, vLLM) |
| **Status** | <span style="color: green;">**Mature**</span> — Documented by examples on this site |
| **Scale** | Single machine |

---

## When to Use

- Rapid prototyping and development
- Applications where data can remain local
- Testing agent logic before deploying to HPC
- Scenarios with modest inference demands

---

## Architecture

```
+─────────────────────────────────────────+
│           Your Laptop/Workstation       │
│  +─────────+    +─────────+             │
│  │  Agent  │───>│  Tools  │             │
│  │ (LangGraph)  │ (local) │             │
│  +────┬────+    +─────────+             │
│       │                                 │
│       v                                 │
│  +─────────+                            │
│  │   LLM   │  (API or local)            │
│  +─────────+                            │
+─────────────────────────────────────────+
```

---

## Getting Started

### Option 1: LLM via API

```bash
cd Frameworks/AgentsLangChain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your_key>
python main_langchain.py
```

### Option 2: Local LLM

```bash
# Start Ollama with a model
ollama run llama3

# Point your agent to local endpoint
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=unused
python main_langchain.py
```

---

## Examples on This Site

- [**AgentsLangChain**](/Frameworks/AgentsLangChain/) — 5-agent pipeline for catalyst discovery
- [**AgentsExample**](/Frameworks/AgentsExample/) — Dashboard demo with configurable goals
- [**CharacterizeChemicals**](/Frameworks/CharacterizeChemicals/) — Molecular property agent with local Phi-3.5

---

## Next Steps

Once your agent works locally, consider:
- [**Agents with HPC Tools**](hpc-tools.md) — Invoke simulations on DOE systems
- [**Massively Parallel Inference**](parallel-inference.md) — Scale to thousands of LLM instances
