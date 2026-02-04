# LLM Configuration

The examples in this directory support multiple LLM backends. You can use OpenAI, FIRST (HPC inference), or run without any API key using mock responses.

## Supported Modes

| Mode | Environment Variable | Description |
|------|---------------------|-------------|
| **OpenAI** | `OPENAI_API_KEY` | Uses OpenAI (gpt-4o-mini by default) |
| **FIRST** | `FIRST_API_KEY` | Uses FIRST HPC inference service |
| **Mock** | (none) | Demonstrates patterns with hardcoded responses |

## Configuration

### OpenAI

```bash
export OPENAI_API_KEY=<your_key>
python main.py
```

### FIRST (HPC Inference)

```bash
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://your-first-endpoint/v1
export FIRST_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
python main.py
```

### Mock Mode (No API Key)

```bash
python main.py
```

Mock mode runs without any API key, showing realistic example outputs to demonstrate the patterns.

## Mode Detection Output

When you run an example, it prints which mode was selected and why:

```
============================================================
LLM Mode: OpenAI (gpt-4o-mini)
  Reason: OPENAI_API_KEY found in environment
============================================================
```

Or in mock mode:

```
============================================================
LLM Mode: Mock
  Reason: No OPENAI_API_KEY or FIRST_API_KEY found; using hardcoded responses
============================================================
```

## Using Argonne's FIRST Service

Argonne National Laboratory provides access to FIRST through the ALCF (Argonne Leadership Computing Facility).

### Getting Access

1. **Get an ALCF account** if you don't have one
2. **Obtain an API token** following the instructions at:
   [docs.alcf.anl.gov/services/inference-endpoints/#api-access](https://docs.alcf.anl.gov/services/inference-endpoints/#api-access)

### Argonne Configuration

```bash
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://inference-api.alcf.anl.gov/resource_server/metis/api/v1
export FIRST_MODEL=gpt-oss-120b
python main.py
```

### Available Models at Argonne

Check the [ALCF documentation](https://docs.alcf.anl.gov/services/inference-endpoints/) for the current list of available models. Common options include:

- `gpt-oss-120b` - Large general-purpose model
- `meta-llama/Meta-Llama-3.1-70B-Instruct` - Llama 3.1 70B
