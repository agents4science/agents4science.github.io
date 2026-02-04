# Using Argonne's FIRST Inference Service

The examples in this directory support [FIRST](https://www.alcf.anl.gov/alcf-resources/first), Argonne's HPC inference service. This page explains how to configure access.

## Getting Access

1. **Get an ALCF account** if you don't have one
2. **Obtain an API token** following the instructions at:
   [docs.alcf.anl.gov/services/inference-endpoints/#api-access](https://docs.alcf.anl.gov/services/inference-endpoints/#api-access)

## Configuration

Set these environment variables:

```bash
export FIRST_API_KEY=<your_token>
export FIRST_API_BASE=https://inference-api.alcf.anl.gov/resource_server/metis/api/v1
export FIRST_MODEL=gpt-oss-120b
```

Then run any of the examples:

```bash
python main.py
```

## Available Models

Check the [ALCF documentation](https://docs.alcf.anl.gov/services/inference-endpoints/) for the current list of available models. Common options include:

- `gpt-oss-120b` - Large general-purpose model
- `meta-llama/Meta-Llama-3.1-70B-Instruct` - Llama 3.1 70B

## Verification

When configured correctly, the examples will show:

```
============================================================
LLM Mode: FIRST (gpt-oss-120b)
  Reason: FIRST_API_KEY found in environment (model: gpt-oss-120b)
============================================================
```
