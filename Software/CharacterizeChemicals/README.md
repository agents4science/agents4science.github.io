# Notes

Install and run as described below. Defaults will characterize three simple molecules with a simple LLM. The simple LLM sometimes fails to generate valid JSON.

The code has little error handling.

# To get started

```
conda create -n chem-agent python=3.11 rdkit xtb -c conda-forge
conda activate chem-agent

pip install \
  torch \
  transformers \
  accelerate \
  academy-py \
  hugging

python run_chem_agent_phi.py -h
```
