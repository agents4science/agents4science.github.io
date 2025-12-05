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
