# CharacterizeChemicals  
### An LLM-Planned, Tool-Executing Molecular Property Agent

This project demonstrates a full **LLM-planned computational chemistry agent** built on the **Academy** agent framework.  
Given one or more **SMILES strings**, the agent:

1. Uses a **local LLM** (default: `microsoft/Phi-3.5-mini-instruct`) to plan a multi-step workflow.  
2. Executes real computational tools (**RDKit**, **xTB**, **xTB GBSA**) according to that plan.  
3. Parses quantum-chemical outputs (energies, dipole, HOMO–LUMO gap, solvation).  
4. Returns a structured **properties dictionary** per molecule.  
5. Logs the entire plan and every step executed for transparency and debugging.

Main entry point:

```
python run_chem_agent.py
```

**Source code:** [View on GitHub](https://github.com/agents4science/agents4science.github.io/tree/main/Capabilities/federated-agents/CharacterizeChemicals)

---

## ✨ Features

### 🔬 RDKit descriptors
- `logP`
- `TPSA`
- Internal descriptors (e.g., exact MolWt)
- 3D geometry generation via RDKit embedding

### ⚛️ xTB (GFN2-xTB) quantum properties
Extracted from real xTB output:

- `E_scc_hartree`, `E_scc_eV`  
- `E_total_hartree`, `E_total_eV`  
- `HOMO_LUMO_gap_eV`  
- `dipole_moment_D` (from the “molecular dipole” `full:` line)

### 🌊 xTB GBSA solvation
- `solvation_free_energy_kcal_per_mol`  
  Parsed from the `Gsolv` term in the SUMMARY block.

### 📋 Transparency and debugging
- Raw LLM-generated JSON plan printed before execution  
- Parsed execution DAG printed  
- Per-step logging of inputs and outputs  
- xTB command lines logged  
- Plan validation + stalled-plan detection  
- Final aggregated properties for each molecule  

---

## 🔧 Installation

The easiest way to install RDKit and xTB is with **conda**.

### 1. Create conda env

```
conda create -n chem-agent python=3.11 rdkit xtb -c conda-forge
conda activate chem-agent
```

### 2. Install Python dependencies

```
pip install torch transformers accelerate academy-py huggingface_hub
```

### 3. Verify xTB installation

```
xtb --version
```

If this fails, your agent will not be able to run xTB-dependent steps.

---

## 🚀 Usage

Run with default settings:

```
python run_chem_agent.py
```

### Command-line arguments

```
python run_chem_agent.py     [--model MODEL]     [--smiles SMILES ...]     [--props PROPERTIES ...]     [--accuracy-profile {fast,balanced,high}]
```

---

## 🔣 Arguments

### `--model`, `-m`
Hugging Face model ID for the **planner LLM**. Default:

```
microsoft/Phi-3.5-mini-instruct
```

### `--smiles`, `-s`
One or more SMILES strings. Default:

```
CCO  c1ccccc1  CC(=O)O
```

### `--props`, `-p`
Desired properties (used as hints for the planner). Default:

```
logP dipole_moment solvation_free_energy
```

### `--accuracy-profile`, `-a`
Planner behavior hint:  
`fast | balanced | high`  
Default: `balanced`

---

## 🧪 Examples

Default run:

```
python run_chem_agent.py
```

Single molecule:

```
python run_chem_agent.py --smiles "CCO"
```

Custom planner model:

```
python run_chem_agent.py --model Qwen/Qwen2.5-7B-Instruct
```

Multiple molecules:

```
python run_chem_agent.py -s CCO "c1ccccc1" "CC(=O)O"
```

---

## 🧱 Architecture Overview

### 1. Planner LLM → JSON Plan

The LLM receives:
- tool catalog  
- SMILES string  
- requested properties  
- accuracy profile  

It outputs a structured plan like:

```json
{
  "steps": [
    {
      "id": "s1_rdkit",
      "tool": "rdkit_descriptors",
      "inputs": { "smiles": "CCO", "descriptor_set": ["logP", "TPSA"] },
      "depends_on": []
    },
    {
      "id": "s2_xtb",
      "tool": "xtb_opt",
      "inputs": { "smiles": "CCO", "level": "GFN2-xTB" },
      "depends_on": ["s1_rdkit"]
    },
    {
      "id": "s3_solv",
      "tool": "solvation_energy_from_xtb",
      "inputs": {
        "geometry_path": "step:s2_xtb.optimized_geometry",
        "solvent": "water"
      },
      "depends_on": ["s2_xtb"]
    }
  ]
}
```

The agent prints the raw plan and parsed version.

---

### 2. Execution DAG

The executor:

- resolves references like  
  `"step:s2_xtb.optimized_geometry"`
- executes steps in dependency order  
- logs before/after each step  
- aborts on dependency cycles or stalled execution  

---

### 3. Tool Layer

#### RDKit
- SMILES → molecule  
- descriptor calculation  
- 3D embedding for xTB

#### xTB optimization

```
xtb structure.xyz --opt --gfn 2
```

Parsers extract:
- SCC energy  
- total energy  
- HOMO-LUMO gap  
- dipole moment  

#### xTB GBSA solvation

```
xtb structure.xyz --gfn 2 --gbsa water
```

Parses:
- Gsolv (Eh → kcal/mol)

---

## 📊 Property Aggregation

Final results include:

- `logP`
- `dipole_moment_D`
- `solvation_free_energy_kcal_per_mol`
- `E_total_hartree`, `E_total_eV`
- `E_scc_hartree`, `E_scc_eV`
- `HOMO_LUMO_gap_eV`

Returned as:

```
{
  "status": "success",
  "molecule_smiles": "...",
  "properties": {...},
  "plan_used": {...},
  "provenance": {...}
}
```

---

## 🐞 Logging & Debugging

You will see:

- LLM planner call timing  
- Raw JSON plan and parsed DAG  
- Inputs/outputs for every tool step  
- All xTB command lines  
- Parsing warnings (missing fields, regex mismatches)  
- Final property summary  

This makes it easy to identify:
- bad plans,  
- geometry failures,  
- xTB crashes,  
- solver issues.

---

## ⚠️ Caveats

- Planner LLM can be slow on CPU/MPS; reduce `max_new_tokens` or use a smaller model.
- xTB may crash on malformed geometries; RDKit embedding quality matters.
- Regex parsers tuned for xTB 6.7.1; future versions may differ.
- This is an experimental prototype — production systems should add:
  - retries,  
  - plan validation,  
  - caching,  
  - error propagation and recovery.  

---

## 📚 Attribution

This project uses:

- **Academy** — https://github.com/academy-agents/academy  
- **RDKit** — https://www.rdkit.org  
- **xTB** — https://github.com/grimme-lab/xtb  
- **Transformers / Accelerate / torch**  
- **Phi-3.5-mini-instruct** — https://huggingface.co/microsoft/Phi-3.5-mini-instruct  

