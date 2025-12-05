# chem_agent.py
import json
import subprocess
import logging
import tempfile
import time
import re
import os
import asyncio
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Crippen, rdMolDescriptors

from academy.agent import Agent, action

from .local_planner import LocalPlannerLLM

import math

HARTREE_TO_EV = 27.211386245988  # CODATA-ish
HARTREE_TO_KCAL_MOL = 627.509474  # for solvation if you want Eh->kcal/mol


# ---------- Plan data structures ----------

@dataclass
class PlanStep:
    id: str
    tool: str
    inputs: Dict[str, Any]
    outputs: List[str]
    depends_on: List[str]


@dataclass
class Plan:
    steps: List[PlanStep]
    target_properties: List[str]


# ---------- Tool registry (wrap simulation codes) ----------

class ToolRegistry:
    """
    Wraps real chemistry tools:
      - rdkit_descriptors: compute logP, TPSA, MolWt
      - xtb_opt: generate 3D geometry and run xTB optimization
      - solvation_energy_from_xtb: run xTB GBSA and parse solvation energy
    """

    def __init__(self, workdir: Path, logger):
        self.workdir = workdir
        self.logger = logger

    # ---------- 1. RDKit descriptors ----------

    async def rdkit_descriptors(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        smiles = inputs["smiles"]
        descriptor_set = inputs.get("descriptor_set", ["logP", "TPSA", "MolWt"])
        self.logger.info("[rdkit_descriptors] SMILES=%s descriptors=%s", smiles, descriptor_set)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

        # RDKit descriptors
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        molwt = rdMolDescriptors.CalcExactMolWt(mol)

        out: Dict[str, Any] = {
            "smiles": smiles,
        }
        if "logP" in descriptor_set:
            out["logP"] = float(logp)
        if "TPSA" in descriptor_set:
            out["TPSA"] = float(tpsa)
        if "MolWt" in descriptor_set:
            out["MolWt"] = float(molwt)

        return out

    # ---------- 2. xTB geometry optimization ----------

    def _write_xyz_from_smiles(self, smiles: str, label: str) -> Path:
        """Generate a 3D geometry from SMILES and write an XYZ file."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        # Embed and optimize with UFF as a cheap pre-geometry
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise RuntimeError(f"RDKit embedding failed for {smiles}")
        AllChem.UFFOptimizeMolecule(mol)

        xyz_path = self.workdir / f"{label}.xyz"
        with xyz_path.open("w") as f:
            n_atoms = mol.GetNumAtoms()
            f.write(f"{n_atoms}\n")
            f.write(f"{label}\n")
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                f.write(
                    f"{atom.GetSymbol():2s}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}\n"
                )
        return xyz_path

    def _run_xtb(self, xyz_path: Path, extra_args=None, label="xtb_run") -> Path:
        """
        Run xTB on the given xyz file with provided args.
        Returns path to stdout log file.
        """
        if extra_args is None:
            extra_args = []

        log_path = self.workdir / f"{label}.log"
        cmd = ["xtb", str(xyz_path)] + extra_args
        self.logger.info("[xTB] Running command: %s", " ".join(cmd))

        with log_path.open("w") as log_f:
            result = subprocess.run(
                cmd,
                cwd=self.workdir,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )

        if result.returncode != 0:
            self.logger.error(
                "[xTB] Command failed with return code %s; see log %s",
                result.returncode,
                log_path,
            )
            raise RuntimeError(f"xTB failed (return code {result.returncode}) for {xyz_path}")

        return log_path

    def _parse_xtb_dipole(self, text: str) -> float | None:
        """
        Extract dipole moment (Debye) from xTB output.
        Looks for the 'full:' line under 'molecular dipole'.
        """
        import re

        # First find the dipole block
        block = re.search(r"molecular dipole:(.*?)(?=molecular quadrupole|$)", text, re.S)
        if not block:
            return None

        dip_section = block.group(1)

        # Look for the 'full:' line e.g.
        # full:   -0.000   -0.000   -0.000     0.000
        m = re.search(r"full:\s+([-\d\.Ee]+)\s+([-\d\.Ee]+)\s+([-\d\.Ee]+)\s+([-\d\.Ee]+)", dip_section)
        if not m:
            return None

        # total dipole is last column
        tot_dip = float(m.group(4))
        return tot_dip

    def _parse_xtb_energy_gap_and_dipole(self, log_path: Path) -> Dict[str, float]:
        """
        Parse SCC energy, total energy, HOMO-LUMO gap, and dipole from xTB log.
        Returns energies in Hartree and eV, plus gap and dipole (Debye).
        """
        text = log_path.read_text()
    
        # SCC energy line:
        # :: SCC energy               -10.334165418678 Eh    ::
        m_scc = re.search(
            r"::\s*SCC energy\s+(-?\d+\.\d+)\s+Eh", text
        )
        E_scc_h = float(m_scc.group(1)) if m_scc else None
    
        # total energy line:
        # :: total energy             -10.275566002092 Eh    ::
        m_tot = re.search(
            r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh", text
        )
        E_tot_h = float(m_tot.group(1)) if m_tot else None
    
        # HOMO-LUMO gap line:
        # :: HOMO-LUMO gap               6.544150159267 eV    ::
        m_gap = re.search(
            r"::\s*HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV", text
        )
        gap_ev = float(m_gap.group(1)) if m_gap else None
    
        dipole_D = self._parse_xtb_dipole(text)
    
        out: Dict[str, float] = {}
    
        if E_scc_h is not None:
            out["E_scc_hartree"] = E_scc_h
            out["E_scc_eV"] = E_scc_h * HARTREE_TO_EV
    
        if E_tot_h is not None:
            out["E_total_hartree"] = E_tot_h
            out["E_total_eV"] = E_tot_h * HARTREE_TO_EV
    
        if gap_ev is not None:
            out["HOMO_LUMO_gap_eV"] = gap_ev
    
        if dipole_D is not None:
            out["dipole_moment"] = dipole_D  # Debye
    
        return out


    def _parse_xtb_energy_and_dipole(self, log_path: Path) -> Dict[str, float]:
        """
        Parse xTB log file for total energy and dipole magnitude.

        This depends on xTB's output format; adjust regexes as needed.
        """
        text = log_path.read_text()

        dipole_debye = _parse_xtb_dipole(text)

        # Example patterns; you may need to tweak based on your xTB version
        # TOTAL ENERGY      -40.1234560 Eh
        mE = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", text)
        # | dipole moment     1.2345 Debye
        mD = re.search(r"\|\s*dipole moment\s+(\d+\.\d+)\s+Debye", text)

        energy_hartree = float(mE.group(1)) if mE else None
        dipole_debye = float(mD.group(1)) if mD else None

        out: Dict[str, float] = {}
        if energy_hartree is not None:
            # convert to eV if you prefer; here we keep in Hartree for now
            out["E_total_hartree"] = energy_hartree
        if dipole_debye is not None:
            out["dipole_moment"] = dipole_debye
        return out

    async def xtb_opt(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        smiles = inputs["smiles"]
        level = inputs.get("level", "GFN2-xTB")
        label = inputs.get("label", "molecule")
        self.logger.info("[xtb_opt] SMILES=%s level=%s", smiles, level)
    
        xyz_path = self._write_xyz_from_smiles(smiles, f"{label}_opt_in")
        log_path = self._run_xtb(
            xyz_path,
            extra_args=["--opt", "--gfn", "2"],
            label=f"{label}_opt",
        )
    
        parsed = self._parse_xtb_energy_gap_and_dipole(log_path)
    
        out: Dict[str, Any] = {
            "optimized_geometry": str(xyz_path),
            "xtb_log_path": str(log_path),
            "level": level,
            "smiles": smiles,
        }
        out.update(parsed)  # E_total/E_scc/gap/dipole
    
        return out


    # ---------- 3. xTB solvation energy ----------

    def _parse_xtb_gbsa(self, log_path: Path) -> float | None:
        """
        Parse total solvation free energy (Gsolv) from xTB log, return kcal/mol.
        """
        text = log_path.read_text()
    
        # Look for the Gsolv line in the SUMMARY block
        m = re.search(r"::\s*->\s*Gsolv\s+(-?\d+\.\d+)\s+Eh", text)
        if m:
            gsolv_h = float(m.group(1))
            gsolv_kcal = gsolv_h * HARTREE_TO_KCAL_MOL
            return gsolv_kcal
    
        # fallback: last number before 'kcal/mol' anywhere (very defensive)
        matches = re.findall(r"(-?\d+\.\d+)\s*kcal/mol", text)
        if matches:
            return float(matches[-1])
    
        self.logger.warning(
            "[_parse_xtb_gbsa] Could not find Gsolv in %s",
            log_path,
        )
        return None


    async def solvation_energy_from_xtb(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        geometry_path = inputs["geometry_path"]
        solvent = inputs.get("solvent", "water")
        self.logger.info(
            "[solvation_energy_from_xtb] geometry_path=%s solvent=%s",
            geometry_path,
            solvent,
        )
    
        xyz_path = Path(geometry_path)
        if not xyz_path.is_absolute():
            xyz_path = self.workdir / xyz_path
    
        self.logger.info("Resolved geometry_path -> %s", xyz_path)
    
        log_gbsa = self._run_xtb(
            xyz_path,
            extra_args=["--gfn", "2", "--gbsa", solvent],
            label=f"{xyz_path.stem}_gbsa",
        )
        dG_kcal = self._parse_xtb_gbsa(log_gbsa)
    
        return {
            "solvation_free_energy_kcal_per_mol": dG_kcal,
            "solvent": solvent,
            "xtb_gbsa_log": str(log_gbsa),
        }


    # ---------- 4. Dispatcher ----------

    async def run_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        if tool_name == "rdkit_descriptors":
            out = await self.rdkit_descriptors(inputs)
        elif tool_name == "xtb_opt":
            out = await self.xtb_opt(inputs)
        elif tool_name == "solvation_energy_from_xtb":
            out = await self.solvation_energy_from_xtb(inputs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
        elapsed = time.time() - t0
        out["_runtime_seconds"] = elapsed
        out["_tool"] = tool_name
        return out



# ---------- MoleculePropertyAgent with integrated LLM ----------

class MoleculePropertyAgent(Agent):
    """
    Academy agent that:
      - uses a local planner LLM (LocalPlannerLLM) to plan a workflow
      - executes the plan using local tools
      - returns canonical properties + provenance
    """

    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct") -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        model_name = os.getenv("PLANNER_MODEL", "microsoft/Phi-3.5-mini-instruct")
        self.logger.info("Initializing planner LLM with model=%s", model_name)

        # build the planner LLM with the chosen model
        self.llm = LocalPlannerLLM(model_name=model_name)
        self._planner_lock = asyncio.Lock()


    @action
    async def compute_properties(
        self,
        molecule_smiles: str,
        target_properties: List[str] | None = None,
        accuracy_profile: str = "balanced",
        max_wallclock_minutes: int = 60,
    ) -> Dict[str, Any]:
        if target_properties is None:
            target_properties = [
                "logP",
                "dipole_moment",
                "solvation_free_energy",
            ]

        self.logger.info(
            "MoleculePropertyAgent: computing for SMILES=%s props=%s profile=%s",
            molecule_smiles,
            target_properties,
            accuracy_profile,
        )

        # --- u. Call OSS LLM *inside the agent* to get a plan ---
        t0 = time.time()
        plan_dict = await asyncio.to_thread(
            self.llm.plan_workflow,
            molecule_smiles,
            target_properties,
            accuracy_profile,
        )
        t1 = time.time()
        self.logger.info("Planner (Phi) call took %.2f seconds", t1 - t0)

        steps = [
            PlanStep(
                id=s["id"],
                tool=s["tool"],
                inputs=s.get("inputs", {}),
                outputs=s.get("outputs", []),
                depends_on=s.get("depends_on", []),
            )
            for s in plan_dict["steps"]
        ]
        plan = Plan(steps=steps, target_properties=plan_dict["target_properties"])

        self.logger.info(
            "MoleculePropertyAgent: received plan %s",
            json.dumps(
                {
                    "steps": [asdict(st) for st in plan.steps],
                    "target_properties": plan.target_properties,
                },
                indent=2,
            ),
        )

        # Fix up solvation steps to always use the optimized geometry from their dependency.
        for step in plan.steps:
            if step.tool == "solvation_energy_from_xtb":
                if step.depends_on:
                    dep_id = step.depends_on[0]
                    # Force geometry_path to be a reference to the previous xtb_opt step's geometry
                    step.inputs["geometry_path"] = f"step:{dep_id}.optimized_geometry"
                else:
                    # If the LLM forgot depends_on, you can either raise or skip
                    self.logger.warning(
                        "solvation_energy_from_xtb step %s has no depends_on; "
                        "cannot infer geometry_path",
                        step.id,
                    )


        # --- 2. Execute plan in scratch dir ---
        with tempfile.TemporaryDirectory() as tmpdir_str:
            workdir = Path(tmpdir_str)
            registry = ToolRegistry(workdir, self.logger)

            results: Dict[str, Dict[str, Any]] = {}
            provenance_tools: List[Dict[str, Any]] = []
            completed: set[str] = set()

            start_time = time.time()
            deadline = start_time + max_wallclock_minutes * 60

            while len(completed) < len(plan.steps):
                progress = False
                for step in plan.steps:
                    if step.id in completed:
                        continue
                    if any(dep not in completed for dep in step.depends_on):
                        continue
            
                    # Resolve inputs, including "step:s2_xtb.field" and optional "s2_xtb.field"
                    resolved_inputs: Dict[str, Any] = {}
                    for k, v in step.inputs.items():
                        if isinstance(v, str):
                            if v.startswith("step:"):
                                _, ref = v.split(":", 1)
                                if "." in ref:
                                    dep_id, field = ref.split(".", 1)
                                    resolved_inputs[k] = results[dep_id][field]
                                else:
                                    dep_id = ref
                                    resolved_inputs[k] = results[dep_id]
                            elif "." in v:
                                # Fallback: treat "s2_xtb.xtb_output_path" as step reference
                                dep_id, field = v.split(".", 1)
                                if dep_id in results and field in results[dep_id]:
                                    resolved_inputs[k] = results[dep_id][field]
                                else:
                                    resolved_inputs[k] = v  # leave as literal if we can't resolve
                            else:
                                resolved_inputs[k] = v
                        else:
                            resolved_inputs[k] = v

                    # BEFORE: log step + inputs
                    self.logger.info(
                        ">>> Executing step=%s tool=%s depends_on=%s\n    inputs=%s",
                        step.id,
                        step.tool,
                        step.depends_on,
                        resolved_inputs,
                    )
            
                    tool_output = await registry.run_tool(step.tool, resolved_inputs)
            
                    # AFTER: log outputs (hide internal _ fields)
                    self.logger.info(
                        "<<< Finished step=%s tool=%s\n    outputs=%s",
                        step.id,
                        step.tool,
                        {k: v for k, v in tool_output.items() if not k.startswith("_")},
                    )
            
                    results[step.id] = tool_output
                    provenance_tools.append(
                        {
                            "step_id": step.id,
                            "tool": step.tool,
                            "inputs": resolved_inputs,
                            "outputs": list(tool_output.keys()),
                            "runtime_seconds": tool_output.get("_runtime_seconds"),
                        }
                    )
                    completed.add(step.id)
                    progress = True
            
                if not progress:
                    raise RuntimeError("Plan execution stalled (cycle or unresolved dependency).")

        # --- 3. Aggregate outputs → canonical properties ---
        properties: Dict[str, Any] = {}
        for step_id, out in results.items():
            for k, v in out.items():
                if k == "logP":
                    properties["logP"] = v
                elif k == "dipole_moment":
                    # store in Debye
                    properties["dipole_moment_D"] = v
                elif k == "solvation_free_energy_kcal_per_mol":
                    properties["solvation_free_energy_kcal_per_mol"] = v
                elif k == "E_total_hartree":
                    properties["E_total_hartree"] = v
                    properties["E_total_eV"] = out.get("E_total_eV")  # already computed
                elif k == "E_scc_hartree":
                    properties["E_scc_hartree"] = v
                    properties["E_scc_eV"] = out.get("E_scc_eV")
                elif k == "HOMO_LUMO_gap_eV":
                    properties["HOMO_LUMO_gap_eV"] = v

        response = {
            "status": "success",
            "molecule_smiles": molecule_smiles,
            "properties": properties,
            "plan_used": {
                "steps": [asdict(s) for s in plan.steps],
                "target_properties": plan.target_properties,
            },
            "provenance": {"tools": provenance_tools},
            "error_message": None,
        }
        self.logger.info(
            "MoleculePropertyAgent: finished SMILES=%s properties=%s",
            molecule_smiles,
            properties,
        )
        return response

