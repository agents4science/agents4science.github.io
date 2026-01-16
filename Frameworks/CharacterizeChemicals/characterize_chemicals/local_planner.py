# local_planner_llm.py
#
# Minimal wrapper around a local OSS LLM (e.g., oss-20b) using transformers.
# No HTTP, no network; everything in-process.
#
# You probably want to run this in an environment with a GPU and
# a quantized variant of oss20b (e.g. 4-bit).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class PlannerJSONError(Exception):
    """Error raised when the planner LLM fails to produce valid JSON after retries."""
    pass

@dataclass

class ChatMessage:
    role: str   # "system" | "user"
    content: str


class LocalPlannerLLM:
    """
    Tiny chat-style wrapper around a local causal LLM.

    It is *not* chat-native; we just flatten system+user messages into
    a single prompt. For a real chat model, you’d swap in an appropriate
    tokenizer chat-template call.
    """

    def __init__(
        self,
        #model_name: str = "openai/gpt-oss-20b",
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        # Very simple chat → prompt formatting. Adjust if your model
        # has a proper chat template (e.g. tokenizer.apply_chat_template).
        parts: List[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"[SYSTEM]\n{m.content}\n")
            elif m.role == "user":
                parts.append(f"[USER]\n{m.content}\n")
            else:
                parts.append(f"[{m.role.upper()}]\n{m.content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def generate(self, messages: List[ChatMessage], temperature: float = 0.1) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            )

        # Slice off the prompt tokens
        generated = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    def _extract_first_json_object(self, text: str) -> Dict[str, Any]:
        """
        Try to parse the first JSON object found in the text using JSONDecoder.raw_decode.
        Raises json.JSONDecodeError if nothing parseable is found.
        """
        decoder = json.JSONDecoder()
        s = text.lstrip()
    
        # Find first '{'
        start = s.find("{")
        if start == -1:
            raise json.JSONDecodeError("No '{' found", s, 0)
        s = s[start:]
    
        obj, idx = decoder.raw_decode(s)
        return obj

    def plan_workflow(
        self,
        smiles: str,
        target_properties: List[str],
        accuracy_profile: str,
    ) -> Dict[str, Any]:
        """
        Synchronous: given SMILES + props + profile, return plan dict.
        """

        system_prompt = """
You are a planning agent that designs small computational workflows
for molecular property prediction.

You have access to these tools (BUT YOU ONLY OUTPUT A PLAN IN JSON;
YOU DO NOT EXECUTE ANYTHING):

- "rdkit_descriptors(smiles, descriptor_set)": fast, cheap descriptor
  calculation from SMILES; can compute logP, TPSA, MolWt and returns
  those fields in its output.

- "xtb_opt(smiles, level)": generates a 3D geometry, runs a GFN2-xTB
  optimization, and returns at least:
    - "optimized_geometry": path to the XYZ geometry file
    - "xtb_log_path": path to the xTB log file
    - "E_total_hartree" and "E_total_eV": total energy
    - "E_scc_hartree" and "E_scc_eV": SCC electronic energy
    - "HOMO_LUMO_gap_eV": HOMO-LUMO gap
    - "dipole_moment": total dipole moment in Debye

- "solvation_energy_from_xtb(geometry_path, solvent)": runs a GBSA
  solvation calculation starting from an optimized geometry. It returns:
    - "solvation_free_energy_kcal_per_mol": total Gsolv in kcal/mol
    - "xtb_gbsa_log": path to the GBSA log file

Your job: given a SMILES string, a list of target properties, and an
accuracy profile, produce a PLAN that is a JSON object with:

{
  "steps": [
    {
      "id": "s1_rdkit",
      "tool": "rdkit_descriptors",
      "inputs": { "smiles": "<SMILES>", "descriptor_set": ["logP", "TPSA"] },
      "outputs": ["logP", "TPSA"],
      "depends_on": []
    },
    {
      "id": "s2_xtb",
      "tool": "xtb_opt",
      "inputs": { "smiles": "<SMILES>", "level": "GFN2-xTB" },
      "outputs": [
        "optimized_geometry",
        "E_total_hartree",
        "E_scc_hartree",
        "HOMO_LUMO_gap_eV",
        "dipole_moment"
      ],
      "depends_on": ["s1_rdkit"]
    },
    {
      "id": "s3_solvation",
      "tool": "solvation_energy_from_xtb",
      "inputs": {
        "geometry_path": "step:s2_xtb.optimized_geometry",
        "solvent": "water"
      },
      "outputs": ["solvation_free_energy_kcal_per_mol"],
      "depends_on": ["s2_xtb"]
    }
  ],
  "target_properties": [...]
}

Rules:
- Only use tools listed above.
- Each "id" must be unique.
- "depends_on" lists earlier step ids whose outputs are needed.
- If an input needs a field from a previous step, write
  "step:<STEP_ID>.<FIELD_NAME>", e.g. "step:s2_xtb.optimized_geometry".
- Respond with VALID JSON ONLY. No commentary, no markdown, no extra text.
""".strip()


        user_prompt = f"""
SMILES: {smiles}
Target properties: {target_properties}
Accuracy profile: {accuracy_profile}

Produce an efficient plan using the tools, following the required JSON schema.
Make sure all tools you reference are in the catalog.
        """.strip()

        # --- First attempt ---
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
    
        raw = self.generate(messages, temperature=0.1)
        raw_str = raw.strip()
    
        try:
            plan_dict = self._extract_first_json_object(raw_str)
            return plan_dict
        except json.JSONDecodeError as e:
            # Fall through to repair attempts
            last_error = e
    
        # --- Repair attempts ---
        for attempt in range(2):  # e.g. two repair tries
            repair_messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "You previously produced invalid JSON. "
                        "Your ONLY task now is to output a corrected JSON object "
                        "for the workflow plan, with no commentary, no markdown, "
                        "no backticks."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=f"Here is your previous output:\n\n{raw_str}\n\nNow output ONLY valid JSON.",
                ),
            ]
            raw_repair = self.generate(repair_messages, temperature=0.0)
            raw_repair_str = raw_repair.strip()
    
            try:
                plan_dict = self._extract_first_json_object(raw_repair_str)
                return plan_dict
            except json.JSONDecodeError as e:
                last_error = e
                raw_str = raw_repair_str  # so we show the latest in the final error
    
        # If we get here, we failed all attempts
        raise PlannerJSONError(
            f"Failed to parse JSON plan after retries: {last_error}\nLast output was:\n{raw_str!r}"
        )
