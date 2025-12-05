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
  calculation from SMILES; can compute logP, TPSA, MolWt.
- "xtb_opt(smiles, level)": geometry optimization + total energy +
  dipole moment using xTB; 'level' is usually "GFN2-xTB".
- "solvation_energy_from_xtb(geometry_path, solvent)": 
  run GBSA solvation using an optimized geometry in XYZ format.

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
    ...
  ],
  "target_properties": [...]
}

Rules:
- Only use tools listed above.
- Each "id" must be unique.
- "depends_on" lists earlier step ids whose outputs are needed.
- If an input needs a field from a previous step, write
  "step:<STEP_ID>.<FIELD_NAME>", e.g. "step:s2_xtb.xtb_output_path".
- Respond with VALID JSON ONLY. No commentary, no markdown, no extra text.
        """.strip()

        user_prompt = f"""
SMILES: {smiles}
Target properties: {target_properties}
Accuracy profile: {accuracy_profile}

Produce an efficient plan using the tools, following the required JSON schema.
Make sure all tools you reference are in the catalog.
        """.strip()

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        raw = self.generate(messages, temperature=0.1)
        raw_str = raw.strip()

        # Use JSONDecoder to parse the first JSON object and ignore trailing junk
        decoder = json.JSONDecoder()
        s = raw_str.lstrip()

        # Find first '{'
        start = s.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model output:\n{raw_str!r}")

        s = s[start:]

        try:
            obj, idx = decoder.raw_decode(s)
        except json.JSONDecodeError as e:
            # Optional: log the bad output for debugging
            raise ValueError(f"Failed to parse JSON from model output: {e}\nOutput was:\n{s!r}") from e

        plan_dict: Dict[str, Any] = obj
        return plan_dict

        """
        # Simple heuristic to extract JSON
        try:
            start = raw_str.index("{")
            end = raw_str.rindex("}")
            json_str = raw_str[start : end + 1]
        except ValueError:
            json_str = raw_str  # hope it's pure JSON

        plan_dict: Dict[str, Any] = json.loads(json_str)
        return plan_dict
        """

