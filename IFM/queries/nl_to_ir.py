#!/usr/bin/env python3
"""NL front door: parse a natural-language measurement request into the
measurement-request IR (schema-enforced LLM structured extraction).

Usage:
  python3 nl_to_ir.py --prompt                       # print the parsing prompt (for any LLM)
  python3 nl_to_ir.py --via-claude "I want to ..."   # parse via local `claude -p`, print IR JSON
  python3 nl_to_ir.py --validate file.json           # structural check of an IR file

The parsing rules that matter (encoded in the prompt):
  - never invent units or values; unstated quantities go to elicitation.unresolved
  - inferred fields carry their basis in elicitation.inferred
  - hard constraints only when the user's words justify them; else preferences
"""
import json
import subprocess
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "measurement_request.schema.json"

PROMPT_TEMPLATE = """You are the natural-language front door of a scientific-instrument \
selection system. Convert the user's measurement request into a JSON object conforming \
to the Measurement Request schema below.

Rules — these override fluency:
1. NEVER invent values or units. If the user did not state a quantity, do not guess it; \
list the field in elicitation.unresolved instead.
2. Every field you infer (rather than quote) goes in elicitation.inferred with its basis.
3. Mark a requirement "hard": true only when the user's words justify exclusion \
("must", "at least", "cannot survive"); otherwise it is a preference or soft bound.
4. elicitation.stated lists fields the user explicitly stated. \
elicitation.original_text carries the request verbatim.
5. Output ONLY the JSON object. No markdown fences, no commentary.

SCHEMA:
{schema}

USER REQUEST:
{request}"""


def build_prompt(request_text: str) -> str:
    schema = SCHEMA_PATH.read_text()
    return PROMPT_TEMPLATE.format(schema=schema, request=request_text)


def validate(ir: dict) -> list:
    """Minimal structural validation (no jsonschema dependency)."""
    problems = []
    for req in ("observable", "target", "elicitation"):
        if req not in ir:
            problems.append(f"missing required field: {req}")
    if "observable" in ir and "text" not in ir.get("observable", {}):
        problems.append("observable.text missing")
    el = ir.get("elicitation", {})
    for k in ("stated", "inferred", "unresolved"):
        if k not in el:
            problems.append(f"elicitation.{k} missing")
    # units mandatory on any quantity/bound
    def walk(node, path):
        if isinstance(node, dict):
            if ("value" in node or "max" in node or "min" in node) and "unit" not in node:
                problems.append(f"{path}: quantity/bound without unit")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(ir, "$")
    return problems


def via_claude(request_text: str) -> dict:
    prompt = build_prompt(request_text)
    out = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=300,
    )
    text = out.stdout.strip()
    # tolerate accidental fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    ir = json.loads(text)
    problems = validate(ir)
    if problems:
        print("VALIDATION PROBLEMS:", *problems, sep="\n  ", file=sys.stderr)
    return ir


if __name__ == "__main__":
    if "--prompt" in sys.argv:
        print(build_prompt("<request here>"))
    elif "--via-claude" in sys.argv:
        idx = sys.argv.index("--via-claude")
        text = " ".join(sys.argv[idx + 1:])
        print(json.dumps(via_claude(text), indent=2))
    elif "--validate" in sys.argv:
        idx = sys.argv.index("--validate")
        ir = json.load(open(sys.argv[idx + 1]))
        problems = validate(ir)
        print("\n".join(problems) if problems else "ok")
    else:
        print(__doc__)
