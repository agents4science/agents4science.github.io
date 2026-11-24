#!/usr/bin/env python3
"""Prototype matchmaker: 'I want to measure X' -> ranked instruments.

Usage:
  python3 matchmaker.py "3d microstructure of a battery electrode during cycling"
  python3 matchmaker.py --demo          # run the canned demo queries
  python3 matchmaker.py --all "query"   # include non-operational instruments

Scoring is deliberately simple (keyword/synonym overlap, field-weighted).
The point of the prototype is the DATA MODEL: measurables vs techniques vs
constraints, with provenance and record depth. A real system would replace
this scorer with embedding retrieval + an LLM feasibility check; the record
schema is what carries over.
"""
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Query-side synonym expansion: user phrasings -> catalog vocabulary.
# In a real system this becomes an ontology alignment (PaNET) + embeddings.
SYNONYMS = {
    "3d": ["tomography", "tomogram", "microstructure", "internal", "morphology"],
    "microstructure": ["morphology", "tomography", "porosity", "3d"],
    "porosity": ["pore", "microstructure", "tomography"],
    "crack": ["microstructure", "morphology", "in-situ", "evolution"],
    "band": ["electronic", "fermi", "arpes", "dispersion"],
    "electronic": ["band", "fermi", "arpes", "excitations", "states"],
    "magnetic": ["magnetization", "spin", "domain"],
    "spin": ["magnetic", "domain"],
    "protein": ["macromolecular", "macromolecule", "solution", "crystallography"],
    "structure": ["structural", "crystal", "lattice"],
    "crystal": ["crystallography", "diffraction", "lattice", "structure"],
    "pressure": ["gpa", "anvil", "extreme", "compression"],
    "catalyst": ["catalysis", "surface", "reaction", "operando"],
    "surface": ["interface", "interfacial", "adsorbate"],
    "oxidation": ["redox", "valence", "chemical state", "speciation"],
    "redox": ["oxidation", "valence"],
    "chemical": ["speciation", "composition", "bonding"],
    "battery": ["electrode", "cathode", "anode", "electrochemistry", "lithium"],
    "electrode": ["battery", "electrochemistry"],
    "operando": ["in-situ", "in situ", "reaction conditions", "real-time"],
    "in-situ": ["operando", "real-time", "evolution"],
    "element": ["elemental", "composition", "fluorescence"],
    "trace": ["elemental", "fluorescence", "microprobe"],
    "map": ["maps", "mapping", "imaging", "microscopy", "distribution"],
    "image": ["imaging", "microscopy", "map"],
    "nanoscale": ["nanometer", "nano", "nm", "ptychography"],
    "vibrational": ["infrared", "functional-group", "molecular"],
    "biofilm": ["biological", "bacteria", "living"],
    "living": ["biological", "real-time", "hydrated"],
    "phase": ["transition", "crystalline", "diffraction"],
    "strain": ["stress", "lattice", "elastic", "microdiffraction"],
    "solution": ["saxs", "scattering", "conformational"],
    "shape": ["envelope", "conformational", "saxs", "morphology"],
    "flexibility": ["conformational", "ensembles", "solution"],
    "gas": ["ambient-pressure", "reaction conditions"],
    "molecular": ["molar", "molecular weight", "sec", "gpc", "mals", "functional-group", "vibrational", "infrared"],
    "weight": ["molar mass", "mass", "dispersity", "sec"],
    "glass": ["transition", "thermal", "dsc", "calorimetry"],
    "transition": ["glass", "thermal", "melting", "crystallization"],
    "thermal": ["glass transition", "stability", "decomposition", "dsc", "tga", "heat"],
    "viscosity": ["rheology", "flow", "rod-pull", "viscometry"],
    "rheology": ["viscosity", "flow", "jamming", "modulus"],
    "zeta": ["electrophoretic", "charge", "dls", "colloid"],
    "modulus": ["hardness", "mechanical", "nanoindentation", "stiffness"],
    "hardness": ["modulus", "mechanical", "nanoindentation", "scratch"],
    "gloss": ["sheen", "reflectivity", "coating"],
    "synthesize": ["synthesis", "polymerization", "formulation", "library"],
    "polymer": ["copolymer", "latex", "film", "coating", "macromolecule"],
    "film": ["thin film", "coating", "casting"],
    "particle": ["colloid", "dispersion", "size", "zeta"],
    "tissue": ["biological", "chemical maps", "micron"],
}

STOP = set("i want to measure of a the in on at an and for with under during my our its is are how what best".split())

FIELD_WEIGHTS = [
    ("measurables", 3.0),
    ("demonstrated_applications", 2.0),
    ("techniques", 2.0),
    ("sample_environments", 1.5),
    ("name", 1.0),
    ("aliases", 1.0),
]

def load_records():
    records = []
    for f in sorted(DATA_DIR.glob("*.json")):
        records.extend(json.load(open(f)))
    return records

def tokens(text):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower()) if t not in STOP]

def expand(query_tokens):
    expanded = dict.fromkeys(query_tokens, 1.0)  # original terms weight 1
    for t in query_tokens:
        base = t.rstrip("s")
        for key, syns in SYNONYMS.items():
            if t == key or base == key:
                for s in syns:
                    expanded.setdefault(s, 0.5)   # synonyms weight 0.5
    return expanded

def field_text(record, field):
    v = record.get(field, "")
    if field == "techniques":
        parts = []
        for t in v:
            parts.append(t.get("name", ""))
            parts.append(t.get("acronym", ""))
            parts.extend(t.get("variants", []))
        return " ".join(parts)
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)

def score(record, expanded):
    total, matched = 0.0, {}
    for field, weight in FIELD_WEIGHTS:
        text = field_text(record, field).lower()
        for term, tw in expanded.items():
            if term in text:
                total += weight * tw
                matched.setdefault(field, set()).add(term)
    # small boost for deep records: richer, sourced data
    if record.get("record_depth") == "deep":
        total *= 1.15
    return total, matched

def search(query, include_all=False, top=5):
    records = load_records()
    expanded = expand(tokens(query))
    results = []
    for r in records:
        if not include_all and r.get("status") in ("closed", "not-operational"):
            continue
        s, matched = score(r, expanded)
        if s > 0:
            results.append((s, r, matched))
    results.sort(key=lambda x: -x[0])
    return results[:top], expanded

def show(query, include_all=False):
    results, expanded = search(query, include_all)
    print(f'\nQUERY: "{query}"')
    if not results:
        print("  no matches")
        return
    for rank, (s, r, matched) in enumerate(results, 1):
        status = r.get("status", "?")
        depth = r.get("record_depth", "?")
        print(f"  {rank}. [{s:5.1f}] {r['id']}  {r['name']}")
        print(f"         status={status}  record_depth={depth}")
        techs = ", ".join(t.get("acronym") or t["name"] for t in r["techniques"][:4])
        print(f"         techniques: {techs}")
        for field in ("measurables", "demonstrated_applications"):
            if field in matched:
                hits = sorted(matched[field])
                print(f"         matched {field}: {', '.join(hits[:6])}")
        if depth == "directory":
            print("         (directory record: measurables derived from technique map, verify on beamline page)")

DEMO_QUERIES = [
    "3d microstructure of a battery electrode during in-situ cycling",
    "electronic band structure and fermi surface of a 2d quantum material",
    "protein solution shape and conformational flexibility",
    "surface oxidation state of a catalyst under reaction gas conditions",
    "crystal structure phase transition at high pressure",
    "chemical composition map of living bacterial biofilm",
    "magnetic domain structure of a thin film at the nanoscale",
    "trace element distribution in plant roots",
]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all" and a != "--demo"]
    include_all = "--all" in sys.argv
    if "--demo" in sys.argv:
        for q in DEMO_QUERIES:
            show(q, include_all)
    elif args:
        show(" ".join(args), include_all)
    else:
        print(__doc__)
