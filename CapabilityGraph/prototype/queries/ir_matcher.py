#!/usr/bin/env python3
"""Match a measurement-request IR against the capability graph.

Usage:
  python3 ir_matcher.py requests/cu_operando.json
  python3 ir_matcher.py --show-excluded requests/cu_operando.json

Pipeline (the hybrid split is the point):
  1. hard-constraint filter — deterministic, units-normalized, per-constraint
     verdict PASS / FAIL / UNKNOWN with a reason string. Only definite
     conflicts exclude; missing record data yields UNKNOWN, never FAIL.
  2. soft ranking — verified-pass count, observable/measurable overlap,
     preference matches, deep-record bonus.
  3. reporting — per-candidate constraint table (explainable acceptance),
     per-exclusion reasons (explainable rejection), near-misses (excluded by
     exactly one hard constraint), and clarifying-question suggestions from
     elicitation.unresolved.

The LLM feasibility adjudication of the top-k (stage 4 in the design) is not
implemented here; this module produces the evidence package it would consume.
"""
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- unit normalization ------------------------------------------------------

ENERGY_TO_EV = {"ev": 1.0, "kev": 1e3, "mev": 1e6, "mev_": 1e-3}
LENGTH_TO_M = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "micron": 1e-6, "nm": 1e-9,
               "angstrom": 1e-10, "a": 1e-10, "pm": 1e-12}
PRESSURE_TO_TORR = {"torr": 1.0, "mbar": 0.750062, "bar": 750.062, "pa": 7.50062e-3,
                    "kpa": 7.50062, "atm": 760.0}

def to_torr(value, unit):
    return value * PRESSURE_TO_TORR[unit.lower()]

def to_meters(value, unit):
    return value * LENGTH_TO_M[unit.lower()]

# Absorption edges (eV), K and L3, for edge-coverage checks. Extend as needed.
EDGES_EV = {
    "C":  {"K": 284.2},           "N":  {"K": 409.9},          "O": {"K": 543.1},
    "Fe": {"K": 7112.0, "L3": 706.8},
    "Cu": {"K": 8979.0, "L3": 932.7},
    "Ni": {"K": 8333.0, "L3": 852.7},
    "Mn": {"K": 6539.0, "L3": 638.7},
    "Co": {"K": 7709.0, "L3": 778.1},
}

# Technique-level property: does the technique resolve chemical elements?
ELEMENT_SPECIFIC_TECHS = {"xas", "xes", "xfs", "rixs", "apxps", "xps",
                          "x-ray absorption", "x-ray emission", "x-ray fluorescence",
                          "photoelectron", "photoemission"}
NOT_ELEMENT_SPECIFIC_TECHS = {"ftir", "infrared", "saxs", "waxs", "gisaxs", "tomography",
                              "micro-ct", "radiography"}

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"

# --- constraint checks: each returns (verdict, reason) ------------------------

def tech_blob(record):
    parts = []
    for t in record.get("techniques", []):
        parts += [t.get("name", ""), t.get("acronym", "") or ""] + t.get("variants", [])
    return " ".join(parts).lower()

def env_blob(record):
    sc = record.get("sample_constraints", {})
    return " ".join(record.get("sample_environments", [])
                    + sc.get("special_requirements", [])
                    + [sc.get("size_notes", "")]
                    + record.get("demonstrated_applications", [])).lower()

def check_status(record, ir):
    s = record.get("status", "unknown")
    if s == "operational":
        return PASS, "operational"
    if s in ("closed", "not-operational"):
        return FAIL, f"status={s}"
    return UNKNOWN, f"status={s}"

def check_state(record, ir):
    want = ir.get("target", {}).get("state")
    if not want:
        return UNKNOWN, "sample state unresolved in request"
    states = record.get("sample_constraints", {}).get("states")
    if not states:
        return UNKNOWN, "record has no accepted-states data"
    want_l = want.lower()
    for s in states:
        if want_l in s.lower() or s.lower() in want_l:
            return PASS, f"accepts '{s}'"
    return FAIL, f"sample state '{want}' not in accepted states {states}"

def check_vacuum(record, ir):
    tol = ir.get("target", {}).get("tolerates", {})
    if tol.get("vacuum") is not False:
        return UNKNOWN, "vacuum tolerance not constrained"
    if record.get("sample_constraints", {}).get("vacuum_required") is True:
        return FAIL, "instrument requires vacuum; sample cannot survive vacuum"
    if record.get("sample_constraints", {}).get("vacuum_required") is False:
        return PASS, "no vacuum required"
    return UNKNOWN, "record does not state vacuum requirement"

def check_gas_environment(record, ir):
    env = ir.get("conditions", {}).get("environment", {})
    gas, pressure = env.get("gas"), env.get("pressure")
    if not gas and not ir.get("conditions", {}).get("in_situ"):
        return UNKNOWN, "no in-situ gas environment requested"
    blob = env_blob(record)
    if pressure:
        want_torr = to_torr(pressure["value"], pressure["unit"])
        # look for a stated ceiling like "up to 10 Torr" / "above 2 mbar"
        for m in re.finditer(r"(?:up to|to|above)\s+([\d.]+)\s*(torr|mbar|bar|pa|atm)", blob):
            ceiling = to_torr(float(m.group(1)), m.group(2))
            if ceiling >= want_torr:
                return PASS, f"stated gas-pressure capability {m.group(1)} {m.group(2)} >= requested {pressure['value']} {pressure['unit']}"
        # no explicit ceiling: fall through to keyword evidence
    for kw in ("operando", "in situ", "in-situ", "reaction conditions", "gas", "ambient-pressure"):
        if kw in blob:
            return UNKNOWN, f"in-situ evidence ('{kw}') but no stated pressure ceiling covering the request"
    return FAIL, "no in-situ/gas-environment capability found in record"

def check_element_specific(record, ir):
    if ir.get("observable", {}).get("element_specific") is not True:
        return UNKNOWN, "element specificity not required"
    blob = tech_blob(record)
    if any(t in blob for t in ELEMENT_SPECIFIC_TECHS):
        return PASS, "offers an element-specific technique"
    if any(t in blob for t in NOT_ELEMENT_SPECIFIC_TECHS) :
        return FAIL, "techniques offered are not element-specific"
    return UNKNOWN, "cannot determine element specificity of techniques"

def check_edge_coverage(record, ir):
    elements = ir.get("observable", {}).get("elements_of_interest", [])
    if not elements:
        return UNKNOWN, "no elements of interest stated"
    rng = record.get("probe", {}).get("energy_range_eV")
    if not rng:
        return UNKNOWN, "record has no probe energy range"
    lo, hi = rng
    covered = []
    for el in elements:
        for edge, ev in EDGES_EV.get(el, {}).items():
            if lo <= ev <= hi:
                covered.append(f"{el} {edge} ({ev:g} eV)")
    if covered:
        return PASS, "edge(s) in range: " + ", ".join(covered)
    # Absorption-edge coverage is the wrong test for photoemission: XPS probes
    # core-level binding energies (e.g. Cu 3p at ~75 eV), not absorption edges.
    if any(t in tech_blob(record) for t in ("photoemission", "photoelectron", "xps", "apxps")):
        return UNKNOWN, (f"no absorption edge of {elements} in {lo}-{hi} eV, but photoemission probes "
                         "core-level binding energies instead - needs adjudication against core levels")
    return FAIL, f"no absorption edge of {elements} within probe range {lo}-{hi} eV"

def check_spatial_resolution(record, ir):
    bound = ir.get("requirements", {}).get("spatial_resolution")
    if not bound or "max" not in bound:
        return UNKNOWN, "no spatial-resolution requirement"
    want_m = to_meters(bound["max"], bound["unit"])
    have = record.get("resolution", {}).get("spatial_m")
    if have is None:
        return UNKNOWN, "record has no spatial-resolution figure"
    if have <= want_m:
        return PASS, f"resolution {have:g} m <= required {want_m:g} m"
    verdict = FAIL if bound.get("hard", True) else UNKNOWN
    return verdict, f"resolution {have:g} m coarser than required {want_m:g} m" + ("" if verdict == FAIL else " (soft)")

CHECKS = [
    ("status", check_status),
    ("sample state", check_state),
    ("vacuum tolerance", check_vacuum),
    ("in-situ gas environment", check_gas_environment),
    ("element specificity", check_element_specific),
    ("edge coverage", check_edge_coverage),
    ("spatial resolution", check_spatial_resolution),
]

# --- soft ranking --------------------------------------------------------------

def soft_score(record, ir, results):
    score = 2.0 * sum(1 for v, _ in results.values() if v == PASS)
    # observable overlap with measurables + demonstrated applications
    text = " ".join(ir.get("observable", {}).get("vocab", [])
                    + [ir.get("observable", {}).get("text", "")]).lower()
    terms = {w for w in re.findall(r"[a-z][a-z\-]{3,}", text)}
    blob = " ".join(record.get("measurables", [])).lower()
    apps = " ".join(record.get("demonstrated_applications", [])).lower()
    score += 1.5 * sum(1 for t in terms if t in blob)
    score += 1.0 * sum(1 for t in terms if t in apps)
    comp = (ir.get("target", {}).get("composition_hint") or "").lower()
    score += 2.0 * sum(1 for w in re.findall(r"[a-z]{3,}", comp) if w in apps)
    if record.get("record_depth") == "deep":
        score *= 1.15
    return score

# --- clarification suggestions -------------------------------------------------

CLARIFY_QUESTIONS = {
    "target.state": "What is the sample's physical state (solid / solution / powder / single crystal / thin film)?",
    "requirements.spatial_resolution": "Do you need this spatially resolved, and at what resolution?",
    "target.tolerates.vacuum": "Can the sample survive vacuum?",
    "conditions.environment.temperature": "At what temperature must the measurement run?",
    "target.quantity_available": "How much sample is available?",
}

def clarifications(ir, candidates):
    out = []
    for field in ir.get("elicitation", {}).get("unresolved", []):
        q = CLARIFY_QUESTIONS.get(field)
        if not q:
            continue
        # crude partition estimate: does the related property differ across candidates?
        if field == "target.state":
            vals = {tuple(c["record"].get("sample_constraints", {}).get("states", ["?"])) for c in candidates}
            if len(vals) > 1:
                q += f"  [would partition {len(candidates)} candidates]"
        out.append(q)
    return out

# --- driver --------------------------------------------------------------------

def match(ir, show_excluded=False):
    records = []
    for f in sorted(DATA_DIR.glob("*.json")):
        records.extend(json.load(open(f)))
    candidates, excluded = [], []
    for r in records:
        results = {name: fn(r, ir) for name, fn in CHECKS}
        fails = [(n, reason) for n, (v, reason) in results.items() if v == FAIL]
        entry = {"record": r, "results": results, "fails": fails}
        (excluded if fails else candidates).append(entry)
    for c in candidates:
        c["score"] = soft_score(c["record"], ir, c["results"])
    candidates.sort(key=lambda c: -c["score"])

    print(f"REQUEST: {ir.get('elicitation', {}).get('original_text') or ir['observable']['text']}")
    print(f"\n=== CANDIDATES ({len(candidates)}) ===")
    for i, c in enumerate(candidates[:5], 1):
        r = c["record"]
        print(f"\n{i}. [{c['score']:5.1f}] {r['id']}  {r['name']}  (depth={r.get('record_depth')})")
        for name, (v, reason) in c["results"].items():
            mark = {"PASS": "+", "UNKNOWN": "?"}[v]
            print(f"     {mark} {name:26s} {reason}")
    near = [e for e in excluded if len(e["fails"]) == 1
            and e["record"].get("status") not in ("closed",)]
    if near:
        print(f"\n=== NEAR MISSES (excluded by exactly one constraint) ===")
        for e in near[:6]:
            n, reason = e["fails"][0]
            print(f"  {e['record']['id']:14s} {n}: {reason}")
    if show_excluded:
        print(f"\n=== EXCLUDED ({len(excluded)}) ===")
        for e in excluded:
            reasons = "; ".join(f"{n}: {reason}" for n, reason in e["fails"])
            print(f"  {e['record']['id']:14s} {reasons}")
    qs = clarifications(ir, candidates)
    if qs:
        print("\n=== CLARIFYING QUESTIONS (from unresolved fields) ===")
        for q in qs:
            print(f"  - {q}")
    return candidates

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(0)
    ir = json.load(open(args[0]))
    match(ir, show_excluded="--show-excluded" in sys.argv)
