#!/usr/bin/env python3
"""Convert aps_directory_raw.json (agent-assembled APS beamline list,
sourced from search snippets, CAT partner sites, SEES/HPCAT status pages,
and publications on 2026-08-27) into schema-conforming directory records.

Beamlines with deep records (2-BM, 2-ID, 13-ID-E) are skipped here.
measurables are DERIVED from a technique->measurable keyword map and
flagged as such in provenance.
"""
import json
import os
import re

RETRIEVED = "2026-08-27"
SKIP = {"2-BM", "2-ID", "13-ID-E"}  # covered by deep records in aps_deep.json

# keyword -> derived measurables (matched case-insensitively against technique strings)
KEYWORD_MEASURABLES = [
    ("fluorescence", ["elemental composition", "trace element distribution"]),
    ("xafs", ["oxidation state", "local chemical environment and speciation"]),
    ("xanes", ["oxidation state", "chemical speciation"]),
    ("exafs", ["local coordination environment"]),
    ("absorption", ["oxidation state", "unoccupied electronic states"]),
    ("emission spectroscopy", ["occupied electronic states", "chemical bonding states"]),
    ("rixs", ["electronic excitations", "chemical bonding states"]),
    ("inelastic", ["electronic and vibrational excitations (phonons)"]),
    ("nuclear resonant", ["phonon density of states of Moessbauer isotopes", "hyperfine/magnetic state"]),
    ("photoemission", ["electronic band structure", "surface chemical composition"]),
    ("arpes", ["electronic band structure", "fermi surface"]),
    ("dichroism", ["element-specific magnetization", "magnetic ordering"]),
    ("magnetic", ["magnetic ordering and domain correlations"]),
    ("photon correlation", ["equilibrium and non-equilibrium dynamics (ns-hours)"]),
    ("coherent", ["nanoscale structure, strain and dynamics"]),
    ("ptychography", ["nanoscale morphology (highest-resolution x-ray imaging)"]),
    ("tomography", ["3d microstructure", "internal morphology (non-destructive)"]),
    ("imaging", ["2d/3d morphology"]),
    ("radiography", ["time-resolved internal dynamics (2d projection)"]),
    ("topography", ["crystal defect distribution"]),
    ("laue", ["local strain and crystal orientation maps"]),
    ("microdiffraction", ["local strain and crystal orientation maps"]),
    ("diffraction microscopy", ["grain-resolved 3d microstructure and strain (bulk)"]),
    ("powder", ["crystal structure", "phase identification", "lattice parameters"]),
    ("pair distribution", ["local atomic structure of disordered/nanocrystalline materials"]),
    ("total scattering", ["local atomic structure of disordered/nanocrystalline materials"]),
    ("diffraction", ["crystal structure", "phase identification"]),
    ("usaxs", ["structure spanning ~1 nm to ~1 um (continuous q)"]),
    ("saxs", ["nanoscale size and shape distributions"]),
    ("waxs", ["crystalline phase identification", "molecular packing"]),
    ("gisaxs", ["thin-film nanostructure and morphology"]),
    ("reflectivity", ["thin-film thickness, roughness and density profiles"]),
    ("liquid surface", ["liquid surface/interface structure"]),
    ("crystallography", ["crystal structure"]),
    ("macromolecular", ["macromolecular crystal structure"]),
    ("serial", ["room-temperature/time-resolved macromolecular structure"]),
    ("shock", ["material response under dynamic compression (structure, density)"]),
    ("dynamic", ["time-resolved structural evolution"]),
    ("pump-probe", ["ultrafast structural and electronic dynamics"]),
    ("high-pressure", ["structure and equation of state at extreme pressure"]),
    ("anvil", ["structure and equation of state at extreme pressure"]),
    ("multi-anvil", ["structure at high pressure-temperature (large volume)"]),
    ("energy-dispersive", ["structure at high pressure (energy-dispersive)"]),
    ("optics testing", ["x-ray optics and detector characterization"]),
    ("detector testing", ["x-ray optics and detector characterization"]),
    ("spray", ["fluid/spray dynamics"]),
    ("fiber", ["fiber/muscle diffraction structure"]),
]

def derive_measurables(techniques):
    out = []
    blob = " ".join(techniques).lower()
    for kw, ms in KEYWORD_MEASURABLES:
        if kw in blob:
            for m in ms:
                if m not in out:
                    out.append(m)
    return out or ["(unmapped technique - needs deep extraction)"]

def slug(beamline):
    return "aps-" + re.sub(r"[^a-z0-9]+", "-", beamline.lower()).strip("-")

def main():
    here = os.path.dirname(__file__)
    raw = json.load(open(os.path.join(here, "sources/aps_directory_raw.json")))
    records = []
    for b in raw:
        if b["beamline"] in SKIP:
            continue
        rec = {
            "id": slug(b["beamline"]),
            "facility": "APS",
            "institution": "ANL",
            "name": f'{b["name"]} (Beamline {b["beamline"]})',
            "instrument_class": "synchrotron-beamline",
            "status": b.get("status", "unknown"),
            "techniques": [{"name": t} for t in b["techniques"]],
            "measurables": derive_measurables(b["techniques"]),
            "access": {"mechanism": "general user proposal", "proposal_system": "APS User Portal"},
            "provenance": [{
                "source_url": u, "retrieved": RETRIEVED,
                "notes": "Directory-level record assembled from search snippets / partner sites / status pages "
                         "(aps.anl.gov blocks automated fetch). measurables DERIVED from technique keywords. "
                         + b.get("notes", "")
            } for u in b.get("source_urls", ["(none)"])[:2]],
            "record_depth": "directory",
        }
        if b.get("energy_range_eV"):
            rec["probe"] = {"type": "photon", "energy_range_eV": b["energy_range_eV"],
                            "tunable": b["energy_range_eV"][0] != b["energy_range_eV"][1]}
        records.append(rec)
    out = os.path.join(here, "aps_directory.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} APS directory records to {out}")

if __name__ == "__main__":
    main()
