#!/usr/bin/env python3
"""Build directory-depth capability records for ALS beamlines from the
facility directory table (retrieved from https://als.lbl.gov/beamlines/ on
2026-08-27). Beamlines that already have deep records are skipped.

Directory records carry only what the directory table states, plus
measurables DERIVED from a technique->measurables map (marked in provenance).
"""
import json
import os

RETRIEVED = "2026-08-27"
SOURCE = "https://als.lbl.gov/beamlines/"

# (id, name, energy_min_eV, energy_max_eV, status, [techniques])
DIRECTORY = [
    ("10.0.1", "Angle- and Spin-Resolved Photoelectron Spectroscopy", 17, 350, "operational", ["ARPES", "spin-resolved photoemission"]),
    ("10.3.2", "X-Ray Fluorescence Microprobe", 2100, 14000, "not-operational", ["XAS", "XFS"]),
    ("11.0.1.1", "PEEM-3 Photoemission Electron Microscope", 160, 1800, "operational", ["magnetic microscopy", "PEEM"]),
    ("11.0.1.2", "Resonant Soft X-Ray Scattering", 165, 1500, "operational", ["coherent scattering", "SAXS", "WAXS"]),
    ("11.0.2.1", "Molecular Environmental Science - APXPS/GIXS", 160, 2000, "operational", ["APXPS", "XAS", "GIXS"]),
    ("11.0.2.2", "Molecular Environmental Science - STXM", 160, 2000, "operational", ["STXM", "XAS"]),
    ("11.3.1", "Tender Nano-Tomography", 5000, 17000, "operational", ["full-field X-ray microscopy", "tomography"]),
    ("11.3.2", "EUV Lithography Photomask Imaging (SHARP)", 50, 1000, "operational", ["full-field X-ray microscopy"]),
    ("12.0.1", "EUV Lithography Nanopatterning", 92, 92, "operational", ["photoelectron spectroscopy"]),
    ("12.0.2", "Coherent X-Ray Scattering", 400, 1300, "operational", ["coherent scattering", "magnetic scattering"]),
    ("12.2.1", "Small-Molecule Crystallography", 6000, 28000, "operational", ["small-molecule crystallography"]),
    ("12.3.2", "Microdiffraction", 6000, 22000, "closed", ["X-ray microdiffraction"]),
    ("2.0.1", "Macromolecular Crystallography (GEMINI)", 3000, 18000, "operational", ["protein crystallography"]),
    ("2.1", "National Center for X-Ray Tomography", 400, 1300, "closed", ["full-field X-ray microscopy", "tomography"]),
    ("2.4", "Synchrotron Infrared Nanospectroscopy (SINS)", 0.02, 0.50, "operational", ["infrared microscopy", "infrared spectroscopy"]),
    ("3.2.1", "LIGA", 2000, 15000, "operational", ["microscopy", "X-ray footprinting"]),
    ("3.3.1", "X-Ray Footprinting", 2000, 12000, "operational", ["X-ray footprinting"]),
    ("3.3.2", "General X-Ray Testing Station", 4000, 20000, "operational", ["XRD", "full-field X-ray microscopy"]),
    ("4.0.2", "Magnetic Spectroscopy and Scattering", 400, 1500, "operational", ["magnetic spectroscopy", "resonant scattering"]),
    ("4.2.2", "Macromolecular Crystallography (MBC)", 7000, 15000, "operational", ["protein crystallography"]),
    ("5.0.1", "Macromolecular Crystallography (BCSB)", 12700, 12700, "operational", ["protein crystallography"]),
    ("5.0.2", "Macromolecular Crystallography (BCSB)", 5000, 16000, "operational", ["protein crystallography"]),
    ("5.0.3", "Macromolecular Crystallography (BCSB)", 12700, 12700, "operational", ["protein crystallography"]),
    ("5.3.2.1", "Scanning Transmission X-Ray Microscopy", 600, 2000, "restricted", ["STXM", "ptychography"]),
    ("5.3.2.2", "Polymer STXM", 250, 780, "operational", ["STXM", "XAS"]),
    ("5.4", "Synchrotron Infrared Nanospectroscopy (SINS)", 0.05, 1.40, "operational", ["infrared microscopy", "infrared spectroscopy"]),
    ("6.0.1", "Energy, Catalytic, and Chemical Science (AMBER)", 250, 2500, "commissioning", ["XAS", "XES", "XFS", "RIXS"]),
    ("6.0.2", "Double-Dispersion RIXS (QERLIN)", 290, 1500, "commissioning", ["RIXS", "XAS", "XES"]),
    ("6.1.2", "Full-Field Transmission Soft X-Ray Microscopy", 300, 1000, "operational", ["magnetic microscopy", "full-field X-ray microscopy"]),
    ("6.3.1", "Magnetic Spectroscopy/Materials Science", 250, 2000, "operational", ["magnetic spectroscopy", "XAS"]),
    ("6.3.2", "Calibration, Optics Testing, Spectroscopy", 25, 1300, "operational", ["spectroscopy"]),
    ("7.0.1.1", "COSMIC Scattering", 250, 1600, "operational", ["SAXS", "XRD", "coherent scattering"]),
    ("7.0.2", "Quantum Materials Growth (MAESTRO)", 20, 1000, "operational", ["ARPES", "PEEM", "SPEM"]),
    ("7.3.1", "High-Pressure In Situ Soft X-Ray Spectroscopy", 250, 1500, "operational", ["XAS"]),
    ("7.3.3", "Small- and Wide-Angle X-Ray Scattering", 10000, 10000, "operational", ["SAXS", "WAXS", "GISAXS"]),
    ("8.2.1", "Macromolecular Crystallography (BCSB/HHMI)", 6000, 18000, "operational", ["protein crystallography"]),
    ("8.2.2", "Macromolecular Crystallography (BCSB/HHMI)", 6000, 18000, "operational", ["protein crystallography"]),
    ("8.3.1", "Macromolecular Crystallography (TomAlberTron)", 5000, 17000, "operational", ["protein crystallography", "XRD"]),
    ("9.0.1", "Chemical Transformations (Soft X-Ray)", 15, 750, "operational", ["photoelectron spectroscopy", "XAS"]),
    ("9.0.2", "Chemical Transformations (Vacuum Ultraviolet)", 7.50, 24, "operational", ["mass spectrometry"]),
    ("9.3.1", "Tender X-Ray Spectroscopy", 2320, 6000, "unknown", ["APXPS", "XAS"]),
]

# Technique -> derived measurables (what a user learns). Used only for
# directory-depth records; deep records carry sourced measurables.
TECH_MEASURABLES = {
    "ARPES": ["electronic band structure", "fermi surface"],
    "spin-resolved photoemission": ["spin texture of electronic bands"],
    "XAS": ["oxidation state", "local chemical environment and speciation", "unoccupied electronic states"],
    "XFS": ["elemental composition", "trace element distribution"],
    "XES": ["occupied electronic states", "chemical bonding states"],
    "RIXS": ["electronic excitations", "chemical bonding states"],
    "magnetic microscopy": ["magnetic domain structure"],
    "PEEM": ["surface electronic and magnetic structure maps"],
    "SPEM": ["spatially resolved electronic structure"],
    "coherent scattering": ["nanoscale structure and dynamics", "domain correlations"],
    "magnetic scattering": ["magnetic ordering and domain correlations"],
    "resonant scattering": ["element-specific magnetic and charge order"],
    "SAXS": ["nanoscale size and shape distributions", "particle/pore structure 1-100 nm"],
    "WAXS": ["crystalline phase identification", "molecular packing"],
    "GISAXS": ["thin-film nanostructure and morphology"],
    "GIXS": ["thin-film and interface structure"],
    "APXPS": ["surface chemical composition under reaction conditions", "oxidation states"],
    "STXM": ["nanoscale chemical composition maps", "chemical speciation maps"],
    "ptychography": ["nanoscale morphology (highest-resolution x-ray imaging)"],
    "tomography": ["3d microstructure", "internal morphology (non-destructive)"],
    "full-field X-ray microscopy": ["2d/3d morphology at nanometer to micron scale"],
    "XRD": ["crystal structure", "phase identification", "lattice parameters"],
    "X-ray microdiffraction": ["local strain and crystal orientation maps"],
    "small-molecule crystallography": ["small-molecule crystal structure"],
    "protein crystallography": ["macromolecular crystal structure"],
    "X-ray footprinting": ["solvent-accessible surface / macromolecular conformation in solution"],
    "infrared microscopy": ["chemical composition maps at micron scale"],
    "infrared spectroscopy": ["molecular functional-group composition", "molecular vibrational spectra"],
    "photoelectron spectroscopy": ["electronic structure", "surface chemical composition"],
    "mass spectrometry": ["molecular mass and fragmentation (gas-phase chemistry)"],
    "spectroscopy": ["optical constants and calibration spectra"],
    "microscopy": ["microstructure imaging"],
}

TECH_FULLNAMES = {
    "ARPES": "angle-resolved photoemission spectroscopy",
    "APXPS": "ambient-pressure X-ray photoelectron spectroscopy",
    "XAS": "X-ray absorption spectroscopy",
    "XES": "X-ray emission spectroscopy",
    "XFS": "X-ray fluorescence spectroscopy",
    "RIXS": "resonant inelastic X-ray scattering",
    "SAXS": "small-angle X-ray scattering",
    "WAXS": "wide-angle X-ray scattering",
    "GISAXS": "grazing-incidence small-angle X-ray scattering",
    "GIXS": "grazing-incidence X-ray scattering",
    "STXM": "scanning transmission X-ray microscopy",
    "XRD": "X-ray diffraction",
    "PEEM": "photoemission electron microscopy",
    "SPEM": "scanning photoemission microscopy",
}

def main():
    records = []
    for bid, name, emin, emax, status, techs in DIRECTORY:
        measurables = []
        for t in techs:
            for m in TECH_MEASURABLES.get(t, []):
                if m not in measurables:
                    measurables.append(m)
        if not measurables:
            measurables = ["(unmapped technique - needs deep extraction)"]
        records.append({
            "id": f"als-{bid}",
            "facility": "ALS",
            "institution": "LBNL",
            "name": f"{name} (Beamline {bid})",
            "instrument_class": "synchrotron-beamline",
            "status": status,
            "techniques": [
                {"name": TECH_FULLNAMES.get(t, t), **({"acronym": t} if t in TECH_FULLNAMES else {})}
                for t in techs
            ],
            "measurables": measurables,
            "probe": {"type": "photon", "energy_range_eV": [emin, emax], "tunable": emin != emax},
            "access": {"mechanism": "general user proposal", "proposal_system": "ALSHub"},
            "urls": [f"https://als.lbl.gov/beamlines/{bid.replace('.', '-')}/"],
            "provenance": [{
                "source_url": SOURCE,
                "retrieved": RETRIEVED,
                "notes": "Directory-table extraction only (name, techniques, energy range, status). measurables are DERIVED from a technique->measurable map, not sourced from beamline documentation."
            }],
            "record_depth": "directory",
        })
    out = os.path.join(os.path.dirname(__file__), "als_directory.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} directory records to {out}")

if __name__ == "__main__":
    main()
