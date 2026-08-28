#!/usr/bin/env python3
"""Rebuild aps_directory.json from the user-saved copy of the official APS
Beamline Directory (www.aps.anl.gov/Beamlines/Directory, saved 2026-08-28;
Cloudflare blocks automated fetch). Authoritative for beamline list,
disciplines, techniques, energy ranges, access modes. The table does not
carry status; statuses are retained from prior per-beamline sources where
known, else 'unknown'."""
import json, re, html, os
from build_aps_directory import derive_measurables, slug

SAVED = "/Users/ian/AAA/Code/GM_Instruments/APS_Info/APS Beamline Directory | Advanced Photon Source.html"
RET = "2026-08-28"
SKIP_PREFIX = ("2-BM", "2-ID-E", "13-ID-E")   # deep records in aps_deep.json

def lis(cell):
    items = re.findall(r'<li>(.*?)</li>', cell, flags=re.S)
    if not items: items = [cell]
    return [' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', i)).split()) for i in items if i.strip()]

def kev_to_ev(ranges):
    vals = []
    for r in ranges:
        for m in re.finditer(r'([\d.]+)\s*-\s*([\d.]+)\s*keV', r):
            vals += [float(m.group(1))*1000, float(m.group(2))*1000]
        for m in re.finditer(r'^([\d.]+)\s*keV$', r.strip()):
            vals.append(float(m.group(1))*1000)
    return [min(vals), max(vals)] if vals else None

raw = open(SAVED, encoding="utf-8", errors="replace").read()
old = {r["id"]: r for r in json.load(open("aps_directory.json"))}
rows = re.findall(r'<tr[^>]*class="beamline-directory beamline"[^>]*>(.*?)</tr>', raw, flags=re.S)
records = []
for row in rows:
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, flags=re.S)
    if len(cells) < 5: continue
    name_m = re.search(r'href="(/Beamlines/Beamline-Directory/\d+)">([^<]+)', cells[0])
    label = name_m.group(2).strip(); url = "https://www.aps.anl.gov" + name_m.group(1)
    if any(label.startswith(p) for p in SKIP_PREFIX): continue
    disciplines, techs, energy, access = lis(cells[1]), lis(cells[2]), lis(cells[3]), lis(cells[4])
    rid = slug(label.split(",")[0])
    prior = old.get(rid) or next((v for k, v in old.items() if k.startswith(rid) or rid.startswith(k)), None)
    rec = {"id": rid, "facility": "APS", "institution": "ANL",
           "name": f"APS Beamline {label} ({'; '.join(disciplines)})",
           "instrument_class": "synchrotron-beamline",
           "status": prior["status"] if prior else "unknown",
           "techniques": [{"name": t} for t in techs],
           "measurables": derive_measurables(techs),
           "access": {"mechanism": "APS general user proposal; modes: " + ", ".join(access),
                      "proposal_system": "APS User Portal"},
           "urls": [url],
           "provenance": ([{"source_url": "https://www.aps.anl.gov/Beamlines/Directory",
             "retrieved": RET,
             "notes": "Official APS Beamline Directory, user-saved copy (Cloudflare blocks automated fetch). Authoritative: beamline list, disciplines, techniques, energy ranges, access modes. Status not in table; " + ("retained from prior sources." if prior else "unknown.")}]
             + (prior["provenance"] if prior else [])),
           "record_depth": "directory"}
    ev = kev_to_ev(energy)
    if ev: rec["probe"] = {"type": "photon", "energy_range_eV": ev, "tunable": ev[0] != ev[1],
                           "flux_notes": "energy ranges verbatim: " + "; ".join(energy)}
    records.append(rec)
json.dump(records, open("aps_directory.json", "w"), indent=1)
print(f"wrote {len(records)} records from official directory")
