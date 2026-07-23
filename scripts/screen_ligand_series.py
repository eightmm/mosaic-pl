#!/usr/bin/env python3
"""Fragment-library screening driver for CASP17 ligand-series targets (L01/L02).

Fixed-receptor docking of the full SMILES library into multiple pockets, reusing
this repo's docking stack (meeko ligand prep, P2Rank pockets, AutoDock-GPU +
Vina). SLURM-array friendly: `dock`/`prep-ligands` chunk the fragment list by
``--chunk`` of ``--nchunks`` so an array task handles a slice.

Subcommands:
  prep-ligands  SMILES csv -> per-fragment SDF + PDBQT       (needs pred venv tools)
  pockets       receptor PDB -> top-N pocket centers (P2Rank) -> pockets.json
  dock          dock a fragment chunk into every pocket with ADG (nrun) + Vina

Layout under inputs/ligand_series/<T>/:
  receptor.fasta, ligands.csv, receptor.pdb (linked from fold),
  ligands_pdbqt/<id>.pdbqt, pockets.json,
  poses/<engine>/pocket<k>/<id>/...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PREP_PY = REPO / ".venvs/protenix-dock/bin/python"   # rdkit+meeko+ambertools (ligand prep)
VINA_PY = REPO / ".venvs/protenix-dock/bin/python"
ADG = REPO / ".local/bin/autodock_gpu_128wi"
AUTOGRID = REPO / ".local/bin/autogrid4"
PRANK = REPO / ".local/p2rank_2.5/prank"
BASE = REPO / "inputs/ligand_series"
BOX = [22.5, 22.5, 22.5]
SPACING = 0.375


def load_ligs(target: str) -> list[tuple[str, str]]:
    out = []
    for line in (BASE / target / "ligands.csv").read_text().splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        lid, smi = line.split(",", 1)
        out.append((lid.strip(), smi.strip()))
    return out


def chunk_of(items, chunk, nchunks):
    return [x for i, x in enumerate(items) if i % nchunks == chunk]


# --------------------------------------------------------------------------- #
def prep_ligands(target, chunk, nchunks):
    ligs = chunk_of(load_ligs(target), chunk, nchunks)
    outdir = BASE / target / "ligands_pdbqt"; outdir.mkdir(parents=True, exist_ok=True)
    sdfdir = BASE / target / "ligands_sdf"; sdfdir.mkdir(parents=True, exist_ok=True)
    payload = [{"id": lid, "smi": smi} for lid, smi in ligs]
    script = (
        "import sys; sys.path.insert(0, %r)\n" % str(REPO / "scripts") +
        "from prepare_docking_inputs import smiles_to_sdf, sdf_to_pdbqt\n"
        "from pathlib import Path\n"
        "recs = %r\n" % payload +
        "ok = 0\n"
        "for r in recs:\n"
        "    lid, smi = r['id'], r['smi']\n"
        "    pq = Path(%r)/f'{lid}.pdbqt'\n" % str(outdir) +
        "    if pq.exists(): ok += 1; continue\n"
        "    sdf = Path(%r)/f'{lid}.sdf'\n" % str(sdfdir) +
        "    try:\n"
        "        if smiles_to_sdf(smi, sdf, lid) and sdf_to_pdbqt(sdf, pq): ok += 1\n"
        "        else: print('SKIP', lid)\n"
        "    except Exception as e: print('FAIL', lid, e)\n"
        "print(f'prepped {ok}/{len(recs)}')\n"
    )
    subprocess.run([str(PREP_PY), "-c", script], check=False)


# --------------------------------------------------------------------------- #
def pockets(target, receptor_pdb, topn):
    tdir = BASE / target
    rec = Path(receptor_pdb)
    pdir = tdir / "_prank"; pdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(PRANK), "predict", "-f", str(rec), "-o", str(pdir)], check=True)
    # p2rank writes <name>.pdb_predictions.csv
    pred = next(pdir.glob("*_predictions.csv"), None) or next(pdir.rglob("*_predictions.csv"), None)
    centers = []
    if pred:
        rows = pred.read_text().splitlines()
        hdr = [h.strip() for h in rows[0].split(",")]
        cx, cy, cz = hdr.index("center_x"), hdr.index("center_y"), hdr.index("center_z")
        for r in rows[1:topn + 1]:
            c = r.split(",")
            centers.append([float(c[cx]), float(c[cy]), float(c[cz])])
    (tdir / "pockets.json").write_text(json.dumps({"receptor": str(rec), "centers": centers}, indent=2))
    print(f"{target}: {len(centers)} pockets -> {tdir/'pockets.json'}")


# --------------------------------------------------------------------------- #
def _parse_types(pdbqt: Path):
    types, seen = [], set()
    for line in pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            tok = line[77:79].strip() if len(line) >= 79 else line.split()[-1].strip()
            if tok and tok not in seen:
                seen.add(tok); types.append(tok)
    return types


def _dock_adg(rec_pdbqt, lig_pdbqt, center, out_dir, nrun, seed=101):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = out_dir / "grid"; grid.mkdir(exist_ok=True)
    lig_types = _parse_types(lig_pdbqt) or ["A", "C", "HD", "N", "NA", "OA", "SA"]
    rec_types = list(dict.fromkeys(["A", "C", "HD", "N", "NA", "OA", "SA"]
                                   + _parse_types(rec_pdbqt) + lig_types))
    npts = [max(1, int(s / SPACING)) for s in BOX]
    gpf = [f"npts {npts[0]} {npts[1]} {npts[2]}", "gridfld receptor.maps.fld",
           f"spacing {SPACING}", f"receptor_types {' '.join(rec_types)}",
           f"ligand_types {' '.join(lig_types)}", f"receptor {rec_pdbqt}",
           f"gridcenter {center[0]} {center[1]} {center[2]}", "smooth 0.5"]
    gpf += [f"map receptor.{t}.map" for t in lig_types]
    gpf += ["elecmap receptor.e.map", "dsolvmap receptor.d.map", "dielectric -0.1465"]
    (grid / "receptor.gpf").write_text("\n".join(gpf) + "\n")
    subprocess.run([str(AUTOGRID), "-p", "receptor.gpf", "-l", "autogrid.log"],
                   cwd=str(grid), check=True)
    subprocess.run([str(ADG), "--ffile", str(grid / "receptor.maps.fld"),
                    "--lfile", str(lig_pdbqt), "--nrun", str(nrun),
                    "--nev", "1500000", "--ngen", "27000", "--heuristics", "1",
                    "--autostop", "1", "--seed", str(seed),
                    "--resnam", str(out_dir / "docking")], check=True)


def _dock_vina(rec_pdbqt, lig_pdbqt, center, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    snippet = (
        "from vina import Vina\n"
        "v = Vina(sf_name='vina', seed=101, verbosity=0)\n"
        "v.set_receptor(%r)\n" % str(rec_pdbqt) +
        "v.set_ligand_from_file(%r)\n" % str(lig_pdbqt) +
        "v.compute_vina_maps(center=%r, box_size=%r)\n" % (list(center), BOX) +
        "v.dock(exhaustiveness=8, n_poses=9)\n"       # Vina defaults
        "v.write_poses(%r, n_poses=9, overwrite=True)\n" % str(out_dir / "vina_out.pdbqt")
    )
    subprocess.run([str(VINA_PY), "-c", snippet], check=False)


def dock(target, engine, chunk, nchunks, nrun):
    tdir = BASE / target
    pk = json.loads((tdir / "pockets.json").read_text())
    rec_pdbqt = tdir / "receptor.pdbqt"
    centers = pk["centers"]
    ligs = chunk_of(load_ligs(target), chunk, nchunks)
    lpdir = tdir / "ligands_pdbqt"
    done = 0
    for lid, _ in ligs:
        lig_pdbqt = lpdir / f"{lid}.pdbqt"
        if not lig_pdbqt.exists():
            print(f"  {lid}: no pdbqt, skip"); continue
        for k, c in enumerate(centers):
            out = tdir / "poses" / engine / f"pocket{k}" / lid
            if (out / ("docking.dlg" if engine == "adg" else "vina_out.pdbqt")).exists():
                continue
            try:
                if engine == "adg":
                    _dock_adg(rec_pdbqt, lig_pdbqt, c, out, nrun)
                else:
                    _dock_vina(rec_pdbqt, lig_pdbqt, c, out)
                done += 1
            except Exception as e:
                print(f"  {lid} pocket{k} {engine} FAIL: {e}", file=sys.stderr)
    print(f"{target}[{engine}] chunk {chunk}/{nchunks}: docked {done}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-ligands"); p.add_argument("--target", required=True)
    p.add_argument("--chunk", type=int, default=0); p.add_argument("--nchunks", type=int, default=1)
    p = sub.add_parser("pockets"); p.add_argument("--target", required=True)
    p.add_argument("--receptor", required=True); p.add_argument("--topn", type=int, default=4)
    p = sub.add_parser("dock"); p.add_argument("--target", required=True)
    p.add_argument("--engine", choices=["adg", "vina"], required=True)
    p.add_argument("--chunk", type=int, default=0); p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("--nrun", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "prep-ligands":
        prep_ligands(a.target, a.chunk, a.nchunks)
    elif a.cmd == "pockets":
        pockets(a.target, a.receptor, a.topn)
    elif a.cmd == "dock":
        dock(a.target, a.engine, a.chunk, a.nchunks, a.nrun)


if __name__ == "__main__":
    raise SystemExit(main())
