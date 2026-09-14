#!/usr/bin/env python3
"""Bundle the R2386 donor crystals, aligned into one RNA frame, with their consensus.

The solvent prediction rests on two things that live only inside
``cluster_solvent_donors.py`` as numbers: *where each donor crystal lands* once
it is superposed onto the reference frame, and *which sites survive* the van der
Waals merge. Neither is inspectable — the pipeline emits a JSON of coordinates
and nothing that can be opened in a viewer.

This writes both out as structures in a single frame:

``aligned/<PDB>_aligned.pdb``
    Each donor's intron chain renumbered onto the target and transformed into
    the frame, with its own MG / K / HOH kept as HETATM. Loading all of them at
    once shows how tightly the crystals agree and where the solvent shells
    overlap.
``consensus_sites.pdb``
    One HETATM per merged site. ``occupancy`` carries the number of distinct
    donor crystals supporting it and ``B`` the distance to the nearest RNA atom,
    so a viewer can colour or filter by evidence without re-reading the JSON.
``frame_<ID>.pdb``
    The reference frame's own RNA, target-numbered — the common ground every
    other file is expressed in.
``transforms.json``
    Per donor: rotation, translation, P-atom RMSD and shared-atom count, so the
    same placement can be reproduced on the original mmCIF rather than trusted.

The donor set is gated exactly as the clustering step gates it (coverage window
and superposition RMSD), so the bundle contains the same crystals the consensus
was actually built from. Rejected entries are still listed in ``manifest.tsv``
with the reason, because a donor silently missing from a bundle is
indistinguishable from one that was never considered.

Usage:
    uv run python scripts/export_solvent_donor_bundle.py \
      --templates 'data/solvent_templates/R2386/*.cif' \
      --sequence inputs/R2386.fasta \
      --consensus data/solvent_templates/R2386/consensus_sites.json \
      --frame 3G78 \
      --out experiments/CASP17/R2386/R2386_donor_bundle.tgz
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from make_solvent_ts_submission import (  # noqa: E402
    SOLVENT_ALIASES,
    SOLVENT_NAMES,
    load_template,
    superpose,
)

#: PDB residue names for the solvent species we carry.
RESNAME = {"HOH": "HOH", "MG": "MG", "K": "K", "NA": "NA"}
#: Element symbol written in columns 77-78.
ELEMENT = {"HOH": "O", "MG": "MG", "K": "K", "NA": "NA"}
#: Atom name written in columns 13-16.
ATOMNAME = {"HOH": "O", "MG": "MG", "K": "K", "NA": "NA"}


def read_sequence(path: Path) -> str:
    return "".join(ln.strip() for ln in path.read_text().splitlines()
                   if ln.strip() and not ln.startswith(">"))


def atom_element(name: str) -> str:
    """Element of an RNA atom name — the leading alphabetic character."""
    for ch in name:
        if ch.isalpha():
            return ch
    return "C"


def pdb_atom(record: str, serial: int, name: str, resname: str, chain: str,
             resnum: int, xyz, occ: float, bfac: float, element: str) -> str:
    # A 4-character atom name starts in column 13; shorter ones start in 14,
    # which is what viewers use to tell CA (calcium) from C-alpha.
    aname = name if len(name) >= 4 else f" {name:<3s}"
    return (f"{record:<6s}{serial:>5d} {aname:<4s} {resname:>3s} {chain:1s}"
            f"{resnum:>4d}    {xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
            f"{occ:>6.2f}{bfac:>6.2f}          {element:>2s}")


def write_donor_pdb(path: Path, tpl, rot: np.ndarray, trans: np.ndarray,
                    header: list[str]) -> tuple[int, Counter]:
    """Write one donor's RNA + solvent, transformed into the frame."""
    lines = [f"REMARK   1 {h}" for h in header]
    serial = 0
    for num in sorted(tpl.residues):
        resname = tpl.resnames.get(num, "N")
        for aname, xyz in tpl.residues[num].items():
            serial += 1
            p = np.asarray(xyz, dtype=float) @ rot + trans
            lines.append(pdb_atom("ATOM", serial, aname, resname, "A", num, p,
                                  1.0, 0.0, atom_element(aname)))
    lines.append("TER")

    counts: Counter = Counter()
    water_num = 0
    for kind, x, y, z, b in tpl.solvent:
        kind = SOLVENT_ALIASES.get(kind, kind)
        if kind not in SOLVENT_NAMES:
            continue
        serial += 1
        water_num += 1
        counts[kind] += 1
        p = np.array([x, y, z], dtype=float) @ rot + trans
        lines.append(pdb_atom("HETATM", serial, ATOMNAME[kind], RESNAME[kind],
                              "S", water_num, p, 1.0, float(b), ELEMENT[kind]))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return serial, counts


def rna_lines(tpl, chain: str = "A", start: int = 0) -> list[str]:
    """The frame's RNA as ATOM records, target-numbered."""
    out = []
    serial = start
    for num in sorted(tpl.residues):
        resname = tpl.resnames.get(num, "N")
        for aname, xyz in tpl.residues[num].items():
            serial += 1
            out.append(pdb_atom("ATOM", serial, aname, resname, chain, num,
                                [float(v) for v in xyz], 1.0, 0.0,
                                atom_element(aname)))
    return out


def write_consensus_pdb(path: Path, sites: list[dict], frame, header: list[str]) -> Counter:
    """The frame RNA plus one HETATM per merged site.

    The RNA travels with the sites rather than living only in ``frame_*.pdb``:
    opened on its own, a file of bare solvent atoms is an unreadable cloud of
    dots, and whether a site sits in the major groove or out in bulk is the
    whole point. Chain A is the RNA, chain S the sites, so either can be styled
    alone in a viewer.
    """
    lines = [f"REMARK   1 {h}" for h in header]
    rna = rna_lines(frame)
    lines += rna
    lines.append("TER")
    counts: Counter = Counter()
    for i, s in enumerate(sites, start=1):
        kind = s["kind"]
        counts[kind] += 1
        lines.append(pdb_atom("HETATM", len(rna) + i, ATOMNAME[kind], RESNAME[kind],
                              "S", i, [float(v) for v in s["xyz"]],
                              float(min(s["n_donors"], 99)),
                              float(min(s.get("rna_min", 0.0), 999.99)),
                              ELEMENT[kind]))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return counts


def write_frame_pdb(path: Path, tpl, header: list[str]) -> None:
    lines = [f"REMARK   1 {h}" for h in header]
    lines += rna_lines(tpl)
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", required=True, help="glob of donor mmCIFs (quote it)")
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--consensus", type=Path, default=None,
                    help="consensus_sites.json from cluster_solvent_donors.py")
    ap.add_argument("--frame", default="3G78")
    ap.add_argument("--min-coverage", type=float, default=80.0)
    ap.add_argument("--max-coverage", type=float, default=105.0)
    ap.add_argument("--max-rmsd", type=float, default=3.0)
    ap.add_argument("--min-support", type=int, default=1,
                    help="drop consensus sites supported by fewer donor crystals")
    ap.add_argument("--out", type=Path, required=True,
                    help="bundle path; .zip writes a zip, anything else a .tgz")
    args = ap.parse_args()

    seq = read_sequence(args.sequence)
    paths = sorted(Path(p) for p in glob.glob(args.templates))
    if not paths:
        raise SystemExit(f"no templates matched {args.templates!r}")

    frame = None
    loaded: list[tuple[float, object]] = []
    manifest: list[dict] = []
    for p in paths:
        try:
            tpl = load_template(p)
        except Exception as exc:
            manifest.append({"pdb_id": p.stem.upper(), "status": "unreadable",
                             "reason": f"{type(exc).__name__}: {exc}"})
            continue
        cov = 100.0 * len(tpl.p_map()) / max(len(seq), 1)
        if tpl.pdb_id == args.frame.upper():
            frame = tpl
        n_solvent = sum(1 for k, *_ in tpl.solvent
                        if SOLVENT_ALIASES.get(k, k) in SOLVENT_NAMES)
        if not (args.min_coverage <= cov <= args.max_coverage):
            manifest.append({"pdb_id": tpl.pdb_id, "status": "rejected",
                             "coverage": round(cov, 1), "n_solvent": n_solvent,
                             "reason": f"coverage {cov:.1f}% outside "
                                       f"[{args.min_coverage}, {args.max_coverage}]"})
            continue
        if not n_solvent:
            manifest.append({"pdb_id": tpl.pdb_id, "status": "rejected",
                             "coverage": round(cov, 1), "n_solvent": 0,
                             "reason": "no modelled MG / K / HOH"})
            continue
        loaded.append((cov, tpl))
    if frame is None:
        raise SystemExit(f"frame {args.frame} not among the templates")
    loaded.sort(key=lambda t: -t[0])

    work = Path(tempfile.mkdtemp(prefix="donor_bundle_"))
    stem = args.out.name.split(".")[0]
    root = work / stem
    aligned = root / "aligned"
    aligned.mkdir(parents=True)

    transforms: dict[str, dict] = {}
    kept: list[str] = []
    for cov, tpl in loaded:
        if tpl.pdb_id == frame.pdb_id:
            rot, trans, rmsd, nshared = np.eye(3), np.zeros(3), 0.0, len(tpl.p_map())
        else:
            try:
                rot, trans, rmsd, nshared = superpose(tpl, frame)
            except ValueError as exc:
                manifest.append({"pdb_id": tpl.pdb_id, "status": "rejected",
                                 "coverage": round(cov, 1), "reason": str(exc)})
                continue
        if rmsd > args.max_rmsd:
            manifest.append({"pdb_id": tpl.pdb_id, "status": "rejected",
                             "coverage": round(cov, 1), "rmsd": round(rmsd, 2),
                             "n_shared_P": nshared,
                             "reason": f"P-RMSD {rmsd:.2f} A > {args.max_rmsd} — "
                                       f"not this molecule's frame"})
            continue
        header = [
            f"{tpl.pdb_id} superposed onto {frame.pdb_id} on shared P atoms",
            f"P-RMSD {rmsd:.2f} A over {nshared} atoms; coverage {cov:.1f}%",
            "chain A = intron RNA renumbered onto the R2386 target",
            "chain S = this crystal's own MG / K / HOH, B = experimental B-iso",
        ]
        natoms, counts = write_donor_pdb(aligned / f"{tpl.pdb_id}_aligned.pdb",
                                         tpl, rot, trans, header)
        transforms[tpl.pdb_id] = {
            "rotation": [[round(float(v), 6) for v in row] for row in rot],
            "translation": [round(float(v), 6) for v in trans],
            "p_rmsd": round(float(rmsd), 3), "n_shared_P": int(nshared),
            "convention": "frame_xyz = donor_xyz @ rotation + translation",
        }
        manifest.append({
            "pdb_id": tpl.pdb_id, "status": "kept", "coverage": round(cov, 1),
            "resolution": round(float(tpl.resolution), 2), "rmsd": round(rmsd, 2),
            "n_shared_P": int(nshared), "n_residues": len(tpl.residues),
            "n_atoms": natoms, "n_solvent": int(sum(counts.values())),
            "MG": counts.get("MG", 0), "K": counts.get("K", 0),
            "HOH": counts.get("HOH", 0), "reason": "",
        })
        kept.append(tpl.pdb_id)
        print(f"  {tpl.pdb_id:6} cov {cov:5.1f}%  P-RMSD {rmsd:5.2f} A "
              f"({nshared} shared)  solvent {sum(counts.values()):4d}")

    write_frame_pdb(root / f"frame_{frame.pdb_id}.pdb", frame, [
        f"{frame.pdb_id} — reference frame, intron RNA renumbered onto R2386",
        "every other structure in this bundle is expressed in this frame",
    ])

    site_counts: Counter = Counter()
    n_sites = 0
    if args.consensus and args.consensus.exists():
        sites = json.loads(args.consensus.read_text())
        sites = [s for s in sites if s["n_donors"] >= args.min_support]
        n_sites = len(sites)
        site_counts = write_consensus_pdb(root / "consensus_sites.pdb", sites, frame, [
            f"chain A = {frame.pdb_id} intron RNA, target-numbered",
            f"chain S = consensus solvent sites merged from {len(kept)} donors",
            "greedy van der Waals merge: two observations are one site when",
            "closer than the shortest separation their species pair allows",
            "occupancy = number of distinct donor crystals supporting the site",
            "B-factor  = distance to the nearest RNA atom of the frame",
        ])
        shutil.copy(args.consensus, root / "consensus_sites.json")

    (root / "transforms.json").write_text(json.dumps(transforms, indent=1) + "\n")

    cols = ["pdb_id", "status", "resolution", "coverage", "rmsd", "n_shared_P",
            "n_residues", "n_atoms", "n_solvent", "MG", "K", "HOH", "reason"]
    rows = ["\t".join(cols)]
    for m in sorted(manifest, key=lambda m: (m["status"] != "kept", m["pdb_id"])):
        rows.append("\t".join(str(m.get(c, "")) for c in cols))
    (root / "manifest.tsv").write_text("\n".join(rows) + "\n")

    total_solvent = sum(m.get("n_solvent", 0) for m in manifest
                        if m["status"] == "kept")
    (root / "README.txt").write_text(f"""\
R2386 solvent donors, aligned and merged
========================================

frame_{frame.pdb_id}.pdb   reference frame: the intron RNA of {frame.pdb_id},
                   renumbered onto the R2386 target. Everything else here is
                   expressed in this frame, so the files can be opened together
                   without further fitting.

aligned/           {len(kept)} donor crystals, each superposed onto the frame on
                   its shared P atoms. Chain A is the renumbered intron; chain S
                   is that crystal's own MG / K / HOH with the experimental
                   B-factor preserved. {total_solvent} solvent atoms in total.

consensus_sites.pdb  the frame RNA (chain A) plus {n_sites} merged sites
                   (chain S: {', '.join(f'{v} {k}' for k, v in sorted(site_counts.items()))}).
                   It carries the RNA so it can be opened on its own — bare
                   solvent atoms are an unreadable cloud of dots, and whether a
                   site sits in a groove or out in bulk is the whole question.
                   occupancy = how many distinct donor crystals support the site;
                   B-factor = distance to the nearest RNA atom. Two observations
                   were merged when they sat closer than the shortest separation
                   their species pair physically allows, so anything that
                   survives can be written out together without a clash.

transforms.json    the rotation and translation applied to each donor, with the
                   P-atom RMSD and shared-atom count. Convention:
                   frame_xyz = donor_xyz @ rotation + translation. Apply these to
                   the original mmCIF to reproduce the placement independently.

manifest.tsv       every entry considered, kept or rejected, with the reason.
                   Gates: coverage in [{args.min_coverage}, {args.max_coverage}]%
                   of the target, at least one modelled MG / K / HOH, and P-atom
                   superposition RMSD <= {args.max_rmsd} A.

Caveats worth knowing before trusting a site:

- Only the longest nucleic-acid chain of each entry is read. The ligated exon
  and any protein chains are dropped, because the target carries neither.
- Residue numbering is the donor's own plus a fixed offset onto the target. It
  is checked by coverage, not by alignment, so an entry whose construct differs
  will show up as a high P-RMSD rather than a silent mis-numbering.
- A high-occupancy consensus site means many crystals agree, not that the site
  is right: the donors are not independent — several are re-refinements of the
  same crystal form.
""")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".zip":
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(work))
    else:
        with tarfile.open(args.out, "w:gz") as tf:
            tf.add(root, arcname=stem)
    shutil.rmtree(work)

    n_reject = sum(1 for m in manifest if m["status"] != "kept")
    print(f"\ndonors aligned: {len(kept)}   rejected: {n_reject}")
    print(f"consensus sites written: {n_sites}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
