#!/usr/bin/env python3
"""Frame-consistency audit for a finished CASP17 run.

Independent QA checker — no fixes, just reports. Verifies every coordinate
artefact of a target run lives in a single common frame:

  1. Cofold reference cif used by docking-prep is ``*_aligned.cif``.
  2. ``binding_site_predictions[*]['center']`` are within receptor-extent + 30 Å
     of the receptor heavy-atom centroid (strong frame-bug signal otherwise).
  3. Track 1 / Track 2 docked poses have heavy-atom centroids near the
     receptor (sanity bound: 30 Å past the receptor extent, same gate).
  4. ``alignment_summary.json`` rmsd_after is bounded — large values mean
     the common frame itself is poorly defined.

Run on a single target dir or a whole runs root. Exits non-zero only when
the user passes ``--strict`` and at least one target fails.

Usage::

    .venv/bin/python scripts/check_frame_consistency.py \\
        --runs-dir experiments/novel2025_runs/runs

    .venv/bin/python scripts/check_frame_consistency.py \\
        --target-dir experiments/novel2025_runs/runs/7hqq_input/7hqq_input
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np


@dataclass
class TargetReport:
    target: str
    cofold_ref_aligned: bool | None = None
    cofold_ref_path: str | None = None
    receptor_centroid: list[float] | None = None
    receptor_extent: float | None = None
    n_sources: int = 0
    n_source_outliers: int = 0
    source_outliers: list[dict] = field(default_factory=list)
    n_pose_files: int = 0
    n_pose_outliers: int = 0
    pose_outliers: list[dict] = field(default_factory=list)
    alignment_n: int | None = None
    alignment_rmsd_max: float | None = None
    issues: list[str] = field(default_factory=list)


def _polymer_heavy_atoms(path: Path) -> np.ndarray | None:
    """Return (N, 3) heavy-atom array of the polymer chains in a structure
    (mmCIF or PDB). Excludes hydrogens, waters, ligands, ions."""
    try:
        st = gemmi.read_structure(str(path))
    except Exception:
        return None
    pts: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.entity_type != gemmi.EntityType.Polymer:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    pts.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    if not pts:
        return None
    return np.asarray(pts, dtype=float)


def _sdf_centroid(path: Path) -> np.ndarray | None:
    """Return (3,) centroid of the first molecule's heavy atoms in an SDF."""
    try:
        from rdkit import Chem  # type: ignore
        suppl = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=False)
        for mol in suppl:
            if mol is None:
                continue
            try:
                conf = mol.GetConformer()
            except ValueError:
                continue
            coords = np.asarray(
                [[conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol.GetNumAtoms())]
            )
            if len(coords) == 0:
                continue
            return coords.mean(axis=0)
    except Exception:
        return None
    return None


def check_target(run: Path, max_padding: float = 30.0) -> TargetReport:
    """Audit a single ``<target>_input/<target>_input/`` run dir."""
    rep = TargetReport(target=run.name)
    summary_path = run / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary_path.exists():
        rep.issues.append("docking_prep_summary.json missing")
        return rep
    summary = json.loads(summary_path.read_text())

    # 1. Cofold reference cif used.
    cof_ref = summary.get("cofolding_structure")
    if cof_ref:
        rep.cofold_ref_path = cof_ref
        rep.cofold_ref_aligned = "_aligned" in Path(cof_ref).name
        if not rep.cofold_ref_aligned:
            rep.issues.append(
                f"cofold reference is UNALIGNED: {Path(cof_ref).name}"
            )

    # 2. Receptor centroid + extent from receptor.pdb (the canonical Track 1
    #    receptor; everything must be within range of this).
    rec_pdb = run / "inputs" / "docking" / "receptor.pdb"
    rec_pts = _polymer_heavy_atoms(rec_pdb) if rec_pdb.exists() else None
    if rec_pts is None or len(rec_pts) == 0:
        rep.issues.append("receptor.pdb missing or has no polymer atoms")
        return rep
    rec_centroid = rec_pts.mean(axis=0)
    rec_extent = float(np.linalg.norm(rec_pts - rec_centroid, axis=1).max())
    rep.receptor_centroid = [round(float(x), 3) for x in rec_centroid]
    rep.receptor_extent = round(rec_extent, 2)
    max_allowed = rec_extent + max_padding

    # 3. Each binding-site source center vs receptor.
    bs_pred = summary.get("binding_site_predictions") or {}
    rep.n_sources = len(bs_pred)
    for name, payload in bs_pred.items():
        c = payload.get("center")
        if not c or len(c) != 3:
            continue
        d = float(np.linalg.norm(np.asarray(c) - rec_centroid))
        if d > max_allowed:
            rep.source_outliers.append({
                "source": name,
                "center": [round(float(x), 3) for x in c],
                "dist": round(d, 2),
            })
    rep.n_source_outliers = len(rep.source_outliers)
    if rep.n_source_outliers:
        rep.issues.append(
            f"{rep.n_source_outliers}/{rep.n_sources} binding-site sources "
            f"> {max_allowed:.0f} Å from receptor centroid"
        )

    # 4. Spot-check Track 1 + Track 2 staged poses (a handful per source).
    pose_root = run / "outputs"
    pose_files = []
    for sub in (
        "vina_*", "autodock_gpu_*", "protenix_dock", "template_docking",
    ):
        for p in pose_root.glob(f"{sub}/**/*.sdf"):
            pose_files.append(p)
            if len(pose_files) >= 80:
                break
        if len(pose_files) >= 80:
            break
    rep.n_pose_files = len(pose_files)
    for sdf in pose_files:
        c = _sdf_centroid(sdf)
        if c is None:
            continue
        d = float(np.linalg.norm(c - rec_centroid))
        if d > max_allowed:
            rep.pose_outliers.append({
                "sdf": str(sdf.relative_to(run)),
                "centroid": [round(float(x), 3) for x in c],
                "dist": round(d, 2),
            })
    rep.n_pose_outliers = len(rep.pose_outliers)
    if rep.n_pose_outliers:
        rep.issues.append(
            f"{rep.n_pose_outliers}/{rep.n_pose_files} sampled pose SDFs "
            f"have centroid > {max_allowed:.0f} Å from receptor centroid"
        )

    # 4b. Cross-source consistency. If ``cofolding_1`` (cofold's predicted
    #     ligand position — definitely in the aligned receptor's frame) is
    #     within the receptor but ``template_consensus_1`` (highest-evidence
    #     template cluster) is on the OTHER side of the receptor, that's a
    #     strong frame-mismatch hint even when both pass the gross
    #     receptor-centroid range gate. Threshold 30 Å between them is
    #     generous for real multi-pocket / cryptic-site cases.
    cof1 = bs_pred.get("cofolding_1", {}).get("center")
    tc1 = bs_pred.get("template_consensus_1", {}).get("center")
    if cof1 and tc1:
        d_cof_tc = float(np.linalg.norm(np.asarray(cof1) - np.asarray(tc1)))
        if d_cof_tc > 30.0:
            rep.issues.append(
                f"cofolding_1 ↔ template_consensus_1 = {d_cof_tc:.1f} Å apart "
                f"(possible frame mismatch — they should usually share a pocket)"
            )

    # 5. Alignment quality from the bridge's own summary.
    align_path = run / "outputs" / "alignment_summary.json"
    if align_path.exists():
        try:
            align = json.loads(align_path.read_text())
            details = align.get("details", [])
            rmsds = [d.get("rmsd_after") for d in details
                     if d.get("rmsd_after") is not None]
            rep.alignment_n = align.get("n_aligned", 0)
            if rmsds:
                rep.alignment_rmsd_max = round(max(rmsds), 3)
                # >5 Å after Kabsch on CA atoms suggests sequence-mismatch /
                # very different conformations — the common frame is wobbly.
                if rep.alignment_rmsd_max > 5.0:
                    rep.issues.append(
                        f"alignment rmsd_after max = {rep.alignment_rmsd_max} Å "
                        f"(common frame may be unreliable)"
                    )
        except Exception as e:
            rep.issues.append(f"alignment_summary.json unreadable: {e}")
    return rep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--target-dir", type=Path, default=None,
                   help="Single ``<target>_input/<target>_input/`` dir.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-padding", type=float, default=30.0,
                   help="Distance past receptor extent considered outlier (Å).")
    p.add_argument("--output-tsv", type=Path,
                   default=Path("/tmp/frame_consistency.tsv"))
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any target has issues.")
    args = p.parse_args()

    if args.target_dir:
        runs = [args.target_dir]
    elif args.runs_dir:
        runs = []
        for outer in sorted(args.runs_dir.iterdir()):
            if not outer.is_dir() or not outer.name.endswith("_input"):
                continue
            nested = outer / outer.name
            if nested.exists():
                runs.append(nested)
            elif (outer / "inputs" / "docking").exists() or (outer / "outputs").exists():
                runs.append(outer)
    else:
        print("error: provide --runs-dir or --target-dir")
        return 2
    print(f"[frame-check] {len(runs)} target(s), max_padding={args.max_padding} Å")

    reports: list[TargetReport] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_target, r, args.max_padding): r for r in runs}
        for fut in as_completed(futs):
            try:
                reports.append(fut.result())
            except Exception as e:
                reports.append(TargetReport(
                    target=futs[fut].name,
                    issues=[f"exception: {e}"],
                ))

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "target", "cofold_ref_aligned", "n_sources",
            "n_source_outliers", "n_pose_files", "n_pose_outliers",
            "alignment_rmsd_max", "n_issues", "issues",
        ])
        for r in sorted(reports, key=lambda r: r.target):
            w.writerow([
                r.target, r.cofold_ref_aligned, r.n_sources,
                r.n_source_outliers, r.n_pose_files, r.n_pose_outliers,
                r.alignment_rmsd_max, len(r.issues),
                " | ".join(r.issues),
            ])
    print(f"[frame-check] tsv: {args.output_tsv}")

    n_clean = sum(1 for r in reports if not r.issues)
    n_dirty = len(reports) - n_clean
    print()
    print(f"=== Frame-consistency summary ===")
    print(f"  clean: {n_clean}/{len(reports)}")
    print(f"  with issues: {n_dirty}")
    if n_dirty:
        print()
        print("issue breakdown (top 10):")
        from collections import Counter
        per_issue = Counter()
        for r in reports:
            for issue in r.issues:
                # Bucket by short prefix so similar variants collapse.
                key = issue.split(":")[0]
                per_issue[key] += 1
        for k, v in per_issue.most_common(10):
            print(f"  {v:>4}× {k}")
        print()
        print("first 5 dirty targets:")
        for r in reports:
            if r.issues:
                print(f"  {r.target}: {' | '.join(r.issues)}")
                if reports.index(r) >= 4:
                    break
    return 1 if (args.strict and n_dirty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
