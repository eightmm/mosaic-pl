"""Supplement per_pose_scores.csv with missing pxdock entries.

The original CSV was generated when ``_resolve_pose_file`` had a bug that
returned ``Path("")`` for ``protenix_dock_{lig_id}`` keys (only matched the
bare ``protenix_dock``). Result: pxdock poses had no resolvable SDF, true_rmsd
was never computed, those rows never made it into the CSV.

Fix is in compute_submission_scores.py (Apr 27 patch). This script does an
incremental supplement: for each target, find pxdock poses in
``collect_pose_scores`` output, compute true_rmsd via the same USalign +
RDKit pipeline used by ``score_per_metric.py``, and append rows to CSV.

Skips targets that already have at least one pxdock row (idempotent).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from compute_submission_scores import collect_pose_scores  # noqa: E402
from evaluate import (  # noqa: E402
    cif_path_for, dump_crystal_protein_pdb, load_crystal_ligand,
    load_targets, run_usalign,
    _largest_template_matching_fragment, _reassign_with_fallback,
    _sanitize_loose, _mcs_rmsd,
)
from casp17.geometry import pose_rmsd, transform_mol  # noqa: E402

CSV_PATH = ROOT / "per_pose_scores.csv"
WORK = ROOT / "_per_metric_work"
RUNS = REPO / "experiments" / "runs"


def _conformer_at(sdf: Path, idx: int) -> Chem.Mol | None:
    try:
        supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
        for i, m in enumerate(supp):
            if i == idx:
                return m
    except Exception:
        return None
    return None


def _compute_rmsd(pred_mol, ref_lig, eff_template):
    pred_raw = _largest_template_matching_fragment(pred_mol, eff_template)
    if pred_raw is None:
        return None
    pred_recast, _ = _reassign_with_fallback(pred_raw, eff_template)
    if pred_recast is None:
        pred_recast = _sanitize_loose(pred_raw)
    if pred_recast is None:
        return None
    try:
        r = pose_rmsd(pred_recast, ref_lig)
        if not math.isnan(r):
            return float(r)
    except Exception:
        pass
    return _mcs_rmsd(pred_recast, ref_lig)


def main():
    targets = load_targets()
    print(f"loaded {len(targets)} targets from rcsb_index")

    # Existing pxdock-having targets (skip)
    have_pxdock: set[str] = set()
    target_zone: dict[str, str] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for r in csv.DictReader(f):
                target_zone[r["target"]] = r["seq_zone"]
                if "protenix_dock" in r["source"]:
                    have_pxdock.add(r["target"])

    print(f"already have pxdock: {len(have_pxdock)} targets")
    novel_lgs = sorted((REPO / "experiments" / "submissions").glob("*_input.lg"))
    novel_targets = [p.stem.replace("_input", "") for p in novel_lgs if not p.name.startswith("L10")]
    novel_targets = [t for t in novel_targets if t in targets]
    todo = [t for t in novel_targets if t not in have_pxdock]
    print(f"to supplement: {len(todo)} targets")
    WORK.mkdir(exist_ok=True)

    added_rows = 0
    crashed: list[str] = []

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for i, tgt in enumerate(todo, 1):
            run_dir = RUNS / f"{tgt}_input"
            if not run_dir.exists():
                continue

            try:
                meta = targets[tgt]
                smi = meta["candidate_smiles"][0] if meta["candidate_smiles"] else ""
                ccd = meta["candidate_ccd_codes"][0] if meta["candidate_ccd_codes"] else ""
                if not smi or not ccd:
                    continue
                template = Chem.MolFromSmiles(smi)
                if template is None:
                    continue

                cif = cif_path_for(tgt)
                if cif is None:
                    continue

                # Crystal ligand + protein PDB
                lig_data = load_crystal_ligand(cif, ccd, template)
                if lig_data is None:
                    continue
                ref_lig, eff_template = lig_data

                crystal_pdb = WORK / f"{tgt}_crystal.pdb"
                if not crystal_pdb.exists():
                    if not dump_crystal_protein_pdb(cif, crystal_pdb):
                        continue

                # Cofold protein PDB (required for USalign)
                # score_per_metric.py uses sorted rglob *_aligned.cif first; we mirror
                cof_cifs = sorted(run_dir.glob("outputs/*/**/*_aligned.cif"))
                if not cof_cifs:
                    continue
                first_cif = cof_cifs[0]
                cof_pdb = WORK / f"{tgt}_cofold.pdb"
                if not cof_pdb.exists():
                    import gemmi
                    try:
                        st = gemmi.read_structure(str(first_cif))
                        st.write_pdb(str(cof_pdb))
                    except Exception:
                        continue

                # USalign
                us = run_usalign(cof_pdb, crystal_pdb)
                if us is None:
                    continue
                R, t, _, _ = us

                # Iterate pxdock poses only
                poses = collect_pose_scores(run_dir)
                pxdock_poses = [p for p in poses if p.source.startswith("protenix_dock")]
                if not pxdock_poses:
                    continue

                rmsd_cache: dict = {}
                seq_zone = target_zone.get(tgt, meta.get("seq_zone", "?"))
                wrote = 0
                for p in pxdock_poses:
                    sdf = p.pose_file
                    if sdf is None or str(sdf) in ("", "."):
                        continue
                    if sdf.suffix.lower() != ".sdf":
                        sib = sdf.with_suffix(".sdf")
                        if not sib.exists():
                            continue
                        sdf = sib
                    if not sdf.exists():
                        continue
                    pose_tail = p.pose_name.rsplit("_", 1)[-1]
                    idx = int(pose_tail) if pose_tail.isdigit() else 0
                    cache_key = (str(sdf), idx)
                    if cache_key in rmsd_cache:
                        rmsd = rmsd_cache[cache_key]
                    else:
                        mol = _conformer_at(sdf, idx)
                        if mol is None:
                            rmsd_cache[cache_key] = None
                            continue
                        pred_xfm = transform_mol(mol, R, t)
                        rmsd = _compute_rmsd(pred_xfm, ref_lig, eff_template)
                        rmsd_cache[cache_key] = rmsd
                    if rmsd is None:
                        continue

                    writer.writerow([
                        tgt, seq_zone, p.source, p.pose_name,
                        p.ba_pred_pkd, p.rmsd_pred, p.lscore,
                        p.iptm, p.ptm, p.plddt, p.conf,
                        p.boltz_aff_log_kd_nM, p.boltz_binder_prob,
                        rmsd,
                    ])
                    wrote += 1
                added_rows += wrote
                f.flush()
                if i % 25 == 0 or wrote > 0:
                    print(f"  [{i}/{len(todo)}] {tgt}: +{wrote} pxdock rows (total +{added_rows})", flush=True)
            except Exception as e:
                crashed.append(tgt)
                print(f"  [{tgt}] crashed: {e!r}", flush=True)

    print(f"\nDone. Added {added_rows} pxdock rows. Crashed: {len(crashed)}")
    if crashed:
        print(f"  crashed targets: {crashed[:10]}")


if __name__ == "__main__":
    main()
