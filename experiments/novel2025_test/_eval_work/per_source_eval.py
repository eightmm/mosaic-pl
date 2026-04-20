"""Per-source standalone evaluation: for each target × source, sort poses by
the source's predicted pRMSD ascending and measure top-1 / best-of-5 RMSD
vs the RCSB crystal ligand. Compare to our ensemble selector's output.

Differs from evaluate.py, which only measures the final LG top-5. Here we
pretend each source is a standalone tool: take every pose it generated,
rank by its own pRMSD-Pred score, record what it would have submitted
independently.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from rdkit import Chem
from rdkit.RDLogger import DisableLog
DisableLog("rdApp.*")

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import pose_rmsd, reassign_bonds, transform_mol  # noqa: E402

# Import from evaluate.py
_EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EVAL_DIR))
from evaluate import (  # noqa: E402
    load_crystal_ligand,
    cif_path_for,
    load_targets,
    dump_crystal_protein_pdb,
    run_usalign,
)


def parse_rmsd_pred_tsv(path: Path) -> dict[str, float]:
    """name → pRMSD (ascending = better)."""
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                out[row["Name"]] = float(row["pRMSD"])
            except (KeyError, ValueError, TypeError):
                pass
    return out


def load_pose_record(sdf_path: Path, record_idx: int, template: Chem.Mol) -> Chem.Mol | None:
    """Load the record_idx-th mol from an SDF, reassign bond orders.

    Meeko's mk_export output for Vina/ADG keeps explicit hydrogens; RDKit's
    ``SDMolSupplier(removeHs=True)`` sometimes leaves behind hydrogens when
    sanitize=False fails. Explicitly run ``RemoveHs`` after loading so the
    heavy-atom count aligns with the SMILES template that
    ``reassign_bonds`` requires.
    """
    try:
        sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    except Exception:
        return None
    for i, mol in enumerate(sup):
        if i != record_idx or mol is None:
            continue
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            pass
        return reassign_bonds(mol, template)
    return None


def resolve_pose_file(run_dir: Path, pose_name: str) -> tuple[Path | None, int]:
    """From ``{stem}_{idx}`` walk down parts until a staged SDF is found.
    Returns (sdf_path, record_index)."""
    parts = pose_name.split("_")
    staged = run_dir / "outputs" / "analysis" / "poses"
    for i in range(len(parts), 0, -1):
        prefix = "_".join(parts[:i])
        sdf = staged / f"{prefix}.sdf"
        if sdf.exists():
            tail = parts[i] if i < len(parts) and parts[i].isdigit() else "0"
            return sdf, int(tail)
    # pxdock fallback
    pxdock_sdf = run_dir / "outputs" / "protenix_dock" / "poses.sdf"
    if pxdock_sdf.exists() and "poses" in pose_name:
        m = re.search(r"_(\d+)$", pose_name)
        idx = int(m.group(1)) if m else 0
        return pxdock_sdf, idx
    return None, 0


def source_of(tsv_stem: str) -> str:
    """rmsd_pred_vina_cofolding.tsv → vina_cofolding."""
    return tsv_stem.replace("rmsd_pred_", "")


def load_first_ligand_from_yaml(yaml_path: Path) -> tuple[str, str] | None:
    """Read the first ``- ligand:`` entry from the pipeline YAML (already
    smart-split by build_inputs._smart_split_smiles), return (smiles, ccd_if_known)."""
    try:
        import yaml as _yaml
    except Exception:
        return None
    d = _yaml.safe_load(yaml_path.read_text())
    for entry in (d.get("sequences") or []):
        if "ligand" in entry:
            lig = entry["ligand"] or {}
            smi = lig.get("smiles", "")
            if smi:
                return smi, lig.get("ccd", "")
    return None


def eval_target(pdb_id: str, meta: dict | None, submission_lg: Path, work_dir: Path) -> dict:
    """Return {source: {"top1_rmsd": float, "best5_rmsd": float, "n_poses": int}}."""
    # Prefer pipeline YAML (smart-split SMILES) over raw TSV (may have
    # broken HEM-like entries the build stage already repaired)
    yaml_path = Path("experiments/novel2025_test/pipeline") / f"{pdb_id}_input.yaml"
    smi = ""
    if yaml_path.exists():
        info = load_first_ligand_from_yaml(yaml_path)
        if info:
            smi = info[0]
    if not smi and meta:
        smi = meta["candidate_smiles"][0] if meta["candidate_smiles"] else ""
    ccd = meta["candidate_ccd_codes"][0] if meta and meta["candidate_ccd_codes"] else ""
    if not smi or not ccd:
        return {"__error__": "no candidate"}
    template = Chem.MolFromSmiles(smi)
    if template is None:
        return {"__error__": "bad SMILES"}
    cif = cif_path_for(pdb_id)
    if cif is None:
        return {"__error__": "no mmCIF"}
    ref_lig = load_crystal_ligand(cif, ccd, template)
    if ref_lig is None:
        return {"__error__": f"crystal ligand {ccd} parse failed"}

    run_dir = Path("experiments/runs") / f"{pdb_id}_input"
    analysis_dir = run_dir / "outputs" / "analysis"
    if not analysis_dir.exists():
        return {"__error__": "no analysis dir"}

    # US-align predicted receptor → crystal protein for one-shot transform
    # (all docking + cofolding poses share the same cofolding-aligned frame)
    crystal_pdb = work_dir / f"{pdb_id}_crystal.pdb"
    if not crystal_pdb.exists() and not dump_crystal_protein_pdb(cif, crystal_pdb):
        return {"__error__": "crystal protein dump failed"}
    pred_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not pred_pdb.exists():
        return {"__error__": "no docking receptor pdb"}
    us = run_usalign(pred_pdb, crystal_pdb)
    if us is None:
        return {"__error__": "usalign failed"}
    R, t, _tm, _prot_rmsd = us

    out: dict[str, dict] = {}
    for tsv in sorted(analysis_dir.glob("rmsd_pred_*.tsv")):
        src = source_of(tsv.stem)
        prmsds = parse_rmsd_pred_tsv(tsv)
        if not prmsds:
            continue
        # Sort by predicted pRMSD ascending (source's own best ranking)
        sorted_poses = sorted(prmsds.items(), key=lambda kv: kv[1])
        rmsds = []
        for pose_name, _pred_prmsd in sorted_poses[:5]:
            sdf, rec_idx = resolve_pose_file(run_dir, pose_name)
            if sdf is None:
                continue
            mol = load_pose_record(sdf, rec_idx, template)
            if mol is None:
                continue
            try:
                transform_mol(mol, R, t)
                r = pose_rmsd(mol, ref_lig)
            except Exception:
                continue
            rmsds.append(round(r, 3))
        if not rmsds:
            continue
        out[src] = {
            "top1_rmsd": rmsds[0],
            "best5_rmsd": min(rmsds),
            "n_poses_in_pool": len(prmsds),
            "n_evaluated": len(rmsds),
        }
    return out


def main():
    targets_meta = load_targets()
    work_dir = Path("experiments/novel2025_test/_eval_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "per_source_eval.json"

    submissions = sorted(Path("experiments/submissions").glob("*.lg"))
    submissions = [s for s in submissions if not s.name.startswith("L1")]
    print(f"evaluating {len(submissions)} submissions across per-source pose pools...")

    results = {}
    for i, lg in enumerate(submissions, 1):
        pdb_id = lg.stem.replace("_input", "")
        meta = targets_meta.get(pdb_id)
        try:
            r = eval_target(pdb_id, meta, lg, work_dir)
        except Exception as e:
            r = {"__error__": str(e)}
        results[pdb_id] = r
        if i % 10 == 0 or i == len(submissions):
            print(f"  [{i}/{len(submissions)}] {pdb_id}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # Aggregate per source
    source_rows: dict[str, dict[str, list]] = defaultdict(lambda: {"top1": [], "best5": [], "pool_sizes": []})
    for pdb_id, per_source in results.items():
        if "__error__" in per_source:
            continue
        for src, s in per_source.items():
            source_rows[src]["top1"].append(s["top1_rmsd"])
            source_rows[src]["best5"].append(s["best5_rmsd"])
            source_rows[src]["pool_sizes"].append(s["n_poses_in_pool"])

    print()
    print("=" * 90)
    print(f"{'Source':<28} {'n_tgt':>6} {'avg_pool':>8}  {'med T1':>7}  {'<2Å T1':>7}  {'med B5':>7}  {'<2Å B5':>7}")
    print("-" * 90)
    rows = []
    for src, s in source_rows.items():
        n = len(s["top1"])
        med_t1 = median(s["top1"])
        med_b5 = median(s["best5"])
        lt2_t1 = sum(1 for r in s["top1"] if r < 2.0) / n * 100
        lt2_b5 = sum(1 for r in s["best5"] if r < 2.0) / n * 100
        avg_pool = sum(s["pool_sizes"]) / len(s["pool_sizes"])
        rows.append((src, n, avg_pool, med_t1, lt2_t1, med_b5, lt2_b5))
    rows.sort(key=lambda r: r[3])  # sort by median top-1 asc
    for src, n, avg_pool, t1, lt2_t1, b5, lt2_b5 in rows:
        print(f"{src:<28} {n:>6} {avg_pool:>7.1f}   {t1:>6.2f}   {lt2_t1:>5.1f}%   {b5:>6.2f}   {lt2_b5:>5.1f}%")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
