"""Per-source standalone evaluation using each tool's NATIVE score (not BA-
Pred / pRMSD-Pred). For each target × source, rank that source's pose pool
by its built-in confidence/energy, take top-1 and best-of-5, measure against
the crystal ligand. This answers: if we used only this model alone with its
own scoring, how often would we land within 2Å?

Native scores per source:
  - Vina variants : meeko SDF tag .free_energy            (lower=better)
  - ADG variants  : meeko SDF tag .free_energy            (lower=better)
  - PxDock        : meeko SDF / _out.json pscore          (lower=better)
  - cofold_boltz2 : confidence_boltz_input_model_*.json::confidence_score (higher=better)
  - cofold_boltz2x: same as boltz2 under boltz2x/         (higher=better)
  - cofold_protenix: *_summary_confidence_sample_*.json::(plddt or ptm+iptm mix) (higher=better)
  - cofold_af3    : *_summary_confidences.json::ranking_score (higher=better)
"""
from __future__ import annotations

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

_EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EVAL_DIR))
from evaluate import (  # noqa: E402
    load_crystal_ligand,
    cif_path_for,
    load_targets,
    dump_crystal_protein_pdb,
    run_usalign,
)

import yaml


def load_template_smi(pdb_id: str) -> str | None:
    p = Path("experiments/novel2025_test/pipeline") / f"{pdb_id}_input.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text())
    for e in (d.get("sequences") or []):
        if "ligand" in e and "smiles" in e["ligand"]:
            return e["ligand"]["smiles"]
    return None


def pose_from_sdf_record(sdf: Path, idx: int, template: Chem.Mol) -> Chem.Mol | None:
    try:
        sup = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    except Exception:
        return None
    for i, mol in enumerate(sup):
        if i != idx or mol is None:
            continue
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            pass
        return reassign_bonds(mol, template)
    return None


def meeko_score(sdf: Path, idx: int) -> float | None:
    """Read meeko tag JSON on the idx-th SDF record, return free_energy."""
    try:
        sup = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    except Exception:
        return None
    for i, mol in enumerate(sup):
        if i != idx or mol is None:
            continue
        if mol.HasProp("meeko"):
            try:
                blob = json.loads(mol.GetProp("meeko"))
                return float(blob.get("free_energy", 0))
            except Exception:
                return None
    return None


def pxdock_scores(pxdock_json: Path) -> list[tuple[int, float]]:
    """Return [(pose_idx, pscore), ...] lower=better."""
    try:
        d = json.loads(pxdock_json.read_text())
    except Exception:
        return []
    poses = d.get("poses", [])
    return [(i, float(p.get("pscore", 0))) for i, p in enumerate(poses)]


# ---------- cofolding per-sample score extractors ----------

def _boltz_sample_scores(outputs_root: Path, tool_dir: str) -> list[tuple[Path, float]]:
    """Return [(aligned_cif, confidence), ...] for all samples under outputs/{tool_dir}/."""
    out = []
    for conf_json in (outputs_root / tool_dir).rglob("confidence_boltz_input_model_*.json"):
        try:
            d = json.loads(conf_json.read_text())
        except Exception:
            continue
        score = float(d.get("confidence_score", 0))
        # Find corresponding aligned CIF in same directory (predictions/boltz_input/)
        predictions_dir = conf_json.parent
        m = re.search(r"_model_(\d+)\.json$", conf_json.name)
        if not m:
            continue
        sample_idx = int(m.group(1))
        cif = predictions_dir / f"boltz_input_model_{sample_idx}_aligned.cif"
        if cif.exists():
            out.append((cif, score))
    return out


def _protenix_sample_scores(outputs_root: Path) -> list[tuple[Path, float]]:
    out = []
    for conf_json in (outputs_root / "protenix").rglob("*_summary_confidence_sample_*.json"):
        try:
            d = json.loads(conf_json.read_text())
        except Exception:
            continue
        # Use (plddt/100 + ptm + iptm) / 3 as aggregate; lower NaN fallback
        plddt = float(d.get("plddt", 0)) / 100.0
        ptm = float(d.get("ptm", 0))
        iptm = float(d.get("iptm", 0))
        score = (plddt + ptm + iptm) / 3.0
        m = re.search(r"_sample_(\d+)\.json$", conf_json.name)
        if not m:
            continue
        sample_idx = int(m.group(1))
        stem = conf_json.parent
        cif_candidates = sorted(stem.glob(f"*_sample_{sample_idx}_aligned.cif"))
        if cif_candidates:
            out.append((cif_candidates[0], score))
    return out


def _af3_sample_scores(outputs_root: Path) -> list[tuple[Path, float]]:
    out = []
    for summary_json in (outputs_root / "alphafold3").rglob("*_summary_confidences.json"):
        try:
            d = json.loads(summary_json.read_text())
        except Exception:
            continue
        score = float(d.get("ranking_score", 0))
        sample_dir = summary_json.parent
        # AF3 outputs the sample CIF in the same dir as summary, named *_model.cif. Aligned version lives under _aligned.cif.
        aligned = sorted(sample_dir.glob("*_model_aligned.cif"))
        if aligned:
            out.append((aligned[0], score))
    return out


# ---------- cofolding ligand extraction (same logic as run_post_analysis) ----------

def extract_cofolding_ligand(cif: Path, template: Chem.Mol) -> Chem.Mol | None:
    """Extract the PRIMARY binder (chain ``L``) from a cofolding aligned CIF.

    Multi-ligand targets keep metals/cofactors on chains ``X2``/``X3``/...
    Earlier versions matched anything with ``het_flag == 'H'`` and lumped all
    heteroatoms into a single HETATM block, which made the atom count
    disagree with the template SMILES (L ligand only) → reassign_bonds
    returned None. Restrict to the dedicated ``L`` chain to pull only the
    candidate ligand; fall back to LIG*/UNK/UNL residue names when ``L`` is
    absent (legacy single-ligand layouts).
    """
    try:
        import gemmi
    except Exception:
        return None
    try:
        st = gemmi.read_structure(str(cif))
    except Exception:
        return None

    def _collect(predicate) -> list[str]:
        lines: list[str] = []
        serial = 1
        for model in st:
            for chain in model:
                for res in chain:
                    if not predicate(chain, res):
                        continue
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        elem = atom.element.name
                        lines.append(
                            f"HETATM{serial:>5d} {atom.name:>4s} LIG L"
                            f"{1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2s}\n"
                        )
                        serial += 1
            break
        return lines

    # Primary: chain == "L"
    lines = _collect(lambda ch, res: ch.name == "L")
    # Fallback: named LIG/UNL/UNK residues
    if not lines:
        lines = _collect(lambda ch, res: res.name in ("LIG", "LIG1", "UNL", "UNK"))
    if not lines:
        return None
    raw = Chem.MolFromPDBBlock("".join(lines) + "END\n", removeHs=True, sanitize=False)
    if raw is None:
        return None
    return reassign_bonds(raw, template)


# ---------- eval per target ----------

def eval_target(pdb_id: str, meta: dict | None, work_dir: Path) -> dict:
    smi = load_template_smi(pdb_id) or (meta["candidate_smiles"][0] if meta and meta["candidate_smiles"] else "")
    ccd = meta["candidate_ccd_codes"][0] if meta and meta["candidate_ccd_codes"] else ""
    if not smi or not ccd:
        return {"__error__": "no candidate smi/ccd"}
    template = Chem.MolFromSmiles(smi)
    if template is None:
        return {"__error__": "bad smiles"}
    cif = cif_path_for(pdb_id)
    if cif is None:
        return {"__error__": "no cif"}
    ref_lig = load_crystal_ligand(cif, ccd, template)
    if ref_lig is None:
        return {"__error__": f"crystal {ccd} parse failed"}

    run_dir = Path("experiments/runs") / f"{pdb_id}_input"
    outputs_root = run_dir / "outputs"
    if not outputs_root.exists():
        return {"__error__": "no outputs dir"}

    # US-align predicted protein → crystal once
    crystal_pdb = work_dir / f"{pdb_id}_crystal.pdb"
    if not crystal_pdb.exists() and not dump_crystal_protein_pdb(cif, crystal_pdb):
        return {"__error__": "crystal pdb dump failed"}
    pred_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not pred_pdb.exists():
        return {"__error__": "no docking receptor pdb"}
    us = run_usalign(pred_pdb, crystal_pdb)
    if us is None:
        return {"__error__": "usalign failed"}
    R, t, _tm, _prot_rmsd = us

    out: dict[str, dict] = {}

    # --- Vina variants + ADG variants (meeko tag) ---
    meeko_sources = []
    for d in outputs_root.glob("vina_*"):
        if d.is_dir():
            meeko_sources.append(d.name)
    for d in outputs_root.glob("autodock_gpu_*"):
        if d.is_dir():
            meeko_sources.append(d.name)
    for src in meeko_sources:
        staged_sdf = run_dir / "outputs" / "analysis" / "poses"
        # SDFs are per-seed
        per_seed_files = sorted(staged_sdf.glob(f"{src}_seed_*.sdf"))
        records = []  # (score, sdf, idx)
        for sdf in per_seed_files:
            try:
                sup = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
            except Exception:
                continue
            for i, mol in enumerate(sup):
                if mol is None or not mol.HasProp("meeko"):
                    continue
                try:
                    fe = float(json.loads(mol.GetProp("meeko")).get("free_energy", 0))
                except Exception:
                    continue
                records.append((fe, sdf, i))
        if not records:
            continue
        records.sort(key=lambda r: r[0])  # ascending (lower FE = better)
        rmsds = []
        for _score, sdf, i in records[:5]:
            mol = pose_from_sdf_record(sdf, i, template)
            if mol is None:
                continue
            try:
                transform_mol(mol, R, t)
                rmsds.append(round(pose_rmsd(mol, ref_lig), 3))
            except Exception:
                continue
        if rmsds:
            out[src] = {
                "n_pool": len(records),
                "n_evaluated": len(rmsds),
                "top1_rmsd": rmsds[0],
                "best5_rmsd": min(rmsds),
                "score_type": "vina/adg free_energy (kcal/mol)",
            }

    # --- Protenix-Dock ---
    pxdock_sdf = outputs_root / "protenix_dock" / "poses.sdf"
    pxdock_json_files = list((outputs_root / "protenix_dock").glob("*_out.json"))
    if pxdock_sdf.exists() and pxdock_json_files:
        score_by_idx = dict(pxdock_scores(pxdock_json_files[0]))
        if score_by_idx:
            order = sorted(score_by_idx.items(), key=lambda kv: kv[1])  # lower=better
            rmsds = []
            for idx, _sc in order[:5]:
                mol = pose_from_sdf_record(pxdock_sdf, idx, template)
                if mol is None:
                    continue
                try:
                    transform_mol(mol, R, t)
                    rmsds.append(round(pose_rmsd(mol, ref_lig), 3))
                except Exception:
                    continue
            if rmsds:
                out["protenix_dock"] = {
                    "n_pool": len(score_by_idx),
                    "n_evaluated": len(rmsds),
                    "top1_rmsd": rmsds[0],
                    "best5_rmsd": min(rmsds),
                    "score_type": "pxdock pscore",
                }

    # --- Cofolding: per-sample confidence ---
    cofold_jobs = [
        ("cofold_boltz2",   _boltz_sample_scores(outputs_root, "boltz2"),   "higher"),
        ("cofold_boltz2x",  _boltz_sample_scores(outputs_root, "boltz2x"),  "higher"),
        ("cofold_protenix", _protenix_sample_scores(outputs_root),          "higher"),
        ("cofold_af3",      _af3_sample_scores(outputs_root),               "higher"),
    ]
    for name, samples, order in cofold_jobs:
        if not samples:
            continue
        if order == "higher":
            ranked = sorted(samples, key=lambda kv: -kv[1])
        else:
            ranked = sorted(samples, key=lambda kv: kv[1])
        rmsds = []
        for cif_path, _sc in ranked[:5]:
            mol = extract_cofolding_ligand(cif_path, template)
            if mol is None:
                continue
            try:
                transform_mol(mol, R, t)
                rmsds.append(round(pose_rmsd(mol, ref_lig), 3))
            except Exception:
                continue
        if rmsds:
            out[name] = {
                "n_pool": len(samples),
                "n_evaluated": len(rmsds),
                "top1_rmsd": rmsds[0],
                "best5_rmsd": min(rmsds),
                "score_type": "cofolding confidence (higher=better)",
            }

    return out


def main():
    targets_meta = load_targets()
    work_dir = Path("experiments/novel2025_test/_eval_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "per_source_native.json"

    submissions = sorted(Path("experiments/submissions").glob("*.lg"))
    submissions = [s for s in submissions if not s.name.startswith("L1")]
    print(f"evaluating {len(submissions)} submissions using NATIVE per-source scores...")

    results = {}
    for i, lg in enumerate(submissions, 1):
        pdb_id = lg.stem.replace("_input", "")
        try:
            r = eval_target(pdb_id, targets_meta.get(pdb_id), work_dir)
        except Exception as e:
            r = {"__error__": str(e)}
        results[pdb_id] = r
        if i % 10 == 0 or i == len(submissions):
            print(f"  [{i}/{len(submissions)}] {pdb_id}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    # Aggregate
    agg: dict[str, dict] = defaultdict(lambda: {"top1": [], "best5": [], "pool_sizes": []})
    for pdb_id, per_source in results.items():
        if "__error__" in per_source:
            continue
        for src, s in per_source.items():
            agg[src]["top1"].append(s["top1_rmsd"])
            agg[src]["best5"].append(s["best5_rmsd"])
            agg[src]["pool_sizes"].append(s["n_pool"])

    print()
    print("=" * 92)
    print(f"{'Source (native score)':<28} {'n_tgt':>6} {'avg_pool':>8}  {'med T1':>7}  {'<2Å T1':>7}  {'med B5':>7}  {'<2Å B5':>7}")
    print("-" * 92)
    rows = []
    for src, s in agg.items():
        n = len(s["top1"])
        if n == 0: continue
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
