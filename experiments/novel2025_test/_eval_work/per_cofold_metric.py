"""Per-cofolding-model × per-metric standalone evaluation. For every
cofolding model, iterate every available confidence metric it exposes,
rank the 25 samples by that metric, take the #1 pose, and measure RMSD vs
the crystal ligand. Surfaces which metric is most discriminating.

Metrics per model (from the per-sample JSON files):
  boltz2, boltz2x (confidence_boltz_input_model_*.json):
    confidence_score, ptm, iptm, ligand_iptm, protein_iptm,
    complex_plddt, complex_iplddt, -complex_pde, -complex_ipde
  protenix (*_summary_confidence_sample_*.json):
    plddt, ptm, iptm, -gpde
  af3 (*_summary_confidences.json):
    ranking_score, ptm, iptm, -fraction_disordered
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
sys.path.insert(0, str(_EVAL_DIR / "_eval_work"))
from per_source_native import extract_cofolding_ligand, load_template_smi  # noqa: E402


# Metric specs: (metric_name, extractor fn, direction). Higher-is-better → +1;
# lower-is-better (e.g. pde/gpde) → -1 so we can always sort descending.
def _neg(key):
    return lambda d: -float(d.get(key, 0)) if d.get(key) is not None else None


def _pos(key):
    return lambda d: float(d.get(key, 0)) if d.get(key) is not None else None


BOLTZ_METRICS = {
    "confidence_score": _pos("confidence_score"),
    "ptm":              _pos("ptm"),
    "iptm":             _pos("iptm"),
    "ligand_iptm":      _pos("ligand_iptm"),
    "complex_plddt":    _pos("complex_plddt"),
    "complex_iplddt":   _pos("complex_iplddt"),
    "neg_complex_pde":  _neg("complex_pde"),
    "neg_complex_ipde": _neg("complex_ipde"),
}

PROTENIX_METRICS = {
    "plddt":   _pos("plddt"),
    "ptm":     _pos("ptm"),
    "iptm":    _pos("iptm"),
    "neg_gpde": _neg("gpde"),
}

AF3_METRICS = {
    "ranking_score":            _pos("ranking_score"),
    "ptm":                      _pos("ptm"),
    "iptm":                     _pos("iptm"),
    "neg_fraction_disordered":  _neg("fraction_disordered"),
}


# ---------- per-sample iterators ----------

def _boltz_samples(outputs_root: Path, tool_dir: str) -> list[tuple[Path, dict]]:
    """[(aligned_cif, metrics_dict), ...]"""
    out = []
    for conf_json in (outputs_root / tool_dir).rglob("confidence_boltz_input_model_*.json"):
        try:
            d = json.loads(conf_json.read_text())
        except Exception:
            continue
        m = re.search(r"_model_(\d+)\.json$", conf_json.name)
        if not m:
            continue
        idx = int(m.group(1))
        cif = conf_json.parent / f"boltz_input_model_{idx}_aligned.cif"
        if cif.exists():
            out.append((cif, d))
    return out


def _protenix_samples(outputs_root: Path) -> list[tuple[Path, dict]]:
    out = []
    for conf_json in (outputs_root / "protenix").rglob("*_summary_confidence_sample_*.json"):
        try:
            d = json.loads(conf_json.read_text())
        except Exception:
            continue
        m = re.search(r"_sample_(\d+)\.json$", conf_json.name)
        if not m:
            continue
        idx = int(m.group(1))
        cifs = sorted(conf_json.parent.glob(f"*_sample_{idx}_aligned.cif"))
        if cifs:
            out.append((cifs[0], d))
    return out


def _af3_samples(outputs_root: Path) -> list[tuple[Path, dict]]:
    out = []
    for conf_json in (outputs_root / "alphafold3").rglob("*_summary_confidences.json"):
        try:
            d = json.loads(conf_json.read_text())
        except Exception:
            continue
        sample_dir = conf_json.parent
        aligned = sorted(sample_dir.glob("*_model_aligned.cif"))
        if aligned:
            out.append((aligned[0], d))
    return out


MODEL_SPECS = [
    ("cofold_boltz2",   "boltz2",   BOLTZ_METRICS,    _boltz_samples),
    ("cofold_boltz2x",  "boltz2x",  BOLTZ_METRICS,    _boltz_samples),
    ("cofold_protenix", "protenix", PROTENIX_METRICS, _protenix_samples),
    ("cofold_af3",      "af3",      AF3_METRICS,      _af3_samples),
]


def eval_target(pdb_id: str, meta: dict | None, work_dir: Path) -> dict:
    smi = load_template_smi(pdb_id) or (meta["candidate_smiles"][0] if meta and meta["candidate_smiles"] else "")
    ccd = meta["candidate_ccd_codes"][0] if meta and meta["candidate_ccd_codes"] else ""
    if not smi or not ccd:
        return {"__error__": "no candidate"}
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

    crystal_pdb = work_dir / f"{pdb_id}_crystal.pdb"
    if not crystal_pdb.exists() and not dump_crystal_protein_pdb(cif, crystal_pdb):
        return {"__error__": "crystal dump failed"}
    pred_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not pred_pdb.exists():
        return {"__error__": "no receptor pdb"}
    us = run_usalign(pred_pdb, crystal_pdb)
    if us is None:
        return {"__error__": "usalign failed"}
    R, t, _tm, _prot_rmsd = us

    # Pre-compute RMSD for every sample per model (sample → rmsd) once;
    # then pick winners per metric ranking without re-computing RMSD.
    out: dict[str, dict] = {}
    for model_name, tool_dir, metrics, sampler in MODEL_SPECS:
        if tool_dir == "boltz2":
            samples = _boltz_samples(outputs_root, "boltz2")
        elif tool_dir == "boltz2x":
            samples = _boltz_samples(outputs_root, "boltz2x")
        else:
            samples = sampler(outputs_root)
        if not samples:
            continue
        # Compute RMSD for each sample once.
        sample_rmsds: list[tuple[int, dict, float]] = []  # (idx, metrics_dict, rmsd)
        for i, (cif_path, mdict) in enumerate(samples):
            mol = extract_cofolding_ligand(cif_path, template)
            if mol is None:
                continue
            try:
                transform_mol(mol, R, t)
                r = pose_rmsd(mol, ref_lig)
            except Exception:
                continue
            sample_rmsds.append((i, mdict, round(r, 3)))
        if not sample_rmsds:
            continue
        out[model_name] = {"n_pool": len(samples), "metrics": {}}
        for mname, extractor in metrics.items():
            scored = []
            for idx, mdict, rmsd in sample_rmsds:
                v = extractor(mdict)
                if v is None:
                    continue
                scored.append((v, rmsd))
            if not scored:
                continue
            # Sort by score desc (since we negated lower-is-better)
            scored.sort(key=lambda p: -p[0])
            rmsds_in_order = [p[1] for p in scored]
            out[model_name]["metrics"][mname] = {
                "top1_rmsd": rmsds_in_order[0],
                "best5_rmsd": min(rmsds_in_order[:5]),
                "n_scored": len(rmsds_in_order),
            }
        # Oracle (best possible ligand rmsd from this model's pool)
        out[model_name]["oracle_best"] = min(r for _, _, r in sample_rmsds)
    return out


def main():
    targets_meta = load_targets()
    work_dir = Path("experiments/novel2025_test/_eval_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "per_cofold_metric.json"

    submissions = sorted(Path("experiments/submissions").glob("*.lg"))
    submissions = [s for s in submissions if not s.name.startswith("L1")]
    print(f"evaluating {len(submissions)} submissions per cofold model × metric...")

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

    # Aggregate: per (model, metric) → rmsd lists
    agg: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: {"top1": [], "best5": []}))
    oracle_agg: dict[str, list] = defaultdict(list)
    for pdb_id, per_model in results.items():
        if "__error__" in per_model:
            continue
        for model_name, info in per_model.items():
            if "metrics" not in info:
                continue
            oracle_agg[model_name].append(info["oracle_best"])
            for mname, s in info["metrics"].items():
                agg[model_name][mname]["top1"].append(s["top1_rmsd"])
                agg[model_name][mname]["best5"].append(s["best5_rmsd"])

    print()
    for model_name, ms in agg.items():
        oracle_rmsds = oracle_agg.get(model_name, [])
        if oracle_rmsds:
            ora_med = median(oracle_rmsds)
            ora_lt2 = sum(1 for r in oracle_rmsds if r < 2.0) / len(oracle_rmsds) * 100
            print(f"\n=== {model_name} (n_tgt={len(oracle_rmsds)}, oracle med={ora_med:.2f}Å, oracle <2Å={ora_lt2:.1f}%) ===")
        else:
            print(f"\n=== {model_name} ===")
        print(f"  {'Metric':<26} {'n':>4}  {'med T1':>7}  {'<2Å T1':>7}  {'med B5':>7}  {'<2Å B5':>7}")
        print(f"  {'-'*72}")
        rows = []
        for mname, s in ms.items():
            n = len(s["top1"])
            if n == 0: continue
            med_t1 = median(s["top1"])
            med_b5 = median(s["best5"])
            lt2_t1 = sum(1 for r in s["top1"] if r < 2.0) / n * 100
            lt2_b5 = sum(1 for r in s["best5"] if r < 2.0) / n * 100
            rows.append((mname, n, med_t1, lt2_t1, med_b5, lt2_b5))
        rows.sort(key=lambda r: r[2])
        for mname, n, t1, lt2_t1, b5, lt2_b5 in rows:
            print(f"  {mname:<26} {n:>4}   {t1:>6.2f}   {lt2_t1:>5.1f}%   {b5:>6.2f}   {lt2_b5:>5.1f}%")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
