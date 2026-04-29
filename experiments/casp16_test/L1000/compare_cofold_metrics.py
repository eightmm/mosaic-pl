"""Compare cofolding-model-internal confidence metrics as pose selectors.

For each L1xxx target we have 100 cofolding poses (boltz2/boltz2x/protenix/af3,
5 seeds × 5 samples each). Every cofolding sample ships a per-structure
confidence JSON with metrics like pLDDT, pTM, ipTM, ligand_ipTM. This script
ranks the cofolding pool by each metric in isolation and reports the top-1
and best-of-top-5 actual RMSD (vs. the experimental crystal) per metric.

Metrics extracted (higher = more confident):
    - plddt       (boltz: complex_plddt; protenix: plddt/100 if scaled; af3: ranking_score)
    - ptm         (all three)
    - iptm        (all three)
    - ligand_iptm (boltz only — closest to "pose-level" signal)
    - confidence  (boltz confidence_score; protenix/af3 synthesize via ipTM*ipLDDT etc.)

Each metric is evaluated across ALL cofolding sources pooled together, so the
ranking also implicitly chooses the model.

Usage:
    /home/jaemin/project/CASP17/.venvs/protenix-dock/bin/python compare_cofold_metrics.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import (  # noqa: E402
    kabsch,
    parse_ca,
    pose_rmsd,
    reassign_bonds,
    transform_mol,
)

ROOT = Path(__file__).parent
EXPER = ROOT / "L1000_prepared"
RUNS = ROOT.parents[1] / "runs"
AFF_CSV = ROOT / "L1000_exper_affinity.csv"


def load_smiles() -> dict[str, str]:
    out = {}
    with AFF_CSV.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["Target ID"].strip()] = row["ligand_smiles"].strip()
    return out


# ---------- ligand extraction (from aligned CIFs) ----------

def cofold_ligand(cif: Path, template) -> Chem.Mol | None:
    """Extract the bound ligand from an aligned cofolding CIF and assign
    bond orders from the SMILES template. Returns None on any failure."""
    import gemmi
    try:
        st = gemmi.read_structure(str(cif))
    except Exception:
        return None
    lines, serial = [], 1
    for model in st:
        for chain in model:
            for res in chain:
                is_lig = (res.name in ("LIG", "LIG1", "UNL", "UNK")
                          or getattr(res, "het_flag", "") == "H"
                          or chain.name == "L")
                if not is_lig:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    lines.append(
                        f"HETATM{serial:>5d} {atom.name:>4s} LIG L"
                        f"{1:>4d}    {atom.pos.x:>8.3f}{atom.pos.y:>8.3f}"
                        f"{atom.pos.z:>8.3f}  1.00  0.00          {atom.element.name:>2s}\n"
                    )
                    serial += 1
        break
    if not lines:
        return None
    raw = Chem.MolFromPDBBlock("".join(lines) + "END\n", removeHs=True, sanitize=False)
    return reassign_bonds(raw, template)


# ---------- cofolding confidence loaders ----------

def load_boltz_conf(cif: Path) -> dict | None:
    """boltz_input_model_N_aligned.cif → confidence_boltz_input_model_N.json"""
    stem = cif.stem.replace("_aligned", "")
    conf_path = cif.parent / f"confidence_{stem}.json"
    if not conf_path.exists():
        return None
    try:
        d = json.loads(conf_path.read_text())
    except Exception:
        return None
    return {
        "plddt": d.get("complex_plddt"),
        "ptm": d.get("ptm"),
        "iptm": d.get("iptm"),
        "ligand_iptm": d.get("ligand_iptm"),
        "confidence": d.get("confidence_score"),
    }


def load_protenix_conf(cif: Path) -> dict | None:
    """L1001_input_sample_N_aligned.cif → L1001_input_summary_confidence_sample_N.json"""
    stem = cif.stem.replace("_aligned", "")  # e.g. L1001_input_sample_0
    m = stem.split("_sample_")
    if len(m) != 2:
        return None
    base, idx = m
    conf_path = cif.parent / f"{base}_summary_confidence_sample_{idx}.json"
    if not conf_path.exists():
        return None
    try:
        d = json.loads(conf_path.read_text())
    except Exception:
        return None
    plddt = d.get("plddt")
    # Protenix reports plddt in 0..100; normalize to 0..1 for comparability
    if plddt is not None and plddt > 1:
        plddt = plddt / 100.0
    return {
        "plddt": plddt,
        "ptm": d.get("ptm"),
        "iptm": d.get("iptm"),
        "ligand_iptm": None,
        "confidence": d.get("ptm"),  # protenix has no single-scalar confidence; use ptm as proxy
    }


def load_af3_conf(cif: Path) -> dict | None:
    """*_model_aligned.cif → *_summary_confidences.json in same dir"""
    stem = cif.stem.replace("_model_aligned", "")
    conf_path = cif.parent / f"{stem}_summary_confidences.json"
    if not conf_path.exists():
        return None
    try:
        d = json.loads(conf_path.read_text())
    except Exception:
        return None
    return {
        "plddt": None,  # AF3 summary doesn't publish a single plddt scalar
        "ptm": d.get("ptm"),
        "iptm": d.get("iptm"),
        "ligand_iptm": None,
        "confidence": d.get("ranking_score"),
    }


# ---------- per-target evaluation ----------

METRICS = ("plddt", "ptm", "iptm", "ligand_iptm", "confidence")


def evaluate_target(tid: str, smiles: str) -> dict:
    run_dir = RUNS / f"{tid}_input"
    tdir = EXPER / tid
    rec_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not (run_dir.exists() and tdir.exists() and rec_pdb.exists()):
        return {"target": tid, "status": "missing dirs"}
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return {"target": tid, "status": "bad template"}

    ref_raw = Chem.MolFromPDBFile(str(next(tdir.glob("ligand_*.pdb"))),
                                   removeHs=False, sanitize=False)
    try:
        ref_raw = Chem.RemoveHs(ref_raw, sanitize=False)
    except Exception:
        return {"target": tid, "status": "exper RemoveHs failed"}
    ref_lig = reassign_bonds(ref_raw, template)
    if ref_lig is None:
        return {"target": tid, "status": "exper reassign failed"}

    pred_ca = parse_ca(rec_pdb)
    ref_ca = parse_ca(tdir / "protein_aligned.pdb")
    common = sorted(set(pred_ca) & set(ref_ca))
    if len(common) < 50:
        return {"target": tid, "status": "too few common CAs"}
    P = np.array([pred_ca[r] for r in common])
    Q = np.array([ref_ca[r] for r in common])
    R, t, _ = kabsch(P, Q)

    # Collect cofolding poses + metrics + actual RMSD
    sources = [
        ("boltz2", (run_dir / "outputs" / "boltz2").rglob("*_aligned.cif"), load_boltz_conf),
        ("boltz2x", (run_dir / "outputs" / "boltz2x").rglob("*_aligned.cif"), load_boltz_conf),
        ("protenix", (run_dir / "outputs" / "protenix").rglob("*_aligned.cif"), load_protenix_conf),
        ("af3", (run_dir / "outputs" / "alphafold3").rglob("*_aligned.cif"), load_af3_conf),
    ]
    pool = []
    for model, cifs, loader in sources:
        for cif in sorted(cifs):
            lig = cofold_ligand(cif, template)
            if lig is None:
                continue
            lig_t = transform_mol(lig, R, t, inplace=False)
            rmsd = pose_rmsd(lig_t, ref_lig)
            if rmsd != rmsd:
                continue
            conf = loader(cif) or {}
            entry = {"model": model, "cif": str(cif), "rmsd": round(rmsd, 3)}
            for m in METRICS:
                entry[m] = conf.get(m)
            pool.append(entry)

    if not pool:
        return {"target": tid, "status": "empty cofold pool"}

    # For each metric, rank descending (higher = more confident), pick top5
    per_metric = {}
    for m in METRICS:
        scored = [p for p in pool if p[m] is not None]
        if not scored:
            per_metric[m] = {"n": 0}
            continue
        ordered = sorted(scored, key=lambda p: p[m], reverse=True)
        top5 = ordered[:5]
        per_metric[m] = {
            "n": len(scored),
            "top1_rmsd": top5[0]["rmsd"],
            "best5_rmsd": round(min(p["rmsd"] for p in top5), 3),
            "top5_rmsds": [p["rmsd"] for p in top5],
            "top5_models": [p["model"] for p in top5],
        }

    overall = min(p["rmsd"] for p in pool)
    return {
        "target": tid,
        "n_pool": len(pool),
        "overall_best_rmsd": round(overall, 3),
        "per_metric": per_metric,
    }


def main():
    smiles = load_smiles()
    results = []
    targets = sorted(smiles.keys())

    header = (f"{'target':<7} | {'pool':>4} | {'best':>5} | "
              + " | ".join(f"{m:>6}" for m in METRICS))
    print(header)
    print("-" * len(header))
    for tid in targets:
        r = evaluate_target(tid, smiles[tid])
        results.append(r)
        if "status" in r:
            print(f"{tid:<7} | {r['status']}")
            continue
        row = f"{tid:<7} | {r['n_pool']:>4} | {r['overall_best_rmsd']:>5} | "
        for m in METRICS:
            v = r["per_metric"][m]
            row += f"{v.get('top1_rmsd', '-'):>6} | " if v.get("top1_rmsd") is not None else f"{'-':>6} | "
        print(row.rstrip(" |"))

    # Aggregate
    ok = [r for r in results if "per_metric" in r]
    print()
    print("Success rates (top-1 < 2 Å | best-5 < 2 Å) across {} targets:".format(len(ok)))
    for m in METRICS:
        n_top1 = sum(
            1 for r in ok
            if r["per_metric"][m].get("top1_rmsd") is not None
            and r["per_metric"][m]["top1_rmsd"] < 2.0
        )
        n_b5 = sum(
            1 for r in ok
            if r["per_metric"][m].get("best5_rmsd") is not None
            and r["per_metric"][m]["best5_rmsd"] < 2.0
        )
        n_have = sum(1 for r in ok if r["per_metric"][m].get("top1_rmsd") is not None)
        print(f"  {m:<12} top1={n_top1}/{n_have}  best5={n_b5}/{n_have}")

    out = ROOT / "compare_cofold_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
