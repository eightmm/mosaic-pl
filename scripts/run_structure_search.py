#!/usr/bin/env python3
"""Run Foldseek structure search on cofolding outputs and compare hits.

For each cofolding model (boltz2, boltz2x, protenix, alphafold3):
1. Extract best structure (CIF)
2. Run Foldseek against RCSB structure DB
3. Filter hits by ligand presence (via rcsb_index.db)
4. Compare overlapping hits across models

Usage:
    python run_structure_search.py \
        --run-dir experiments/runs/full_pipeline_test \
        --foldseek-db data/search_dbs/structure/rcsb_structDB \
        --rcsb-db /home/jaemin/DB/RCSB/processed/rcsb_index.db
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def find_best_structure(model_dir: Path, model: str) -> Path | None:
    """Find best CIF from a cofolding model output."""
    if model.startswith("boltz"):
        for cif in sorted(model_dir.rglob("predictions/**/*.cif")):
            return cif
    elif model == "protenix":
        for cif in sorted(model_dir.rglob("*.cif")):
            return cif
    elif model == "alphafold3":
        for cif in sorted(model_dir.rglob("*model*.cif")):
            return cif
    for cif in sorted(model_dir.rglob("*.cif")):
        return cif
    return None


def run_foldseek(
    query_cif: Path, db_path: str, output_dir: Path, model_name: str,
    foldseek_bin: str = "foldseek", sensitivity: float = 9.5, max_hits: int = 200,
) -> Path | None:
    """Run Foldseek easy-search on a query structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_tsv = output_dir / f"foldseek_{model_name}.tsv"
    tmp_dir = output_dir / f"tmp_{model_name}"

    cmd = [
        foldseek_bin, "easy-search",
        str(query_cif), db_path, str(result_tsv), str(tmp_dir),
        "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,lddt,alntmscore",
        "-s", str(sensitivity),
        "--max-seqs", str(max_hits),
        "--alignment-type", "1",
        "--threads", "8",
    ]
    print(f"  Foldseek {model_name}: {query_cif.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        if result_tsv.exists():
            hit_count = sum(1 for _ in open(result_tsv))
            print(f"    → {hit_count} hits")
            return result_tsv
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    FAILED: {e}")
    return None


def parse_foldseek_hits(tsv_path: Path) -> list[dict]:
    """Parse Foldseek output TSV."""
    hits = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 14:
                continue
            target = parts[1]
            pdb_id = target.split("_")[0].lower() if "_" in target else target[:4].lower()
            chain = target.split("_")[1] if "_" in target else ""
            hits.append({
                "target": target,
                "pdb_id": pdb_id,
                "chain": chain,
                "pident": float(parts[2]),
                "evalue": float(parts[10]),
                "lddt": float(parts[14]) if len(parts) > 14 and parts[14] else 0.0,
                "tmscore": float(parts[15]) if len(parts) > 15 and parts[15] else 0.0,
            })
    return hits


def lookup_ligands_bulk(db_path: Path, pdb_ids: set[str]) -> dict[str, list[dict]]:
    """Bulk lookup ligands for multiple PDB IDs."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    result: dict[str, list[dict]] = defaultdict(list)

    placeholders = ",".join("?" * len(pdb_ids))
    cur.execute(
        f"""SELECT pdb_id, ccd_code, ligand_type, is_candidate, smiles, molecular_weight
            FROM ligand_instances
            WHERE pdb_id IN ({placeholders}) AND is_candidate = 1""",
        list(pdb_ids),
    )
    for row in cur.fetchall():
        result[row[0]].append({
            "ccd_code": row[1],
            "ligand_type": row[2],
            "smiles": row[4],
            "mw": row[5],
        })
    conn.close()
    return dict(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Structure search on cofolding outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--foldseek-db", type=str, required=True)
    parser.add_argument("--rcsb-db", type=Path, required=True)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--models", nargs="+", default=["boltz2", "boltz2x", "protenix", "alphafold3"])
    args = parser.parse_args()

    search_dir = args.run_dir / "outputs" / "structure_search"
    search_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Foldseek for each model
    model_hits: dict[str, list[dict]] = {}
    for model in args.models:
        model_dir = args.run_dir / "outputs" / model
        if not model_dir.exists():
            print(f"  {model}: output not found, skipping")
            continue

        structure = find_best_structure(model_dir, model)
        if not structure:
            print(f"  {model}: no structure found, skipping")
            continue

        result_tsv = run_foldseek(
            structure, args.foldseek_db, search_dir, model,
            foldseek_bin=args.foldseek_bin,
        )
        if result_tsv:
            model_hits[model] = parse_foldseek_hits(result_tsv)

    if not model_hits:
        print("No Foldseek results. Exiting.")
        return 1

    # 2. Collect all PDB IDs and lookup ligands
    all_pdb_ids = set()
    for hits in model_hits.values():
        for h in hits:
            all_pdb_ids.add(h["pdb_id"])

    print(f"\nLooking up ligands for {len(all_pdb_ids)} unique PDB IDs...")
    ligand_map = lookup_ligands_bulk(args.rcsb_db, all_pdb_ids)
    print(f"  {len(ligand_map)} PDBs have candidate ligands")

    # 3. Find overlapping hits across models
    pdb_to_models: dict[str, dict[str, dict]] = defaultdict(dict)
    for model, hits in model_hits.items():
        for h in hits:
            pdb_to_models[h["pdb_id"]][model] = h

    # Sort by number of models that found it (consensus), then by avg pident
    consensus = []
    for pdb_id, models in pdb_to_models.items():
        ligands = ligand_map.get(pdb_id, [])
        avg_pident = sum(m["pident"] for m in models.values()) / len(models)
        avg_tmscore = sum(m.get("tmscore", 0) for m in models.values()) / len(models)
        consensus.append({
            "pdb_id": pdb_id,
            "num_models": len(models),
            "models": sorted(models.keys()),
            "avg_pident": avg_pident,
            "avg_tmscore": avg_tmscore,
            "has_ligand": len(ligands) > 0,
            "ligands": ligands,
            "per_model": {m: {"pident": d["pident"], "tmscore": d.get("tmscore", 0)} for m, d in models.items()},
        })

    consensus.sort(key=lambda x: (-x["num_models"], -x["has_ligand"], -x["avg_tmscore"]))

    # 4. Write results
    summary_path = search_dir / "consensus_summary.json"
    json.dump(consensus[:50], open(summary_path, "w"), indent=2)  # top 50

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Structure Search Consensus ({len(consensus)} unique PDBs)")
    print(f"{'='*70}")
    print(f"{'PDB':>6s} {'Models':>7s} {'AvgPident':>10s} {'AvgTMscore':>11s} {'Ligands':>8s} {'Details'}")
    print(f"{'-'*70}")
    for entry in consensus[:20]:
        models_str = ",".join(entry["models"])
        lig_str = ";".join(l["ccd_code"] for l in entry["ligands"][:3]) or "-"
        print(f"{entry['pdb_id']:>6s} {entry['num_models']:>7d} {entry['avg_pident']:>10.1f} {entry['avg_tmscore']:>11.3f} {lig_str:>8s} [{models_str}]")

    # Highlight: ligand-containing hits found by all models
    all_model_set = set(args.models) & set(model_hits.keys())
    full_consensus = [e for e in consensus if set(e["models"]) >= all_model_set and e["has_ligand"]]
    if full_consensus:
        print(f"\n  ★ {len(full_consensus)} PDBs with ligands found by ALL models:")
        for e in full_consensus[:10]:
            lig_str = ", ".join(f"{l['ccd_code']}({l['ligand_type']})" for l in e["ligands"])
            print(f"    {e['pdb_id']} — pident={e['avg_pident']:.1f}% TM={e['avg_tmscore']:.3f} — {lig_str}")

    print(f"\nResults: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
