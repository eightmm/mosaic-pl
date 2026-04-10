#!/usr/bin/env python3
"""Collect ion/metal positions from aligned template structures.

For metals/ions that are hard to position via cofolding alone,
this script finds templates containing the target ion, aligns them
to the cofolding best model, and collects all ion positions in the
cofolding reference frame.

Usage:
    python collect_template_ions.py \
        --run-dir experiments/runs/target \
        --input-yaml experiments/runs/target/inputs/boltz_input.yaml \
        --rcsb-dir ~/DB/RCSB/raw/mmCIF_data \
        --rcsb-db ~/DB/RCSB/processed/rcsb_index.db

Output:
    <run-dir>/outputs/ion_placement/
        ion_placement_summary.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IonPosition:
    template_pdb_id: str
    pident: float
    chain_id: str
    residue_name: str
    x: float
    y: float
    z: float
    alignment_rmsd: float
    aligned_residues: int

    def to_dict(self) -> dict:
        return {
            "template_pdb_id": self.template_pdb_id,
            "pident": self.pident,
            "chain_id": self.chain_id,
            "residue_name": self.residue_name,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "alignment_rmsd": round(self.alignment_rmsd, 3),
            "aligned_residues": self.aligned_residues,
        }


def extract_ion_ccd_codes(yaml_path: Path) -> list[str]:
    """Extract CCD codes of ion/metal ligands from unified input YAML."""
    known_ions = {
        "ZN", "MG", "CA", "FE", "MN", "CO", "CU", "NI", "NA", "K",
        "FE2", "FE3", "CU1", "CU2", "ZN2", "MG2", "CA2", "MN2",
        "CD", "HG", "PT", "W", "MO", "SE",
    }
    codes = []
    text = yaml_path.read_text()
    in_ligand = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ligand:") or stripped == "ligand:":
            in_ligand = True
            continue
        if in_ligand:
            if stripped.startswith("ccd:"):
                ccd = stripped.split(":", 1)[1].strip().strip("'\"").upper()
                if ccd in known_ions:
                    codes.append(ccd)
                in_ligand = False
            elif stripped.startswith("smiles:"):
                in_ligand = False
            elif stripped.startswith("id:"):
                continue  # skip id field, stay in ligand block
            elif stripped.startswith("- ") or (stripped and not stripped.startswith((" ", "#"))):
                in_ligand = False
    return codes


def find_template_cif(pdb_id: str, rcsb_dir: Path) -> Path | None:
    """Find CIF file for a PDB ID in the RCSB raw data directory."""
    pdb_id = pdb_id.lower()
    hash_dir = pdb_id[1:3]
    cif_gz = rcsb_dir / hash_dir / f"{pdb_id}.cif.gz"
    if cif_gz.exists():
        return cif_gz
    cif = rcsb_dir / hash_dir / f"{pdb_id}.cif"
    if cif.exists():
        return cif
    return None


def extract_cif(cif_path: Path, output_dir: Path) -> Path:
    """Extract gzipped CIF to output directory."""
    output = output_dir / cif_path.name.replace(".gz", "")
    if output.exists():
        return output
    if cif_path.suffix == ".gz":
        with gzip.open(cif_path, "rb") as f_in, open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(cif_path, output)
    return output


def find_ions_in_structure(cif_path: Path, target_ions: list[str]) -> list[dict]:
    """Find specific ions in a CIF structure, return their info."""
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    results = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name.upper() in target_ions:
                    for atom in residue:
                        results.append({
                            "chain_id": chain.name,
                            "residue_name": residue.name,
                            "atom_name": atom.name,
                            "x": atom.pos.x,
                            "y": atom.pos.y,
                            "z": atom.pos.z,
                        })
        break  # first model only
    return results


def align_template_to_reference(
    template_cif: Path, reference_cif: Path
) -> tuple[float, int, "gemmi.Transform | None"]:
    """Superpose template onto reference using gemmi CA superposition.

    Returns (rmsd, num_aligned_residues, transform) or (inf, 0, None) on failure.
    """
    import gemmi

    ref_st = gemmi.read_structure(str(reference_cif))
    mov_st = gemmi.read_structure(str(template_cif))

    # Get first protein chain from each
    ref_chain = None
    for model in ref_st:
        for chain in model:
            polymer = chain.get_polymer()
            if polymer and len(polymer) > 0:
                ref_chain = chain
                break
        if ref_chain:
            break

    mov_chain = None
    for model in mov_st:
        for chain in model:
            polymer = chain.get_polymer()
            if polymer and len(polymer) > 0:
                mov_chain = chain
                break
        if mov_chain:
            break

    if not ref_chain or not mov_chain:
        return float("inf"), 0, None

    ref_polymer = ref_chain.get_polymer()
    mov_polymer = mov_chain.get_polymer()

    try:
        sup = gemmi.calculate_superposition(
            ref_polymer, mov_polymer,
            gemmi.PolymerType.PeptideL,
            gemmi.SupSelect.CaP,
        )
        return sup.rmsd, sup.count, sup.transform
    except Exception as e:
        print(f"    Superposition failed: {e}")
        return float("inf"), 0, None


def transform_position(transform, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply gemmi Transform to a 3D position."""
    import gemmi
    pos = gemmi.Position(x, y, z)
    new_pos = transform.apply(pos)
    return new_pos.x, new_pos.y, new_pos.z


def cluster_positions(
    positions: list[IonPosition], threshold: float = 2.0
) -> list[dict]:
    """Cluster ion positions by distance. Simple greedy clustering."""
    if not positions:
        return []

    import numpy as np
    coords = np.array([[p.x, p.y, p.z] for p in positions])
    used = [False] * len(positions)
    clusters = []

    for i in range(len(positions)):
        if used[i]:
            continue
        cluster_members = [i]
        used[i] = True
        for j in range(i + 1, len(positions)):
            if used[j]:
                continue
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= threshold:
                cluster_members.append(j)
                used[j] = True

        member_coords = coords[cluster_members]
        centroid = member_coords.mean(axis=0)
        clusters.append({
            "centroid": [round(c, 3) for c in centroid.tolist()],
            "num_templates": len(cluster_members),
            "spread_angstrom": round(float(member_coords.std()), 3),
            "members": [positions[m].to_dict() for m in cluster_members],
        })

    clusters.sort(key=lambda c: -c["num_templates"])
    return clusters


def find_best_cofolding_structure(run_dir: Path) -> Path | None:
    """Find best cofolding CIF structure."""
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue
        if model.startswith("boltz"):
            for cif in sorted(model_dir.rglob("predictions/**/*.cif")):
                return cif
        elif model == "alphafold3":
            for cif in sorted(model_dir.rglob("*model*.cif")):
                return cif
        else:
            for cif in sorted(model_dir.rglob("*.cif")):
                return cif
    return None


def parse_mmseqs_hits(tsv_path: Path) -> list[dict]:
    """Parse MMseqs2 hits TSV (raw or filtered)."""
    hits = []
    if not tsv_path.exists():
        return hits

    # Try as filtered TSV first (has header)
    with open(tsv_path) as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.startswith("query\t"):
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                hits.append(row)
            return hits

    # Raw MMseqs2 output (no header)
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            target = parts[1]
            pdb_id = target.split("_")[0].lower() if "_" in target else target[:4].lower()
            hits.append({
                "pdb_id": pdb_id,
                "pident": parts[2],
                "target": target,
            })
    return hits


def check_ion_in_rcsb_db(db_path: Path, pdb_id: str, target_ions: list[str]) -> bool:
    """Check if a PDB has any of the target ions via rcsb_index.db."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in target_ions)
    cur.execute(
        f"SELECT COUNT(*) FROM ligand_instances WHERE pdb_id = ? AND ccd_code IN ({placeholders})",
        (pdb_id.lower(), *target_ions),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count > 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect ion/metal positions from aligned template structures."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--rcsb-dir", type=Path, default=Path.home() / "DB/RCSB/raw/mmCIF_data")
    parser.add_argument("--rcsb-db", type=Path, default=Path.home() / "DB/RCSB/processed/rcsb_index.db")
    parser.add_argument("--max-templates", type=int, default=50)
    parser.add_argument("--min-pident", type=float, default=30.0)
    parser.add_argument("--cluster-threshold", type=float, default=2.0,
                        help="Distance threshold for position clustering (Angstrom).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # 1. Detect target ions from input YAML
    target_ions = extract_ion_ccd_codes(args.input_yaml)
    if not target_ions:
        print("No ion/metal entities found in input YAML, skipping ion placement.")
        return 0

    print(f"Target ions: {', '.join(target_ions)}")

    # 2. Find cofolding reference structure
    ref_cif = find_best_cofolding_structure(run_dir)
    if not ref_cif:
        print("No cofolding structure found. Run cofolding first.")
        return 1
    print(f"Reference structure: {ref_cif}")

    # 3. Find template search hits
    filtered_tsv = run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
    raw_tsv = run_dir / "outputs" / "template_search_sequence" / "mmseqs_hits.tsv"
    hits_tsv = filtered_tsv if filtered_tsv.exists() else raw_tsv
    if not hits_tsv.exists():
        print("No template search results found.")
        return 0

    hits = parse_mmseqs_hits(hits_tsv)
    print(f"Total template hits: {len(hits)}")

    # 4. Filter hits by pident and ion presence
    output_dir = run_dir / "outputs" / "ion_placement"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp_cifs"
    tmp_dir.mkdir(exist_ok=True)

    all_positions: dict[str, list[IonPosition]] = {ion: [] for ion in target_ions}
    processed = 0

    for hit in hits:
        pdb_id = hit.get("pdb_id", "")
        if not pdb_id or len(pdb_id) != 4:
            continue
        pident = float(hit.get("pident", 0))
        if pident < args.min_pident:
            continue

        # Check if this PDB has the target ion (via DB or direct CIF check)
        has_ion = False
        if args.rcsb_db.exists():
            has_ion = check_ion_in_rcsb_db(args.rcsb_db, pdb_id, target_ions)
        if not has_ion:
            continue

        # Find and extract template CIF
        cif_path = find_template_cif(pdb_id, args.rcsb_dir)
        if not cif_path:
            continue

        template_cif = extract_cif(cif_path, tmp_dir)

        # Check which ions are actually in this structure
        ions_found = find_ions_in_structure(template_cif, target_ions)
        if not ions_found:
            continue

        # Align template to reference
        rmsd, count, transform = align_template_to_reference(template_cif, ref_cif)
        if transform is None or count < 20:
            continue

        # Transform ion coordinates
        for ion in ions_found:
            tx, ty, tz = transform_position(transform, ion["x"], ion["y"], ion["z"])
            ion_name = ion["residue_name"].upper()
            pos = IonPosition(
                template_pdb_id=pdb_id,
                pident=pident,
                chain_id=ion["chain_id"],
                residue_name=ion_name,
                x=tx, y=ty, z=tz,
                alignment_rmsd=rmsd,
                aligned_residues=count,
            )
            if ion_name in all_positions:
                all_positions[ion_name].append(pos)

        processed += 1
        if processed >= args.max_templates:
            break

    if processed == 0:
        print("No templates with target ions found.")
        return 0

    print(f"\nProcessed {processed} templates with target ions")

    # 5. Cluster and group by confidence
    summary: dict[str, dict] = {}
    for ion_name, positions in all_positions.items():
        if not positions:
            continue

        print(f"\n{'='*60}")
        print(f"  {ion_name}: {len(positions)} positions from {len(set(p.template_pdb_id for p in positions))} templates")
        print(f"{'='*60}")

        # Group by confidence
        high = [p for p in positions if p.pident >= 70]
        medium = [p for p in positions if 50 <= p.pident < 70]
        low = [p for p in positions if p.pident < 50]

        # Cluster all positions
        all_clusters = cluster_positions(positions, args.cluster_threshold)

        # Cluster per confidence group
        high_clusters = cluster_positions(high, args.cluster_threshold)
        medium_clusters = cluster_positions(medium, args.cluster_threshold)
        low_clusters = cluster_positions(low, args.cluster_threshold)

        for label, group, clusters in [
            ("high (pident >= 70%)", high, high_clusters),
            ("medium (50-70%)", medium, medium_clusters),
            ("low (30-50%)", low, low_clusters),
        ]:
            if not group:
                print(f"  {label}: no templates")
                continue
            n_templates = len(set(p.template_pdb_id for p in group))
            print(f"  {label}: {len(group)} positions from {n_templates} templates, {len(clusters)} cluster(s)")
            for ci, cl in enumerate(clusters[:3]):
                c = cl["centroid"]
                print(f"    cluster {ci+1}: [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] "
                      f"({cl['num_templates']} positions, spread={cl['spread_angstrom']:.1f}A)")

        summary[ion_name] = {
            "total_positions": len(positions),
            "total_templates": len(set(p.template_pdb_id for p in positions)),
            "clusters": all_clusters,
            "by_confidence": {
                "high": {
                    "pident_range": ">=70%",
                    "num_positions": len(high),
                    "num_templates": len(set(p.template_pdb_id for p in high)),
                    "clusters": high_clusters,
                },
                "medium": {
                    "pident_range": "50-70%",
                    "num_positions": len(medium),
                    "num_templates": len(set(p.template_pdb_id for p in medium)),
                    "clusters": medium_clusters,
                },
                "low": {
                    "pident_range": "30-50%",
                    "num_positions": len(low),
                    "num_templates": len(set(p.template_pdb_id for p in low)),
                    "clusters": low_clusters,
                },
            },
        }

    # 6. Write summary
    summary_path = output_dir / "ion_placement_summary.json"
    result = {
        "target_ions": target_ions,
        "reference_structure": str(ref_cif),
        "num_templates_processed": processed,
        "min_pident": args.min_pident,
        "cluster_threshold_angstrom": args.cluster_threshold,
        "ions": summary,
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSummary written to {summary_path}")

    # Cleanup temp CIFs
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
