"""Filter template search hits using RCSB ligand index DB.

Given MMseqs2/Foldseek hit list, looks up each PDB in the SQLite index
to find which hits have drug-like ligands, cofactors, etc.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LigandHit:
    pdb_id: str
    ccd_code: str
    ligand_type: str
    is_candidate: bool
    smiles: str | None
    molecular_weight: float | None
    contact_chain_ids: str | None


@dataclass(frozen=True, slots=True)
class TemplateHit:
    query: str
    target: str  # pdb_id_chain format (e.g., "1abc_A")
    pdb_id: str
    chain_id: str
    pident: float
    evalue: float
    qlen: int
    tlen: int
    ligands: list[LigandHit] = field(default_factory=list)


def parse_mmseqs_hits(tsv_path: Path) -> list[dict[str, str]]:
    """Parse MMseqs2 easy-search output TSV."""
    hits = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            hits.append({
                "query": parts[0],
                "target": parts[1],
                "pident": parts[2],
                "alnlen": parts[3],
                "evalue": parts[10],
                "bits": parts[11],
                "qlen": parts[12] if len(parts) > 12 else "0",
                "tlen": parts[13] if len(parts) > 13 else "0",
            })
    return hits


def lookup_ligands(db_path: Path, pdb_id: str) -> list[LigandHit]:
    """Look up ligand instances for a PDB ID from the RCSB index DB."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """SELECT ccd_code, ligand_type, is_candidate, smiles,
                  molecular_weight, contact_chain_ids
           FROM ligand_instances
           WHERE pdb_id = ? AND is_candidate = 1""",
        (pdb_id.lower(),),
    )
    results = [
        LigandHit(
            pdb_id=pdb_id.lower(),
            ccd_code=row[0],
            ligand_type=row[1],
            is_candidate=bool(row[2]),
            smiles=row[3],
            molecular_weight=row[4],
            contact_chain_ids=row[5],
        )
        for row in cur.fetchall()
    ]
    conn.close()
    return results


def filter_hits_with_ligands(
    hits_tsv: Path,
    db_path: Path,
    ligand_types: set[str] | None = None,
    output_path: Path | None = None,
) -> list[TemplateHit]:
    """Filter template search hits to those with candidate ligands.

    Args:
        hits_tsv: MMseqs2/Foldseek output TSV.
        db_path: Path to rcsb_index.db SQLite database.
        ligand_types: Optional set of ligand types to keep.
            Default: {"small_molecule", "cofactor", "metabolite", "nucleotide_like"}
        output_path: Optional path to write filtered results TSV.

    Returns:
        List of TemplateHit with ligand information.
    """
    if ligand_types is None:
        ligand_types = {"small_molecule", "cofactor", "metabolite", "nucleotide_like", "peptide_like"}

    raw_hits = parse_mmseqs_hits(hits_tsv)
    results: list[TemplateHit] = []

    for hit in raw_hits:
        target = hit["target"]
        # Parse PDB ID from target name (format: "1abc_1" or "1abc_A")
        pdb_id = target.split("_")[0].lower() if "_" in target else target[:4].lower()
        chain_id = target.split("_")[1] if "_" in target else ""

        ligands = lookup_ligands(db_path, pdb_id)
        # Filter by requested types
        filtered_ligands = [l for l in ligands if l.ligand_type in ligand_types]

        template_hit = TemplateHit(
            query=hit["query"],
            target=target,
            pdb_id=pdb_id,
            chain_id=chain_id,
            pident=float(hit["pident"]),
            evalue=float(hit["evalue"]),
            qlen=int(hit["qlen"]),
            tlen=int(hit["tlen"]),
            ligands=filtered_ligands,
        )
        results.append(template_hit)

    # Sort: hits with ligands first, then by identity
    results.sort(key=lambda h: (-len(h.ligands), -h.pident))

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "query", "target", "pdb_id", "chain_id", "pident", "evalue",
                "num_ligands", "ligand_codes", "ligand_types", "ligand_smiles",
            ])
            for h in results:
                writer.writerow([
                    h.query, h.target, h.pdb_id, h.chain_id,
                    f"{h.pident:.1f}", f"{h.evalue:.2e}",
                    len(h.ligands),
                    ";".join(l.ccd_code for l in h.ligands),
                    ";".join(l.ligand_type for l in h.ligands),
                    ";".join(l.smiles or "" for l in h.ligands),
                ])

    return results
