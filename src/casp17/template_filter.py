"""Filter template search hits using RCSB ligand index DB.

Given MMseqs2/Foldseek hit list, looks up each PDB in the SQLite index
to find which hits have drug-like ligands, cofactors, etc.
Optionally computes Tanimoto similarity and MCS coverage against a
target ligand SMILES.
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
    tanimoto: float | None = None
    mcs_coverage: float | None = None


@dataclass(frozen=True, slots=True)
class TemplateHit:
    query: str
    target: str
    pdb_id: str
    chain_id: str
    pident: float
    evalue: float
    qlen: int
    tlen: int
    ligands: list[LigandHit] = field(default_factory=list)
    best_tanimoto: float = 0.0
    best_mcs_coverage: float = 0.0


def _compute_ligand_similarity(
    target_smiles: str, template_smiles: str
) -> tuple[float, float]:
    """Compute Tanimoto similarity and MCS coverage between two SMILES.

    Returns:
        (tanimoto, mcs_coverage) where both are in [0, 1].
        mcs_coverage = MCS_atoms / min(target_atoms, template_atoms).
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, rdFMCS

        mol_t = Chem.MolFromSmiles(target_smiles)
        mol_q = Chem.MolFromSmiles(template_smiles)
        if mol_t is None or mol_q is None:
            return 0.0, 0.0

        # Tanimoto (Morgan fingerprint, radius=2)
        fp_t = AllChem.GetMorganFingerprintAsBitVect(mol_t, 2, nBits=2048)
        fp_q = AllChem.GetMorganFingerprintAsBitVect(mol_q, 2, nBits=2048)
        tanimoto = DataStructs.TanimotoSimilarity(fp_t, fp_q)

        # MCS coverage
        mcs = rdFMCS.FindMCS(
            [mol_t, mol_q],
            timeout=5,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        if mcs.numAtoms > 0:
            min_atoms = min(mol_t.GetNumHeavyAtoms(), mol_q.GetNumHeavyAtoms())
            mcs_coverage = mcs.numAtoms / max(min_atoms, 1)
        else:
            mcs_coverage = 0.0

        return round(tanimoto, 4), round(min(mcs_coverage, 1.0), 4)
    except Exception:
        return 0.0, 0.0


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
    target_smiles: str | None = None,
    ligand_types: set[str] | None = None,
    output_path: Path | None = None,
) -> list[TemplateHit]:
    """Filter template search hits to those with candidate ligands.

    Args:
        hits_tsv: MMseqs2/Foldseek output TSV.
        db_path: Path to rcsb_index.db SQLite database.
        target_smiles: Target ligand SMILES for similarity comparison.
        ligand_types: Optional set of ligand types to keep.
        output_path: Optional path to write filtered results TSV.

    Returns:
        List of TemplateHit with ligand information and similarity scores.
    """
    if ligand_types is None:
        ligand_types = {"small_molecule", "cofactor", "metabolite", "nucleotide_like", "peptide_like"}

    raw_hits = parse_mmseqs_hits(hits_tsv)
    results: list[TemplateHit] = []

    for hit in raw_hits:
        target = hit["target"]
        pdb_id = target.split("_")[0].lower() if "_" in target else target[:4].lower()
        chain_id = target.split("_")[1] if "_" in target else ""

        ligands = lookup_ligands(db_path, pdb_id)
        filtered_ligands = [l for l in ligands if l.ligand_type in ligand_types]

        # Compute similarity if target SMILES provided
        scored_ligands = []
        best_tanimoto = 0.0
        best_mcs = 0.0
        for lig in filtered_ligands:
            tanimoto, mcs_cov = 0.0, 0.0
            if target_smiles and lig.smiles:
                tanimoto, mcs_cov = _compute_ligand_similarity(target_smiles, lig.smiles)
            scored = LigandHit(
                pdb_id=lig.pdb_id,
                ccd_code=lig.ccd_code,
                ligand_type=lig.ligand_type,
                is_candidate=lig.is_candidate,
                smiles=lig.smiles,
                molecular_weight=lig.molecular_weight,
                contact_chain_ids=lig.contact_chain_ids,
                tanimoto=tanimoto,
                mcs_coverage=mcs_cov,
            )
            scored_ligands.append(scored)
            best_tanimoto = max(best_tanimoto, tanimoto)
            best_mcs = max(best_mcs, mcs_cov)

        template_hit = TemplateHit(
            query=hit["query"],
            target=target,
            pdb_id=pdb_id,
            chain_id=chain_id,
            pident=float(hit["pident"]),
            evalue=float(hit["evalue"]),
            qlen=int(hit["qlen"]),
            tlen=int(hit["tlen"]),
            ligands=scored_ligands,
            best_tanimoto=best_tanimoto,
            best_mcs_coverage=best_mcs,
        )
        results.append(template_hit)

    # Sort: best tanimoto first, then ligand count, then identity
    results.sort(key=lambda h: (-h.best_tanimoto, -h.best_mcs_coverage, -len(h.ligands), -h.pident))

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "query", "target", "pdb_id", "chain_id", "pident", "evalue",
                "num_ligands", "best_tanimoto", "best_mcs_coverage",
                "ligand_codes", "ligand_types", "ligand_smiles",
                "ligand_tanimotos", "ligand_mcs_coverages",
            ])
            for h in results:
                writer.writerow([
                    h.query, h.target, h.pdb_id, h.chain_id,
                    f"{h.pident:.1f}", f"{h.evalue:.2e}",
                    len(h.ligands),
                    f"{h.best_tanimoto:.4f}", f"{h.best_mcs_coverage:.4f}",
                    ";".join(l.ccd_code for l in h.ligands),
                    ";".join(l.ligand_type for l in h.ligands),
                    ";".join(l.smiles or "" for l in h.ligands),
                    ";".join(f"{l.tanimoto:.4f}" for l in h.ligands),
                    ";".join(f"{l.mcs_coverage:.4f}" for l in h.ligands),
                ])

    return results
