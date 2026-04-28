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
    # Provenance (union of mmseqs + foldseek). MCS/Tanimoto are kept as
    # *metadata only* here — no gating happens in this filter, so callers
    # like Track-2 box docking and pocket extraction see every hit. Only
    # lig-mcs-align (Track 3) inspects best_mcs_coverage downstream.
    in_mmseqs: bool = False
    in_foldseek: bool = False
    qtmscore: float = 0.0      # foldseek query-side TM-score (1.0 = identical fold)
    ttmscore: float = 0.0      # foldseek target-side TM-score
    alntmscore: float = 0.0    # foldseek alignment TM-score
    prob: float = 0.0          # foldseek HMM probability


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
    """Parse MMseqs2 easy-search output TSV (default 12-column format).

    Each row is tagged with ``source="mmseqs"`` so downstream merge logic
    can reconcile a target with a parallel foldseek hit list.
    """
    hits = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue
            hits.append({
                "source": "mmseqs",
                "query": parts[0],
                "target": parts[1],
                "pident": parts[2],
                "alnlen": parts[3],
                "evalue": parts[10],
                "bits": parts[11],
                "qlen": parts[12] if len(parts) > 12 else "0",
                "tlen": parts[13] if len(parts) > 13 else "0",
                # foldseek-only metrics absent for mmseqs hits
                "qtmscore": "0",
                "ttmscore": "0",
                "alntmscore": "0",
                "prob": "0",
            })
    return hits


def parse_foldseek_hits(tsv_path: Path) -> list[dict[str, str]]:
    """Parse Foldseek easy-search output produced by our adapter.

    Adapter command in ``adapters.py`` uses
    ``--format-output query,target,evalue,bits,alntmscore,qtmscore,ttmscore,prob``
    so each row carries 8 fields. Sequence-only metrics (pident/qlen/tlen)
    are filled with sentinels so the merged schema stays uniform.
    """
    hits = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            hits.append({
                "source": "foldseek",
                "query": parts[0],
                "target": parts[1],
                "evalue": parts[2],
                "bits": parts[3],
                "alntmscore": parts[4],
                "qtmscore": parts[5],
                "ttmscore": parts[6],
                "prob": parts[7],
                # not provided by foldseek easy-search with this format
                "pident": "0",
                "alnlen": "0",
                "qlen": "0",
                "tlen": "0",
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


def lookup_deposition_date(db_path: Path, pdb_id: str) -> str | None:
    """Return the ISO ``deposition_date`` for a PDB entry, or ``None`` if not
    indexed. Used by ``filter_hits_with_ligands`` to drop post-cutoff templates
    for time-split experiments (e.g. exclude 2025+ structures when benchmarking
    on held-out 2025 targets)."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT deposition_date FROM entries WHERE pdb_id = ?",
            (pdb_id.lower(),),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return default


def filter_hits_with_ligands(
    hits_tsv: Path | None = None,
    db_path: Path | None = None,
    target_smiles: str | None = None,
    ligand_types: set[str] | None = None,
    output_path: Path | None = None,
    max_deposition_date: str | None = None,
    foldseek_tsv: Path | None = None,
) -> list[TemplateHit]:
    """Union-merge template hits from MMseqs2 + Foldseek and annotate with ligand info.

    Args:
        hits_tsv: MMseqs2 easy-search output TSV (optional — pass ``None`` to use
            foldseek-only).
        db_path: Path to rcsb_index.db SQLite database.
        target_smiles: Target ligand SMILES — only used to populate Tanimoto/MCS
            metadata. **No gating happens here**; downstream consumers decide
            whether to apply MCS thresholds (only lig-align does, currently).
        ligand_types: Optional set of ligand types to keep.
        output_path: Optional path to write filtered results TSV.
        max_deposition_date: Drop any hit whose RCSB ``deposition_date`` is
            on or after this ISO date. Used for time-split benchmarks.
        foldseek_tsv: Optional Foldseek easy-search output TSV. When supplied,
            its hits are unioned with the mmseqs list keyed on ``(pdb_id,
            chain_id)``. Foldseek-only hits expose ``qtmscore`` etc. as the
            primary structural-similarity signal (pident=0 for those).

    Returns:
        List of TemplateHit with ligand information and similarity scores,
        sorted so candidates with both-source provenance + good metrics
        appear first.
    """
    if ligand_types is None:
        ligand_types = {"small_molecule", "cofactor", "metabolite", "nucleotide_like", "peptide_like"}

    if hits_tsv is None and foldseek_tsv is None:
        raise ValueError("filter_hits_with_ligands: must provide at least one of hits_tsv or foldseek_tsv")
    if db_path is None:
        raise ValueError("filter_hits_with_ligands: db_path is required")

    raw_hits: list[dict[str, str]] = []
    if hits_tsv is not None and Path(hits_tsv).exists():
        raw_hits.extend(parse_mmseqs_hits(hits_tsv))
    if foldseek_tsv is not None and Path(foldseek_tsv).exists():
        raw_hits.extend(parse_foldseek_hits(foldseek_tsv))

    # Union-merge by (pdb_id, chain_id). When both sources hit the same chain
    # we keep both sets of metrics on a single TemplateHit so the consumer can
    # see "this template was supported by both seq+struct" as evidence.
    merged: dict[tuple[str, str], dict] = {}
    for hit in raw_hits:
        target = hit["target"]
        pdb_id = target.split("_")[0].lower() if "_" in target else target[:4].lower()
        chain_id = target.split("_")[1] if "_" in target else ""
        key = (pdb_id, chain_id)
        slot = merged.get(key)
        if slot is None:
            slot = {
                "query": hit["query"],
                "target": target,
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "pident": 0.0,
                "evalue": float("inf"),
                "qlen": 0,
                "tlen": 0,
                "in_mmseqs": False,
                "in_foldseek": False,
                "qtmscore": 0.0,
                "ttmscore": 0.0,
                "alntmscore": 0.0,
                "prob": 0.0,
            }
            merged[key] = slot
        evalue = _safe_float(hit["evalue"], default=float("inf"))
        if evalue < slot["evalue"]:
            slot["evalue"] = evalue
        if hit["source"] == "mmseqs":
            slot["in_mmseqs"] = True
            slot["pident"] = max(slot["pident"], _safe_float(hit["pident"]))
            slot["qlen"] = max(slot["qlen"], _safe_int(hit["qlen"]))
            slot["tlen"] = max(slot["tlen"], _safe_int(hit["tlen"]))
        else:  # foldseek
            slot["in_foldseek"] = True
            slot["qtmscore"] = max(slot["qtmscore"], _safe_float(hit["qtmscore"]))
            slot["ttmscore"] = max(slot["ttmscore"], _safe_float(hit["ttmscore"]))
            slot["alntmscore"] = max(slot["alntmscore"], _safe_float(hit["alntmscore"]))
            slot["prob"] = max(slot["prob"], _safe_float(hit["prob"]))

    results: list[TemplateHit] = []
    date_skipped = 0

    for slot in merged.values():
        pdb_id = slot["pdb_id"]
        if max_deposition_date:
            dep = lookup_deposition_date(db_path, pdb_id)
            if dep and dep >= max_deposition_date:
                date_skipped += 1
                continue

        ligands = lookup_ligands(db_path, pdb_id)
        filtered_ligands = [l for l in ligands if l.ligand_type in ligand_types]

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
            query=slot["query"],
            target=slot["target"],
            pdb_id=slot["pdb_id"],
            chain_id=slot["chain_id"],
            pident=float(slot["pident"]),
            evalue=float(slot["evalue"]) if slot["evalue"] != float("inf") else 0.0,
            qlen=int(slot["qlen"]),
            tlen=int(slot["tlen"]),
            ligands=scored_ligands,
            best_tanimoto=best_tanimoto,
            best_mcs_coverage=best_mcs,
            in_mmseqs=bool(slot["in_mmseqs"]),
            in_foldseek=bool(slot["in_foldseek"]),
            qtmscore=float(slot["qtmscore"]),
            ttmscore=float(slot["ttmscore"]),
            alntmscore=float(slot["alntmscore"]),
            prob=float(slot["prob"]),
        )
        results.append(template_hit)

    if max_deposition_date and date_skipped:
        print(f"  date-filter: dropped {date_skipped} hit(s) with deposition_date >= {max_deposition_date}")

    # Sort by evidence breadth first (both sources > one source), then by the
    # best structural/sequence similarity available, then ligand count.
    # Tanimoto/MCS used as secondary tiebreakers ONLY — they do not gate.
    def _evidence_key(h: TemplateHit) -> tuple:
        sources = int(h.in_mmseqs) + int(h.in_foldseek)
        struct_sim = max(h.qtmscore, h.alntmscore)
        return (
            -sources,
            -struct_sim,
            -h.pident,
            -len(h.ligands),
            -h.best_tanimoto,
            -h.best_mcs_coverage,
        )

    results.sort(key=_evidence_key)

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "query", "target", "pdb_id", "chain_id", "pident", "evalue",
                "num_ligands", "best_tanimoto", "best_mcs_coverage",
                "ligand_codes", "ligand_types", "ligand_smiles",
                "ligand_tanimotos", "ligand_mcs_coverages",
                # Provenance + foldseek metrics (appended to keep older
                # consumers reading the first 14 columns intact).
                "in_mmseqs", "in_foldseek",
                "qtmscore", "ttmscore", "alntmscore", "prob",
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
                    int(h.in_mmseqs), int(h.in_foldseek),
                    f"{h.qtmscore:.4f}", f"{h.ttmscore:.4f}",
                    f"{h.alntmscore:.4f}", f"{h.prob:.4f}",
                ])

    return results
