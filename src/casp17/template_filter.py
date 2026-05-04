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


def _compute_target_fp(target_smiles: str | None):
    """Compute the query (target) Morgan FP once per filter run.

    Returns ``None`` if the SMILES doesn't parse — callers treat that as
    "no ligand-similarity signal available" and skip the per-hit lookup.
    """
    if not target_smiles:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


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


def lookup_ligands(conn_or_path, pdb_id: str) -> list[LigandHit]:
    """Look up ligand instances for a PDB ID from the RCSB index DB.

    Accepts either an open ``sqlite3.Connection`` or a path. The connection
    overload is the hot path: ``filter_hits_with_ligands`` can be called
    with 1500+ hits, and opening a fresh ``sqlite3.connect`` per hit was
    burning 5-20 ms × N_hits in pure connection setup overhead — visible
    as 16× slowdown when 11 jobs ran concurrently. Pass an already-open
    connection to amortise that to a single setup per filter run.
    """
    if isinstance(conn_or_path, sqlite3.Connection):
        conn = conn_or_path
        owns_conn = False
    else:
        conn = sqlite3.connect(str(conn_or_path))
        owns_conn = True
    try:
        cur = conn.execute(
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
    finally:
        if owns_conn:
            conn.close()
    return results


def lookup_deposition_date(conn_or_path, pdb_id: str) -> str | None:
    """Return the ISO ``deposition_date`` for a PDB entry, or ``None`` if not
    indexed. Used by ``filter_hits_with_ligands`` to drop post-cutoff templates
    for time-split experiments (e.g. exclude 2025+ structures when benchmarking
    on held-out 2025 targets). Accepts a connection or path — the connection
    overload reuses an open connection across all hits in the filter loop."""
    if isinstance(conn_or_path, sqlite3.Connection):
        conn = conn_or_path
        owns_conn = False
    else:
        conn = sqlite3.connect(str(conn_or_path))
        owns_conn = True
    try:
        cur = conn.execute(
            "SELECT deposition_date FROM entries WHERE pdb_id = ?",
            (pdb_id.lower(),),
        )
        row = cur.fetchone()
    finally:
        if owns_conn:
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
    foldseek_qtmscore_min: float = 0.0,
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
        foldseek_qtmscore_min: TM-score floor applied to **foldseek-only** hits
            (mmseqs hits and dual-source hits bypass it). Compares against
            ``max(qtmscore, ttmscore)`` so a small template covering a query
            domain (high qtm) and a big template whose domain is our query
            (high ttm) both qualify. Default 0.0 = no gate; set ≥ 0.5 to
            enforce same-fold (canonical Zhang/Skolnick threshold).

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

    # Pre-compute the target FP once + open the CCD-keyed Morgan FP cache.
    # MCS used to be metadata here too (rdFMCS.FindMCS per hit × candidate)
    # but that blew up to 30-90 min per target on 20-30 atom drug ligands;
    # MCS only actually gates Track 3 lig-align downstream, where the
    # candidates are already filtered to a handful — compute lazily there.
    # Tanimoto via cache-resident Morgan FPs is O(1) lookup + ~1ms per
    # comparison, so the whole filter runs in seconds regardless of the
    # query molecule's size.
    from casp17.ligand_fp_cache import LigandFPCache, tanimoto as _tanimoto
    _target_fp = _compute_target_fp(target_smiles)
    _fp_cache = LigandFPCache()

    raw_hits: list[dict[str, str]] = []
    if hits_tsv is not None and Path(hits_tsv).exists():
        raw_hits.extend(parse_mmseqs_hits(hits_tsv))
    foldseek_dropped = 0
    if foldseek_tsv is not None and Path(foldseek_tsv).exists():
        for hit in parse_foldseek_hits(foldseek_tsv):
            if foldseek_qtmscore_min > 0.0:
                qtm = _safe_float(hit.get("qtmscore", "0"))
                ttm = _safe_float(hit.get("ttmscore", "0"))
                if max(qtm, ttm) < foldseek_qtmscore_min:
                    foldseek_dropped += 1
                    continue
            raw_hits.append(hit)
    if foldseek_dropped:
        print(
            f"  foldseek-floor: dropped {foldseek_dropped} hit(s) below "
            f"max(qtmscore, ttmscore) >= {foldseek_qtmscore_min}"
        )

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

    # Open one RCSB connection for the whole filter run, plus a tiny
    # per-PDB cache. lookup_ligands used to ``sqlite3.connect`` per hit
    # — at 1500 hits × 5-20 ms setup that already added ~15 s on its own,
    # but with N concurrent jobs the OS-level open/close churn pushed
    # filter wall-time from 5 min (single) to >1 h (×11 jobs). One conn
    # + dict cache collapses both axes back to a few seconds.
    rcsb_conn = sqlite3.connect(str(db_path))
    ligand_cache: dict[str, list[LigandHit]] = {}
    dep_cache: dict[str, str | None] = {}
    try:
        for slot in merged.values():
            pdb_id = slot["pdb_id"]
            if max_deposition_date:
                if pdb_id not in dep_cache:
                    dep_cache[pdb_id] = lookup_deposition_date(rcsb_conn, pdb_id)
                dep = dep_cache[pdb_id]
                if dep and dep >= max_deposition_date:
                    date_skipped += 1
                    continue

            if pdb_id not in ligand_cache:
                ligand_cache[pdb_id] = lookup_ligands(rcsb_conn, pdb_id)
            ligands = ligand_cache[pdb_id]
            filtered_ligands = [l for l in ligands if l.ligand_type in ligand_types]

            scored_ligands = []
            best_tanimoto = 0.0
            best_mcs = 0.0  # MCS coverage stays at 0 from filter — Track 3
                            # fills this lazily on its picked templates only.
            for lig in filtered_ligands:
                tanimoto = 0.0
                if _target_fp is not None and lig.ccd_code:
                    cand_fp = _fp_cache.get_fp(lig.ccd_code, lig.smiles)
                    tanimoto = round(_tanimoto(_target_fp, cand_fp), 4)
                scored = LigandHit(
                    pdb_id=lig.pdb_id,
                    ccd_code=lig.ccd_code,
                    ligand_type=lig.ligand_type,
                    is_candidate=lig.is_candidate,
                    smiles=lig.smiles,
                    molecular_weight=lig.molecular_weight,
                    contact_chain_ids=lig.contact_chain_ids,
                    tanimoto=tanimoto,
                    mcs_coverage=0.0,
                )
                scored_ligands.append(scored)
                if tanimoto > best_tanimoto:
                    best_tanimoto = tanimoto

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
    finally:
        rcsb_conn.close()

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
