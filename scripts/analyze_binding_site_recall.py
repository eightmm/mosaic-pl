#!/usr/bin/env python3
"""Binding-site recall (DCC ≤ 4 Å) on a batch of pipeline runs.

Methodology (agreed 2026-05-04):
- Ground truth: heavy-atom centroid of the native primary ligand. Primary =
  the first ``ligands`` entry in ``docking_prep_summary.json`` (== first
  SMILES-bearing entity in the input YAML, == the ligand actually docked).
- For homomers / multiple crystal instances of the same CCD, every
  native instance counts — DCC is min over all instances.
- DCC: Euclidean distance from a registered binding-site source's center
  to the nearest native instance centroid.
- Cutoff: ≤ 4 Å.
- Rollup:
  * per-source SR (across all targets that registered that source),
  * per-category oracle (any source in the category within 4 Å),
  * overall oracle (any source at all within 4 Å).

Crystal → cofold frame: USalign chain A of the crystal mmCIF onto a cofold
``*_aligned.cif`` (Stage 2.5 output) — same approach as
``scripts/analyze_batch_rmsd.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from analyze_batch_rmsd import _usalign_xform  # noqa: E402

RCSB_DB = Path("/home/jaemin/DB/RCSB/processed/rcsb_index.db")

CATEGORIES = {
    "cofolding": tuple(f"cofolding_{i}" for i in (1, 2, 3)),
    "swinsite": tuple(f"swinsite_{i}" for i in (1, 2, 3)),
    "p2rank": tuple(f"p2rank_{i}" for i in (1, 2, 3)),
    "template_consensus": tuple(f"template_consensus_{i}" for i in range(1, 11)),
}
ALL_SOURCE_NAMES = [s for grp in CATEGORIES.values() for s in grp]


def _category_of(src: str) -> str | None:
    for cat, members in CATEGORIES.items():
        if src in members:
            return cat
    return None


def _canonical(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _native_centroids(target: str, primary_smiles: str | None,
                      cofold_ref_cif: Path) -> list[tuple[str, np.ndarray]]:
    """Return list of (ccd, centroid in cofold frame) for primary-matching
    crystal ligand instances. Falls back to all candidate ligands when SMILES
    match fails (e.g. RCSB stored a slightly different SMILES) so we don't
    silently turn the target into a missing-native row."""
    conn = sqlite3.connect(str(RCSB_DB))
    try:
        row = conn.execute(
            "SELECT cif_path FROM entries WHERE pdb_id=?",
            (target.lower(),),
        ).fetchone()
        if not row:
            return []
        crystal_cif = Path(row[0])
        candidate_rows = conn.execute(
            "SELECT DISTINCT ccd_code, smiles FROM ligand_instances "
            "WHERE pdb_id=? AND is_candidate=1",
            (target.lower(),),
        ).fetchall()
    finally:
        conn.close()
    if not crystal_cif.exists() or not candidate_rows:
        return []

    primary_canon = _canonical(primary_smiles)
    matching_ccds = {
        ccd.upper()
        for ccd, smi in candidate_rows
        if primary_canon and _canonical(smi) == primary_canon
    }
    if not matching_ccds:
        # SMILES mismatch (RCSB sometimes stores a slightly different
        # representation, or primary is a ccd-only entry). Fall back to all
        # candidate CCDs so the target still contributes a measurement.
        matching_ccds = {ccd.upper() for ccd, _ in candidate_rows}

    td = Path(tempfile.mkdtemp(prefix=f"bsr_{target}_"))
    R, t = _usalign_xform(crystal_cif, cofold_ref_cif, td)
    if R is None:
        return []

    centroids: list[tuple[str, np.ndarray]] = []
    crystal = gemmi.read_structure(str(crystal_cif))
    for ch in crystal[0]:
        for res in ch:
            if res.name.upper() not in matching_ccds:
                continue
            heavy = []
            for a in res:
                if a.element.name == "H":
                    continue
                heavy.append((a.pos.x, a.pos.y, a.pos.z))
            if not heavy:
                continue
            centroid_crystal = np.asarray(heavy).mean(axis=0)
            centroid_cofold = R @ centroid_crystal + t
            centroids.append((res.name.upper(), centroid_cofold))
    return centroids


def analyze_one(target: str, runs_root: Path) -> dict:
    run = runs_root / f"{target}_input" / f"{target}_input"
    summary_json = run / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary_json.exists():
        return {"target": target, "status": "no_docking_prep", "rows": []}
    summary = json.loads(summary_json.read_text())
    bs_pred = summary.get("binding_site_predictions") or {}
    ligs = summary.get("ligands") or []
    if not ligs:
        return {"target": target, "status": "no_ligand", "rows": []}
    primary_smiles = ligs[0].get("smiles")

    cofold_ref = next((run / "outputs/protenix").rglob("*_aligned.cif"), None)
    if cofold_ref is None:
        cofold_ref = next((run / "outputs").rglob("*_aligned.cif"), None)
    if cofold_ref is None:
        return {"target": target, "status": "no_aligned_cif", "rows": []}

    natives = _native_centroids(target, primary_smiles, cofold_ref)
    if not natives:
        return {"target": target, "status": "no_native", "rows": []}

    native_arr = np.asarray([c for _, c in natives])  # (N, 3)
    rows = []
    for src, payload in bs_pred.items():
        c = payload.get("center")
        if c is None or len(c) != 3:
            continue
        center = np.asarray(c, dtype=float)
        d = float(np.linalg.norm(native_arr - center, axis=1).min())
        rows.append({
            "target": target,
            "source": src,
            "category": _category_of(src) or "other",
            "dcc": round(d, 3),
        })
    return {"target": target, "status": "ok",
            "n_natives": len(natives), "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path,
                   default=Path("experiments/msa_e2e_test/runs"))
    p.add_argument("--targets", nargs="*", default=None,
                   help="Optional explicit target list (default: every "
                        "<id>_input subdir under runs-dir)")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-tsv", type=Path,
                   default=Path("/tmp/binding_site_recall.tsv"))
    p.add_argument("--per-target-tsv", type=Path,
                   default=Path("/tmp/binding_site_recall_per_target.tsv"))
    args = p.parse_args()

    if args.targets:
        targets = args.targets
    else:
        targets = sorted(
            d.name.removesuffix("_input")
            for d in args.runs_dir.iterdir()
            if d.is_dir() and d.name.endswith("_input")
        )
    print(f"[bsr] {len(targets)} targets, cutoff={args.cutoff} Å, "
          f"workers={args.workers}")

    all_rows: list[dict] = []
    status_counter: dict[str, int] = defaultdict(int)
    target_stats: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(analyze_one, t, args.runs_dir): t for t in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                status_counter["exception"] += 1
                print(f"  [{i}/{len(targets)}] {t}: EXCEPTION {e}")
                continue
            status_counter[res["status"]] += 1
            if res["status"] != "ok":
                continue
            all_rows.extend(res["rows"])
            cat_min: dict[str, float] = {}
            for row in res["rows"]:
                cat = row["category"]
                cat_min[cat] = min(cat_min.get(cat, 1e9), row["dcc"])
            overall = min((r["dcc"] for r in res["rows"]), default=None)
            target_stats.append({
                "target": t,
                "n_natives": res["n_natives"],
                "n_sources_registered": len(res["rows"]),
                "min_dcc_overall": overall,
                "min_dcc_cofolding": cat_min.get("cofolding"),
                "min_dcc_swinsite": cat_min.get("swinsite"),
                "min_dcc_p2rank": cat_min.get("p2rank"),
                "min_dcc_template_consensus": cat_min.get("template_consensus"),
            })

    print()
    print("status counts:")
    for k, v in sorted(status_counter.items()):
        print(f"  {k}: {v}")

    if not all_rows:
        print("[bsr] no data — bail.")
        return 1

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["target", "source", "category", "dcc"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(all_rows)
    print(f"[bsr] per-source TSV: {args.output_tsv} ({len(all_rows)} rows)")

    args.per_target_tsv.parent.mkdir(parents=True, exist_ok=True)
    if target_stats:
        with args.per_target_tsv.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=list(target_stats[0].keys()), delimiter="\t",
            )
            w.writeheader()
            w.writerows(target_stats)
        print(f"[bsr] per-target TSV: {args.per_target_tsv}")

    cutoff = args.cutoff
    n_targets = len(target_stats)
    print()
    print(f"=== Binding-site recall (DCC ≤ {cutoff} Å) — N={n_targets} targets ===")
    print()

    # Per-source SR
    by_src: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in all_rows:
        s = by_src[r["source"]]
        s["n"] += 1
        s["hit"] += int(r["dcc"] <= cutoff)
    print("per-source SR (rows = targets that registered that source):")
    print(f"  {'source':<24} {'n':>5} {'hit':>5} {'SR%':>6}  median_dcc")
    rows_sorted = sorted(
        by_src.items(),
        key=lambda kv: (
            ALL_SOURCE_NAMES.index(kv[0]) if kv[0] in ALL_SOURCE_NAMES else 999
        ),
    )
    for src, s in rows_sorted:
        med = float(np.median([r["dcc"] for r in all_rows if r["source"] == src]))
        sr = 100.0 * s["hit"] / s["n"] if s["n"] else 0.0
        print(f"  {src:<24} {s['n']:>5} {s['hit']:>5} {sr:>6.1f}  {med:>9.2f}")

    # Per-category oracle (per target)
    print()
    print("per-category oracle (any source in category ≤ cutoff):")
    print(f"  {'category':<24} {'n_targets':>10} {'hit':>5} {'SR%':>6}  median_min_dcc")
    for cat in ("cofolding", "swinsite", "p2rank", "template_consensus"):
        col = f"min_dcc_{cat}"
        vals = [t[col] for t in target_stats if t.get(col) is not None]
        n_have = len(vals)
        n_hit = sum(1 for v in vals if v <= cutoff)
        sr = 100.0 * n_hit / n_have if n_have else 0.0
        med = float(np.median(vals)) if vals else float("nan")
        print(f"  {cat:<24} {n_have:>10} {n_hit:>5} {sr:>6.1f}  {med:>9.2f}")

    # Overall oracle
    overall_vals = [t["min_dcc_overall"] for t in target_stats
                    if t.get("min_dcc_overall") is not None]
    n_hit_overall = sum(1 for v in overall_vals if v <= cutoff)
    sr_overall = 100.0 * n_hit_overall / len(overall_vals) if overall_vals else 0.0
    med_overall = float(np.median(overall_vals)) if overall_vals else float("nan")
    print()
    print(f"OVERALL oracle (any source ≤ {cutoff} Å):")
    print(f"  {n_hit_overall}/{len(overall_vals)} = {sr_overall:.1f}% "
          f"(median min_dcc = {med_overall:.2f} Å)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
