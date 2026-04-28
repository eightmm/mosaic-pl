#!/usr/bin/env python3
"""Cluster pocket centers from extract_template_pockets and emit top-K consensus.

After ``extract_template_pockets.py`` produces ``template_pockets.json`` (a
flat list of bound-ligand centroids in the cofolding frame), this script
runs single-link agglomerative clustering by Euclidean distance and emits
the top-K cluster centroids with evidence scores. Each centroid becomes a
``template_consensus_N`` source in ``docking_prep_summary.binding_site_predictions``
downstream.

Pocket-point evidence weight:

    w = (in_mmseqs + in_foldseek) + max(alignment_tmscore, pident/100)

Range ≈ [0, 3]. Both-source hits + high structural/sequence similarity get
the highest weight. ``alignment_tmscore`` is USalign's actual TM-score
from the pocket-extraction step (the real quantity, not foldseek's
estimate); using it makes mmseqs-only hits — which have ``qtmscore=0``
in the filter row — also score correctly here.

Usage::

    python cluster_template_pockets.py \
        --pockets-json experiments/runs/<target>/outputs/template_pockets/template_pockets.json \
        --cutoff 5.0 \
        --top-k 5

Output (next to input by default)::

    template_pocket_clusters.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _pocket_weight(p: dict) -> float:
    sources = int(bool(p.get("in_mmseqs"))) + int(bool(p.get("in_foldseek")))
    # USalign's actual TM-score for the template→cofold superposition.
    # Falls back to foldseek's qtmscore field if alignment_tmscore is
    # absent (older JSON written before the USalign switch), and to
    # pident/100 as a last-resort similarity proxy.
    actual_tm = float(p.get("alignment_tmscore", 0.0) or 0.0)
    fold_tm = float(p.get("qtmscore", 0.0) or 0.0)
    seq_sim = float(p.get("pident", 0.0) or 0.0) / 100.0
    sim = max(actual_tm, fold_tm, seq_sim)
    return float(sources + sim)


def _euclid(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cluster_pockets(pockets: list[dict], cutoff: float) -> list[dict]:
    """Greedy agglomerative clustering: each new pocket joins the *first*
    existing cluster whose centroid is within ``cutoff`` Å, otherwise spawns
    a new cluster. The order of input matters — the extract step already
    sorts by evidence (both-source > one-source > qtm/pident), so seed
    clusters favor strong evidence.
    """
    if not pockets:
        return []

    clusters: list[dict] = []
    for p in pockets:
        center = (p["centroid_x"], p["centroid_y"], p["centroid_z"])
        weight = _pocket_weight(p)
        joined = False
        for cl in clusters:
            if _euclid(center, cl["_centroid"]) <= cutoff:
                cl["_members"].append(p)
                cl["_weights"].append(weight)
                # Re-compute weighted centroid incrementally
                total_w = sum(cl["_weights"])
                if total_w == 0:
                    total_w = len(cl["_weights"])  # fallback to unweighted
                    cl["_centroid"] = tuple(
                        sum(m[k] for m in cl["_members"]) / len(cl["_members"])
                        for k in ("centroid_x", "centroid_y", "centroid_z")
                    )
                else:
                    cl["_centroid"] = tuple(
                        sum(m[k] * w for m, w in zip(cl["_members"], cl["_weights"])) / total_w
                        for k in ("centroid_x", "centroid_y", "centroid_z")
                    )
                joined = True
                break
        if not joined:
            clusters.append({
                "_centroid": center,
                "_members": [p],
                "_weights": [weight],
            })
    return clusters


def _summarize(clusters: list[dict]) -> list[dict]:
    out: list[dict] = []
    for cl in clusters:
        members = cl["_members"]
        weights = cl["_weights"]
        n = len(members)
        evidence = float(sum(weights))
        in_both = sum(1 for m in members if m.get("in_mmseqs") and m.get("in_foldseek"))
        in_seq_only = sum(1 for m in members if m.get("in_mmseqs") and not m.get("in_foldseek"))
        in_struct_only = sum(1 for m in members if m.get("in_foldseek") and not m.get("in_mmseqs"))
        unique_pdb = len({m.get("template_pdb_id") for m in members})
        unique_ccd = sorted({m.get("ligand_ccd") for m in members if m.get("ligand_ccd")})
        # Spread = max distance from centroid to any member
        c = cl["_centroid"]
        spread = 0.0
        for m in members:
            d = _euclid(c, (m["centroid_x"], m["centroid_y"], m["centroid_z"]))
            if d > spread:
                spread = d
        # Best per-member metrics — useful for downstream filtering / debugging
        best_tm = max((float(m.get("alignment_tmscore", 0.0) or 0.0) for m in members), default=0.0)
        best_qtm = max((float(m.get("qtmscore", 0.0) or 0.0) for m in members), default=0.0)
        best_pident = max((float(m.get("pident", 0.0) or 0.0) for m in members), default=0.0)
        best_tanimoto = max((float(m.get("best_tanimoto", 0.0) or 0.0) for m in members), default=0.0)
        out.append({
            "centroid": [round(c[0], 3), round(c[1], 3), round(c[2], 3)],
            "n_members": n,
            "n_unique_pdb": unique_pdb,
            "evidence_score": round(evidence, 3),
            "spread_angstrom": round(spread, 3),
            "in_both_sources": in_both,
            "in_mmseqs_only": in_seq_only,
            "in_foldseek_only": in_struct_only,
            "unique_ccds": unique_ccd,
            "best_alignment_tmscore": round(best_tm, 3),
            "best_qtmscore": round(best_qtm, 3),
            "best_pident": round(best_pident, 1),
            "best_tanimoto": round(best_tanimoto, 3),
            "members": [
                {
                    "template_pdb_id": m.get("template_pdb_id"),
                    "ligand_ccd": m.get("ligand_ccd"),
                    "ligand_chain": m.get("ligand_chain"),
                    "alignment_rmsd": m.get("alignment_rmsd"),
                    "in_mmseqs": m.get("in_mmseqs"),
                    "in_foldseek": m.get("in_foldseek"),
                }
                for m in members
            ],
        })
    out.sort(key=lambda c: -c["evidence_score"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pockets-json", type=Path, required=True)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom centroid distance (Å) below which two pocket points "
             "join the same cluster. Default 5.0 — typical druglike binding "
             "pockets are 8-15Å diameter, so well-aligned templates land "
             "within ~5Å.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Emit at most K clusters (sorted by evidence_score desc).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.pockets_json.exists():
        print(f"[cluster] no input at {args.pockets_json}; nothing to cluster.")
        return 0

    data = json.loads(args.pockets_json.read_text())
    pockets = data.get("pockets") or []
    if not pockets:
        print("[cluster] empty pocket list; emitting empty cluster set.")
        clusters = []
    else:
        raw_clusters = _cluster_pockets(pockets, cutoff=args.cutoff)
        clusters = _summarize(raw_clusters)

    output = args.output_json or args.pockets_json.with_name("template_pocket_clusters.json")
    summary = {
        "input": str(args.pockets_json),
        "n_pockets": len(pockets),
        "cutoff_angstrom": args.cutoff,
        "top_k": args.top_k,
        "n_clusters": len(clusters),
        "clusters": clusters[: args.top_k],
        "all_clusters": clusters,
    }
    output.write_text(json.dumps(summary, indent=2))
    print(
        f"[cluster] {len(pockets)} pockets → {len(clusters)} clusters "
        f"(cutoff={args.cutoff}Å, top-{args.top_k} kept) → {output}"
    )
    for i, cl in enumerate(clusters[: args.top_k], 1):
        c = cl["centroid"]
        print(
            f"  #{i} centroid=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
            f"n={cl['n_members']} unique_pdb={cl['n_unique_pdb']} "
            f"evidence={cl['evidence_score']:.2f} spread={cl['spread_angstrom']:.1f}Å "
            f"both/seq/struct={cl['in_both_sources']}/{cl['in_mmseqs_only']}/{cl['in_foldseek_only']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
