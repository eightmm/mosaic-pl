#!/usr/bin/env python3
"""Cluster pocket centers from extract_template_pockets and emit top-K consensus.

After ``extract_template_pockets.py`` produces ``template_pockets.json`` (a
flat list of bound-ligand centroids in the cofolding frame), this script
runs hierarchical single-linkage clustering and emits the top-K
evidence-weighted cluster centroids with evidence scores. Each centroid becomes a
``template_consensus_N`` source in ``docking_prep_summary.binding_site_predictions``
downstream.

Pocket-point evidence weight:

    w = (in_mmseqs + in_foldseek) + max(alignment_tmscore, qtmscore, pident/100)
        + max(best_tanimoto, best_mcs_coverage)

Range ≈ [0, 4]. Search support, protein similarity, and ligand similarity give
the highest weight. ``alignment_tmscore`` is USalign's actual TM-score
from the pocket-extraction step (the real quantity, not foldseek's
estimate); using it makes mmseqs-only hits — which have ``qtmscore=0``
in the filter row — also score correctly here.

Usage::

    python cluster_template_pockets.py \
        --pockets-json experiments/runs/<target>/outputs/template_pockets/template_pockets.json \
        --cutoff 5.0 \
        --top-k 10

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
    struct_sim = max(actual_tm, fold_tm, seq_sim)

    # Ligand similarity to the query (target SMILES) — boosts clusters where
    # homologous proteins were crystallised with chemically similar ligands.
    # That's a strong "same-binding-site" signal: even when two folds share
    # the global TM-score they may have diverged into different substrates,
    # and the cluster whose ligands match the query is the one most likely
    # to mark the actual site we want to dock against.
    # ``best_tanimoto`` (Morgan FP, [0,1]) captures global chemotype overlap;
    # ``best_mcs_coverage`` (MCS atoms / min(target, template) heavy atoms)
    # captures shared scaffold size. We use ``max`` of the two so a small
    # fragment with high MCS isn't penalised by a low fingerprint Tanimoto
    # (and a large flexible compound with high Tanimoto isn't penalised by
    # a tiny MCS).
    tanimoto = float(p.get("best_tanimoto", 0.0) or 0.0)
    mcs_cov = float(p.get("best_mcs_coverage", 0.0) or 0.0)
    lig_sim = max(tanimoto, mcs_cov)

    # Additive — keeps each signal's contribution independently inspectable
    # in evidence_score. Range ≈ [0, 4]: sources 0-2, struct_sim 0-1, lig 0-1.
    return float(sources + struct_sim + lig_sim)


def _euclid(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cluster_pockets(pockets: list[dict], cutoff: float) -> list[dict]:
    """Hierarchical agglomerative single-link clustering.

    Greedy first-match (the previous algorithm) drifts the running centroid
    as members get added, so two pockets seeded into separate clusters can
    end up with centroids closer than ``cutoff`` (validated 2026-05-02 on
    7hqq: clusters #1 ↔ #4 ended up 2.95 Å apart with cutoff=5 Å — same
    binding site, two redundant cluster slots).

    Single-link agglomerative builds the full distance matrix once and
    merges any two clusters whose closest member-pair is below ``cutoff``,
    so by construction the final clusters' member-pair distances are all
    strictly above ``cutoff`` between any two distinct clusters. Final
    centroid is the evidence-weighted mean of the cluster members.
    """
    if not pockets:
        return []
    if len(pockets) == 1:
        p = pockets[0]
        return [{
            "_centroid": (p["centroid_x"], p["centroid_y"], p["centroid_z"]),
            "_members": [p],
            "_weights": [_pocket_weight(p)],
        }]

    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    coords = np.array([(p["centroid_x"], p["centroid_y"], p["centroid_z"]) for p in pockets])
    Z = linkage(pdist(coords), method="single")
    labels = fcluster(Z, t=cutoff, criterion="distance")

    label_to_cluster: dict[int, dict] = {}
    for label, p in zip(labels, pockets):
        cl = label_to_cluster.setdefault(int(label), {"_members": [], "_weights": []})
        cl["_members"].append(p)
        cl["_weights"].append(_pocket_weight(p))

    out: list[dict] = []
    for cl in label_to_cluster.values():
        members = cl["_members"]; weights = cl["_weights"]
        total_w = sum(weights)
        if total_w == 0:
            cx = sum(m["centroid_x"] for m in members) / len(members)
            cy = sum(m["centroid_y"] for m in members) / len(members)
            cz = sum(m["centroid_z"] for m in members) / len(members)
        else:
            cx = sum(m["centroid_x"] * w for m, w in zip(members, weights)) / total_w
            cy = sum(m["centroid_y"] * w for m, w in zip(members, weights)) / total_w
            cz = sum(m["centroid_z"] * w for m, w in zip(members, weights)) / total_w
        cl["_centroid"] = (cx, cy, cz)
        out.append(cl)
    return out


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
        best_mcs = max((float(m.get("best_mcs_coverage", 0.0) or 0.0) for m in members), default=0.0)
        # Cluster-level ligand similarity ≡ max(member tanimoto, member MCS).
        # Folded into ``evidence_score`` already (via _pocket_weight) but
        # exposed here so downstream consumers (docking-prep box selection,
        # ranker training data) can filter on the lig-sim signal directly.
        best_lig_sim = max(best_tanimoto, best_mcs)
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
            "best_mcs_coverage": round(best_mcs, 3),
            "best_ligand_similarity": round(best_lig_sim, 3),
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
    # Pair summaries with their raw cluster index *before* sorting so
    # callers (cluster_member_to_index in main) can map raw_clusters[ci]
    # → final-rank index after evidence_score sort.
    indexed = list(enumerate(out))
    indexed.sort(key=lambda pair: -pair[1]["evidence_score"])
    sorted_out = [item for _, item in indexed]
    # Stash raw→sorted index mapping on the first cluster's metadata so
    # main() can splice cluster_index without recomputing the sort.
    if sorted_out:
        raw_to_sorted = {raw_idx: new_idx for new_idx, (raw_idx, _) in enumerate(indexed)}
        sorted_out[0]["_raw_to_sorted"] = raw_to_sorted
    return sorted_out


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
        default=10,
        help="Emit at most K clusters (sorted by evidence_score desc). "
             "Default 10 matches the docking-prep consensus slot count "
             "(prepare_docking_inputs registers up to 10 sources, "
             "filtering singletons via n_members floor).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--surface-margin",
        type=float,
        default=10.0,
        help="Drop pocket centroids whose nearest-protein-heavy-atom "
             "distance exceeds this (Å). Shape-following filter — uses "
             "per-pocket KDTree query against the cofold receptor heavy "
             "atoms, so it tracks the protein surface (no axis-aligned "
             "bbox dead corners on elongated folds). Catches multi-chain "
             "template artifacts where USalign aligns one protomer but "
             "the ligand belongs to another (centroid lands 50-200 Å "
             "from any atom). 0 disables. 10 Å covers buried + surface "
             "+ shallow-cleft pockets while excluding clearly-outlier "
             "cases (validated on 7hqq: real binding pocket centroids "
             "land 1.9-2.8 Å from the nearest atom; interface-style "
             "candidates ~11 Å; chain-mismatch artifacts 20+ Å).",
    )
    args = parser.parse_args()

    if not args.pockets_json.exists():
        print(f"[cluster] no input at {args.pockets_json}; nothing to cluster.")
        return 0

    data = json.loads(args.pockets_json.read_text())
    pockets = data.get("pockets") or []

    if pockets and args.surface_margin > 0:
        ref_cif = data.get("reference_cif")
        if ref_cif and Path(ref_cif).exists():
            try:
                import gemmi
                import numpy as np
                from scipy.spatial import cKDTree
                s = gemmi.read_structure(ref_cif)
                pts = []
                for model in s:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                if atom.element.atomic_number <= 1: continue
                                pts.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    break
                if pts:
                    tree = cKDTree(np.asarray(pts))
                    inside = []
                    max_dist = 0.0
                    for p in pockets:
                        cx, cy, cz = p.get("centroid_x"), p.get("centroid_y"), p.get("centroid_z")
                        if cx is None: continue
                        d, _ = tree.query([cx, cy, cz], k=1)
                        if d > max_dist: max_dist = d
                        if d <= args.surface_margin:
                            inside.append(p)
                    n_dropped = len(pockets) - len(inside)
                    if n_dropped > 0:
                        print(f"[cluster] surface filter (margin={args.surface_margin}Å): "
                              f"dropped {n_dropped}/{len(pockets)} pockets too far from any "
                              f"protein heavy atom (worst pocket was {max_dist:.1f} Å away)")
                        pockets = inside
                        data["pockets"] = pockets
            except Exception as e:
                print(f"[cluster] WARNING: surface filter skipped ({e})")
    raw_clusters: list[dict] = []
    if not pockets:
        print("[cluster] empty pocket list; emitting empty cluster set.")
        clusters = []
    else:
        raw_clusters = _cluster_pockets(pockets, cutoff=args.cutoff)
        clusters = _summarize(raw_clusters)

    # Tag each pocket record with its cluster index — but the index has
    # to match the *sorted* cluster list because that's what
    # ``select_cluster_representative_templates`` indexes into. The raw
    # build order (input pocket order) is meaningless to the consumer.
    raw_to_sorted: dict[int, int] = {}
    if clusters and "_raw_to_sorted" in clusters[0]:
        raw_to_sorted = clusters[0].pop("_raw_to_sorted")
    cluster_member_to_index: dict[int, int] = {}
    for raw_ci, cl in enumerate(raw_clusters):
        sorted_ci = raw_to_sorted.get(raw_ci, raw_ci)
        for member in cl.get("_members", []):
            cluster_member_to_index[id(member)] = sorted_ci
    for p in pockets:
        idx = cluster_member_to_index.get(id(p))
        if idx is not None:
            p["cluster_index"] = idx

    # Rewrite the pockets json with cluster_index annotations so future
    # readers don't have to re-derive the mapping.
    if pockets:
        try:
            data["pockets"] = pockets
            args.pockets_json.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[cluster] WARNING: failed to annotate pockets json: {e}")

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
