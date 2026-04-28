#!/usr/bin/env python3
"""Cluster novel2025 targets by protein sequence identity.

Many of the 499 novel2025 inputs are fragment-screen series (XChem)
where the same enzyme is solved with hundreds of different fragments
bound. Running the full pipeline on all of them is wasteful — for any
similarity-based analysis (oracle SR, ranker comparison, scorer
calibration) we want to count each unique enzyme once.

This script reads every ``experiments/novel2025_test/pipeline/<name>_input.yaml``,
takes the first protein chain, builds a FASTA, and runs ``mmseqs
easy-cluster`` at several identity thresholds. Each threshold gets a
report of cluster counts + top clusters, plus a per-target cluster
assignment CSV (``cluster_targets_<thr>.csv``) usable as a join key
for ``experiments/poses_unified/_summary.csv``.

Usage::

    python scripts/cluster_novel2025_targets.py
        --pipeline-dir experiments/novel2025_test/pipeline \
        --output-dir   experiments/novel2025_test \
        --thresholds   1.0 0.95 0.7 0.5 0.3
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MMSEQS = REPO / ".local" / "bin" / "mmseqs"


def extract_sequences(pipeline_dir: Path) -> dict[str, str]:
    """Return ``{target: first_protein_sequence}`` for every input YAML.

    Multi-chain targets keep only the first chain — we're after a
    coarse "is this the same enzyme as that one" classification, and
    the first chain is almost always the catalytic/binder one.
    """
    import yaml

    seqs: dict[str, str] = {}
    skipped = 0
    for yaml_path in sorted(pipeline_dir.glob("*_input.yaml")):
        target = yaml_path.stem  # e.g. "10sl_input"
        try:
            data = yaml.safe_load(yaml_path.read_text()) or {}
        except Exception as e:
            print(f"  skip {target}: yaml load failed ({e})")
            skipped += 1
            continue
        for entry in data.get("sequences", []) or []:
            if isinstance(entry, dict) and "protein" in entry:
                seq = (entry["protein"] or {}).get("sequence")
                if seq:
                    seqs[target] = str(seq).strip()
                    break
    print(f"  extracted {len(seqs)} sequences (skipped {skipped})")
    return seqs


def write_fasta(seqs: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n{seq}\n")


def run_mmseqs_cluster(
    fasta_path: Path, work_dir: Path, threshold: float, coverage: float = 0.9
) -> Path:
    """Run mmseqs easy-cluster and return path to the cluster TSV."""
    cluster_prefix = work_dir / f"cluster_{int(threshold * 100):03d}"
    tmp_dir = work_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    cmd = [
        str(MMSEQS),
        "easy-cluster",
        str(fasta_path),
        str(cluster_prefix),
        str(tmp_dir),
        "--min-seq-id", str(threshold),
        "-c", str(coverage),
        "--cov-mode", "0",
        "--threads", "8",
        "-v", "1",
    ]
    print(f"  → mmseqs easy-cluster --min-seq-id {threshold} -c {coverage}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  mmseqs failed: {res.stderr[-500:]}")
        raise RuntimeError("mmseqs easy-cluster failed")
    tsv = Path(f"{cluster_prefix}_cluster.tsv")
    if not tsv.exists():
        raise RuntimeError(f"expected mmseqs output not found: {tsv}")
    return tsv


def parse_cluster_tsv(tsv_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return ``(target→rep, rep→members)`` from ``rep<TAB>member`` rows."""
    target_to_rep: dict[str, str] = {}
    rep_to_members: dict[str, list[str]] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            target_to_rep[member] = rep
            rep_to_members.setdefault(rep, []).append(member)
    return target_to_rep, rep_to_members


def report_cluster_stats(
    threshold: float, target_to_rep: dict[str, str],
    rep_to_members: dict[str, list[str]], n_total: int,
) -> dict:
    n_clusters = len(rep_to_members)
    sizes = sorted((len(m) for m in rep_to_members.values()), reverse=True)
    n_singletons = sum(1 for s in sizes if s == 1)
    n_multi = n_clusters - n_singletons
    biggest = sizes[:10]
    reduction = 1 - (n_clusters / n_total) if n_total else 0
    print(
        f"\n  threshold={threshold:.2f}: {n_clusters} clusters from {n_total} targets "
        f"(-{n_total - n_clusters}, {reduction:.0%} reduction)"
    )
    print(
        f"    singletons={n_singletons}, multi-member={n_multi}, "
        f"largest={biggest[0] if biggest else 0}"
    )
    print("    top-10 cluster sizes:")
    for rep, members in sorted(
        rep_to_members.items(), key=lambda kv: -len(kv[1])
    )[:10]:
        sample = ", ".join(members[:5]) + (
            f", ... +{len(members) - 5}" if len(members) > 5 else ""
        )
        print(f"      {len(members):>3}  rep={rep:<20}  members: {sample}")
    return {
        "threshold": threshold,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "n_multi": n_multi,
        "biggest": biggest[:5],
    }


def write_cluster_csv(
    threshold: float, target_to_rep: dict[str, str],
    rep_to_members: dict[str, list[str]], output_dir: Path,
) -> Path:
    out = output_dir / f"cluster_targets_{int(threshold * 100):03d}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "cluster_rep", "cluster_size"])
        for target, rep in sorted(target_to_rep.items()):
            w.writerow([target, rep, len(rep_to_members[rep])])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir", type=Path,
        default=REPO / "experiments" / "novel2025_test" / "pipeline",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "experiments" / "novel2025_test",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[1.0, 0.95, 0.7, 0.5, 0.3],
        help="mmseqs --min-seq-id values to try.",
    )
    parser.add_argument("--coverage", type=float, default=0.9)
    args = parser.parse_args()

    work_dir = args.output_dir / "_cluster_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cluster] pipeline_dir={args.pipeline_dir}")
    print(f"[cluster] output_dir={args.output_dir}")
    print(f"[cluster] thresholds={args.thresholds}")
    print(f"[cluster] coverage={args.coverage}")

    seqs = extract_sequences(args.pipeline_dir)
    n_total = len(seqs)
    if n_total == 0:
        print("no sequences extracted; nothing to cluster")
        return 1

    fasta = work_dir / "all_proteins.fasta"
    write_fasta(seqs, fasta)

    summary_rows: list[dict] = []
    for thr in args.thresholds:
        try:
            tsv = run_mmseqs_cluster(fasta, work_dir, thr, args.coverage)
        except Exception as e:
            print(f"  threshold {thr}: FAILED — {e}")
            continue
        t2r, r2m = parse_cluster_tsv(tsv)
        stats = report_cluster_stats(thr, t2r, r2m, n_total)
        summary_rows.append(stats)
        write_cluster_csv(thr, t2r, r2m, args.output_dir)
        print(f"    wrote {args.output_dir}/cluster_targets_{int(thr * 100):03d}.csv")

    print("\n  summary across thresholds:")
    print(f"  {'threshold':>10}  {'n_clusters':>10}  {'singletons':>10}  {'multi':>6}  {'biggest':>10}")
    for s in summary_rows:
        biggest_str = ", ".join(str(x) for x in s["biggest"])
        print(
            f"  {s['threshold']:>10.2f}  {s['n_clusters']:>10}  "
            f"{s['n_singletons']:>10}  {s['n_multi']:>6}  {biggest_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
