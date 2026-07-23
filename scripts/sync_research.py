#!/usr/bin/env python3
"""Vendor per-target research briefings from the CASP17_own run tree.

The research itself (Google DeepMind science-skills driven via ``claude -p``) is
generated in the sibling ``CASP17_own`` repo. This pipeline only *consumes* the
structured ``research.json`` — so we copy it into this repo's run tree, keeping
CASP17 self-contained and its submissions reproducible (no live cross-repo path
dependency at scoring time).

For each ``<target>``: copy
``<own>/runs/<target>/research/research.json`` → ``<run-dir>/research.json``.
Missing sources are skipped (fail-open); nothing else is touched.

Usage:
    uv run python scripts/sync_research.py --targets T2409 T2410 ... \
        --own ../CASP17_own --casp-root experiments/CASP17
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _run_dir(casp_root: Path, target: str) -> Path | None:
    """CASP17 per-target run dir: experiments/CASP17/<t>/run/<t>."""
    cand = casp_root / target / "run" / target
    return cand if cand.is_dir() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--own", type=Path, default=Path("../CASP17_own"))
    ap.add_argument("--casp-root", type=Path, default=Path("experiments/CASP17"))
    args = ap.parse_args()

    n_ok = 0
    for t in args.targets:
        src = args.own / "runs" / t / "research" / "research.json"
        if not src.is_file():
            print(f"  {t}: no research.json at {src} — skip")
            continue
        run_dir = _run_dir(args.casp_root, t)
        if run_dir is None:
            print(f"  {t}: no run dir under {args.casp_root}/{t}/run/{t} — skip")
            continue
        dst = run_dir / "research.json"
        shutil.copyfile(src, dst)
        print(f"  {t}: {src} -> {dst}")
        n_ok += 1
    print(f"synced {n_ok}/{len(args.targets)} research briefings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
