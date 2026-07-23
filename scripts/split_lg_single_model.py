#!/usr/bin/env python3
"""Split a multi-MODEL ``{target}_LCDD.lg`` into single-MODEL submission files.

CASP17 LG validator (server-verified 2026-05-06, see ``docs/casp17_lg_format.md``
section 0) accepts only ONE model per file — a multi-MODEL file triggers
"validation script crashed". To submit the 5 alternates you upload 5 separate
single-MODEL files. This tool produces them:

    {target}_LCDD.lg  (5 MODEL blocks)  ->  {target}_1.lg .. {target}_5.lg

Each output keeps the original header (PFRMAT / TARGET / AUTHOR / METHOD) and one
MODEL block verbatim, with the ``MODEL k`` line renumbered to ``MODEL 1`` (each
file is a standalone single-model submission; the ``_k`` suffix carries the
alternate rank). The ``_1`` file is the primary prediction.

Usage:
    uv run python scripts/split_lg_single_model.py <lg_file> [<lg_file> ...] \
        [--out-dir DIR]
    # default out-dir = <lg_file_dir>/split
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def split_lg(text: str) -> tuple[list[str], list[list[str]]]:
    """Return (header_lines, [model_block_lines, ...]).

    header = every line before the first ``MODEL``; each model block runs from
    its ``MODEL k`` line through its terminating ``END`` (inclusive)."""
    lines = text.splitlines()
    header: list[str] = []
    models: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if ln.startswith("MODEL"):
            if cur is not None:
                models.append(cur)
            cur = [ln]
            continue
        if cur is None:
            header.append(ln)
            continue
        cur.append(ln)
        if ln.strip() == "END":
            models.append(cur)
            cur = None
    if cur is not None:  # tolerate a missing final END
        models.append(cur)
    return header, models


def write_single_models(lg_path: Path, out_dir: Path) -> list[Path]:
    target = lg_path.stem.replace("_LCDD", "")
    header, models = split_lg(lg_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for k, block in enumerate(models, 1):
        # keep the original MODEL number: the validator/linter expects file
        # {target}_{k}.lg to carry MODEL k (filename rank == MODEL number).
        body = "\n".join(header + list(block)).rstrip() + "\n"
        out = out_dir / f"{target}_{k}.lg"
        out.write_text(body)
        written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lg", type=Path, nargs="+", help="multi-MODEL *_LCDD.lg file(s)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: <lg_dir>/split)")
    args = ap.parse_args()
    for lg in args.lg:
        if not lg.exists():
            print(f"skip {lg}: not found")
            continue
        out_dir = args.out_dir or (lg.parent / "split")
        files = write_single_models(lg, out_dir)
        print(f"{lg.name} -> {len(files)} files in {out_dir}/")
        for f in files:
            print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
