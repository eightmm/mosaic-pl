"""LG (CASP ligand-submission) format parser.

The CASP17 LG format is a PDB-ish multi-MODEL text file with an embedded
MDL (SDF) block inside each MODEL for the bound ligand plus a few header
records (``PFRMAT``, ``TARGET``, ``AUTHOR``, ``METHOD``, ``AFFNTY``, and
per-MODEL ``LSCORE``/``LIGAND``/``REMARK``).

A single ``parse_lg`` was duplicated in two evaluation scripts
(``experiments/casp16_test/L1000/evaluate.py`` and
``experiments/novel2025_test/evaluate.py``). This module consolidates it
so both scripts, and any future LG consumers, share one implementation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_lg(path: Path) -> dict[str, Any]:
    """Parse a CASP LG file into a structured dict.

    Returns ``{"affnty": float | None, "models": [...]}`` where each
    model entry is::

        {
          "idx":        int,     # MODEL number (1-indexed)
          "name":       str|None, # content of the LIGAND line (e.g. "vina_seed_101")
          "lscore":     float|None,
          "atom_lines": list[str],  # raw ATOM/HETATM/TER records
          "mdl_text":   str,     # MDL block body (starts at "     RDKit ..." header)
        }

    The complex-level ``AFFNTY`` record is captured regardless of whether
    it appears before MODEL 1 (legacy) or right before the final ``END``
    (current ``make_casp_submission.py`` output).
    """
    text = path.read_text().splitlines()
    out: dict[str, Any] = {"affnty": None, "models": []}
    cur: dict[str, Any] | None = None
    mode = "header"
    mdl_buf: list[str] = []

    for line in text:
        # Complex-level AFFNTY (can appear pre-MODEL or before END)
        if line.startswith("AFFNTY"):
            try:
                out["affnty"] = float(line.split()[1])
            except (ValueError, IndexError):
                pass
            if cur is None:
                continue
            # Inside a MODEL block we still want to fall through to capture
            # the per-MODEL AFFNTY separately below.

        m = re.match(r"^MODEL\s+(\d+)", line)
        if m:
            if cur is not None:
                cur["mdl_text"] = "\n".join(mdl_buf)
                out["models"].append(cur)
            cur = {
                "idx": int(m.group(1)),
                "atom_lines": [],
                "lscore": None,
                "name": None,
                "mdl_text": "",
            }
            mode = "atoms"
            mdl_buf = []
            continue
        if cur is None:
            continue

        if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
            cur["atom_lines"].append(line)
            continue
        if line.startswith("LIGAND"):
            mode = "lig_meta"
            continue
        if line.startswith("LSCORE"):
            try:
                cur["lscore"] = float(line.split()[1])
            except (ValueError, IndexError):
                pass
            continue
        if line.startswith("AFFNTY"):
            try:
                cur["affnty_model"] = float(line.split()[1])
            except (ValueError, IndexError):
                pass
            continue

        if mode == "lig_meta" and cur["name"] is None and line.strip():
            cur["name"] = line.strip()
            mode = "mdl"
            mdl_buf = []
            continue
        if mode == "mdl":
            mdl_buf.append(line)

    if cur is not None:
        cur["mdl_text"] = "\n".join(mdl_buf)
        out["models"].append(cur)
    return out
