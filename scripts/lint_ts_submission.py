#!/usr/bin/env python3
"""Validate a CASP17 ``PFRMAT TS`` submission before it is uploaded.

Checks the format rules that the prediction center's verification server
enforces, plus the extra ones an ordered-solvent target adds (R2386):

* header block — ``PFRMAT TS`` / ``TARGET`` / ``AUTHOR`` / ``METHOD``, once,
  above the first ``MODEL``
* at most 5 models, indices ``1..n``, each opened by ``MODEL`` and closed by
  ``END`` (CASP closes a TS model with ``END``, not ``ENDMDL``)
* no target residue repetition inside a model
* fixed-column PDB records that actually parse
* ligands as ``HETATM``, restricted to the requested set, every one carrying a
  B-factor, and — the rule the target page spells out — total occupancy equal
  to the number of ligands

Exit code is 1 when any ERROR is reported; WARNs do not fail the run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

#: Ligands R2386 asks for. Override with --allowed-het for another target.
DEFAULT_ALLOWED_HET = ("MG", "K", "NA", "HOH")

MAX_MODELS = 5


def _parse_atom(line: str) -> dict | None:
    """Split a fixed-column ATOM/HETATM record. ``None`` when it does not parse."""
    if len(line) < 54:
        return None
    try:
        return {
            "record": line[0:6].strip(),
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "altloc": line[16:17],
            "resname": line[17:20].strip(),
            "chain": line[21:22],
            "resnum": int(line[22:26]),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "occ": line[54:60].strip(),
            "bfac": line[60:66].strip(),
            "element": line[76:78].strip() if len(line) >= 78 else "",
        }
    except ValueError:
        return None


def lint(path: Path, *, allowed_het: tuple[str, ...] = DEFAULT_ALLOWED_HET) -> tuple[int, int]:
    errors: list[str] = []
    warns: list[str] = []
    lines = path.read_text().splitlines()

    # ---- header ----------------------------------------------------------
    first_model = next((i for i, ln in enumerate(lines) if ln.startswith("MODEL")), None)
    if first_model is None:
        errors.append("no MODEL record")
        first_model = len(lines)
    header = lines[:first_model]
    for key in ("PFRMAT", "TARGET", "AUTHOR", "METHOD"):
        n = sum(1 for ln in header if ln.startswith(key + " "))
        if n == 0:
            errors.append(f"missing {key} record in the header")
        elif n > 1 and key != "METHOD":
            errors.append(f"{key} appears {n} times; exactly one is allowed")
    if header and not header[0].startswith("PFRMAT TS"):
        errors.append(f"file must start with 'PFRMAT TS'; got {header[0]!r}")
    for i, line in enumerate(lines[first_model:], start=first_model):
        if line.startswith(("PFRMAT ", "TARGET ", "AUTHOR ")):
            errors.append(f"line {i + 1}: header record {line.split()[0]} below the "
                          "first MODEL")

    # ---- models ----------------------------------------------------------
    models: list[dict] = []
    cur: dict | None = None
    for i, line in enumerate(lines, start=1):
        if line.startswith("MODEL"):
            if cur is not None:
                errors.append(f"line {i}: MODEL opened while model "
                              f"{cur['index']} was still open (missing END)")
            parts = line.split()
            try:
                index = int(parts[1])
            except (IndexError, ValueError):
                errors.append(f"line {i}: cannot read a model index from {line!r}")
                index = -1
            cur = {"index": index, "line": i, "atoms": [], "het": []}
        elif line.startswith("ENDMDL"):
            errors.append(f"line {i}: ENDMDL — CASP TS closes a model with END")
        elif line.startswith("END"):
            if cur is None:
                errors.append(f"line {i}: END without an open MODEL")
            else:
                models.append(cur)
                cur = None
        elif line.startswith(("ATOM", "HETATM")):
            if cur is None:
                errors.append(f"line {i}: coordinate record outside any MODEL")
                continue
            rec = _parse_atom(line)
            if rec is None:
                errors.append(f"line {i}: unparseable coordinate record")
                continue
            rec["line"] = i
            (cur["het"] if rec["record"] == "HETATM" else cur["atoms"]).append(rec)
    if cur is not None:
        errors.append(f"model {cur['index']} (line {cur['line']}) is never closed by END")

    if not models:
        errors.append("no complete MODEL...END block")
    if len(models) > MAX_MODELS:
        errors.append(f"{len(models)} models; TS allows at most {MAX_MODELS}")
    for pos, model in enumerate(models, start=1):
        if model["index"] != pos:
            errors.append(f"model at line {model['line']} is numbered "
                          f"{model['index']}; TS models must be 1..n in order")

    # ---- per-model contents ---------------------------------------------
    for model in models:
        tag = f"MODEL {model['index']}"
        if not model["atoms"]:
            errors.append(f"{tag}: no ATOM records")

        seen: Counter = Counter()
        for rec in model["atoms"]:
            seen[(rec["chain"], rec["resnum"])] += 1
        resnums = [k for k in seen]
        dup = [k for k in resnums if resnums.count(k) > 1]
        if dup:
            errors.append(f"{tag}: repeated residues {sorted(dup)[:5]}")

        # A residue that appears in two separate runs is a repetition even
        # though its atoms are individually unique.
        order: list[tuple[str, int]] = []
        for rec in model["atoms"]:
            key = (rec["chain"], rec["resnum"])
            if not order or order[-1] != key:
                order.append(key)
        counts = Counter(order)
        split = [k for k, n in counts.items() if n > 1]
        if split:
            errors.append(f"{tag}: residues appear in more than one block "
                          f"{sorted(split)[:5]} — TS forbids residue repetition")

        chains = {rec["chain"] for rec in model["atoms"]}
        if len(chains) > 1:
            warns.append(f"{tag}: {len(chains)} chains {sorted(chains)}")

        bad_b = [r["line"] for r in model["atoms"] if not r["bfac"]]
        if bad_b:
            errors.append(f"{tag}: {len(bad_b)} ATOM records without a B-factor "
                          f"(first at line {bad_b[0]})")

        # ---- ligands -----------------------------------------------------
        het_names = Counter(r["resname"] for r in model["het"])
        unexpected = {n: c for n, c in het_names.items() if n not in allowed_het}
        if unexpected:
            errors.append(f"{tag}: HETATM residues outside the requested set: "
                          f"{unexpected}")
        missing_b = [r["line"] for r in model["het"] if not r["bfac"]]
        if missing_b:
            errors.append(f"{tag}: {len(missing_b)} ligand atoms without a B-factor "
                          f"(first at line {missing_b[0]}) — the target page "
                          "requires one for every ligand")
        no_element = [r["line"] for r in model["het"] if not r["element"]]
        if no_element:
            errors.append(f"{tag}: {len(no_element)} ligand atoms without an element "
                          f"symbol (first at line {no_element[0]})")

        # "total occupancy must be equal to the number of ligands" — with no
        # AltLoc every ligand contributes exactly 1.00.
        total_occ = 0.0
        bad_occ = []
        for rec in model["het"]:
            try:
                total_occ += float(rec["occ"])
            except ValueError:
                bad_occ.append(rec["line"])
        if bad_occ:
            errors.append(f"{tag}: {len(bad_occ)} ligand atoms with an unreadable "
                          f"occupancy (first at line {bad_occ[0]})")
        altloc = {r["altloc"] for r in model["het"]} - {" ", ""}
        n_lig = len(model["het"])
        if not bad_occ and abs(total_occ - n_lig) > 0.01 * max(1, n_lig):
            errors.append(f"{tag}: ligand occupancies sum to {total_occ:.2f} but "
                          f"there are {n_lig} ligands — the target page requires "
                          "them to be equal")

        print(f"  {tag}: {len(model['atoms'])} ATOM, {len(model['het'])} HETATM "
              f"{dict(het_names)}"
              + (f", altloc {sorted(altloc)}" if altloc else "")
              + f", occ sum {total_occ:.2f}")

    for e in errors:
        print(f"  ERROR: {e}")
    for w in warns:
        print(f"  WARN:  {w}")
    return len(errors), len(warns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--allowed-het", nargs="*", default=list(DEFAULT_ALLOWED_HET),
                        help="permitted HETATM residue names "
                             f"(default: {' '.join(DEFAULT_ALLOWED_HET)})")
    args = parser.parse_args()

    total_e = total_w = 0
    for path in args.files:
        print(f"=== {path} ===")
        e, w = lint(path, allowed_het=tuple(n.upper() for n in args.allowed_het))
        if not e and not w:
            print("  (no issues)")
        total_e += e
        total_w += w
        print()
    print(f"Summary: {total_e} ERROR(s), {total_w} WARN(s) across {len(args.files)} file(s).")
    return 1 if total_e else 0


if __name__ == "__main__":
    raise SystemExit(main())
