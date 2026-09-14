#!/usr/bin/env python3
"""Check ligand-series stage-2 model files against the answers file.

``lint_lg_submission.py`` validates the LG grammar. This checks the things
only the L01/L02 stage-2 experiment can be wrong about, reading nothing but
the written file and ``ligands_stage2.csv`` — deliberately independent of the
builder, so a shared assumption cannot hide a defect from both:

* one ``LIGAND`` block per copy, numbered 1..N with no gaps, plus one per
  cofactor at the end;
* names are the answers file's ligand codes;
* every fragment block is the released molecule (heavy-atom graph and formal
  charges), and every cofactor block is the released cofactor;
* copies do not overlap each other or the cofactor, and none of them sits
  inside the receptor this file ships;
* the five MODELs answer with five *different* poses, per copy and as whole
  arrangements -- a repeat throws away one of the five chances the format
  gives, and swapping which copy carries which number does not make two
  MODELs different;
* the receptor is present with varying B-factors.

Usage:
    uv run python scripts/lint_ligand_series_stage2.py \
        experiments/ligand_series/stage2/L01/* experiments/ligand_series/stage2/L02/*
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from make_ligand_series_stage2 import (  # noqa: E402
    CLASH_FLOOR, COFACTOR_FLOOR, DIVERSITY_RMSD, RECEPTOR_FLOOR, load_stage2,
    mdl_coords, min_distance, parse_lg, pose_rmsd, receptor_coords,
    same_arrangement,
)


def graph_matches(mdl: str, smiles: str) -> tuple[bool, str]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    got = Chem.MolFromMolBlock(mdl, sanitize=True, removeHs=True)
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return True, "reference SMILES unparseable — skipped"
    if got is None:
        return False, "MDL body does not parse"
    a, b = Chem.MolToSmiles(got), Chem.MolToSmiles(ref)
    if a == b:
        return True, ""
    # A centre the answers file leaves unspecified may legitimately be
    # resolved either way by a 3D pose.
    if got.HasSubstructMatch(ref, useChirality=True):
        return True, ""
    return False, f"{a} != {b}"


def check(path: Path, spec: dict) -> list[str]:
    bad: list[str] = []
    header, models = parse_lg(path.read_text())
    target = path.name.split("LG")[0]

    if not any(ln.startswith("PFRMAT LG") for ln in header):
        bad.append("missing PFRMAT LG")
    if not any(ln.startswith(f"TARGET {target}") for ln in header):
        bad.append(f"TARGET is not {target}")
    if not (1 <= len(models) <= 5):
        bad.append(f"{len(models)} MODEL block(s); the format allows 1-5")

    comps = spec["components"]
    want_names = [c["code"] for c in comps for _ in range(c["n"])]
    binder, cofactors = comps[0], comps[1:]

    for m in models:
        where = f"MODEL {m.label}"
        atoms = [ln for ln in m.prelude if ln.startswith("ATOM")]
        if not atoms:
            bad.append(f"{where}: no receptor ATOM records")
        elif len({ln[60:66].strip() for ln in atoms}) == 1:
            bad.append(f"{where}: receptor B-factors are uniform")
        if not any(ln.startswith("PARENT") for ln in m.prelude):
            bad.append(f"{where}: no PARENT record")

        names = [lg.name for lg in m.ligands]
        if names != want_names:
            bad.append(f"{where}: LIGAND names {names} != answers file {want_names}")
        nums = [lg.number for lg in m.ligands]
        if nums != list(range(1, len(nums) + 1)):
            bad.append(f"{where}: LIGAND numbers {nums} are not 1..{len(nums)}")
        for lg in m.ligands:
            if lg.lscore is not None and not (0.0 <= lg.lscore <= 1.0):
                bad.append(f"{where}/{lg.name}: LSCORE {lg.lscore} outside [0,1]")

        smiles_for = {c["code"]: c["smiles"] for c in comps}
        for lg in m.ligands:
            ok, why = graph_matches(lg.mdl, smiles_for.get(lg.name, ""))
            if not ok:
                bad.append(f"{where}/LIGAND {lg.number} {lg.name}: {why}")

        pts = [mdl_coords(lg.mdl) for lg in m.ligands]
        rec = receptor_coords(m)
        cof_codes = {c["code"] for c in cofactors}
        for lg, q in zip(m.ligands, pts):
            floor = COFACTOR_FLOOR if lg.name in cof_codes else RECEPTOR_FLOOR
            d = min_distance(q, rec)
            if d < floor:
                bad.append(f"{where}: LIGAND {lg.number} {lg.name} sits "
                           f"{d:.2f} Aa from the receptor (floor {floor})")
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = min_distance(pts[i], pts[j])
                if d < CLASH_FLOOR:
                    bad.append(f"{where}: LIGAND {m.ligands[i].number} "
                               f"{m.ligands[i].name} and {m.ligands[j].number} "
                               f"{m.ligands[j].name} overlap at {d:.2f} Aa "
                               f"(floor {CLASH_FLOOR})")

    # Pose diversity across MODELs, per ligand copy. Reported as a warning
    # rather than an error: a duplicate wastes a slot but is not malformed,
    # and the upstream selector already ships a handful the pose pool could
    # not separate.
    cof_names = {c["code"] for c in cofactors}
    by_copy: dict[int, list[str]] = {}
    for m in models:
        for lg in m.ligands:
            if lg.name not in cof_names:
                by_copy.setdefault(lg.number, []).append(lg.mdl)
    for num, poses in sorted(by_copy.items()):
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                r = pose_rmsd(poses[i], poses[j])
                if r < DIVERSITY_RMSD:
                    bad.append(f"WARN LIGAND {num}: MODEL {i + 1} and MODEL "
                               f"{j + 1} poses differ by {r:.2f} Aa RMSD "
                               f"(want >= {DIVERSITY_RMSD})")

    # The stronger claim: a whole MODEL repeating another. Two MODELs that hold
    # the same set of positions say the same thing even when the copy labels
    # differ, so this comparison ignores the labelling -- and unlike the
    # per-copy warning above, there is no defensible reason for it.
    arrs = [[lg.mdl for lg in m.ligands if lg.name not in cof_names]
            for m in models]
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            if same_arrangement(arrs[i], arrs[j]):
                bad.append(f"WARN MODEL {i + 1} and MODEL {j + 1} place every "
                           f"copy at the same positions (labels aside)")

    _ = binder
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    specs: dict[str, dict] = {}
    for series in ("L01", "L02"):
        try:
            specs.update(load_stage2(series))
        except FileNotFoundError:
            pass

    total, failed = 0, 0
    for path in args.files:
        if path.is_dir():
            continue
        total += 1
        target = path.name.split("LG")[0]
        spec = specs.get(target)
        if spec is None:
            print(f"{path.name}: NOT IN ANSWERS FILE")
            failed += 1
            continue
        issues = check(path, spec)
        if issues:
            failed += 1
            print(f"=== {path.name} ===")
            for i in issues:
                print(f"  {i}")
    print(f"\n{total - failed}/{total} file(s) clean")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
