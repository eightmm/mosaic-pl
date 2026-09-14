#!/usr/bin/env python3
"""Give a TS model every residue and atom the CASP target template declares.

The R2386 verification server rejected our 2026-08-26 submission with 509
``ERROR! ... atom present in the template not found in the model`` lines. CASP
publishes a zero-coordinate template on the target page — 417 residues, chain 0,
one canonical atom set per base — and the check is exact: **every** template atom
must be present. "Omitting residues is allowed" is true of the TS format in
general and false for this check.

That is a problem the solvent pipeline cannot solve on its own. The crystal
frames cover roughly target residues 7-395; residues 397-417 are the ligated 3'
exon, disordered in all 49 usable entries, and no experimental structure in the
set places them. They are also all inside the declared non-core regions, so
nothing here is scored — the RNA there is context that has to exist, be
chemically valid, and stay out of the way.

Three strategies, in decreasing order of how much real structure they carry:

1. **Transplant.** A donor that models the missing run in its own main chain is
   superposed on the residues flanking the gap — a *local* fit, not the global
   one, so the junction bond comes out at 1.6 Å — and the run is copied across.
2. **Extend.** Otherwise a real helical stretch of the frame itself becomes the
   geometric unit: its first residue is fitted onto the anchor residue, and the
   rest of the stretch becomes the new residues. Internal geometry and the
   junction bond are real because the source stretch is real. Candidate
   stretches are tried until one lands clash-free.
3. **Fill.** Any atom still missing inside a residue is copied from another
   residue of the same base, fitted on the atoms they share — this is what puts
   a 5'-phosphate on a terminal residue that was deposited without one.

Bases are then swapped to the target sequence with the same ring-anchored graft
the builder uses, so a transplanted or extended residue carries the right base
without disturbing the backbone it was built on.

Usage:
    uv run python scripts/complete_ts_chain.py \
      --ts experiments/CASP17/R2386/submissions/R2386_LCDD.ts \
      --template data/solvent_templates/R2386/R2386_casp_template.pdb \
      --out R2386_complete.ts
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from make_solvent_ts_submission import (  # noqa: E402
    BACKBONE_ATOMS, _atom_line, graft_base, kabsch,
)

#: Atoms used to seat one residue on another when only the backbone is shared.
FIT_BACKBONE = ("P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'")

#: Seating a residue for an extension uses exactly the three atoms around the
#: bond that is about to form. Three non-collinear points fix a rigid transform
#: exactly, so the source stretch's own O3'-P bond is carried over at its real
#: length; fitting on more atoms averages the sugar pucker difference into the
#: junction and leaves the bond long or short.
FIT_3PRIME = ("C3'", "O3'", "C4'")
FIT_5PRIME = ("P", "O5'", "C5'")

#: A built residue closer than this to an atom it is not bonded to is a clash.
CLASH = 2.5

#: An O3'(i)-P(i+1) bond outside this range is not a bond.
BOND_RANGE = (1.40, 1.80)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def read_template(path: Path):
    """resnum → (resname, [atom names]), in the order the template lists them."""
    order: dict[int, list[str]] = {}
    names: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        num = int(line[22:26])
        order.setdefault(num, []).append(line[12:16].strip())
        names[num] = line[17:20].strip()
    return {n: (names[n], order[n]) for n in order}


def read_models(path: Path):
    """Split a TS file into (header lines, [(remark, rna, solvent), ...])."""
    header, models, cur = [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            cur = {"remark": None, "rna": defaultdict(dict), "resname": {},
                   "bfac": {}, "solvent": []}
            models.append(cur)
            continue
        if cur is None:
            header.append(line)
            continue
        if line.startswith("REMARK"):
            cur["remark"] = line[7:]
        elif line.startswith("ATOM"):
            num = int(line[22:26])
            cur["rna"][num][line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
            cur["resname"][num] = line[17:20].strip()
            cur["bfac"].setdefault(num, {})[line[12:16].strip()] = float(line[60:66])
        elif line.startswith("HETATM"):
            cur["solvent"].append((line[17:20].strip(),
                                   float(line[30:38]), float(line[38:46]),
                                   float(line[46:54]), float(line[60:66])))
    return header, models


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _fit(donor: dict, target: dict, atoms) -> tuple | None:
    shared = [a for a in atoms if a in donor and a in target]
    if len(shared) < 3:
        return None
    return kabsch(np.array([donor[a] for a in shared], dtype=float),
                  np.array([target[a] for a in shared], dtype=float))[:2]


def _apply(res: dict, rt) -> dict:
    r, t = rt
    return {a: tuple(map(float, np.asarray(xyz, dtype=float) @ r + t))
            for a, xyz in res.items()}


def _runs(missing: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for n in sorted(missing):
        if out and n == out[-1][-1] + 1:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def _clash_tree(have: dict[int, dict], run: list[int]):
    ends = {min(run) - 1, max(run) + 1}
    other = np.array([xyz for n, res in have.items() if n not in ends
                      for xyz in res.values()], dtype=float)
    return cKDTree(other) if len(other) else None


def _clashes(new: dict[int, dict], tree, run: list[int],
             cutoff: float = CLASH) -> bool:
    pts = np.array([xyz for n in run for xyz in new[n].values()], dtype=float)
    if not len(pts):
        return True
    if tree is None:
        return False
    return bool((tree.query(pts)[0] < cutoff).any())


# --------------------------------------------------------------------------- #
# the three strategies
# --------------------------------------------------------------------------- #
def transplant_candidates(rna, run, donors, flank=14):
    """Yield copies of ``run`` from donors that model it, fitted on the flanks."""
    for donor_id, dres, dnames in donors:
        if any(n not in dres for n in run):
            continue
        anchors = [n for n in range(min(run) - flank, max(run) + flank + 1)
                   if n in rna and n in dres]
        if len(anchors) < 4:
            continue
        d = {f"{n}:{a}": xyz for n in anchors for a, xyz in dres[n].items()
             if a in FIT_BACKBONE}
        f = {f"{n}:{a}": xyz for n in anchors for a, xyz in rna[n].items()
             if a in FIT_BACKBONE}
        rt = _fit(d, f, list(d))
        if rt is None:
            continue
        yield ({n: _apply(dres[n], rt) for n in run},
               {n: dnames.get(n) for n in run}, f"donor {donor_id}")


def junction_candidates(rna, run, donors, anchor: int, step: int):
    """Yield donor copies seated on the anchor residue by an exact 3-atom fit.

    A flank fit averages the frame difference over a dozen residues and leaves
    the junction bond long; fitting the donor's own anchor residue onto ours
    with the three atoms around the bond reproduces the donor's real
    O3'-P distance exactly, and the rest of its run rides along.
    """
    fit_on = FIT_3PRIME if step > 0 else FIT_5PRIME
    for donor_id, dres, dnames in donors:
        if anchor not in dres or any(n not in dres for n in run):
            continue
        rt = _fit(dres[anchor], rna[anchor], fit_on)
        if rt is None:
            continue
        yield ({n: _apply(dres[n], rt) for n in run},
               {n: dnames.get(n) for n in run}, f"donor {donor_id} (junction fit)")


def extend_candidates(rna, run, anchor: int, step: int):
    """Yield builds of ``run`` made from real stretches of the frame itself.

    ``step`` is +1 when the run continues past the anchor's 3' side and -1 for
    the 5' side. The source stretch's own bonds carry over, junction included,
    because the stretch's first residue is seated on the anchor.
    """
    need = len(run)
    fit_on = FIT_3PRIME if step > 0 else FIT_5PRIME
    for start in sorted(rna):
        stretch = [start + i * step for i in range(need + 1)]
        if any(n not in rna for n in stretch) or anchor in stretch:
            continue
        rt = _fit(rna[stretch[0]], rna[anchor], fit_on)
        if rt is None:
            continue
        built = {}
        for target_num, src in zip(sorted(run, reverse=step < 0), stretch[1:]):
            built[target_num] = _apply(rna[src], rt)
        yield built, {n: None for n in run}, f"the stretch at {start}"


def fill_atoms(rna, resname, required):
    """Copy any atom a residue is missing from another residue of the same base."""
    filled = 0
    by_base = defaultdict(list)
    for n, res in rna.items():
        by_base[resname.get(n)].append(n)
    for num, (base, atoms) in sorted(required.items()):
        if num not in rna:
            continue
        missing = [a for a in atoms if a not in rna[num]]
        if not missing:
            continue
        for src in by_base.get(base, []):
            if src == num or any(a not in rna[src] for a in missing):
                continue
            rt = _fit(rna[src], rna[num], list(rna[num]))
            if rt is None:
                continue
            moved = _apply({a: rna[src][a] for a in missing}, rt)
            rna[num].update(moved)
            filled += len(missing)
            break
    return filled


# --------------------------------------------------------------------------- #
def _swap_base(res: dict, want: str, rna, resname, donors, num: int):
    """Put the target's base on a residue, leaving its backbone alone."""
    for _id, dres, dnames in donors:
        if dnames.get(num) == want and num in dres:
            out = graft_base(res, dres[num])
            if out is not None:
                return out
    for src, base in resname.items():
        if base == want and src in rna:
            out = graft_base(res, rna[src])
            if out is not None:
                return out
    return None


def _fill_one(res: dict, base: str, atoms, rna, resname):
    """Copy any atom this residue is missing from another residue of its base."""
    missing = [a for a in atoms if a not in res]
    if not missing:
        return 0
    for src, sbase in resname.items():
        if sbase != base or src not in rna:
            continue
        if any(a not in rna[src] for a in missing):
            continue
        rt = _fit(rna[src], res, [a for a in res])
        if rt is None:
            continue
        res.update(_apply({a: rna[src][a] for a in missing}, rt))
        return len(missing)
    return 0


def _junction_ok(built, rna, run) -> bool:
    """Both ends of a built run must join the chain with a real bond."""
    for a, b in ((min(run) - 1, min(run)), (max(run), max(run) + 1)):
        left = built.get(a) or rna.get(a)
        right = built.get(b) or rna.get(b)
        if left is None or right is None:
            continue
        if "O3'" not in left or "P" not in right:
            return False
        d = float(np.linalg.norm(np.asarray(left["O3'"], dtype=float)
                                 - np.asarray(right["P"], dtype=float)))
        if not BOND_RANGE[0] <= d <= BOND_RANGE[1]:
            return False
    return True



def _widen(run, rna, required, step, extra):
    """Grow a run into the residues already modelled on its anchor side."""
    if step > 0:
        start = min(run) - extra
        return [n for n in range(start, max(run) + 1) if n in required]
    end = max(run) + extra
    return [n for n in range(min(run), end + 1) if n in required]


def _anchor_for(run, rna, step):
    if step > 0:
        below = [n for n in rna if n < min(run)]
        return max(below) if below else None
    above = [n for n in rna if n > max(run)]
    return min(above) if above else None


def _build_run(rna, resname, required, donors, run, anchor, step, max_tries):
    """Return (built, how) for the first candidate with clean bonds and clearance."""
    if anchor is None:
        return None
    tree = _clash_tree({n: v for n, v in rna.items() if n not in run}, run)
    for cutoff in (CLASH, 2.2, 1.9):
        pool = (list(junction_candidates(rna, run, donors, anchor, step))
                + list(transplant_candidates(rna, run, donors))
                + list(extend_candidates(rna, run, anchor, step)))
        for built, names, how in pool[:max_tries]:
            built = {n: dict(v) for n, v in built.items()}
            ok = True
            for n in run:
                if n not in built:
                    ok = False
                    break
                want = required[n][0]
                if names.get(n) != want:
                    out = _swap_base(built[n], want, rna, resname, donors, n)
                    if out is not None:
                        built[n] = out
                _fill_one(built[n], want, required[n][1], rna, resname)
            if not ok or not _junction_ok(built, rna, run):
                continue
            if _clashes(built, tree, run, cutoff):
                continue
            return built, how + (f" (clearance {cutoff} A)" if cutoff != CLASH else "")
    return None


def complete(rna, resname, required, donors, log, max_tries=400):
    """Build every residue the template declares, accepting only clean geometry.

    A candidate is finished before it is judged — bases swapped to the target,
    missing atoms filled — because both steps move atoms, and a run that was
    clash-free as bare backbone can stop being so once its base is on.
    """
    # Fill inside existing residues first. A crystal's 5'-terminal residue is
    # deposited without its phosphate, and that phosphate is the atom a 5'
    # extension has to anchor on — leave the fill until last and the whole 5'
    # end is unbuildable.
    pre = sum(_fill_one(rna[n], required[n][0], required[n][1], rna, resname)
              for n in sorted(rna) if n in required)
    if pre:
        log.append(f"    atoms filled before building: {pre}")

    for run in _runs([n for n in required if n not in rna]):
        below = [n for n in rna if n < min(run)]
        above = [n for n in rna if n > max(run)]
        if above and not below:
            anchor, step = min(above), -1
        elif below:
            anchor, step = max(below), 1
        else:
            log.append(f"    residues {min(run)}-{max(run)}: no anchor")
            continue

        # If the run cannot leave the frame cleanly, let it start earlier:
        # rebuilding a few residues we already have hands the donor the exit
        # vector as well as the tail. Only ever widens into the terminus.
        for widen in (0, 2, 4, 6, 8, 12):
            work = run if not widen else _widen(run, rna, required, step, widen)
            chosen = _build_run(rna, resname, required, donors, work, anchor
                                if not widen else _anchor_for(work, rna, step),
                                step, max_tries)
            if chosen:
                built, how = chosen
                if widen:
                    how += f" (run widened by {widen})"
                for n in work:
                    rna[n] = built[n]
                    resname[n] = required[n][0]
                log.append(f"    residues {min(work)}-{max(work)} ({len(work)}): {how}")
                break
        else:
            log.append(f"    residues {min(run)}-{max(run)}: COULD NOT BUILD CLEANLY")
        continue

        chosen = None
        tree = _clash_tree(rna, run)
        # A long tail has to leave a crowded surface, so if nothing clears the
        # full van der Waals bar, take the best that clears a tighter one rather
        # than shipping a hole.
        for cutoff in (CLASH, 2.2, 1.9):
            pool = (list(junction_candidates(rna, run, donors, anchor, step))
                    + list(transplant_candidates(rna, run, donors))
                    + list(extend_candidates(rna, run, anchor, step)))
            for built, names, how in pool[:max_tries]:
                built = {n: dict(v) for n, v in built.items()}
                for n in run:
                    want = required[n][0]
                    if names.get(n) != want:
                        out = _swap_base(built[n], want, rna, resname, donors, n)
                        if out is not None:
                            built[n] = out
                    _fill_one(built[n], want, required[n][1], rna, resname)
                if not _junction_ok(built, rna, run):
                    continue
                if _clashes(built, tree, run, cutoff):
                    continue
                chosen = (built, how + (f" (clearance {cutoff} A)"
                                        if cutoff != CLASH else ""))
                break
            if chosen:
                break

        if chosen is None:
            log.append(f"    residues {min(run)}-{max(run)}: COULD NOT BUILD CLEANLY")
            continue
        built, how = chosen
        for n in run:
            rna[n] = built[n]
            resname[n] = required[n][0]
        log.append(f"    residues {min(run)}-{max(run)} ({len(run)}): {how}")

    swapped = 0
    for num, (base, _atoms) in required.items():
        if num in rna and resname.get(num) != base:
            out = _swap_base(rna[num], base, rna, resname, donors, num)
            if out is not None:
                rna[num], resname[num] = out, base
                swapped += 1
    filled = sum(_fill_one(rna[n], required[n][0], required[n][1], rna, resname)
                 for n in sorted(rna) if n in required)
    if swapped:
        log.append(f"    bases swapped to the target sequence: {swapped}")
    log.append(f"    atoms filled inside residues: {filled}")


def built_confidence(rna, bfac, required):
    """Give built residues a confidence that falls off along the built run.

    The B-factor column is CASP's error estimate, and a model that repeats one
    value over every residue draws a verification warning — the assessors read
    the column and cannot use a flat one. Residues carried over keep the
    per-atom confidence the builder measured from cross-structure spread; a
    residue we built has no such evidence, so it starts just under its anchor
    and decays with every step further into the terminus, which is honestly how
    much less we know about it.
    """
    known = sorted(n for n in rna if bfac.get(n))
    if not known:
        return
    for num in sorted(rna):
        if bfac.get(num):
            continue
        anchor = min(known, key=lambda k: abs(k - num))
        vals = bfac.get(anchor) or {}
        base = float(np.mean(list(vals.values()))) if vals else 30.0
        conf = max(2.0, min(base, 40.0) - 4.0 * abs(num - anchor))
        bfac[num] = {a: round(conf, 2) for a in rna[num]}


def render(rna, resname, solvent, required, chain: str, bfac=None) -> list[str]:
    """Emit ATOM records in the template's own atom order, then the solvent."""
    bfac = bfac or {}
    lines, serial = [], 1
    for num, (base, atoms) in sorted(required.items()):
        res = rna.get(num)
        if not res:
            continue
        for aname in atoms:
            if aname not in res:
                continue
            x, y, z = res[aname]
            el = aname[0] if aname[0].isalpha() else aname[1]
            b = (bfac.get(num) or {}).get(aname, 50.0)
            lines.append(_atom_line("ATOM", serial, aname, base, chain, num,
                                    x, y, z, 1.00, b, el))
            serial += 1
    last = max(n for n in required if n in rna)
    lines.append(f"TER   {serial:>5d}      {required[last][0]:>3s} {chain:1s}"
                 f"{last:>4d}")
    serial += 1
    # Solvent is numbered past the last target residue. Sharing the RNA's
    # numbering is what broke the first submission: the verification server
    # keys residues on (chain, resseq), so a water numbered 394 hid the
    # nucleotide numbered 394 and every one of its atoms was reported missing.
    num = max(required) + 1
    for kind, x, y, z, bfac in solvent:
        el = "O" if kind == "HOH" else kind
        lines.append(_atom_line("HETATM", serial, el, kind, chain, num,
                                x, y, z, 1.00, bfac, el))
        serial += 1
        num += 1
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True,
                    help="the zero-coordinate PDB the CASP target page publishes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chain", default="0")
    ap.add_argument("--donor-ts", type=Path, nargs="*", default=[],
                    help="other TS files whose models may donate missing residues. "
                         "A model that already covers the whole target - a folded "
                         "3' tail grafted onto a frame, say - is a better source "
                         "for the disordered termini than any stretch recycled "
                         "from elsewhere in the same chain.")
    args = ap.parse_args()

    required = read_template(args.template)
    header, models = read_models(args.ts)
    print(f"template: {len(required)} residues, "
          f"{sum(len(a) for _, a in required.values())} atoms")

    # Every model is a donor for every other: the frames differ, but a run one
    # frame is missing is often modelled by another.
    pool = [(f"MODEL {i + 1}", m["rna"], m["resname"])
            for i, m in enumerate(models)]
    # External donors lead: they are consulted before this file's own models.
    external = []
    for path in args.donor_ts:
        _hdr, dmodels = read_models(path)
        for j, dm in enumerate(dmodels, start=1):
            cover = sum(1 for n in required if n in dm["rna"])
            external.append((f"{path.stem}#{j}", dm["rna"], dm["resname"]))
            print(f"donor {path.stem} model {j}: covers {cover}/{len(required)} residues")
    pool = external + pool

    out_models = []
    for i, m in enumerate(models, start=1):
        rna, resname = m["rna"], m["resname"]
        before = len(rna)
        log: list[str] = []
        donors = [d for d in pool if d[0] != f"MODEL {i}"]
        donors.sort(key=lambda d: -sum(1 for n in required if n in d[1]))
        complete(rna, resname, required, donors, log)
        gaps = [n for n in required if n not in rna]
        print(f"  MODEL {i}: {before} -> {len(rna)} residues"
              + (f", STILL MISSING {gaps}" if gaps else ", complete"))
        for line in log:
            print(line)
        built_confidence(rna, m["bfac"], required)
        out_models.append((render(rna, resname, m["solvent"], required,
                                  args.chain, m["bfac"]), m["remark"]))

    text = "\n".join(header).rstrip("\n") + "\n"
    for idx, (lines, remark) in enumerate(out_models, start=1):
        text += f"MODEL {idx}\n"
        if remark:
            text += f"REMARK {remark}\n"
        text += "PARENT N/A\n" + "\n".join(lines) + "\nEND\n"
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
