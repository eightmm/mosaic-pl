#!/usr/bin/env python3
"""Replace the receptor of chosen LG MODELs with a differently-arranged prediction.

Companion to ``build_alt_dimer_conformers.py``, for the case that script cannot
serve. That one keeps our protomer and moves it, which is right when the target
arrangement is close to our own — and wrong when it is not, because a rigid
protomer cannot adapt its interface. On CASP17 T2451 every EAL-family
arrangement beyond ~2.7 A buried 64-252 heavy-atom pairs when transplanted, and
relaxing that funnelled straight back to conformation 1.

Template-forced co-folding solves it properly: Boltz's ``force: true`` template
restraint pulls the prediction into the target arrangement and lets side chains
and loops accommodate, so the model *builds* the alternative interface (measured
on T2451: 3.4-3.9 A from conformation 1, 1 inter-chain clash, iptm 0.755). To use
such a structure the receptor has to be swapped wholesale rather than re-posed.

How the ligand survives:

The replacement is superposed onto the LG MODEL's **ligand-hosting chain**, so
the pocket stays where the pose was docked and the ligand block is copied
untouched. That is only sound while the replacement's protomer is close to ours
— ``--max-host-rmsd`` enforces it, and the ligand-to-receptor clearance is
reported after the swap so a pose that ends up jammed into the new backbone is
visible rather than silent.

Usage:
    uv run python scripts/swap_lg_receptor.py \
      --lg experiments/CASP17/T2451/submissions/T2451_LCDD.lg \
      --swap 7:outputs/boltz_forced/4lyk/seed_42/.../model_0.cif \
      --swap 8:outputs/boltz_forced/6ih1/seed_42/.../model_0.cif \
      --out experiments/CASP17/T2451/submissions/T2451_LCDD.lg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gemmi
import numpy as np

_RESIDUE_INFO = gemmi.find_tabulated_residue


class Model:
    def __init__(self, number: str):
        self.number = number
        self.lines: list[str] = []

    def atom_rows(self, chain: str) -> list[int]:
        return [i for i, ln in enumerate(self.lines)
                if ln.startswith(("ATOM", "HETATM")) and ln[21] == chain]

    def chains(self) -> list[str]:
        seen = []
        for ln in self.lines:
            if ln.startswith("ATOM") and ln[21] not in seen:
                seen.append(ln[21])
        return seen

    def coords(self, chain: str) -> np.ndarray:
        return np.array([[float(self.lines[i][30:38]), float(self.lines[i][38:46]),
                          float(self.lines[i][46:54])] for i in self.atom_rows(chain)])

    def ca(self, chain: str) -> dict[int, np.ndarray]:
        return {int(ln[22:26]): np.array([float(ln[30:38]), float(ln[38:46]),
                                          float(ln[46:54])])
                for ln in self.lines
                if ln.startswith("ATOM") and ln[21] == chain
                and ln[12:16].strip() == "CA"}

    def ligand_coords(self) -> np.ndarray:
        out, in_mdl = [], False
        for ln in self.lines:
            if ln.startswith("LIGAND"):
                in_mdl = True
                continue
            if not in_mdl:
                continue
            if ln.startswith("M  END"):
                in_mdl = False
                continue
            p = ln.split()
            if len(p) >= 4 and p[3].isalpha() and len(p[3]) <= 2:
                try:
                    out.append([float(p[0]), float(p[1]), float(p[2])])
                except ValueError:
                    pass
        return np.array(out) if out else np.empty((0, 3))


def parse_lg(path: Path) -> tuple[list[str], list[Model]]:
    header, models, cur = [], [], None
    for raw in path.read_text().splitlines():
        if raw.startswith("MODEL"):
            cur = Model(raw.split()[1])
            cur.lines.append(raw)
            continue
        if cur is None:
            header.append(raw)
            continue
        cur.lines.append(raw)
        if raw.startswith("END") and not raw.startswith("ENDMDL"):
            models.append(cur)
            cur = None
    return header, models


def read_replacement(path: Path, min_res: int = 100):
    """{chain: {(resnum, atom_name): (xyz, bfactor)}} for protein chains."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    out = {}
    for model in st:
        for chain in model:
            atoms, n = {}, 0
            for res in chain:
                tab = _RESIDUE_INFO(res.name)
                if not (tab and tab.is_amino_acid()):
                    continue
                n += 1
                for atom in res:
                    atoms[(res.seqid.num, atom.name)] = (
                        np.array(atom.pos.tolist()), float(atom.b_iso))
            if n >= min_res:
                out[chain.name] = atoms
        break
    return out


def kabsch(a: np.ndarray, b: np.ndarray):
    """Rotation+translation mapping a onto b, plus the RMSD."""
    ca, cb = a.mean(0), b.mean(0)
    v, _, wt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1, 1, d]) @ wt
    rms = float(np.sqrt(((((a - ca) @ rot) - (b - cb)) ** 2).sum() / len(a)))
    return rot, cb - ca @ rot, rms


def set_xyz_b(line: str, xyz: np.ndarray, bfac: float | None) -> str:
    out = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
    if bfac is not None and len(out) >= 66:
        out = f"{out[:60]}{bfac:6.2f}{out[66:]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--swap", action="append", default=[], metavar="MODEL:PATH",
                    help="'<model number>:<replacement structure>', repeatable")
    ap.add_argument("--max-host-rmsd", type=float, default=2.0,
                    help="refuse when the replacement's protomer is further than "
                         "this from the ligand-hosting protomer it replaces")
    ap.add_argument("--min-ligand-clearance", type=float, default=2.0,
                    help="refuse when the carried-over pose ends up this close to "
                         "the new receptor")
    ap.add_argument("--bfactor", choices=["keep", "replacement"], default="replacement",
                    help="'replacement' copies the prediction's per-atom confidence")
    args = ap.parse_args()

    header, models = parse_lg(args.lg)
    by = {m.number: m for m in models}

    for spec in args.swap:
        number, _, rel = spec.partition(":")
        src = Path(rel)
        if number not in by:
            raise SystemExit(f"MODEL {number} not in {args.lg}")
        if not src.is_file():
            raise SystemExit(f"missing replacement: {src}")

        dst = by[number]
        lg_chains = dst.chains()
        rep = read_replacement(src)
        if len(rep) != len(lg_chains):
            raise SystemExit(f"MODEL {number}: {len(lg_chains)} chains in the LG file "
                             f"but {len(rep)} in {src.name}")

        lig = dst.ligand_coords()
        host = lg_chains[0]
        if len(lig):
            host = min(lg_chains,
                       key=lambda c: np.linalg.norm(
                           lig[:, None, :] - dst.coords(c)[None, :, :], axis=-1).min())

        # Superpose on the hosting chain so the pocket — and therefore the pose —
        # stays put; try every replacement chain and keep the best fit.
        best = None
        host_ca = dst.ca(host)
        for rc, atoms in rep.items():
            rca = {r: xyz for (r, name), (xyz, _) in atoms.items() if name == "CA"}
            keys = sorted(set(host_ca) & set(rca))
            if len(keys) < 50:
                continue
            rot, trans, rms = kabsch(np.array([rca[k] for k in keys]),
                                     np.array([host_ca[k] for k in keys]))
            if best is None or rms < best[3]:
                best = (rc, rot, trans, rms)
        if best is None:
            raise SystemExit(f"MODEL {number}: no replacement chain aligned to {host}")
        rep_host, rot, trans, host_rms = best
        if host_rms > args.max_host_rmsd:
            raise SystemExit(
                f"MODEL {number}: replacement protomer is {host_rms:.2f} A from the "
                f"hosting protomer (limit {args.max_host_rmsd}); the pose cannot be "
                f"carried over safely")

        placed = {c: {k: (xyz @ rot + trans, b) for k, (xyz, b) in atoms.items()}
                  for c, atoms in rep.items()}
        others = [c for c in placed if c != rep_host]
        mapping = dict(zip(lg_chains, [rep_host] + others)) if host == lg_chains[0] \
            else dict(zip(lg_chains, others + [rep_host]))
        mapping[host] = rep_host

        missing = 0
        for lg_chain, rep_chain in mapping.items():
            atoms = placed[rep_chain]
            for i in dst.atom_rows(lg_chain):
                key = (int(dst.lines[i][22:26]), dst.lines[i][12:16].strip())
                if key not in atoms:
                    missing += 1
                    continue
                xyz, b = atoms[key]
                dst.lines[i] = set_xyz_b(
                    dst.lines[i], xyz,
                    b * 100.0 if args.bfactor == "replacement" and b <= 1.0
                    else (b if args.bfactor == "replacement" else None))

        recept = np.vstack([dst.coords(c) for c in lg_chains])
        gap = (float(np.linalg.norm(lig[:, None, :] - recept[None, :, :], axis=-1).min())
               if len(lig) else float("nan"))
        print(f"MODEL {number} <- {src.parent.parent.parent.name}/{src.stem}  "
              f"host {host}<-{rep_host} ({host_rms:.2f} A)  "
              f"atoms not in replacement: {missing}  "
              f"ligand clearance {gap:.2f} A")
        if len(lig) and gap < args.min_ligand_clearance:
            raise SystemExit(
                f"MODEL {number}: ligand sits {gap:.2f} A from the swapped receptor "
                f"(floor {args.min_ligand_clearance})")
        if missing:
            print(f"  note: {missing} LG atoms had no counterpart and kept their "
                  f"original coordinates")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + [ln for m in models for ln in m.lines]) + "\n")
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
