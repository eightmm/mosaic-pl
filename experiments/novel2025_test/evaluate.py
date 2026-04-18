"""Evaluate novel2025 LG submissions against RCSB crystal structures.

For each completed submission (`experiments/submissions/{pdb}_input.lg`),
loads the corresponding RCSB mmCIF via rcsb_index.db, extracts the
experimental protein CA coordinates + the first candidate ligand's
atoms, then compares against every LG MODEL block using the same bond
reassignment + Kabsch + symmetric RMSD pipeline used for L1000.

Produces:
  - `evaluation.json`  — per-target MODEL breakdown
  - stdout summary     — top-1 / best-of-5 rates by seq_zone
"""
from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import (  # noqa: E402
    THREE_TO_ONE,
    kabsch,
    mol_from_mdl_body,
    parse_ca,
    pose_rmsd,
    reassign_bonds,
    transform_mol,
)
from casp17.lg_format import parse_lg  # noqa: E402

ROOT = Path(__file__).parent
TSV = Path("/home/jaemin/DB/RCSB/processed/seqid_zones_2025_nonredundant.tsv")
RCSB_DB = Path("/home/jaemin/DB/RCSB/processed/rcsb_index.db")
SUBMISSIONS = ROOT.parent / "submissions"
USALIGN_BIN = Path("/home/jaemin/project/CASP17/.local/bin/USalign")


def load_targets() -> dict[str, dict]:
    out = {}
    with TSV.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["pdb_id"]] = {
                "seq_zone": row["seq_zone"],
                "max_seq_identity": float(row.get("max_seq_identity") or 0),
                "deposition_date": row.get("deposition_date", ""),
                "candidate_ccd_codes": row["candidate_ccd_codes"].split("|"),
                "candidate_smiles": row["candidate_smiles"].split("|"),
                "ligand_ccd_codes": row["ligand_ccd_codes"].split("|"),
                "sequences": row["sequences"].split("|"),
                "organism": row.get("organism_name", ""),
            }
    return out


def cif_path_for(pdb_id: str) -> Path | None:
    conn = sqlite3.connect(str(RCSB_DB))
    try:
        row = conn.execute(
            "SELECT cif_path FROM entries WHERE pdb_id = ?", (pdb_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    p = Path(row[0])
    return p if p.exists() else None


def load_crystal(cif: Path, target_ccd: str, template: Chem.Mol, pred_seq: str | None = None):
    """Return (ca_dict, ref_lig_mol) from the RCSB mmCIF.

    ``ca_dict``: {resid: np.array(x,y,z)} for the first protein chain's
    CA atoms. Keys are **renumbered to align with ``pred_seq``**: we
    extract the crystal's one-letter sequence, find it as a substring
    inside the predicted TSV sequence, and remap each crystal residue
    id to its 1-indexed position in the TSV sequence. This makes the
    predicted-chain resid numbering (starts at 1) directly comparable
    to the crystal's.
    ``ref_lig_mol``: RDKit Mol for the first residue whose name matches
    ``target_ccd``, with bond orders reassigned from ``template``.
    """
    st = gemmi.read_structure(str(cif))
    raw_ca: list[tuple[int, str, np.ndarray]] = []  # (resid, 1-letter, xyz)
    lig_lines: list[str] = []
    serial = 1
    ref_chain_name = None

    for model in st:
        for chain in model:
            for res in chain:
                if res.name == target_ccd and not lig_lines:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        lig_lines.append(
                            f"HETATM{serial:>5d} {atom.name:>4s} LIG L   1    "
                            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                            f"{atom.element.name:>2s}\n"
                        )
                        serial += 1
                    continue
                if ref_chain_name is None or chain.name == ref_chain_name:
                    one = THREE_TO_ONE.get(res.name)
                    if not one:
                        continue
                    for atom in res:
                        if atom.name == "CA":
                            raw_ca.append((
                                res.seqid.num,
                                one,
                                np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                            ))
                            if ref_chain_name is None:
                                ref_chain_name = chain.name
                            break
        break

    # Renumber crystal CAs to match the predicted sequence's 1-indexed
    # residue numbering by aligning one-letter strings.
    ca: dict[int, np.ndarray] = {}
    if raw_ca and pred_seq:
        crystal_seq = "".join(o for _, o, _ in raw_ca)
        # Try direct substring match first
        idx = pred_seq.find(crystal_seq)
        if idx < 0:
            # Fall back: sliding window with mismatches — look for the longest
            # exact prefix-aligned offset that matches >= 90% of positions.
            best_off, best_hits = None, 0
            for off in range(len(pred_seq) - len(crystal_seq) + 1):
                hits = sum(
                    1 for k, c in enumerate(crystal_seq)
                    if off + k < len(pred_seq) and pred_seq[off + k] == c
                )
                if hits > best_hits:
                    best_off, best_hits = off, hits
            if best_off is not None and best_hits >= int(0.9 * len(crystal_seq)):
                idx = best_off
        if idx >= 0:
            # crystal residue i (0-indexed in raw_ca list) → predicted resid idx + i + 1
            for i, (_orig_id, _one, xyz) in enumerate(raw_ca):
                ca[idx + i + 1] = xyz
        else:
            # Fallback: keep original numbering (will likely mis-align)
            for rid, _one, xyz in raw_ca:
                ca[rid] = xyz
    else:
        for rid, _one, xyz in raw_ca:
            ca[rid] = xyz

    if not lig_lines or not ca:
        return ca, None

    raw = Chem.MolFromPDBBlock("".join(lig_lines) + "END\n", removeHs=True, sanitize=False)
    ref_lig = reassign_bonds(raw, template)
    return ca, ref_lig


def run_usalign(pred_pdb: Path, crystal_pdb: Path) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Run USalign and return (R, t, tm_score, rmsd_after_align).

    The matrix transforms ``pred`` coordinates onto ``crystal``:
    ``crystal ≈ R @ pred + t``. TM-score is normalised by the longer
    of the two chains (Chain_1). RMSD is reported on the aligned
    positions (not structure-wide). Returns None on USalign failure.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mat", delete=False) as matf:
        mat_path = Path(matf.name)
    try:
        res = subprocess.run(
            [str(USALIGN_BIN), str(pred_pdb), str(crystal_pdb),
             "-m", str(mat_path), "-ter", "1"],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode != 0:
            return None
        text = mat_path.read_text() if mat_path.exists() else ""
        R = np.zeros((3, 3))
        t = np.zeros(3)
        parsed = False
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] in ("0", "1", "2"):
                i = int(parts[0])
                t[i] = float(parts[1])
                R[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
                parsed = True
        if not parsed:
            return None
        tm = None
        rmsd_prot = None
        for line in res.stdout.splitlines():
            # USalign prints two TM-score lines: normalized by Structure_1
            # (query) and by Structure_2 (reference). The reference-normalized
            # one is the canonical value (USalign's own output says so). Old
            # regex looked for "Chain_1" which never matches USalign's actual
            # "Structure_1/Structure_2" wording → tm stayed None → 0.000.
            if tm is None and "TM-score=" in line and "Structure_2" in line:
                m = re.search(r"TM-score=\s*([\d.]+)", line)
                if m:
                    tm = float(m.group(1))
            if rmsd_prot is None and "RMSD=" in line:
                m = re.search(r"RMSD=\s*([\d.]+)", line)
                if m:
                    rmsd_prot = float(m.group(1))
        # Fallback: if Structure_2 header format ever changes upstream, accept
        # any TM-score line so we don't silently degrade to 0.000 again.
        if tm is None:
            for line in res.stdout.splitlines():
                if "TM-score=" in line:
                    m = re.search(r"TM-score=\s*([\d.]+)", line)
                    if m:
                        tm = float(m.group(1))
                        break
        return R, t, (tm or 0.0), (rmsd_prot or 0.0)
    finally:
        try:
            mat_path.unlink()
        except OSError:
            pass





# ---------- per-target evaluation ----------

def _pdb_atom_name(name: str) -> str:
    """Format an atom name into a 4-char PDB atom-name field.

    PDB convention: 1-letter element names have a leading space (so
    ``CA`` becomes `` CA ``), 2-letter element names start at column 13
    (so ``FE`` becomes ``FE  ``). gemmi stores names without the space
    prefix, so 1–3 char names whose first char is a letter get prefixed.
    """
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3s}"


def dump_crystal_protein_pdb(cif: Path, out_pdb: Path) -> bool:
    """Write the first protein chain of a RCSB mmCIF as a plain PDB for
    USalign. Only ATOM records are emitted (skips ligands/waters).
    Keeps one residue per seqid (drops altloc duplicates that confuse
    USalign's sequence parser)."""
    st = gemmi.read_structure(str(cif))
    lines = []
    ref_chain = None
    serial = 1
    seen_atoms: set[tuple[int, str]] = set()
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in THREE_TO_ONE:
                    continue
                if ref_chain is None:
                    ref_chain = chain.name
                if chain.name != ref_chain:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    key = (res.seqid.num, atom.name)
                    if key in seen_atoms:
                        continue
                    seen_atoms.add(key)
                    name = _pdb_atom_name(atom.name)
                    elem = atom.element.name
                    lines.append(
                        f"ATOM  {serial:>5d} {name} {res.name:>3s} A{res.seqid.num:>4d}    "
                        f"{atom.pos.x:>8.3f}{atom.pos.y:>8.3f}{atom.pos.z:>8.3f}"
                        f"  1.00  0.00          {elem:>2s}\n"
                    )
                    serial += 1
        break
    if not lines:
        return False
    out_pdb.write_text("".join(lines) + "END\n")
    return True


def write_pred_protein_pdb(atom_lines: list[str], out_pdb: Path) -> bool:
    """Dump predicted chain A ATOM records to a PDB file for USalign."""
    filtered = [l for l in atom_lines if l.startswith("ATOM")]
    if not filtered:
        return False
    out_pdb.write_text("".join(l + "\n" if not l.endswith("\n") else l for l in filtered) + "END\n")
    return True


def load_crystal_ligand(cif: Path, target_ccd: str, template: Chem.Mol) -> Chem.Mol | None:
    """Extract the first residue named ``target_ccd`` as an RDKit Mol
    with bond orders reassigned from the SMILES template."""
    st = gemmi.read_structure(str(cif))
    lig_lines: list[str] = []
    serial = 1
    for model in st:
        for chain in model:
            for res in chain:
                if res.name != target_ccd:
                    continue
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    lig_lines.append(
                        f"HETATM{serial:>5d} {atom.name:>4s} LIG L   1    "
                        f"{atom.pos.x:>8.3f}{atom.pos.y:>8.3f}{atom.pos.z:>8.3f}"
                        f"  1.00  0.00          {atom.element.name:>2s}\n"
                    )
                    serial += 1
                if lig_lines:
                    break
            if lig_lines:
                break
        break
    if not lig_lines:
        return None
    raw = Chem.MolFromPDBBlock("".join(lig_lines) + "END\n", removeHs=True, sanitize=False)
    return reassign_bonds(raw, template)


def evaluate_target(pdb_id: str, meta: dict, lg_path: Path, work_dir: Path) -> dict:
    out = {"target": pdb_id, "seq_zone": meta["seq_zone"], "max_seq_identity": meta["max_seq_identity"]}
    smi = meta["candidate_smiles"][0] if meta["candidate_smiles"] else ""
    ccd = meta["candidate_ccd_codes"][0] if meta["candidate_ccd_codes"] else ""
    if not smi or not ccd:
        out["error"] = "no candidate"
        return out
    template = Chem.MolFromSmiles(smi)
    if template is None:
        out["error"] = "bad SMILES"
        return out

    cif = cif_path_for(pdb_id)
    if cif is None:
        out["error"] = "no mmCIF"
        return out

    crystal_pdb = work_dir / f"{pdb_id}_crystal.pdb"
    if not dump_crystal_protein_pdb(cif, crystal_pdb):
        out["error"] = "crystal protein dump failed"
        return out

    ref_lig = load_crystal_ligand(cif, ccd, template)
    if ref_lig is None:
        out["error"] = f"exper ligand {ccd} parse/reassign failed"
        return out

    lg = parse_lg(lg_path)
    out["affnty"] = lg["affnty"]
    out["n_models"] = len(lg["models"])
    out["models"] = []

    for m in lg["models"]:
        entry = {"idx": m["idx"], "name": m["name"], "lscore": m["lscore"]}
        pred_pdb = work_dir / f"{pdb_id}_pred_{m['idx']}.pdb"
        if not write_pred_protein_pdb(m["atom_lines"], pred_pdb):
            entry["error"] = "no predicted ATOM lines"
            out["models"].append(entry)
            continue
        us = run_usalign(pred_pdb, crystal_pdb)
        if us is None:
            entry["error"] = "usalign failed"
            out["models"].append(entry)
            continue
        R, t, tm, prot_rmsd = us
        entry["tm_score"] = round(tm, 3)
        entry["protein_rmsd_usalign"] = round(prot_rmsd, 3)
        pred_raw = mol_from_mdl_body(m["mdl_text"])
        pred_mol = reassign_bonds(pred_raw, template)
        if pred_mol is None:
            entry["error"] = "pred ligand parse/reassign failed"
            out["models"].append(entry)
            continue
        transform_mol(pred_mol, R, t)
        try:
            entry["ligand_rmsd"] = round(pose_rmsd(pred_mol, ref_lig), 3)
        except Exception as e:
            entry["error"] = f"rmsd err: {e}"
        out["models"].append(entry)

    rmsds = [m.get("ligand_rmsd") for m in out["models"] if isinstance(m.get("ligand_rmsd"), float)]
    out["top1_rmsd"] = out["models"][0].get("ligand_rmsd") if out["models"] else None
    out["best_of_5_rmsd"] = round(min(rmsds), 3) if rmsds else None
    return out


def main():
    targets = load_targets()
    lgs = sorted(SUBMISSIONS.glob("*_input.lg"))
    novel_lgs = [p for p in lgs if not p.name.startswith("L10")]
    print(f"Evaluating {len(novel_lgs)} novel2025 LG submissions (USalign-based)...")
    work_dir = ROOT / "_eval_work"
    work_dir.mkdir(exist_ok=True)
    results = []
    for lg in novel_lgs:
        pdb = lg.stem.replace("_input", "")
        if pdb not in targets:
            results.append({"target": pdb, "error": "not in TSV"})
            continue
        try:
            r = evaluate_target(pdb, targets[pdb], lg, work_dir)
        except Exception as e:
            r = {"target": pdb, "error": f"crash: {e!r}"}
        results.append(r)
        top1 = r.get("top1_rmsd")
        best5 = r.get("best_of_5_rmsd")
        zone = r.get("seq_zone", "?")
        err = r.get("error")
        if err:
            print(f"  {pdb:<6} [{zone:>7}]  ERROR: {err}")
        elif top1 is None:
            print(f"  {pdb:<6} [{zone:>7}]  no ligand rmsd (all models failed)")
        else:
            flag = "✓" if top1 < 2 else "✗"
            t1s = f"{top1:>6.2f}"
            b5s = f"{best5:>6.2f}" if best5 is not None else "  n/a"
            print(f"  {pdb:<6} [{zone:>7}]  top1={t1s} best5={b5s}  {flag}")

    (ROOT / "evaluation.json").write_text(json.dumps(results, indent=2, default=str))

    # Summary
    ok = [r for r in results if "top1_rmsd" in r and r["top1_rmsd"] is not None]
    print()
    print(f"Evaluated OK: {len(ok)} / {len(novel_lgs)}")

    def stats(vals):
        if not vals:
            return None
        vs = sorted(vals)
        return {"n": len(vs), "min": vs[0], "median": vs[len(vs) // 2], "max": vs[-1]}

    header = f"  {'zone':<8} {'n':>4} {'top1<2':>7} {'best5<2':>8} {'top1<1':>7} {'med_t1':>7} {'med_b5':>7} {'med_TM':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for zone in ("novel", "remote", "related"):
        zoned = [r for r in ok if r.get("seq_zone") == zone]
        if not zoned:
            continue
        n_top1 = sum(1 for r in zoned if r["top1_rmsd"] < 2)
        n_best5 = sum(1 for r in zoned if (r.get("best_of_5_rmsd") or 99) < 2)
        n_t1_1 = sum(1 for r in zoned if r["top1_rmsd"] < 1)
        t1_med = stats([r["top1_rmsd"] for r in zoned])["median"]
        b5_med = stats([r["best_of_5_rmsd"] for r in zoned if r.get("best_of_5_rmsd") is not None])["median"]
        tms = [m["tm_score"] for r in zoned for m in r.get("models", []) if "tm_score" in m and m["idx"] == 1]
        tm_med = stats(tms)["median"] if tms else float("nan")
        print(f"  {zone:<8} {len(zoned):>4} {n_top1:>7} {n_best5:>8} {n_t1_1:>7} {t1_med:>7.2f} {b5_med:>7.2f} {tm_med:>7.3f}")
    # Totals
    n_top1 = sum(1 for r in ok if r["top1_rmsd"] < 2)
    n_best5 = sum(1 for r in ok if (r.get("best_of_5_rmsd") or 99) < 2)
    n_t1_1 = sum(1 for r in ok if r["top1_rmsd"] < 1)
    t1_med = stats([r["top1_rmsd"] for r in ok])["median"]
    b5_med = stats([r["best_of_5_rmsd"] for r in ok if r.get("best_of_5_rmsd") is not None])["median"]
    tms = [m["tm_score"] for r in ok for m in r.get("models", []) if "tm_score" in m and m["idx"] == 1]
    tm_med = stats(tms)["median"] if tms else float("nan")
    print(f"  {'TOTAL':<8} {len(ok):>4} {n_top1:>7} {n_best5:>8} {n_t1_1:>7} {t1_med:>7.2f} {b5_med:>7.2f} {tm_med:>7.3f}")


if __name__ == "__main__":
    main()
