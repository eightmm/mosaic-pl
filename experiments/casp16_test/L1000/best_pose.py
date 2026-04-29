"""For each target, compute the BEST RMSD across ALL generated poses
(not just top-5 in submission) vs experimental ligand.

Frame: poses are in cofolding/docking frame. Receptor PDB at
inputs/docking/receptor.pdb is the same frame. Align it to experimental
protein, apply the transform to every pose, compute symmetric RMSD.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from casp17.geometry import (  # noqa: E402
    kabsch,
    parse_ca,
    pose_rmsd,
    reassign_bonds,
    transform_mol,
)

ROOT = Path(__file__).parent
EXPER_DIR = ROOT / "L1000_prepared"
RUNS_DIR = ROOT.parents[1] / "runs"


def load_smiles_table() -> dict[str, str]:
    out: dict[str, str] = {}
    with (ROOT / "L1000_exper_affinity.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["Target ID"].strip()] = row["ligand_smiles"].strip()
    return out


def reassign_from_template(ref_pdb_mol: Chem.Mol, smiles: str) -> Chem.Mol | None:
    """Reassign bond orders on a PDB-parsed ligand using a SMILES template.

    Small wrapper that builds the template from SMILES first; delegates
    the actual assignment to :func:`casp17.geometry.reassign_bonds`."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None
    return reassign_bonds(ref_pdb_mol, template)


def cofolding_ligand_from_cif(cif_path: Path, smiles: str) -> Chem.Mol | None:
    """Extract HETATM ligand atoms from cofolding CIF and assign bonds via SMILES template."""
    import gemmi
    try:
        st = gemmi.read_structure(str(cif_path))
    except Exception:
        return None
    # Collect heavy ligand atoms (chain L or residues named LIG*/UNK/UNL or anything HETATM)
    lines = []
    serial = 1
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in ("LIG", "LIG1", "UNL", "UNK") or res.het_flag == "H" or chain.name in ("L",):
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        elem = atom.element.name
                        rec = (
                            f"HETATM{serial:>5d} {atom.name:>4s} LIG L"
                            f"{1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2s}\n"
                        )
                        lines.append(rec)
                        serial += 1
        break
    if not lines:
        return None
    pdb_block = "".join(lines) + "END\n"
    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)
    if raw is None:
        return None
    tpl = Chem.MolFromSmiles(smiles)
    if tpl is None:
        return None
    try:
        return AssignBondOrdersFromTemplate(tpl, raw)
    except Exception:
        return None


def load_sdf_all(path: Path):
    """Read every conformer from an SDF.

    ``removeHs=True`` is a no-op when ``sanitize=False`` (RDKit needs
    sanitization to identify which atoms are hydrogens), so we strip
    them manually after parsing. Without this the returned mol carries
    explicit H atoms and heavy-atom-count matching against the SMILES
    template fails, breaking downstream bond reassignment.
    """
    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    out = []
    for m in suppl:
        if m is None:
            continue
        try:
            m = Chem.RemoveHs(m, sanitize=False)
        except Exception:
            pass
        try:
            Chem.SanitizeMol(m)
        except Exception:
            pass
        out.append(m)
    return out


def evaluate_target(target: str):
    exper_dir = EXPER_DIR / target
    run_dir = RUNS_DIR / f"{target}_input"
    if not run_dir.exists():
        return {"target": target, "status": "no run"}
    rec_pdb = run_dir / "inputs" / "docking" / "receptor.pdb"
    if not rec_pdb.exists():
        return {"target": target, "status": "no docking receptor"}
    ref_lig_pdbs = sorted(exper_dir.glob("ligand_*_C_1.pdb"))
    if not ref_lig_pdbs:
        return {"target": target, "status": "no exper ligand"}
    ref_lig_pdb = Chem.MolFromPDBFile(str(ref_lig_pdbs[0]), removeHs=True, sanitize=False)
    smiles = SMILES.get(target, "")
    template = Chem.MolFromSmiles(smiles) if smiles else None
    ref_lig = reassign_bonds(ref_lig_pdb, template) if template else None
    if ref_lig is None:
        # fall back to PDB mol; will likely fail substructure match
        ref_lig = ref_lig_pdb
    try:
        Chem.SanitizeMol(ref_lig)
    except Exception:
        pass

    pred_ca = parse_ca(rec_pdb)
    ref_ca = parse_ca(exper_dir / "protein_aligned.pdb")
    common = sorted(set(pred_ca) & set(ref_ca))
    P = np.array([pred_ca[r] for r in common])
    Q = np.array([ref_ca[r] for r in common])
    R, t, prot_rmsd = kabsch(P, Q)

    pose_dir = run_dir / "outputs" / "analysis" / "poses"
    pxdock_sdf = run_dir / "outputs" / "protenix_dock" / "poses.sdf"
    out_dir = run_dir / "outputs"

    docking_sources = []  # (tool, path, mols)
    for sdf in sorted(pose_dir.glob("vina_seed_*.sdf")):
        docking_sources.append(("vina", sdf, load_sdf_all(sdf)))
    for sdf in sorted(pose_dir.glob("autodock_gpu_seed_*.sdf")):
        docking_sources.append(("adg", sdf, load_sdf_all(sdf)))
    if pxdock_sdf.exists():
        docking_sources.append(("pxdock", pxdock_sdf, load_sdf_all(pxdock_sdf)))

    cofold_sources = []  # (tool, path, mol)
    for cif in sorted((out_dir / "boltz2").rglob("*_aligned.cif")):
        m = cofolding_ligand_from_cif(cif, smiles)
        if m is not None:
            cofold_sources.append(("boltz2", cif, m))
    for cif in sorted((out_dir / "boltz2x").rglob("*_aligned.cif")):
        m = cofolding_ligand_from_cif(cif, smiles)
        if m is not None:
            cofold_sources.append(("boltz2x", cif, m))
    for cif in sorted((out_dir / "protenix").rglob("*_aligned.cif")):
        m = cofolding_ligand_from_cif(cif, smiles)
        if m is not None:
            cofold_sources.append(("protenix", cif, m))
    for cif in sorted((out_dir / "alphafold3").rglob("*_aligned.cif")):
        m = cofolding_ligand_from_cif(cif, smiles)
        if m is not None:
            cofold_sources.append(("af3", cif, m))

    per_tool: dict[str, list] = {k: [] for k in ("vina", "adg", "pxdock", "boltz2", "boltz2x", "protenix", "af3")}
    n_total = 0
    for tool, path, mols in docking_sources:
        for i, mol in enumerate(mols):
            n_total += 1
            mol_fixed = reassign_bonds(mol, template) if template else mol
            if mol_fixed is None:
                continue
            mol_t = transform_mol(mol_fixed, R, t, inplace=False)
            try:
                r = pose_rmsd(mol_t, ref_lig)
            except Exception:
                continue
            if r == r:
                per_tool[tool].append((path.stem, i, r))
    for tool, path, mol in cofold_sources:
        n_total += 1
        # cofold ligands were already reassigned at extraction time
        mol_t = transform_mol(mol, R, t, inplace=False)
        try:
            r = pose_rmsd(mol_t, ref_lig)
        except Exception:
            continue
        if r == r:
            per_tool[tool].append((path.parent.name + "/" + path.name, 0, r))

    res = {
        "target": target,
        "protein_ca_rmsd": round(prot_rmsd, 3),
        "n_total_poses": n_total,
    }
    for tool in ("vina", "adg", "pxdock", "boltz2", "boltz2x", "protenix", "af3"):
        rs = [r for *_, r in per_tool[tool] if r == r]
        if rs:
            best = min(per_tool[tool], key=lambda x: x[2])
            res[tool] = {
                "n": len(rs),
                "best_rmsd": round(min(rs), 3),
                "best_source": f"{best[0]}#{best[1]}",
                "median_rmsd": round(float(np.median(rs)), 3),
                "n_under_2A": sum(1 for r in rs if r < 2.0),
                "n_under_4A": sum(1 for r in rs if r < 4.0),
            }
    all_rs = [r for tool in per_tool.values() for *_, r in tool if r == r]
    if all_rs:
        res["overall_best_rmsd"] = round(min(all_rs), 3)
        res["overall_under_2A"] = sum(1 for r in all_rs if r < 2.0)
        res["overall_under_4A"] = sum(1 for r in all_rs if r < 4.0)
    return res


SMILES: dict[str, str] = {}


def main():
    import json
    global SMILES
    SMILES = load_smiles_table()
    targets = sorted(p.name for p in EXPER_DIR.glob("L*"))
    results = []
    for t in targets:
        try:
            r = evaluate_target(t)
        except Exception as e:
            r = {"target": t, "error": repr(e)}
        results.append(r)
        print(json.dumps(r, indent=2))
    out = ROOT / "best_pose.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
