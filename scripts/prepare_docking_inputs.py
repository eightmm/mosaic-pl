#!/usr/bin/env python3
"""Prepare docking inputs from unified YAML and cofolding outputs.

Converts:
  - Ligand SMILES → 3D SDF → PDBQT (for Vina)
  - Cofolding output CIF → receptor PDB → receptor PDBQT (for Vina)

Protenix-Dock needs receptor PDB + ligand SDF.
Vina needs receptor PDBQT + ligand PDBQT.

Usage:
    python prepare_docking_inputs.py \
        --input-yaml runs/target/inputs/boltz_input.yaml \
        --cofolding-dir runs/target/outputs/boltz \
        --output-dir runs/target/inputs/docking \
        --model boltz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script runs inside ``.venvs/protenix-dock`` (selected by the wrapper
# because it has rdkit/meeko + ambertools), which is NOT the hub venv that
# pip-installs the ``casp17`` package. Make ``src/`` importable so the
# shared ccd_sets / template-filter modules resolve regardless of caller.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def smiles_to_sdf(smiles: str, output_path: Path, name: str = "ligand") -> Path | None:
    """Convert SMILES to 3D SDF using RDKit.

    Returns the output path on success, ``None`` on parse/embedding failure.
    Large lipids (phosphatidylcholine, polyisoprenoids, glycolipids with long
    aliphatic chains) can make ETKDGv3 run for many minutes — we cap each
    embedding attempt and drop the ligand if neither the template nor random
    fallback succeeds. Callers handle ``None`` by skipping the ligand rather
    than aborting the whole run.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ERROR: RDKit could not parse SMILES for {name}: {smiles[:80]}")
        return None

    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", name)

    # Cap embedding cost: long-chain ligands hit pathological runtime with
    # ETKDGv3's stochastic search. ``maxAttempts`` bounds the retry count
    # and keeps total wall-time predictable even for 100+ atom lipids.
    params = AllChem.ETKDGv3()
    params.maxAttempts = 50
    params.useRandomCoords = False
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Template attempt failed → fall back to random coords (still bounded
        # by maxAttempts). This works for chains RDKit can't stereo-match.
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        print(f"  WARNING: could not generate 3D conformer for {name} (chain too long or stereochem ambiguous)")
        return None

    # MMFF optimization is best-effort: raises on missing parameters (some
    # metal coordinations, exotic elements). Missing optimization isn't
    # fatal — the embedded geometry is already usable for docking prep.
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception as e:
        print(f"  WARNING: MMFF optimization failed for {name}: {e} (keeping embedded geometry)")

    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    print(f"  Ligand SDF: {output_path}")
    return output_path


def sdf_to_pdbqt(sdf_path: Path | None, output_path: Path) -> Path | None:
    """Convert SDF to PDBQT using meeko.

    Returns the output path on success, ``None`` if the SDF is missing or
    meeko rejects the molecule. Docking on that ligand is then skipped
    rather than aborting the whole pipeline.
    """
    from meeko import MoleculePreparation
    from rdkit import Chem

    if sdf_path is None or not Path(sdf_path).exists():
        return None
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(supplier, None)
    if mol is None:
        print(f"  WARNING: failed to read SDF {sdf_path}, skipping PDBQT conversion")
        return None

    preparator = MoleculePreparation()
    try:
        preparator.prepare(mol)
    except Exception as e:
        print(f"  WARNING: meeko crashed on {sdf_path.name}: {e}")
        return None
    if not preparator.is_ok:
        print(f"  WARNING: meeko rejected {sdf_path.name}: {preparator.log}")
        return None

    pdbqt_string = preparator.write_pdbqt_string()
    output_path.write_text(pdbqt_string)
    print(f"  Ligand PDBQT: {output_path}")
    return output_path


def cif_to_pdb(
    cif_path: Path,
    output_path: Path,
    dockable_ligand_chains: set[str] | None = None,
) -> Path:
    """Convert mmCIF to PDB, retaining metals/ions as receptor cofactors.

    The legacy implementation called ``structure.remove_ligands_and_waters()``
    which strips *every* non-polymer entity — including metals (MG/ZN/CA/FE/
    MN/NA/CL etc.) and small cofactors. Metalloproteins (kinases binding
    Mg²⁺, zinc proteases, ferredoxin, …) docked against a metal-stripped
    receptor produce poses that ignore the coordination geometry, so the
    result is wrong even when the predictor finds the correct pocket.

    The right behaviour is selective removal: drop **only the dockable
    ligand chains** (those listed in the input YAML, the molecules we're
    about to dock) plus waters. Metals and other cofactors stay in the
    receptor PDB. Downstream pdbqt conversion / pdb2pqr handle the metal
    atom types via existing code paths (AD4 has Mg/Zn/Ca/Fe/Mn types;
    AMBER ignores HETATMs it can't parameterize, leaving the metal as a
    rigid coordinate the docking grid maps still see).

    Args:
        cif_path: input mmCIF.
        output_path: where to write the PDB.
        dockable_ligand_chains: chain ids of the ligands we'll dock — these
            get stripped. ``None`` falls back to legacy
            ``remove_ligands_and_waters`` behaviour for backward compat.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))

    def _fit_chain_names(st) -> None:
        """Rename chains PDB cannot hold, in place.

        gemmi writes a two-character chain id into the PDB chain column pair,
        but raises ``chain name too long for the PDB format`` on three. An
        input that names a cofactor chain ``X11`` therefore takes the whole
        docking preparation down — and the wrapper reports "docking prep
        failed, continuing", so the job still exits 0 and the run looks
        complete with an empty docking stage. Renaming keeps the receptor
        intact; a cofactor's chain letter carries no meaning downstream, and
        the dockable ligands have already been removed by the time this runs.
        """
        used = {ch.name for model in st for ch in model if len(ch.name) <= 2}
        pool = [c for c in "BCDEFGHIJKMNOPQRSTUVW0123456789" if c not in used]
        for model in st:
            for ch in model:
                if len(ch.name) <= 2:
                    continue
                new = pool.pop(0) if pool else ch.name[:2]
                print(f"  chain {ch.name!r} renamed to {new!r} "
                      f"(PDB holds at most two characters)")
                used.add(new)
                ch.name = new

    if dockable_ligand_chains is None:
        # Backward-compat: caller didn't supply the dockable list, fall
        # back to the old whole-non-polymer wipe. Logged so reruns spot
        # the regression.
        structure.remove_ligands_and_waters()
        print("  WARNING: cif_to_pdb called without dockable_ligand_chains; "
              "every non-polymer (incl. metals) is being stripped from receptor")
        _fit_chain_names(structure)
        structure.write_pdb(str(output_path))
        print(f"  Receptor PDB: {output_path}")
        return output_path

    # Selective: walk every non-polymer residue, remove only those whose
    # chain id is in the dockable set, plus waters. Everything else (metals,
    # cofactors, glycans on protein chains) stays.
    n_dropped_lig = 0
    n_dropped_water = 0
    n_kept_metal_or_cofactor = 0
    metal_elements = {"MG", "ZN", "CA", "FE", "MN", "NA", "CL", "K", "CU", "NI",
                      "CO", "CD", "HG", "PB", "BA", "SR", "AL"}
    # gemmi.Chain doesn't expose a remove-residue helper; we delete by
    # index in reverse order so earlier indices stay stable.
    for model in structure:
        for chain in model:
            is_dockable = chain.name in dockable_ligand_chains
            for i in range(len(chain) - 1, -1, -1):
                res = chain[i]
                is_water = (res.entity_type == gemmi.EntityType.Water
                            or res.name == "HOH")
                is_nonpoly = res.entity_type == gemmi.EntityType.NonPolymer
                if is_water:
                    del chain[i]
                    n_dropped_water += 1
                elif is_dockable and is_nonpoly:
                    del chain[i]
                    n_dropped_lig += 1
                elif is_nonpoly:
                    n_kept_metal_or_cofactor += 1
    _fit_chain_names(structure)
    structure.write_pdb(str(output_path))
    print(f"  Receptor PDB: {output_path}")
    print(f"    dropped: {n_dropped_lig} dockable-ligand residue(s), "
          f"{n_dropped_water} water(s)")
    if n_kept_metal_or_cofactor:
        print(f"    kept (cofactors/metals): {n_kept_metal_or_cofactor} residue(s) "
              "(retained for docking grid maps)")
    return output_path


def run_pdb2pqr(pdb_path: Path, pqr_path: Path) -> None:
    """Run pdb2pqr to add hydrogens and assign AMBER charges."""
    import subprocess
    try:
        subprocess.run(
            ["pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(
            [sys.executable, "-m", "pdb2pqr", "--ff=AMBER", "--ffout=AMBER", "--keep-chain",
             str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )


def pqr_to_protonated_pdb(pqr_path: Path, output_path: Path) -> Path:
    """Convert PQR back to PDB format (keeps protonation, HIS→HID/HIE/HIP)."""
    lines = pqr_path.read_text().splitlines()
    pdb_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:1].upper()
            # PDB format: cols 1-54 (coords), 55-60 (occ), 61-66 (bfactor), 77-78 (element)
            pdb_lines.append(f"{line[:54]:<54s}  1.00  0.00          {element:>2s}  ")
        elif line.startswith(("TER", "END")):
            pdb_lines.append(line)
    output_path.write_text("\n".join(pdb_lines) + "\n")
    print(f"  Receptor PDB (protonated): {output_path}")
    return output_path


# Nucleic acid residue names (DNA + RNA + standard variants). When the
# Nucleic residue set lives in ``casp17.ccd_sets`` so Stage 3 and the
# Track 2 prep share the same authoritative list (drift between the two
# previously gave inconsistent obabel-routing behaviour).
from casp17.ccd_sets import NUCLEIC_RESIDUES as _NUCLEIC_RESIDUES


def _has_nucleic_acid(pdb_path: Path) -> bool:
    """Return True when the PDB has any standard RNA/DNA residue.

    Looked up by residue name (cols 17-20 of ATOM/HETATM lines). Used to
    decide whether to route receptor protonation through obabel instead
    of pdb2pqr; protein-only receptors keep the pdb2pqr path so existing
    behaviour stays unchanged.
    """
    if not pdb_path.exists():
        return False
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            res = line[17:20].strip().upper()
            if res in _NUCLEIC_RESIDUES:
                return True
    return False


def _pdb_to_pdbqt_obabel(pdb_path: Path, output_path: Path) -> bool:
    """Single-step PDB → PDBQT for receptors with nucleic acids.

    obabel handles both protein and nucleotide residues, adds hydrogens
    at pH 7.4, assigns Gasteiger charges, and writes AD4 atom types.
    Used as a fallback path because pdb2pqr's AMBER FF doesn't cover
    nucleic acids.

    Returns True on success.
    """
    import subprocess
    obabel = Path(__file__).resolve().parent.parent / ".venvs" / "pred" / "bin" / "obabel"
    if not obabel.exists():
        print(f"  WARNING: obabel not found at {obabel}; nucleic-acid receptor unsupported")
        return False
    try:
        subprocess.run(
            [
                str(obabel), str(pdb_path), "-O", str(output_path),
                "-p", "7.4",                       # protonate at pH 7.4
                "--partialcharge", "gasteiger",
                "-xr",                             # rigid receptor mode
            ],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        print(f"  obabel PDBQT conversion failed: {e}")
        return False
    return output_path.exists()


# AD4 metal atom types + their typical formal charge. AutoDock-GPU and
# AutoDock-Vina ship grid-map types for these out-of-the-box; pdb2pqr
# silently drops them (AMBER FF doesn't know how to assign Gasteiger
# charges to bare metals), so we re-attach them to the PDBQT after
# pdb2pqr finishes — keeps the metal coordinates in the receptor so
# active-site coordination geometry isn't invisible to docking.
_METAL_TYPE_CHARGE = {
    "MG": ("Mg", +2.0), "ZN": ("Zn", +2.0), "CA": ("Ca", +2.0),
    "FE": ("Fe", +3.0), "MN": ("Mn", +2.0), "CU": ("Cu", +2.0),
    "NI": ("Ni", +2.0), "CO": ("Co", +2.0), "CD": ("Cd", +2.0),
    "HG": ("Hg", +2.0), "BA": ("Ba", +2.0), "SR": ("Sr", +2.0),
    "AL": ("Al", +3.0), "K":  ("K",  +1.0), "NA": ("Na", +1.0),
    "CL": ("Cl", -1.0),
}


def _extract_metal_pdbqt_lines(pdb_path: Path) -> list[str]:
    """Pull HETATM metal lines from a PDB file → PDBQT-formatted lines.

    pdb2pqr drops any HETATM whose residue name isn't in its CCD parameter
    file, including bare metals. Without this fallback the metals we
    retained in cif_to_pdb would silently disappear from receptor.pdbqt.

    Returns one PDBQT line per metal atom, ready to splice into the
    PQR-derived PDBQT body. Empty list when the input has no metals.
    """
    out: list[str] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip().upper()
        if resname not in _METAL_TYPE_CHARGE:
            continue
        ad_type, charge = _METAL_TYPE_CHARGE[resname]
        # Re-emit in the same fixed-width PDBQT layout.
        prefix = f"{line[:54]:<54s}"
        out.append(f"{prefix}  0.00  0.00    {charge:+.3f} {ad_type:<2s}")
    return out


def _reattach_dropped_hetatms(pdb_path: Path, pdbqt_lines: list[str],
                              workdir: Path) -> list[str]:
    """PDBQT lines for HETATM residues that pdb2pqr silently discarded.

    ``_extract_metal_pdbqt_lines`` covers bare metals, but pdb2pqr's AMBER
    force field has no parameters for an organic cofactor either — an SFG
    (sinefungin) bound in every L02 receptor vanished between ``receptor.pdb``
    and ``receptor.pdbqt``, so docking treated the occupied cofactor site as
    empty pocket and put fragment poses inside it. Anything the PQR pass lost
    is handed to obabel, which types and charges it properly.
    """
    kept = {(ln[17:20].strip().upper(), ln[21:22], ln[22:26].strip())
            for ln in pdbqt_lines if ln.startswith(("ATOM", "HETATM"))}
    missing: list[str] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        resname = line[17:20].strip().upper()
        if resname in {"HOH", "WAT", "DOD"} or resname in _METAL_TYPE_CHARGE:
            continue
        if (resname, line[21:22], line[22:26].strip()) in kept:
            continue
        missing.append(line)
    if not missing:
        return []

    frag_pdb = workdir / "_dropped_hetatm.pdb"
    frag_pdbqt = workdir / "_dropped_hetatm.pdbqt"
    frag_pdb.write_text("\n".join(missing) + "\nEND\n")
    ok = _pdb_to_pdbqt_obabel(frag_pdb, frag_pdbqt)
    if not ok or not frag_pdbqt.exists():
        print(f"  WARNING: {len(missing)} HETATM line(s) dropped by pdb2pqr could "
              "not be re-typed by obabel — the receptor is missing a cofactor")
        return []
    raw = [ln for ln in frag_pdbqt.read_text().splitlines()
           if ln.startswith(("ATOM", "HETATM"))]
    frag_pdb.unlink(missing_ok=True)
    frag_pdbqt.unlink(missing_ok=True)

    # obabel relabels everything ``UNL 1``. Restore the residue identity so the
    # receptor still says which cofactor this is: input atoms keep their order,
    # and hydrogens obabel added take the label of the atom they are nearest.
    def xyz(ln):
        return (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))

    src_xyz = [xyz(ln) for ln in missing]
    out = []
    for i, ln in enumerate(raw):
        if i < len(missing):
            label = missing[i][17:27]
        else:
            p = xyz(ln)
            j = min(range(len(missing)),
                    key=lambda k: sum((p[d] - src_xyz[k][d]) ** 2 for d in range(3)))
            label = missing[j][17:27]
        out.append(f"HETATM{ln[6:17]}{label}{ln[27:]}")
    names = sorted({ln[17:20].strip() for ln in missing})
    print(f"  Re-attached {len(out)} cofactor atom(s) [{', '.join(names)}] to "
          "receptor PDBQT (pdb2pqr's AMBER FF has no parameters for them)")
    return out


def pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> Path:
    """Convert PDB to PDBQT for receptor: pdb2pqr (protonation + charges) → PDBQT format.

    Metals retained in the input PDB (see ``cif_to_pdb``) are re-attached
    to the PDBQT after pdb2pqr because pdb2pqr's AMBER force field doesn't
    parameterise bare metals — without this fallback the metal coordinates
    would be silently lost between PDB and PDBQT and the docking grid
    maps would never see the active-site coordination geometry.

    For receptors containing nucleic acid (RNA/DNA), the pdb2pqr +
    AMBER path can't parameterize the nucleotide residues — they get
    silently dropped. We auto-detect nucleic acid by residue name and
    route through obabel for the entire receptor. obabel handles both
    protein and nucleotide chemistries plus phosphate backbone charges,
    giving a self-consistent PDBQT in one shot.
    """

    # Auto-route to obabel for nucleic acid receptors. AMBER FF can't
    # parameterize nucleotides so the pdb2pqr path silently drops them.
    if _has_nucleic_acid(pdb_path):
        print("  Nucleic acid detected in receptor → routing through obabel "
              "(pdb2pqr's AMBER FF doesn't cover RNA/DNA)")
        if _pdb_to_pdbqt_obabel(pdb_path, output_path):
            print(f"  Receptor PDBQT (obabel): {output_path}")
            return output_path
        # If obabel failed, fall through to pdb2pqr — last-resort attempt
        # so a misconfigured obabel doesn't strand the whole pipeline.
        print("  obabel path failed; trying pdb2pqr (likely to drop nucleic chains)")

    # Step 1: pdb2pqr adds hydrogens and assigns charges
    pqr_path = output_path.with_suffix(".pqr")
    run_pdb2pqr(pdb_path, pqr_path)

    # Step 2: PQR → PDBQT (PQR has charges in the occupancy/bfactor columns)
    AD_TYPE_MAP = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
        "F": "F", "P": "P", "CL": "Cl", "BR": "Br", "I": "I",
    }
    pqr_lines = pqr_path.read_text().splitlines()
    pdbqt_lines = []
    for line in pqr_lines:
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith(("TER", "END")):
                pdbqt_lines.append(line)
            continue
        # PQR format: cols 55-62 = charge, 63-70 = radius
        atom_name = line[12:16].strip()
        element = atom_name.lstrip("0123456789")[0:2].upper().strip()
        if len(element) > 1 and element not in ("CL", "BR"):
            element = element[0]
        ad_type = AD_TYPE_MAP.get(element, element)

        # Extract charge from PQR (columns vary, parse from end)
        parts = line.split()
        try:
            charge = float(parts[-2])  # second to last = charge
        except (ValueError, IndexError):
            charge = 0.0

        # Build PDBQT line: first 54 chars from PQR + reformatted tail
        pdb_prefix = f"{line[:54]:<54s}"
        pdbqt_lines.append(f"{pdb_prefix}  0.00  0.00    {charge:+.3f} {ad_type:<2s}")

    # Step 3: re-attach metal HETATMs that pdb2pqr dropped (see helper for
    # rationale). Insert before the trailing END so they live in the same
    # ATOM block downstream tools iterate.
    extra = _extract_metal_pdbqt_lines(pdb_path)
    if extra:
        print(f"  Re-attached {len(extra)} metal HETATM(s) to receptor PDBQT "
              f"(pdb2pqr drops bare metals)")
    # Step 3b: same problem, non-metal cofactors — an organic HETATM the AMBER
    # FF doesn't cover is dropped just as silently.
    extra += _reattach_dropped_hetatms(pdb_path, pdbqt_lines, output_path.parent)
    if extra:
        # Splice before END if present, else just append.
        end_idx = next((i for i, line in enumerate(pdbqt_lines)
                        if line.strip().startswith("END")), len(pdbqt_lines))
        pdbqt_lines = pdbqt_lines[:end_idx] + extra + pdbqt_lines[end_idx:]

    output_path.write_text("\n".join(pdbqt_lines) + "\n")
    print(f"  Receptor PDBQT: {output_path} (protonated, Gasteiger charges)")
    return output_path


def find_best_cofolding_structure(cofolding_dir: Path, model: str) -> Path | None:
    """Find the best-ranked structure from cofolding output.

    Returns ``*_aligned.cif`` when present (Stage 2.5 alignment output) so
    receptor coords match every other coordinate consumer in the pipeline
    (binding-site predictors, template-pocket clusters, post-analysis,
    submission). Falls back to the raw cif with a loud warning — running
    with an unaligned receptor while template_consensus_* / docking poses
    are in the aligned frame produces silent geometric mis-docking.
    """
    def _prefer_aligned(cifs: list[Path]) -> Path | None:
        for cif in cifs:
            aligned = cif.with_name(cif.stem + "_aligned.cif")
            if aligned.exists():
                return aligned
        if cifs:
            print(f"  WARNING: no ``*_aligned.cif`` found for {model} under "
                  f"{cofolding_dir}; falling back to {cifs[0].name} (UNALIGNED). "
                  f"Run scripts/align_cofolding_outputs.py first to keep "
                  f"every downstream stage in a common frame.")
            return cifs[0]
        return None

    if model.startswith("boltz"):
        return _prefer_aligned(sorted(cofolding_dir.rglob("predictions/**/*.cif")))
    elif model == "protenix":
        raw = [c for c in sorted(cofolding_dir.rglob("*.cif")) if "_aligned" not in c.name]
        return _prefer_aligned(raw)
    elif model == "alphafold3":
        raw = [c for c in sorted(cofolding_dir.rglob("*model*.cif")) if "_aligned" not in c.name]
        return _prefer_aligned(raw)

    raw = [c for c in sorted(cofolding_dir.rglob("*.cif")) if "_aligned" not in c.name]
    result = _prefer_aligned(raw)
    if result:
        return result
    for pdb in sorted(cofolding_dir.rglob("*.pdb")):
        return pdb
    return None


def read_confidence_score(cofolding_dir: Path, model: str) -> float:
    """Read mean per-residue/atom pLDDT, normalised to the ``[0, 100]`` scale.

    Each cofolding backend stores pLDDT differently:
      * **Boltz-2 / Boltz-2x** write a per-residue array in ``plddt_*.npz`` on
        the ``[0, 1]`` scale. We multiply by 100 so it is comparable with AF3.
      * **Protenix v2** writes a scalar mean pLDDT already on ``[0, 100]``
        inside the per-sample ``confidences.json`` (``"plddt": <float>``).
        The earlier version of this function called ``sum(float)`` on that
        scalar and fell through to the exception handler, returning ``-1`` —
        the bug that made Protenix invisible to the auto-selector.
      * **AlphaFold3** writes a ``"atom_plddts"`` list already on ``[0, 100]``.
    """
    try:
        if model.startswith("boltz"):
            import numpy as np
            for npz in cofolding_dir.rglob("plddt_*model_0.npz"):
                data = np.load(str(npz))
                return float(data[data.files[0]].mean()) * 100.0
        elif model == "protenix":
            for json_f in cofolding_dir.rglob("*confidence*.json"):
                data = json.loads(json_f.read_text())
                if "plddt" in data:
                    val = data["plddt"]
                    if isinstance(val, (list, tuple)) and val:
                        return float(sum(val) / len(val))
                    if isinstance(val, (int, float)):
                        return float(val)
        elif model == "alphafold3":
            for json_f in cofolding_dir.rglob("*confidence*.json"):
                if "summary" in json_f.name:
                    continue
                data = json.loads(json_f.read_text())
                if "atom_plddts" in data and data["atom_plddts"]:
                    return float(sum(data["atom_plddts"]) / len(data["atom_plddts"]))
    except Exception:
        pass
    return -1.0


def select_best_model(output_root: Path) -> tuple[str, Path]:
    """Auto-select best cofolding model by confidence score."""
    candidates = []
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = output_root / "outputs" / model
        if not model_dir.exists():
            continue
        structure = find_best_cofolding_structure(model_dir, model)
        if structure is None:
            continue
        score = read_confidence_score(model_dir, model)
        candidates.append((model, model_dir, structure, score))
        print(f"  Model {model}: pLDDT={score:.1f}, structure={structure.name}")

    if not candidates:
        return ("", Path())

    # Sort by confidence score descending, pick best
    candidates.sort(key=lambda x: x[3], reverse=True)
    best = candidates[0]
    print(f"  Selected: {best[0]} (pLDDT={best[3]:.1f})")
    return (best[0], best[1])


POCKET_PREDICTOR_TOP_K = 3  # how many top-ranked pockets each ML predictor exposes


def _parse_p2rank_predictions(pred_file: Path, top_k: int = POCKET_PREDICTOR_TOP_K) -> list[dict]:
    """Parse P2Rank predictions CSV → top-K pockets sorted by score.

    Returns a list of ``{"center": [x,y,z], "size": [22.5]*3, "score": float, "rank": int}``
    capped at ``top_k``. Empty list when the CSV has no pockets.
    """
    import csv
    out = []
    with open(pred_file) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for i, row in enumerate(reader, start=1):
            if i > top_k:
                break
            row = {k.strip(): v.strip() for k, v in row.items()}
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            cz = float(row["center_z"])
            score = float(row.get("score") or 0.0)
            out.append({
                "center": [cx, cy, cz],
                "size": [22.5, 22.5, 22.5],
                "score": score,
                "rank": i,
            })
            print(f"  P2Rank pocket {i}: center=[{cx:.1f}, {cy:.1f}, {cz:.1f}], score={score:.3f}")
    return out


def run_p2rank(
    pdb_path: Path,
    output_dir: Path,
    reuse_cache: bool = False,
) -> list[dict]:
    """Run P2Rank binding site prediction and return top-K pockets.

    Multi-chain receptors expose distinct binding sites on different chains
    so we no longer collapse the prediction to a single best pocket — each
    of the top-K (default 3) becomes a separate ``p2rank_<rank>`` source.

    When ``reuse_cache`` is True and a previous predictions CSV exists, the
    binary is skipped and the cached CSV is parsed directly. Safe across
    reruns because P2Rank is deterministic for a given receptor PDB.

    Returns an empty list on failure / missing prediction.
    """
    import subprocess

    prank_bin = Path(__file__).resolve().parent.parent / ".local" / "bin" / "prank"
    if not prank_bin.exists():
        print("  P2Rank not found, skipping binding site prediction.")
        return []

    p2rank_out = output_dir / "p2rank"
    pred_file = p2rank_out / f"{pdb_path.name}_predictions.csv"

    if reuse_cache and pred_file.exists():
        print(f"  P2Rank cache hit ({pred_file.name}) — skipping prediction")
        return _parse_p2rank_predictions(pred_file)

    try:
        subprocess.run(
            [str(prank_bin), "predict", "-f", str(pdb_path), "-o", str(p2rank_out)],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  P2Rank failed: {e}")
        return []

    if not pred_file.exists():
        print("  P2Rank produced no predictions file.")
        return []
    return _parse_p2rank_predictions(pred_file)


def run_swinsite(
    pdb_path: Path,
    output_dir: Path,
    reuse_cache: bool = False,
) -> list[dict]:
    """Run SwinSite binding site prediction and return top-K pockets.

    Multi-chain receptors expose distinct binding sites; SwinSite's
    grid*_score_*.pdb output already ranks every pocket it finds, so we
    expose the top-K (default 3) as ``swinsite_<rank>`` sources instead
    of collapsing to a single best.

    When ``reuse_cache`` is True and the SwinSite results dir from a prior
    run is on disk, the GPU prediction is skipped and the cached pocket
    PDBs are parsed directly. Safe across reruns because SwinSite output
    is deterministic for a given receptor PDB.

    Returns an empty list on failure / missing pockets.
    """
    repo_root = Path(__file__).resolve().parent.parent
    swinsite_dir = repo_root / "external" / "swinsite"
    pred_python = repo_root / ".venvs" / "pred" / "bin" / "python"

    if not swinsite_dir.exists() or not pred_python.exists():
        print("  SwinSite not found, skipping.")
        return []

    swinsite_out = output_dir / "swinsite"
    cached_results = swinsite_out / "results" / "input" / "receptor"

    if reuse_cache and cached_results.exists() and any(cached_results.glob("*_score_*.pdb")):
        print(f"  SwinSite cache hit ({cached_results}) — skipping GPU prediction")
        return _parse_swinsite_results(cached_results)

    # SwinSite expects input_dir/<sample_name>/protein.pdb
    input_dir = swinsite_out / "input"
    sample_dir = input_dir / "receptor"
    sample_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(pdb_path), str(sample_dir / "protein.pdb"))

    try:
        import subprocess as _sp
        _sp.run(
            [str(pred_python), str(swinsite_dir / "predict.py"),
             "-i", str(input_dir), "-f", "pdb", "-of", "pdb",
             "-o", str(swinsite_out / "results"),
             "-l", str(swinsite_out / "log.txt"),
             "-m",
             str(swinsite_dir / "model/fold_1/best_epoch.h5"),
             str(swinsite_dir / "model/fold_2/best_epoch.h5"),
             str(swinsite_dir / "model/fold_3/best_epoch.h5"),
             str(swinsite_dir / "model/fold_4/best_epoch.h5"),
            ],
            check=True, capture_output=True, text=True, timeout=300,
        )
    except Exception as e:
        print(f"  SwinSite failed: {e}")
        return []

    results_dir = swinsite_out / "results" / "input" / "receptor"
    if not results_dir.exists():
        print("  SwinSite produced no output.")
        return []
    return _parse_swinsite_results(results_dir)


def _parse_swinsite_results(
    results_dir: Path,
    top_k: int = POCKET_PREDICTOR_TOP_K,
) -> list[dict]:
    """Parse a SwinSite results directory → top-K pockets ranked by score.

    Layout (observed in runs/22mj_input/...):
        results_dir/
            grid0_score_0.7219.pdb     ← HETATM UNL grid points (pocket volume)
            grid1_score_0.3136.pdb
            pocket0_score_0.7219.pdb   ← protein residues near the pocket
            pocket1_score_0.3136.pdb

    ``grid*`` files give the cleanest pocket centroid; falls back to
    ``pocket*`` files if grids are missing. Returns up to ``top_k`` entries
    sorted by score descending. Each entry::

        {"center": [x,y,z], "size": [22.5]*3, "score": float, "rank": int}
    """
    import re as _re
    score_re = _re.compile(r"_score_([0-9.]+)\.pdb$")

    def _score(path: Path) -> float:
        m = score_re.search(path.name)
        try:
            return float(m.group(1)) if m else -1.0
        except ValueError:
            return -1.0

    grid_files = sorted(results_dir.glob("grid*_score_*.pdb"), key=_score, reverse=True)
    pocket_files = sorted(results_dir.glob("pocket*_score_*.pdb"), key=_score, reverse=True)
    candidates = grid_files or pocket_files

    if not candidates:
        print("  SwinSite found no pockets.")
        return []

    import numpy as _np
    out: list[dict] = []
    for rank, pdb_file in enumerate(candidates[:top_k], start=1):
        coords = []
        for line in pdb_file.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    continue
        if not coords:
            continue
        arr = _np.array(coords)
        center = arr.mean(axis=0).tolist()
        score = _score(pdb_file)
        out.append({
            "center": center,
            "size": [22.5, 22.5, 22.5],
            "score": float(score),
            "rank": rank,
        })
        print(f"  SwinSite pocket {rank} {pdb_file.name}: center=[{center[0]:.1f}, "
              f"{center[1]:.1f}, {center[2]:.1f}], atoms={len(coords)}, score={score:.3f}")
    return out


def _load_receptor_chain_atoms(receptor_pdb: Path) -> list[tuple[str, list[float]]]:
    """Parse receptor PDB → list of (chain_id, [x,y,z]) for every CA atom.

    Used to attach a ``nearest_protein_chain`` tag to each binding-site
    centroid so multi-chain post-analysis can distinguish a pocket on
    chain A from a chemically-identical pocket on chain B.
    """
    out = []
    if not receptor_pdb.exists():
        return out
    for line in receptor_pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        chain = line[21:22].strip() or "?"
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        out.append((chain, [x, y, z]))
    return out


def _nearest_chain(point: list[float], ca_atoms: list[tuple[str, list[float]]]) -> tuple[str | None, float]:
    """Return (chain_id, distance) of the nearest CA atom to ``point``.

    Returns ``(None, inf)`` when there are no atoms to compare against.
    """
    import math
    best_chain = None
    best_d2 = math.inf
    for chain, xyz in ca_atoms:
        d2 = ((point[0] - xyz[0]) ** 2
              + (point[1] - xyz[1]) ** 2
              + (point[2] - xyz[2]) ** 2)
        if d2 < best_d2:
            best_d2 = d2
            best_chain = chain
    return best_chain, math.sqrt(best_d2) if best_d2 != math.inf else float("inf")


def extract_smiles_from_yaml(input_yaml: Path) -> list[tuple[str, str]]:
    """Extract ``(ligand_id, smiles)`` pairs from the unified input YAML.

    Uses ``yaml.safe_load`` so block-style nested mappings parse correctly
    and multi-ligand inputs (candidate + cofactors + metals) preserve their
    declared order. Ligand entries that carry only a ``ccd`` field
    (metals/ions addressed by CCD code) are skipped — docking can only
    consume real SMILES.
    """
    try:
        import yaml
    except Exception:
        return []
    try:
        data = yaml.safe_load(input_yaml.read_text()) or {}
    except Exception:
        return []
    results: list[tuple[str, str]] = []
    for entry in data.get("sequences", []) or []:
        if not isinstance(entry, dict) or "ligand" not in entry:
            continue
        lig = entry["ligand"] or {}
        smi = lig.get("smiles")
        if not smi:
            continue  # CCD-only ligand (metal/ion) — not dockable
        # A ligand present in several copies declares a *list* of chain ids.
        # ``str()`` on that produced filenames like ``ligand_['L1', 'L2'].sdf``
        # — a Python repr on disk, which nothing downstream could open. Dock
        # one copy, under the first id.
        lid = _yaml_ligand_ids(lig)[0]
        results.append((lid, str(smi).strip().strip("'\"")))
    return results


def _yaml_ligand_ids(lig: dict) -> list[str]:
    """Every chain id a YAML ligand entry declares, copies included."""
    raw = lig.get("id") or "L"
    ids = raw if isinstance(raw, list) else [raw]
    return [str(i).strip() for i in ids if str(i).strip()] or ["L"]


def extract_smiles_from_json(input_json: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, smiles) from Protenix/AF3 JSON input."""
    data = json.loads(input_json.read_text())
    if isinstance(data, list):
        data = data[0]

    results = []
    for seq in data.get("sequences", []):
        if "ligand" in seq:
            lig = seq["ligand"]
            smiles = lig.get("smiles") or lig.get("ligand", "")
            lig_id = lig.get("id", "L")
            if isinstance(lig_id, list):
                lig_id = lig_id[0]
            if smiles:
                results.append((str(lig_id), smiles))
    return results


def _extract_cofolding_ligand_centroid(
    cif_path: Path,
    dockable_chains: set[str],
) -> list[float] | None:
    """Extract the heavy-atom centroid of the **dockable ligand** in a cofolding CIF.

    Cofold outputs may contain non-polymer entities besides the dockable
    target (metals, ions, cofactors that we kept on the receptor). Averaging
    all non-polymers gives a wrong centroid on metal-bearing or multi-ligand
    targets — the cluster centre then biases docking boxes toward the
    geometric midpoint of "ligand + Mg²⁺ + cofactor" instead of the real
    binding pose. Restricting the average to the YAML's dockable ligand
    chain ids fixes that.

    Args:
        cif_path: cofold CIF.
        dockable_chains: chain ids declared as ligands (with SMILES) in the
            input. Empty set returns ``None`` (no SMILES ligand to dock).

    Returns ``[x, y, z]`` or ``None`` when no qualifying atoms are found.
    """
    if not dockable_chains:
        return None
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        coords: list[tuple[float, float, float]] = []
        for model in st:
            for chain in model:
                if chain.name not in dockable_chains:
                    continue
                for res in chain:
                    # LIG-prefixed residue name kept as a backstop for cofold
                    # outputs that don't pin entity_type properly (rare).
                    is_ligand = (
                        res.entity_type == gemmi.EntityType.NonPolymer
                        or res.name.startswith("LIG")
                    )
                    if not is_ligand:
                        continue
                    for atom in res:
                        if atom.element.is_hydrogen:
                            continue
                        coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
            break  # first model only
        if not coords:
            return None
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        return [round(cx, 4), round(cy, 4), round(cz, 4)]
    except Exception as e:
        print(f"  WARNING: could not extract cofolding ligand centroid: {e}")
        return None


# Cofold-ligand cluster knobs (Stage 4 binding-site source registration). All
# four models × 5 seeds × 5 samples = 100 placements live in the same frame
# after Stage 2.5 alignment, so a greedy first-match centroid cluster on
# their positions captures multi-pocket / inter-model disagreement signal
# that the old "best-model single-centroid" extraction discarded.
COFOLD_CLUSTER_CUTOFF = 5.0       # Å — same as template-pocket cluster
# MIN_MEMBERS scales with the actual placement count instead of being
# hardcoded for a 100-placement default (4 models × 5 seeds × 5 samples).
# Targets that change ``cofolding_seeds`` or ``diffusion_samples`` previously
# kept the same absolute floor → tiny pools (e.g. 25 placements with single
# seed) lost legitimate clusters. Now: max(2, n_placements // 20) → ~5 % of
# the pool, with an absolute minimum of 2 to guarantee multi-placement
# evidence (singleton placements are still rejected).
COFOLD_CLUSTER_MIN_FRACTION = 0.05
COFOLD_CLUSTER_MIN_FLOOR = 2
COFOLD_CLUSTER_TOP_K = 3


def _extract_cofolding_ligand_clusters(
    run_dir: Path,
    dockable_chains: set[str],
) -> list[dict]:
    """Cluster ligand centroids across every aligned cofolding CIF.

    Iterates ``outputs/{boltz2,boltz2x,protenix,alphafold3}/**/*_aligned.cif``
    (typically 4 models × 25 seeds = 100 placements, all in the same
    coordinate frame after ``align_cofolding_outputs.py``), pulls the
    heavy-atom ligand centroid from each (restricted to the YAML's dockable
    ligand chains via ``dockable_chains`` so metals/cofactors don't bias
    the average), and runs greedy first-match centroid clustering with a
    ``COFOLD_CLUSTER_CUTOFF`` Å cutoff.

    Returns the top-``COFOLD_CLUSTER_TOP_K`` clusters with at least
    ``COFOLD_CLUSTER_MIN_MEMBERS`` placements, sorted by ``n_members``
    descending. Each entry::

        {"centroid": [x, y, z],
         "n_members": int,
         "n_unique_models": int,
         "models": [str, ...]}

    Well-converged targets collapse all 100 placements into a single cluster
    → returns one entry (matches the legacy single-centroid behaviour). Multi-
    pocket / inter-model-disagreement targets get 2–3 cluster centroids → each
    becomes its own ``cofolding_N`` binding-site source so docking covers every
    pocket the cofold ensemble agrees on.

    The MIN_MEMBERS filter drops single-placement strays (alternate / spurious
    sites). Real binding pockets typically attract ≥ 5 of 100 placements; the
    threshold is conservative because any signal below that is unlikely to
    survive even one round of seed/sample re-rolling.
    """
    placements: list[tuple[list[float], str]] = []
    for model in ("boltz2", "boltz2x", "protenix", "alphafold3"):
        model_dir = run_dir / "outputs" / model
        if not model_dir.exists():
            continue
        for cif in sorted(model_dir.rglob("*_aligned.cif")):
            c = _extract_cofolding_ligand_centroid(cif, dockable_chains=dockable_chains)
            if c is not None:
                placements.append((c, model))

    # Dynamic min-members: 5 % of placements (floor 2). This degrades
    # gracefully when the user bumps cofold seeds × samples up or down.
    n_placements = len(placements)
    min_members = max(
        COFOLD_CLUSTER_MIN_FLOOR,
        int(round(n_placements * COFOLD_CLUSTER_MIN_FRACTION)),
    )

    if not placements:
        return []

    cutoff_sq = COFOLD_CLUSTER_CUTOFF ** 2
    clusters: list[dict] = []
    for centroid, model in placements:
        joined = False
        for cl in clusters:
            cx, cy, cz = cl["centroid"]
            d2 = (centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2 + (centroid[2] - cz) ** 2
            if d2 <= cutoff_sq:
                cl["points"].append(centroid)
                cl["models"].append(model)
                n = len(cl["points"])
                cl["centroid"] = [
                    sum(p[0] for p in cl["points"]) / n,
                    sum(p[1] for p in cl["points"]) / n,
                    sum(p[2] for p in cl["points"]) / n,
                ]
                joined = True
                break
        if not joined:
            clusters.append({"centroid": list(centroid),
                             "points": [list(centroid)],
                             "models": [model]})

    out = []
    for cl in clusters:
        if len(cl["points"]) < min_members:
            continue
        out.append({
            "centroid": [round(x, 4) for x in cl["centroid"]],
            "n_members": len(cl["points"]),
            "n_unique_models": len(set(cl["models"])),
            "models": sorted(set(cl["models"])),
        })
    out.sort(key=lambda d: -d["n_members"])
    out = out[:COFOLD_CLUSTER_TOP_K]
    diagnostics = {"n_placements": n_placements, "min_members": min_members}
    return out, diagnostics


def summary_pred_iter(binding_site_results):
    """Yield ``(name, (center, size, [metadata]))`` from the in-memory
    ``binding_site_results`` dict, normalising the optional metadata slot
    so the frame-check loop above can read ``payload[0]`` (center)
    uniformly. Standalone helper so the prep main flow stays readable."""
    for k, v in binding_site_results.items():
        yield k, v if isinstance(v, tuple) else (v.get("center"), v.get("size"))


TEMPLATE_CONSENSUS_TOP_K = 10
TEMPLATE_CONSENSUS_MIN_MEMBERS = 2
TEMPLATE_CONSENSUS_BOX_SIZE = [22.5, 22.5, 22.5]  # legacy default; runtime override via _adaptive_box_size below


def _adaptive_box_size(
    ligands: list[dict],
    *,
    base: float = 22.5,
    padding: float = 8.0,
    cap: float = 40.0,
) -> list[float]:
    """Pick a docking box edge that comfortably wraps every dockable ligand.

    For each ligand SDF we compute the max coordinate extent across XYZ,
    take the largest across all ligands, and add ``padding`` Å margin.
    Floored at ``base`` (legacy 22.5 Å so typical druglike behaviour
    stays unchanged), capped at ``cap`` (40 Å — beyond this Vina/ADG
    grid resolution starts to hurt search quality).

    Returns a 3-list usable as ``box_size`` everywhere the prep summary
    is consumed.
    """
    try:
        from rdkit import Chem
    except Exception:
        return [base, base, base]
    max_extent = 0.0
    for lig in ligands:
        sdf = lig.get("sdf") if isinstance(lig, dict) else None
        if not sdf:
            continue
        try:
            supp = Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=False)
            mol = next(iter(supp), None)
            if mol is None or mol.GetNumConformers() == 0:
                continue
            conf = mol.GetConformer()
            xs, ys, zs = [], [], []
            for i in range(mol.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                xs.append(p.x); ys.append(p.y); zs.append(p.z)
            if not xs:
                continue
            extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            if extent > max_extent:
                max_extent = extent
        except Exception:
            continue
    edge = max(base, min(cap, max_extent + padding))
    return [round(edge, 2)] * 3


def _add_template_consensus_sources(
    run_dir: Path | None,
    binding_site_results: dict,
) -> list[str]:
    """Read template_pocket_clusters.json (if produced upstream) and inject the
    top-K consensus pocket centroids as ``template_consensus_N`` entries in
    ``binding_site_results``. Each entry is ``(center, size, metadata)``.

    Single-template clusters (``n_members < TEMPLATE_CONSENSUS_MIN_MEMBERS``)
    are skipped because a binding site supported by only one structure has
    no consensus signal and is more likely to be an alternate/spurious site
    than the target's actual pocket. Real target binding sites typically
    show up in many of the same-fold templates and dominate the
    ``n_members`` ranking.

    Returns the list of source names that were added (in priority order).
    Silently no-ops when the cluster JSON is missing — keeps the script
    backward compatible with runs that did not enable template search.
    """
    if run_dir is None:
        return []
    cluster_json = run_dir / "outputs" / "template_pockets" / "template_pocket_clusters.json"
    if not cluster_json.exists():
        return []
    try:
        data = json.loads(cluster_json.read_text())
    except Exception as e:
        print(f"  WARNING: could not parse {cluster_json}: {e}")
        return []
    clusters = data.get("clusters") or []
    added: list[str] = []
    rank = 0
    for cl in clusters:
        centroid = cl.get("centroid")
        if not centroid or len(centroid) != 3:
            continue
        n_members = int(cl.get("n_members") or 0)
        if n_members < TEMPLATE_CONSENSUS_MIN_MEMBERS:
            continue
        rank += 1
        if rank > TEMPLATE_CONSENSUS_TOP_K:
            break
        src_name = f"template_consensus_{rank}"
        meta = {
            "n_members": cl.get("n_members"),
            "n_unique_pdb": cl.get("n_unique_pdb"),
            "evidence_score": cl.get("evidence_score"),
            "spread_angstrom": cl.get("spread_angstrom"),
            "in_both_sources": cl.get("in_both_sources"),
            "in_mmseqs_only": cl.get("in_mmseqs_only"),
            "in_foldseek_only": cl.get("in_foldseek_only"),
            "best_alignment_tmscore": cl.get("best_alignment_tmscore"),
            "best_qtmscore": cl.get("best_qtmscore"),
            "best_pident": cl.get("best_pident"),
        }
        binding_site_results[src_name] = (
            list(centroid),
            list(TEMPLATE_CONSENSUS_BOX_SIZE),
            meta,
        )
        added.append(src_name)
    return added


def compute_box_from_ligand(sdf_path: Path, box_side: float = 22.5) -> tuple[list[float], list[float]]:
    """Compute docking box center from ligand 3D coordinates. Box size fixed."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next(supplier)
    if mol is None:
        return [0.0, 0.0, 0.0], [box_side, box_side, box_side]

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    min_xyz = positions.min(axis=0)
    max_xyz = positions.max(axis=0)
    center = ((min_xyz + max_xyz) / 2).tolist()

    return center, [box_side, box_side, box_side]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare docking inputs from unified input + cofolding output.")
    parser.add_argument("--input-yaml", type=Path, help="Unified input YAML (Boltz format).")
    parser.add_argument("--input-json", type=Path, help="Protenix/AF3 input JSON (alternative to YAML).")
    parser.add_argument("--cofolding-dir", type=Path, help="Cofolding output directory (single model).")
    parser.add_argument("--run-dir", type=Path, help="Run directory (auto-selects best model from outputs/).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for docking inputs.")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "boltz", "protenix", "alphafold3"],
                        help="Which cofolding model to use. 'auto' selects by confidence score.")
    parser.add_argument("--no-p2rank", action="store_true",
                        help="Disable P2Rank binding site prediction (default: enabled).")
    parser.add_argument("--reuse-binding-site-cache", action="store_true",
                        help="Reuse existing P2Rank/SwinSite outputs from a prior run "
                             "instead of re-invoking the binaries. Lets the prep step "
                             "run on a CPU-only node when the cache was already "
                             "populated by an earlier GPU run. Cache is keyed implicitly "
                             "by path; safe across reruns since receptor.pdb is "
                             "deterministic from _aligned.cif.")
    args = parser.parse_args()

    # Force absolute. Every receptor/ligand path the prep summary emits
    # is derived from args.output_dir, and downstream consumers (notably
    # the ADG runner, which invokes autogrid with cwd=lig_grid_dir) cannot
    # resolve relative paths from their own cwd. Past silent-fail across
    # the 499-target batch on 2026-05-09 traced to this.
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract SMILES
    if args.input_yaml and args.input_yaml.exists():
        ligands = extract_smiles_from_yaml(args.input_yaml)
    elif args.input_json and args.input_json.exists():
        ligands = extract_smiles_from_json(args.input_json)
    else:
        print("  ERROR: No input file found.")
        return 1

    if not ligands:
        print("  No ligands with SMILES found, skipping ligand preparation.")
        return 0

    # 2. Convert SMILES → SDF → PDBQT for ALL ligands. Per-ligand failures
    # (e.g. RDKit cannot embed a polyisoprenoid chain, meeko chokes on an
    # exotic valence) are logged and skipped so the pipeline still reaches
    # receptor prep + summary. ``prepared_ligands`` records which ones
    # actually have usable SDF/PDBQT files for the summary JSON.
    prepared_ligands: list[tuple[str, str, Path | None, Path | None]] = []
    for lig_id, smiles in ligands:
        print(f"  Preparing ligand {lig_id}: {smiles}")
        try:
            sdf_path = smiles_to_sdf(smiles, args.output_dir / f"ligand_{lig_id}.sdf", name=lig_id)
        except Exception as e:
            print(f"  WARNING: SDF generation crashed for {lig_id}: {e}")
            sdf_path = None
        try:
            pdbqt_path = sdf_to_pdbqt(sdf_path, args.output_dir / f"ligand_{lig_id}.pdbqt")
        except Exception as e:
            print(f"  WARNING: PDBQT generation crashed for {lig_id}: {e}")
            pdbqt_path = None
        prepared_ligands.append((lig_id, smiles, sdf_path, pdbqt_path))

    usable = [pl for pl in prepared_ligands if pl[2] is not None or pl[3] is not None]
    if not usable:
        print(
            f"  WARNING: 0/{len(prepared_ligands)} ligands produced usable SDF/PDBQT. "
            "All subsequent docking will be skipped by the docking runners; "
            "submission will fall back to the cofolded ligand pose if available."
        )
    elif len(usable) < len(prepared_ligands):
        print(
            f"  {len(usable)}/{len(prepared_ligands)} ligands prepared successfully "
            f"({len(prepared_ligands) - len(usable)} failed SDF/PDBQT)."
        )

    # 3. Select best cofolding structure
    cofolding_dir = args.cofolding_dir
    model = args.model

    if model == "auto" and args.run_dir:
        # Auto-select by comparing confidence scores across all models
        print("  Auto-selecting best cofolding model...")
        model, cofolding_dir = select_best_model(args.run_dir)
        if not model:
            print("  ERROR: No cofolding outputs found for auto-selection.")
            return 1
    elif model == "auto" and cofolding_dir:
        # Single dir provided, guess model from path
        dirname = cofolding_dir.name
        model = dirname if dirname in ("boltz", "protenix", "alphafold3") else "boltz"

    structure = find_best_cofolding_structure(cofolding_dir, model)
    if structure is None:
        print(f"  WARNING: No cofolding structure found in {cofolding_dir}")
        return 0

    print(f"  Using cofolding structure: {structure}")
    # Dockable ligand chain ids = whatever the input YAML calls out as
    # ligand entries. We strip these from the cofold cif; metals / other
    # cofactors stay so docking sees the active-site coordination.
    dockable_chains: set[str] = set()
    if args.input_yaml and args.input_yaml.exists():
        try:
            import yaml as _yaml
            data = _yaml.safe_load(args.input_yaml.read_text()) or {}
            for entry in data.get("sequences", []) or []:
                if isinstance(entry, dict) and "ligand" in entry:
                    lig = entry["ligand"] or {}
                    # Dockable ligand = entry has SMILES (vs ccd-only entries
                    # like metals which we want to keep on the receptor).
                    if lig.get("smiles") and lig.get("id"):
                        # Every copy has to be stripped, not just the first:
                        # a leftover copy stays baked into the receptor and
                        # the docking box then contains its own answer.
                        dockable_chains.update(_yaml_ligand_ids(lig))
        except Exception as e:
            print(f"  WARNING: could not parse dockable chain ids from YAML: {e}")
    elif args.input_json and args.input_json.exists():
        # Protenix/AF3 JSON-only input — same dockable filter (SMILES ligand)
        # so the cofold cluster centroid average doesn't drift into metals/
        # cofactors. Without this the legacy "any non-polymer" path would fire.
        for lig_id, _smi in extract_smiles_from_json(args.input_json):
            dockable_chains.add(lig_id)

    # AF3 rewrites multi-character chain ids (``L2``, ``X2``) to alphabet
    # letters (``B``, ``C``) because AF3 only accepts upper-case letter ids.
    # When the auto-selected best cofolding model is AF3, the cif on disk
    # uses the *new* ids — so a dockable_chains set sourced from the YAML
    # would miss the renamed ligand and cif_to_pdb would leave it baked
    # into the receptor. Union in the AF3 mapping when applicable.
    if model == "alphafold3" and args.run_dir is not None:
        af3_remap_path = args.run_dir / "inputs" / "alphafold3_chain_remap.json"
        if af3_remap_path.exists():
            try:
                af3_remap = json.loads(af3_remap_path.read_text()).get("yaml_to_af3") or {}
                added = {af3_remap[c] for c in list(dockable_chains) if c in af3_remap}
                added -= dockable_chains
                if added:
                    dockable_chains.update(added)
                    print(f"  AF3 chain remap added dockable ids: {sorted(added)}")
            except Exception as e:
                print(f"  WARNING: could not load AF3 chain remap: {e}")

    if dockable_chains:
        print(f"  Dockable ligand chains (will be stripped from receptor): "
              f"{sorted(dockable_chains)}")

    if structure.suffix in (".cif", ".mmcif"):
        pdb_path = cif_to_pdb(structure, args.output_dir / "receptor.pdb",
                              dockable_ligand_chains=dockable_chains or None)
    else:
        pdb_path = structure

    pdb_to_pdbqt(pdb_path, args.output_dir / "receptor.pdbqt")

    # Also generate protonated PDB for Protenix-Dock (HIS→HID/HIE/HIP)
    pqr_path = args.output_dir / "receptor.pqr"
    if not pqr_path.exists():
        run_pdb2pqr(pdb_path, pqr_path)
    pqr_to_protonated_pdb(pqr_path, args.output_dir / "receptor_protonated.pdb")
    pqr_path.unlink(missing_ok=True)

    # Per-ligand "ligand-aware" receptors for multi-ligand targets.
    # Each per-ligand receptor strips ONLY that ligand from the cofold cif,
    # leaving every other dockable ligand's cofold pose embedded as rigid
    # heavy atoms. Docking ligand i against this receptor sees the other
    # ligands' cofold poses as part of the binding site → site competition
    # is captured automatically (e.g. two ligands at adjacent / overlapping
    # pockets get separated rather than docked into the same spot).
    # Single-ligand targets skip this; the legacy receptor.pdbqt is reused.
    per_ligand_receptors: dict[str, dict[str, str]] = {}
    if structure.suffix in (".cif", ".mmcif") and len(dockable_chains) > 1:
        print(f"  Multi-ligand target ({len(dockable_chains)} dockable chains) → "
              f"building per-ligand receptors with other cofold ligands embedded")
        for chain_id in sorted(dockable_chains):
            target_pdb = args.output_dir / f"receptor_excl_{chain_id}.pdb"
            target_pdbqt = args.output_dir / f"receptor_excl_{chain_id}.pdbqt"
            try:
                # Strip only this chain — other dockable ligand chains stay
                # in the receptor as static (cofold-placed) heavy atoms.
                cif_to_pdb(structure, target_pdb, dockable_ligand_chains={chain_id})
                pdb_to_pdbqt(target_pdb, target_pdbqt)
                per_ligand_receptors[chain_id] = {
                    "receptor_pdb": str(target_pdb),
                    "receptor_pdbqt": str(target_pdbqt),
                }
            except Exception as e:
                print(f"  WARNING: per-ligand receptor for {chain_id} failed ({e}); "
                      f"falling back to apo receptor for that ligand")

    # 4. Determine docking box
    # Priority: cofolding ligand > SwinSite > P2Rank > input ligand coords
    center, size = None, None
    box_method = "fallback"
    binding_site_results = {}

    # 4a. Cofolding predicted ligand centroids — cluster across every
    #     aligned cofold cif (4 models × 25 seeds = 100 placements after
    #     Stage 2.5 alignment). Each top-K cluster centroid becomes its
    #     own ``cofolding_N`` binding-site source so docking covers all
    #     pockets the cofold ensemble agrees on. Well-converged targets
    #     collapse into a single cluster (= legacy behaviour); multi-
    #     pocket / inter-model-disagreement targets get 2-3 sources.
    # Adaptive box size: max-extent + 8 Å padding, floored at 22.5, capped
    # at 40. Macrocyclic / peptide ligands (extents > 14.5 Å) get a larger
    # box; typical druglike (≤600 Da) keep the legacy 22.5 Å.
    default_size = _adaptive_box_size(
        [{"sdf": pl[2]} for pl in prepared_ligands if pl[2]]
    )
    if default_size[0] != 22.5:
        print(f"  Adaptive box size: {default_size[0]:.1f} Å edge (ligand-extent + 8 Å padding)")

    # Receptor CA atoms (chain-aware) for nearest-chain tagging. Real
    # binding-site post-analysis on multi-chain proteins needs to know
    # whether a pocket sits on chain A vs chain B vs an interface, so
    # every centroid we register gets the closest CA's chain id.
    ca_atoms = _load_receptor_chain_atoms(pdb_path)
    print(f"  Receptor CA atoms: {len(ca_atoms)} (chains: "
          f"{sorted({c for c, _ in ca_atoms})})")

    cofold_clusters, cofold_diag = _extract_cofolding_ligand_clusters(
        args.run_dir, dockable_chains=dockable_chains,
    )
    n_placements = cofold_diag["n_placements"]
    min_members = cofold_diag["min_members"]
    for rank, cl in enumerate(cofold_clusters, start=1):
        src_name = f"cofolding_{rank}"
        nearest_chain, nearest_d = _nearest_chain(cl["centroid"], ca_atoms)
        binding_site_results[src_name] = (
            cl["centroid"],
            list(default_size),
            {
                "n_members": cl["n_members"],
                "n_unique_models": cl["n_unique_models"],
                "models": cl["models"],
                "cluster_rank": rank,
                "nearest_protein_chain": nearest_chain,
                "nearest_ca_distance": round(nearest_d, 2),
            },
        )
        print(f"  Cofolding cluster #{rank}: {cl['centroid']} "
              f"(n_members={cl['n_members']}, models={cl['models']}, "
              f"chain={nearest_chain})")
    if not cofold_clusters:
        print("  WARNING: no cofolding ligand clusters passed "
              f"the dynamic min_members floor (5 % of placements, ≥ "
              f"{COFOLD_CLUSTER_MIN_FLOOR})")

    # 4b. SwinSite top-K (ML-based surface pocket predictor, needs GPU
    # unless cache reused). Multi-chain receptors: top-2/3 typically pick
    # up the chain-B/-C equivalent of the chain-A active site.
    print("  Running SwinSite binding site prediction...")
    swinsite_pockets = run_swinsite(
        pdb_path, args.output_dir,
        reuse_cache=args.reuse_binding_site_cache,
    ) or []
    for pkt in swinsite_pockets:
        rank = pkt["rank"]
        src_name = f"swinsite_{rank}"
        nearest_chain, nearest_d = _nearest_chain(pkt["center"], ca_atoms)
        binding_site_results[src_name] = (
            pkt["center"],
            pkt["size"],
            {
                "score": pkt["score"],
                "rank": rank,
                "nearest_protein_chain": nearest_chain,
                "nearest_ca_distance": round(nearest_d, 2),
            },
        )

    # 4c. P2Rank top-K (surface geometry-based)
    if not args.no_p2rank:
        print("  Running P2Rank binding site prediction...")
        p2rank_pockets = run_p2rank(
            pdb_path, args.output_dir,
            reuse_cache=args.reuse_binding_site_cache,
        ) or []
        for pkt in p2rank_pockets:
            rank = pkt["rank"]
            src_name = f"p2rank_{rank}"
            nearest_chain, nearest_d = _nearest_chain(pkt["center"], ca_atoms)
            binding_site_results[src_name] = (
                pkt["center"],
                pkt["size"],
                {
                    "score": pkt["score"],
                    "rank": rank,
                    "nearest_protein_chain": nearest_chain,
                    "nearest_ca_distance": round(nearest_d, 2),
                },
            )

    # 4d. Template-consensus pockets (mmseqs+foldseek union → bound-ligand
    #     centroids → spatial cluster). Each top-K cluster centroid becomes
    #     a separate ``template_consensus_N`` source so vina/adg/pxdock dock
    #     at every plausible pocket the templates agree on.
    consensus_sources = _add_template_consensus_sources(args.run_dir, binding_site_results)
    if consensus_sources:
        print(f"  Template-consensus pockets: {consensus_sources}")
        # Tag the consensus sources too so post-analysis can attribute them
        # to the right chain on multi-chain receptors.
        for src in consensus_sources:
            entry = binding_site_results.get(src)
            if not (isinstance(entry, tuple) and len(entry) >= 3 and entry[2] is not None):
                continue
            center, size, meta = entry
            nearest_chain, nearest_d = _nearest_chain(center, ca_atoms)
            meta["nearest_protein_chain"] = nearest_chain
            meta["nearest_ca_distance"] = round(nearest_d, 2)

    # Pick best for the *fallback* box_center (each source still gets its own
    # docking variant downstream — this picks only the legacy single-box
    # field). Strong template-consensus pockets (n_unique_pdb≥2) win over
    # swinsite/p2rank because multi-template agreement is the highest-quality
    # binding-site signal we have when seq+struct templates align. Within
    # cofold sources, cofolding_1 (largest cluster) is preferred. Ranked
    # predictor variants (swinsite_1..3, p2rank_1..3) fall in after the
    # strong-consensus block.
    priority = [f"cofolding_{i}" for i in range(1, COFOLD_CLUSTER_TOP_K + 1)]
    for src in consensus_sources:
        info = binding_site_results.get(src)
        if isinstance(info, tuple) and len(info) >= 3 and info[2].get("n_unique_pdb", 0) >= 2:
            priority.append(src)
    priority += [f"swinsite_{i}" for i in range(1, POCKET_PREDICTOR_TOP_K + 1)]
    priority += [f"p2rank_{i}" for i in range(1, POCKET_PREDICTOR_TOP_K + 1)]
    for src in consensus_sources:
        if src not in priority:
            priority.append(src)
    for method in priority:
        if method in binding_site_results:
            entry = binding_site_results[method]
            center, size = entry[0], entry[1]
            box_method = method
            break

    if center is None:
        # Last-resort fallback: centre the box on whichever ligand SDF we
        # managed to generate. If none did, leave center/size as None and
        # downstream consumers skip their docking runs (they all check the
        # summary JSON at runtime).
        first_sdf = next((sdf for _lid, _smi, sdf, _pdbqt in prepared_ligands if sdf is not None), None)
        if first_sdf is not None:
            center, size = compute_box_from_ligand(first_sdf)
            box_method = "ligand_coordinates"

    # 5. Write summary JSON — include only the ligands that actually have
    # usable SDF + PDBQT files. Downstream docking tools iterate this list
    # and skip pipeline stages cleanly when the ligand list is empty.
    usable_ligands = []
    for lig_id, smiles, sdf, pdbqt in prepared_ligands:
        if sdf is None or pdbqt is None:
            continue
        entry = {
            "id": lig_id,
            "smiles": smiles,
            "sdf": str(sdf),
            "pdbqt": str(pdbqt),
        }
        # If a per-ligand "ligand-aware" receptor was built (multi-ligand
        # target), point this entry at it so docking variants pick it up
        # automatically. Single-ligand targets fall through to the top-level
        # ``receptor_pdbqt`` (apo receptor — every dockable ligand stripped).
        if lig_id in per_ligand_receptors:
            entry["receptor_pdbqt"] = per_ligand_receptors[lig_id]["receptor_pdbqt"]
            entry["receptor_pdb"] = per_ligand_receptors[lig_id]["receptor_pdb"]
        usable_ligands.append(entry)
    n_dropped_ligands = len(prepared_ligands) - len(usable_ligands)

    # Per-source diagnostic — every potential source is listed even if it
    # ended up missing, so post-mortem runs can see where the silent skip
    # happened without reading the prep log.
    source_status = {
        "cofolding_clusters_registered": [
            k for k in binding_site_results if k.startswith("cofolding_")
        ],
        "swinsite_pockets_registered": [
            k for k in binding_site_results if k.startswith("swinsite_")
        ],
        "p2rank_pockets_registered": [
            k for k in binding_site_results if k.startswith("p2rank_")
        ],
        "template_consensus_registered": [
            k for k in binding_site_results if k.startswith("template_consensus_")
        ],
        "n_cofold_placements_total": n_placements,
        "cofold_min_members_floor": min_members,
        "n_dropped_ligands_at_prep": n_dropped_ligands,
        "dockable_chains_from_yaml": sorted(dockable_chains) if dockable_chains else [],
    }

    # Frame-consistency check: every binding-site center should be within
    # reasonable distance of the receptor's heavy-atom centroid. A source
    # >50 Å away is a strong signal that we picked the wrong cofold ref or
    # a frame mismatch slipped through (e.g. template-consensus written in
    # an unaligned frame). Logging here is the cheapest defence.
    try:
        import gemmi as _gemmi
        _rec_atoms: list[tuple[float, float, float]] = []
        for _m in _gemmi.read_structure(str(args.output_dir / "receptor.pdb")):
            for _ch in _m:
                for _r in _ch:
                    if not _r.entity_type == _gemmi.EntityType.Polymer:
                        continue
                    for _a in _r:
                        if _a.element.name == "H":
                            continue
                        _rec_atoms.append((_a.pos.x, _a.pos.y, _a.pos.z))
            break
        if _rec_atoms:
            import numpy as _np
            _rec = _np.asarray(_rec_atoms)
            _rec_centroid = _rec.mean(axis=0)
            _rec_extent = float(_np.linalg.norm(_rec - _rec_centroid, axis=1).max())
            # 30 Å past receptor's farthest heavy atom is generously beyond
            # any pocket. Anything past that is almost certainly a frame
            # bug, not a real cryptic site.
            _max_allowed = _rec_extent + 30.0
            outliers: list[dict] = []
            for src, payload in summary_pred_iter(binding_site_results):
                ctr = _np.asarray(payload[0])
                d = float(_np.linalg.norm(ctr - _rec_centroid))
                if d > _max_allowed:
                    outliers.append({
                        "source": src,
                        "center": [round(float(x), 3) for x in payload[0]],
                        "dist_from_receptor_centroid": round(d, 2),
                    })
            source_status["frame_check"] = {
                "receptor_centroid": [round(float(x), 3) for x in _rec_centroid],
                "receptor_extent_angstrom": round(_rec_extent, 2),
                "max_allowed_source_distance": round(_max_allowed, 2),
                "n_sources_checked": len(binding_site_results),
                "n_outliers": len(outliers),
                "outliers": outliers,
            }
            if outliers:
                print(f"  WARNING: {len(outliers)} binding-site source(s) > "
                      f"{_max_allowed:.0f} Å from receptor centroid — possible "
                      f"frame mismatch:")
                for o in outliers:
                    print(f"    {o['source']}: dist={o['dist_from_receptor_centroid']} Å")
    except Exception as _e:
        source_status["frame_check"] = {"error": str(_e)}

    # Surface alignment quality so downstream readers see frame health
    # without opening a separate file. ``align_cofolding_outputs.py`` writes
    # an ``alignment_summary.json`` with per-cif RMSD-before/after; we lift
    # the summary stats into source_status. Missing alignment summary is
    # non-fatal — just means the bridge didn't run (template-only pipelines).
    try:
        _align_path = args.run_dir / "outputs" / "alignment_summary.json"
        if _align_path.exists():
            _align = json.loads(_align_path.read_text())
            _details = _align.get("details", [])
            _rmsds = [d.get("rmsd_after") for d in _details
                      if d.get("rmsd_after") is not None]
            source_status["alignment"] = {
                "reference": _align.get("reference"),
                "n_aligned": _align.get("n_aligned", 0),
                "n_skipped": _align.get("n_skipped", 0),
                "rmsd_after_max": round(max(_rmsds), 3) if _rmsds else None,
                "rmsd_after_median": round(sorted(_rmsds)[len(_rmsds) // 2], 3)
                                      if _rmsds else None,
            }
    except Exception:
        pass

    summary = {
        "receptor_pdb": str(args.output_dir / "receptor_protonated.pdb"),
        "receptor_pdb_raw": str(args.output_dir / "receptor.pdb"),
        "receptor_pdbqt": str(args.output_dir / "receptor.pdbqt"),
        "ligands": usable_ligands,
        "box_center": center,
        "box_size": size,
        "box_method": box_method,
        "binding_site_predictions": {
            k: (
                {"center": v[0], "size": v[1], **(v[2] if len(v) >= 3 and isinstance(v[2], dict) else {})}
                if isinstance(v, tuple)
                else {"center": v[0], "size": v[1]}
            )
            for k, v in binding_site_results.items()
        },
        "cofolding_structure": str(structure),
        "cofolding_model": model,
        "source_status": source_status,
    }
    summary_path = args.output_dir / "docking_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  Summary: {summary_path}")
    print(f"  Box center: {center} (method: {box_method})")
    print(f"  Box size: {size}")
    print(f"  Sources registered: cofold={len(source_status['cofolding_clusters_registered'])} "
          f"swinsite={len(source_status['swinsite_pockets_registered'])} "
          f"p2rank={len(source_status['p2rank_pockets_registered'])} "
          f"template_consensus={len(source_status['template_consensus_registered'])}")
    if n_dropped_ligands:
        print(f"  WARNING: {n_dropped_ligands} ligand(s) dropped during prep "
              "(SDF/PDBQT generation failed) — check log above for SMILES errors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
