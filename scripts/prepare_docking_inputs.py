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


def cif_to_pdb(cif_path: Path, output_path: Path) -> Path:
    """Convert mmCIF to PDB using gemmi."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    structure.remove_ligands_and_waters()
    structure.write_pdb(str(output_path))
    print(f"  Receptor PDB: {output_path}")
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


def pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> Path:
    """Convert PDB to PDBQT for receptor: pdb2pqr (protonation + charges) → PDBQT format."""

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

    output_path.write_text("\n".join(pdbqt_lines) + "\n")
    print(f"  Receptor PDBQT: {output_path} (protonated, Gasteiger charges)")
    return output_path


def find_best_cofolding_structure(cofolding_dir: Path, model: str) -> Path | None:
    """Find the best-ranked structure from cofolding output.

    Prefers ``_aligned.cif`` (produced by ``align_cofolding_outputs.py``) over
    the raw CIF so that downstream stages work in the common reference frame.
    """
    def _prefer_aligned(cifs: list[Path]) -> Path | None:
        # First pass: return the first cif that has a sibling ``_aligned``.
        # Second pass: fall back to the first raw cif.
        for cif in cifs:
            aligned = cif.with_name(cif.stem + "_aligned.cif")
            if aligned.exists():
                return aligned
        return cifs[0] if cifs else None

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


def run_p2rank(pdb_path: Path, output_dir: Path) -> tuple[list[float], list[float]] | None:
    """Run P2Rank binding site prediction and return (center, size) of top pocket."""
    import subprocess
    import csv

    prank_bin = Path(__file__).resolve().parent.parent / ".local" / "bin" / "prank"
    if not prank_bin.exists():
        print("  P2Rank not found, skipping binding site prediction.")
        return None

    p2rank_out = output_dir / "p2rank"
    try:
        subprocess.run(
            [str(prank_bin), "predict", "-f", str(pdb_path), "-o", str(p2rank_out)],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  P2Rank failed: {e}")
        return None

    # Parse predictions CSV
    pred_file = p2rank_out / f"{pdb_path.name}_predictions.csv"
    if not pred_file.exists():
        print("  P2Rank produced no predictions file.")
        return None

    with open(pred_file) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            # Normalize keys (P2Rank pads headers with spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}
            # First row = top-ranked pocket
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            cz = float(row["center_z"])
            box_side = 22.5
            print(f"  P2Rank pocket 1: center=[{cx:.1f}, {cy:.1f}, {cz:.1f}], score={row['score'].strip()}")
            return ([cx, cy, cz], [box_side, box_side, box_side])

    print("  P2Rank found no pockets.")
    return None


def run_swinsite(pdb_path: Path, output_dir: Path) -> tuple[list[float], list[float]] | None:
    """Run SwinSite binding site prediction and return (center, size) of top pocket."""
    repo_root = Path(__file__).resolve().parent.parent
    swinsite_dir = repo_root / "external" / "swinsite"
    pred_python = repo_root / ".venvs" / "pred" / "bin" / "python"

    if not swinsite_dir.exists() or not pred_python.exists():
        print("  SwinSite not found, skipping.")
        return None

    swinsite_out = output_dir / "swinsite"
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
        return None

    # Parse SwinSite output. File layout (observed in runs/22mj_input/...):
    #   results/input/receptor/
    #     grid0_score_0.7219.pdb     ← HETATM UNL grid points of pocket volume
    #     grid1_score_0.3136.pdb
    #     pocket0_score_0.7219.pdb   ← protein residues near the pocket
    #     pocket1_score_0.3136.pdb
    # ``grid*`` files give the cleanest pocket centroid (they are literally
    # the predicted pocket cloud); their filenames carry a score suffix we
    # sort on to pick the highest-confidence pocket. The old parser used a
    # ``pocket_*.pdb`` glob which never matched (files have a score suffix,
    # no underscore), so every run silently dropped SwinSite.
    results_dir = swinsite_out / "results" / "input" / "receptor"
    if not results_dir.exists():
        print("  SwinSite produced no output.")
        return None

    import re as _re
    score_re = _re.compile(r"_score_([0-9.]+)\.pdb$")

    def _score(path: Path) -> float:
        m = score_re.search(path.name)
        try:
            return float(m.group(1)) if m else -1.0
        except ValueError:
            return -1.0

    # Prefer grid files (pocket-volume point cloud). Fall back to pocket
    # files (near-pocket protein residues) if grids are missing.
    grid_files = sorted(results_dir.glob("grid*_score_*.pdb"), key=_score, reverse=True)
    pocket_files = sorted(results_dir.glob("pocket*_score_*.pdb"), key=_score, reverse=True)
    candidates = grid_files or pocket_files

    if not candidates:
        print("  SwinSite found no pockets.")
        return None

    best = candidates[0]
    coords = []
    for line in best.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        print(f"  SwinSite top pocket {best.name} had no atoms, skipping.")
        return None

    import numpy as _np
    arr = _np.array(coords)
    center = arr.mean(axis=0).tolist()
    box_side = 22.5
    print(f"  SwinSite top pocket {best.name}: center=[{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}], atoms={len(coords)}, score={_score(best):.3f}")
    return (center, [box_side, box_side, box_side])


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
        lid = str(lig.get("id") or "L").strip()
        results.append((lid, str(smi).strip().strip("'\"")))
    return results


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


def _extract_cofolding_ligand_centroid(cif_path: Path) -> list[float] | None:
    """Extract the centroid of ligand (non-polymer) heavy atoms from a cofolding CIF.

    Boltz/Protenix/AF3 cofolding outputs contain both protein and ligand atoms.
    Non-polymer entities (ligand) are identified by ``entity_type == NonPolymer``
    or residue names starting with ``LIG``. Returns ``[x, y, z]`` or ``None``
    if no ligand atoms are found.
    """
    try:
        import gemmi
        st = gemmi.read_structure(str(cif_path))
        coords: list[tuple[float, float, float]] = []
        for model in st:
            for chain in model:
                for res in chain:
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
# after Stage 2.5 alignment, so a single-link cluster on their centroids
# captures multi-pocket / inter-model disagreement signal that the old
# "best-model single-centroid" extraction discarded.
COFOLD_CLUSTER_CUTOFF = 5.0       # Å — same as template-pocket cluster
COFOLD_CLUSTER_MIN_MEMBERS = 5    # out of ~100 placements
COFOLD_CLUSTER_TOP_K = 3


def _extract_cofolding_ligand_clusters(run_dir: Path) -> list[dict]:
    """Cluster ligand centroids across every aligned cofolding CIF.

    Iterates ``outputs/{boltz2,boltz2x,protenix,alphafold3}/**/*_aligned.cif``
    (typically 4 models × 25 seeds = 100 placements, all in the same
    coordinate frame after ``align_cofolding_outputs.py``), pulls the
    heavy-atom ligand centroid from each, and runs greedy single-link
    clustering with a ``COFOLD_CLUSTER_CUTOFF`` Å cutoff.

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
            c = _extract_cofolding_ligand_centroid(cif)
            if c is not None:
                placements.append((c, model))

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
        if len(cl["points"]) < COFOLD_CLUSTER_MIN_MEMBERS:
            continue
        out.append({
            "centroid": [round(x, 4) for x in cl["centroid"]],
            "n_members": len(cl["points"]),
            "n_unique_models": len(set(cl["models"])),
            "models": sorted(set(cl["models"])),
        })
    out.sort(key=lambda d: -d["n_members"])
    return out[:COFOLD_CLUSTER_TOP_K]


TEMPLATE_CONSENSUS_TOP_K = 10
TEMPLATE_CONSENSUS_MIN_MEMBERS = 2
TEMPLATE_CONSENSUS_BOX_SIZE = [22.5, 22.5, 22.5]


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
    parser.add_argument("--use-p2rank", action="store_true", default=True,
                        help="Use P2Rank for binding site prediction (default: true).")
    parser.add_argument("--no-p2rank", action="store_true", help="Disable P2Rank binding site prediction.")
    args = parser.parse_args()

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
    if structure.suffix in (".cif", ".mmcif"):
        pdb_path = cif_to_pdb(structure, args.output_dir / "receptor.pdb")
    else:
        pdb_path = structure

    pdb_to_pdbqt(pdb_path, args.output_dir / "receptor.pdbqt")

    # Also generate protonated PDB for Protenix-Dock (HIS→HID/HIE/HIP)
    pqr_path = args.output_dir / "receptor.pqr"
    if not pqr_path.exists():
        run_pdb2pqr(pdb_path, pqr_path)
    pqr_to_protonated_pdb(pqr_path, args.output_dir / "receptor_protonated.pdb")
    pqr_path.unlink(missing_ok=True)

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
    default_size = [22.5, 22.5, 22.5]
    cofold_clusters = _extract_cofolding_ligand_clusters(args.run_dir)
    for rank, cl in enumerate(cofold_clusters, start=1):
        src_name = f"cofolding_{rank}"
        binding_site_results[src_name] = (
            cl["centroid"],
            list(default_size),
            {
                "n_members": cl["n_members"],
                "n_unique_models": cl["n_unique_models"],
                "models": cl["models"],
                "cluster_rank": rank,
            },
        )
        print(f"  Cofolding cluster #{rank}: {cl['centroid']} "
              f"(n_members={cl['n_members']}, models={cl['models']})")
    if not cofold_clusters:
        print("  WARNING: no cofolding ligand clusters passed "
              f"min_members={COFOLD_CLUSTER_MIN_MEMBERS} filter")

    # 4b. SwinSite (ML-based surface pocket predictor, needs GPU)
    print("  Running SwinSite binding site prediction...")
    swinsite_result = run_swinsite(pdb_path, args.output_dir)
    if swinsite_result:
        binding_site_results["swinsite"] = swinsite_result

    # 4c. P2Rank (surface geometry-based)
    if not args.no_p2rank:
        print("  Running P2Rank binding site prediction...")
        p2rank_result = run_p2rank(pdb_path, args.output_dir)
        if p2rank_result:
            binding_site_results["p2rank"] = p2rank_result

    # 4d. Template-consensus pockets (mmseqs+foldseek union → bound-ligand
    #     centroids → spatial cluster). Each top-K cluster centroid becomes
    #     a separate ``template_consensus_N`` source so vina/adg/pxdock dock
    #     at every plausible pocket the templates agree on.
    consensus_sources = _add_template_consensus_sources(args.run_dir, binding_site_results)
    if consensus_sources:
        print(f"  Template-consensus pockets: {consensus_sources}")

    # Pick best for the *fallback* box_center (each source still gets its own
    # docking variant downstream — this picks only the legacy single-box
    # field). Strong template-consensus pockets (n_unique_pdb≥2) win over
    # swinsite/p2rank because multi-template agreement is the highest-quality
    # binding-site signal we have when seq+struct templates align. Within
    # cofold sources, cofolding_1 (largest cluster) is preferred.
    priority = [f"cofolding_{i}" for i in range(1, COFOLD_CLUSTER_TOP_K + 1)]
    for src in consensus_sources:
        info = binding_site_results.get(src)
        if isinstance(info, tuple) and len(info) >= 3 and info[2].get("n_unique_pdb", 0) >= 2:
            priority.append(src)
    priority += ["swinsite", "p2rank"]
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
    summary = {
        "receptor_pdb": str(args.output_dir / "receptor_protonated.pdb"),
        "receptor_pdb_raw": str(args.output_dir / "receptor.pdb"),
        "receptor_pdbqt": str(args.output_dir / "receptor.pdbqt"),
        "ligands": [
            {
                "id": lig_id,
                "smiles": smiles,
                "sdf": str(sdf) if sdf is not None else None,
                "pdbqt": str(pdbqt) if pdbqt is not None else None,
            }
            for lig_id, smiles, sdf, pdbqt in prepared_ligands
            if sdf is not None and pdbqt is not None
        ],
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
    }
    summary_path = args.output_dir / "docking_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  Summary: {summary_path}")
    print(f"  Box center: {center} (method: {box_method})")
    print(f"  Box size: {size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
