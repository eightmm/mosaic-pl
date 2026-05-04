#!/usr/bin/env python3
"""Prepare docking inputs from template search hits.

Given filtered template hits (with ligand info), extracts the best
template structure, uses the template ligand position for docking box,
and prepares receptor/ligand files for docking.

Usage:
    python prepare_template_docking.py \
        --hits-tsv outputs/template_search_sequence/filtered_hits.tsv \
        --rcsb-dir ~/DB/RCSB/raw/mmCIF_data \
        --input-yaml inputs/boltz_input.yaml \
        --output-dir inputs/template_docking \
        --rcsb-db ~/DB/RCSB/processed/rcsb_index.db
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _usalign_template_to_cofold(template_cif: Path, cofold_cif: Path):
    """Return ``(R, t, tm, rmsd)`` aligning ``template`` onto ``cofold``.

    Wraps ``casp17.usalign.run_usalign``. Returns ``None`` on failure;
    callers decide whether to fall back to no-transform (legacy behaviour).
    """
    try:
        from casp17.usalign import run_usalign
    except Exception as e:
        print(f"  WARNING: casp17.usalign unavailable ({e}); skipping transform")
        return None
    return run_usalign(template_cif, cofold_cif)


def _transform_cif_in_place(cif_path: Path, R, t) -> bool:
    """Apply ``ref ≈ R @ pred + t`` to every atom in the CIF, in place.
    Used to put a template structure into cofold frame *before* it gets
    converted to receptor PDB / ligand SDF — that way every downstream
    file (receptor.pdb, receptor.pdbqt, template_ligand_*.sdf, vina/adg
    outputs) inherits the cofold coordinate system automatically.
    """
    try:
        import gemmi
        import numpy as np
    except Exception as e:
        print(f"  WARNING: gemmi/numpy unavailable ({e}); cannot transform")
        return False
    try:
        st = gemmi.read_structure(str(cif_path))
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        for model in st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        v = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z])
                        v2 = R @ v + t
                        atom.pos = gemmi.Position(float(v2[0]), float(v2[1]), float(v2[2]))
        st.make_mmcif_document().write_file(str(cif_path))
        return True
    except Exception as e:
        print(f"  WARNING: CIF transform failed for {cif_path}: {e}")
        return False


def find_template_cif(pdb_id: str, rcsb_dir: Path) -> Path | None:
    """Find CIF file for a PDB ID in the RCSB raw data directory."""
    pdb_id = pdb_id.lower()
    hash_dir = pdb_id[1:3]  # middle two characters
    cif_gz = rcsb_dir / hash_dir / f"{pdb_id}.cif.gz"
    if cif_gz.exists():
        return cif_gz
    cif = rcsb_dir / hash_dir / f"{pdb_id}.cif"
    if cif.exists():
        return cif
    return None


def extract_cif(cif_path: Path, output_dir: Path) -> Path:
    """Extract gzipped CIF to output directory."""
    output = output_dir / cif_path.name.replace(".gz", "")
    if cif_path.suffix == ".gz":
        import gzip
        with gzip.open(cif_path, "rb") as f_in, open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy2(cif_path, output)
    return output


def extract_ligand_center(
    cif_path: Path, ligand_ccd: str,
    prefer_near: list[float] | None = None,
) -> list[float] | None:
    """Return one ligand instance's heavy-atom centroid.

    Multimeric templates routinely have the same CCD bound to multiple
    chains (8qrt is a homotetramer with 4 × WP2). Naive
    averaging-across-instances drops the box on the *centroid of all
    binding sites at once*, which is roughly the structural centre of
    the protein and not a binding pocket at all — Vina then docks into
    the protein interior. Pick a single instance instead, optionally
    biased toward ``prefer_near`` (e.g. the cofold-ligand position
    after the template→cofold transform) so we end up at the same
    pocket the cofold model picked.
    """
    import gemmi
    import numpy as np

    structure = gemmi.read_structure(str(cif_path))
    instances: list[np.ndarray] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name != ligand_ccd:
                    continue
                atom_coords = [(a.pos.x, a.pos.y, a.pos.z) for a in residue]
                if atom_coords:
                    instances.append(np.array(atom_coords).mean(axis=0))
        break  # first model only — match docking convention
    if not instances:
        return None
    if prefer_near is not None:
        anchor = np.asarray(prefer_near, dtype=float)
        best = min(instances, key=lambda c: float(np.linalg.norm(c - anchor)))
        return best.tolist()
    # No preference → first instance (chain order). Better than averaging.
    return instances[0].tolist()


def extract_template_ligand_sdf(cif_path: Path, ligand_ccd: str, output_sdf: Path) -> Path | None:
    """Extract template ligand from CIF as SDF with bound-state 3D coordinates.

    Writes the first matching residue as a HETATM PDB block, then converts
    to SDF via RDKit (or openbabel as fallback).  The resulting SDF preserves
    the crystallographic pose, which is needed for lig-align reference.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    target_residue = None
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == ligand_ccd:
                    target_residue = residue
                    break
            if target_residue:
                break
        if target_residue:
            break

    if target_residue is None:
        return None

    # Write ligand atoms as minimal PDB. RCSB CIFs sometimes use multi-char
    # chain names (e.g. "AAA" for a polymer entity) which shift PDB columns
    # and make RDKit's MolFromPDBFile fail with
    # "Problem with residue number". Normalize chain→"X" and resid→1 so
    # the fixed-width PDB layout is always valid; coordinates are all
    # downstream consumers of this file need.
    ligand_pdb = output_sdf.with_suffix(".pdb")
    with open(ligand_pdb, "w") as f:
        for i, atom in enumerate(target_residue):
            name = f" {atom.name:<3s}" if len(atom.name) < 4 else atom.name
            f.write(
                f"HETATM{i + 1:5d} {name:4s} {target_residue.name:>3s}"
                f" X{1:4d}    "
                f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                f"  1.00  0.00          {atom.element.name:>2s}\n"
            )
        f.write("END\n")

    # Convert PDB → SDF
    try:
        from rdkit import Chem
        mol = Chem.MolFromPDBFile(str(ligand_pdb), sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass  # keep unsanitized — coords are what matter
            writer = Chem.SDWriter(str(output_sdf))
            writer.write(mol)
            writer.close()
            return output_sdf
    except Exception:
        pass

    # Fallback: try openbabel
    try:
        subprocess.run(
            ["obabel", str(ligand_pdb), "-O", str(output_sdf)],
            check=True, capture_output=True, text=True,
        )
        if output_sdf.exists():
            return output_sdf
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def cif_to_receptor_pdb(
    cif_path: Path,
    output_pdb: Path,
    target_ccds: set[str] | None = None,
) -> Path:
    """Extract protein chains from CIF to PDB, retaining metals + cofactors.

    RCSB CIFs often use multi-character chain labels (e.g. ``AAA`` for a
    polymer-entity asym_id) which gemmi rejects at PDB serialization with
    ``RuntimeError: chain name too long for the PDB format``. Since the
    docking pipeline only needs receptor coordinates (not the original
    chain identity), rename each retained chain to a fresh single letter
    A..Z before writing.

    Selective stripping rules (matched to ``prepare_docking_inputs.cif_to_pdb``):
        - Waters always dropped.
        - When ``target_ccds`` is provided (the candidate ligand codes
          we're about to dock), residues with those CCD names are dropped.
        - Everything else (metals, cofactors like HEM/NAD/FAD, glycans)
          stays so docking grids see the active-site chemistry.

    The previous heuristic (drop any non-polymer with ≥6 heavy atoms
    that isn't a metal) wiped HEM/NAD/FAD too, contradicting the
    "retain cofactor" goal. Switching to a CCD-based rule makes the
    stripping match the metal-retain comment.
    """
    import gemmi
    import string
    structure = gemmi.read_structure(str(cif_path))
    targets = {c.upper() for c in (target_ccds or set())}
    # gemmi.Chain only exposes index-based delete; iterate in reverse so
    # earlier indices stay stable as we strip residues.
    for model in structure:
        for chain in model:
            for i in range(len(chain) - 1, -1, -1):
                res = chain[i]
                is_water = (res.entity_type == gemmi.EntityType.Water
                            or res.name == "HOH")
                if is_water:
                    del chain[i]
                elif res.name.upper() in targets:
                    # Identified target ligand — strip; we'll dock against
                    # this site, ligand atoms shouldn't sit inside the receptor.
                    del chain[i]
    # Collect already-valid single-letter chain names so we allocate from the
    # unused remainder without clobbering them.
    used = {chain.name for model in structure for chain in model if len(chain.name) == 1}
    pool = (c for c in string.ascii_uppercase if c not in used)
    for model in structure:
        for chain in model:
            if len(chain.name) != 1:
                try:
                    chain.name = next(pool)
                except StopIteration:
                    raise RuntimeError(
                        f"More than 26 protein chains in {cif_path.name}; "
                        "cannot fit PDB single-letter chain ids."
                    )
    structure.write_pdb(str(output_pdb))
    return output_pdb


from casp17.ccd_sets import NUCLEIC_RESIDUES as _NUCLEIC_RESIDUES


def _has_nucleic_acid_pdb(pdb_path: Path) -> bool:
    """RNA/DNA residue check on a PDB path (Track 2 receptor variant)."""
    if not pdb_path.exists():
        return False
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            res = line[17:20].strip().upper()
            if res in _NUCLEIC_RESIDUES:
                return True
    return False


def prepare_receptor_pdbqt(pdb_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Run pdb2pqr and convert to PDBQT.

    Nucleic acid receptors (RNA/DNA) get routed through obabel because
    pdb2pqr's AMBER FF silently drops nucleotide residues. Mirrors the
    behaviour in ``prepare_docking_inputs.pdb_to_pdbqt``.
    """
    pqr_path = output_dir / "receptor.pqr"
    pdbqt_path = output_dir / "receptor.pdbqt"
    protonated_pdb = output_dir / "receptor_protonated.pdb"

    # Nucleic acid path: obabel one-shot.
    if _has_nucleic_acid_pdb(pdb_path):
        repo_root = Path(__file__).resolve().parent.parent
        obabel = repo_root / ".venvs" / "pred" / "bin" / "obabel"
        if obabel.exists():
            print(f"  Nucleic acid detected → obabel path for {pdb_path.name}")
            try:
                subprocess.run(
                    [str(obabel), str(pdb_path), "-O", str(pdbqt_path),
                     "-p", "7.4", "--partialcharge", "gasteiger", "-xr"],
                    check=True, capture_output=True, text=True, timeout=120,
                )
                # Track 2 also needs a protonated PDB for PxDock; obabel
                # gives us both via two calls (pdbqt path + .pdb path).
                subprocess.run(
                    [str(obabel), str(pdb_path), "-O", str(protonated_pdb),
                     "-p", "7.4"],
                    check=True, capture_output=True, text=True, timeout=120,
                )
                return protonated_pdb, pdbqt_path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError) as e:
                print(f"  obabel failed: {e}; falling back to pdb2pqr (likely to drop nucleic)")

    # pdb2pqr (protein path)
    try:
        subprocess.run(
            [sys.executable, "-m", "pdb2pqr", "--ff=AMBER", "--ffout=AMBER",
             "--keep-chain", str(pdb_path), str(pqr_path)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # pdb2pqr can choke on a number of legitimate templates (4v8w, 6gjc
        # observed: nonstandard residues / missing atoms / unusual altlocs).
        # Try obabel as a second-chance protein path before giving up — it
        # accepts much messier PDB input than pdb2pqr/AMBER. Only fail loudly
        # when both tools refuse the receptor.
        repo_root = Path(__file__).resolve().parent.parent
        obabel = repo_root / ".venvs" / "pred" / "bin" / "obabel"
        if obabel.exists():
            try:
                subprocess.run(
                    [str(obabel), str(pdb_path), "-O", str(pdbqt_path),
                     "-p", "7.4", "--partialcharge", "gasteiger", "-xr"],
                    check=True, capture_output=True, text=True, timeout=120,
                )
                subprocess.run(
                    [str(obabel), str(pdb_path), "-O", str(protonated_pdb),
                     "-p", "7.4"],
                    check=True, capture_output=True, text=True, timeout=120,
                )
                print(f"  pdb2pqr failed → obabel fallback succeeded for {pdb_path.name}")
                return protonated_pdb, pdbqt_path
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError):
                pass
        msg = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
        raise RuntimeError(
            f"pdb2pqr (and obabel fallback) failed for {pdb_path.name}; "
            f"cannot produce a Vina-ready PDBQT. Skip Track 2 for this template. "
            f"Underlying error:\n{msg}"
        ) from e

    # PQR → protonated PDB
    pdb_lines = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:1].upper()
            pdb_lines.append(f"{line[:54]:<54s}  1.00  0.00          {element:>2s}  ")
        elif line.startswith(("TER", "END")):
            pdb_lines.append(line)
    protonated_pdb.write_text("\n".join(pdb_lines) + "\n")

    # PQR → PDBQT
    AD_TYPE_MAP = {"C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD"}
    pdbqt_lines = []
    for line in pqr_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = atom_name.lstrip("0123456789")[0:2].upper().strip()
            if len(element) > 1 and element not in ("CL", "BR"):
                element = element[0]
            ad_type = AD_TYPE_MAP.get(element, element)
            parts = line.split()
            try:
                charge = float(parts[-2])
            except (ValueError, IndexError):
                charge = 0.0
            pdb_prefix = f"{line[:54]:<54s}"
            pdbqt_lines.append(f"{pdb_prefix}  0.00  0.00    {charge:+.3f} {ad_type:<2s}")
        elif line.startswith(("TER", "END")):
            pdbqt_lines.append(line)
    pdbqt_path.write_text("\n".join(pdbqt_lines) + "\n")
    pqr_path.unlink(missing_ok=True)

    return protonated_pdb, pdbqt_path


def prepare_ligand_files(smiles: str, lig_id: str, output_dir: Path) -> tuple[Path, Path]:
    """SMILES → 3D SDF → PDBQT."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation

    sdf_path = output_dir / f"ligand_{lig_id}.sdf"
    pdbqt_path = output_dir / f"ligand_{lig_id}.pdbqt"

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", lig_id)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    preparator = MoleculePreparation()
    preparator.prepare(mol)
    if preparator.is_ok:
        pdbqt_path.write_text(preparator.write_pdbqt_string())

    return sdf_path, pdbqt_path


def extract_smiles_from_yaml(yaml_path: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, smiles) from unified YAML.

    Uses ``yaml.safe_load`` so multi-line / quoted-scalar SMILES survive
    intact. The previous line-based parser was fragile to quoting and
    extension fields downstream of the ligand block.
    """
    import yaml as _yaml
    data = _yaml.safe_load(yaml_path.read_text()) or {}
    results: list[tuple[str, str]] = []
    for entry in data.get("sequences", []) or []:
        if not isinstance(entry, dict) or "ligand" not in entry:
            continue
        lig = entry.get("ligand") or {}
        smiles = lig.get("smiles")
        if not smiles:
            continue
        lig_id = lig.get("id", "L")
        if isinstance(lig_id, list):
            lig_id = lig_id[0] if lig_id else "L"
        results.append((str(lig_id), str(smiles)))
    return results


def parse_filtered_hits(tsv_path: Path) -> list[dict]:
    """Parse filtered hits TSV from template_filter."""
    hits = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row.get("num_ligands", 0)) > 0:
                hits.append(row)
    return hits


def select_cluster_representative_templates(
    pockets_json: Path,
    hits_tsv_lookup: dict[str, dict],
    *,
    max_clusters: int = 10,
    require_strong: bool = False,
) -> list[dict]:
    """Pick one representative template per pocket cluster.

    Walks ``template_pockets.json`` (per-instance pocket list with each
    record's evidence_score / TM-score) grouped by cluster index from
    ``template_pocket_clusters.json``, picks the highest-evidence template
    per cluster, and returns the corresponding row from
    ``hits_tsv_lookup`` (keyed by ``pdb_id``).

    The motivation: Track 2 used to take the top-N by filtered_hits.tsv
    sort, which often picks 3 alternate chains of the *same* PDB or 3
    near-identical conformations from one cluster. Spreading across
    clusters gives Track 2 docking the same pocket diversity Track 1's
    ``vina_template_consensus_*`` already enjoys, but with the
    experimental receptor conformation rather than the cofold prediction.

    Args:
        pockets_json: ``outputs/template_pockets/template_pockets.json``
        hits_tsv_lookup: ``{pdb_id: row}`` from filtered_hits.tsv so the
            returned dicts are docking-ready (carry ligand_codes etc.)
        max_clusters: cap on cluster count (matches the docking-prep
            ``TEMPLATE_CONSENSUS_TOP_K``)
        require_strong: when True, only clusters with ``n_unique_pdb >= 2``
            are considered — drops single-template clusters that are more
            likely alternate/spurious sites.

    Returns the per-cluster list of hit rows in cluster-rank order.
    Falls back to an empty list if ``template_pockets.json`` is missing
    so callers can degrade gracefully (e.g. to legacy sort top-N).
    """
    if not pockets_json.exists():
        return []
    try:
        data = json.loads(pockets_json.read_text())
    except Exception as e:
        print(f"  WARNING: pockets json unreadable ({e}); cluster-aware off")
        return []
    pockets = data.get("pockets") or []
    clusters_path = pockets_json.parent / "template_pocket_clusters.json"
    if not clusters_path.exists():
        return []
    try:
        cl_data = json.loads(clusters_path.read_text())
    except Exception as e:
        print(f"  WARNING: clusters json unreadable ({e}); cluster-aware off")
        return []
    clusters = cl_data.get("clusters") or []

    # Group pockets by cluster_index (greedy centroid cluster builder writes
    # this onto each pocket record). Fallback: euclidean nearest centroid.
    by_cluster: dict[int, list[dict]] = {}
    for p in pockets:
        ci = p.get("cluster_index")
        if ci is None:
            # Fall back: assign to nearest cluster centroid by Euclidean.
            best_i, best_d2 = None, float("inf")
            for i, cl in enumerate(clusters):
                cx, cy, cz = cl["centroid"]
                d2 = ((p["centroid_x"] - cx) ** 2
                      + (p["centroid_y"] - cy) ** 2
                      + (p["centroid_z"] - cz) ** 2)
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            ci = best_i
        if ci is None:
            continue
        by_cluster.setdefault(ci, []).append(p)

    selected: list[dict] = []
    for rank, cl in enumerate(clusters[:max_clusters]):
        if require_strong and (cl.get("n_unique_pdb") or 0) < 2:
            continue
        members = by_cluster.get(rank, [])
        if not members:
            continue
        # Best member: highest evidence weight (in_mmseqs+in_foldseek
        # + max(alignment_tmscore, qtmscore, pident/100))
        def _weight(m: dict) -> float:
            base = (1 if m.get("in_mmseqs") else 0) + (1 if m.get("in_foldseek") else 0)
            tm = float(m.get("alignment_tmscore") or 0.0)
            qtm = float(m.get("qtmscore") or 0.0)
            pid = float(m.get("pident") or 0.0)
            return base + max(tm, qtm, pid / 100.0)

        members_sorted = sorted(members, key=_weight, reverse=True)
        for member in members_sorted:
            pdb_id = (member.get("template_pdb_id") or "").strip().lower()
            row = hits_tsv_lookup.get(pdb_id)
            if row is None:
                continue
            # De-dupe: same PDB across clusters is OK (different binding
            # mode in different cluster), but skip if we already selected
            # this exact row in this cluster (shouldn't happen but cheap).
            if any(r.get("pdb_id", "").strip().lower() == pdb_id
                   and r.get("_cluster_rank") == rank for r in selected):
                continue
            row = dict(row)
            row["_cluster_rank"] = rank
            row["_cluster_evidence"] = round(_weight(member), 3)
            row["_cluster_n_unique_pdb"] = cl.get("n_unique_pdb")
            selected.append(row)
            break  # one representative per cluster
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare template-based docking inputs.")
    parser.add_argument("--hits-tsv", type=Path, required=True, help="Filtered template hits TSV.")
    parser.add_argument("--rcsb-dir", type=Path, default=Path.home() / "DB/RCSB/raw/mmCIF_data")
    parser.add_argument("--input-yaml", type=Path, help="Unified input YAML (for ligand SMILES).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-templates", type=int, default=10,
                        help="Max templates to prepare. With --pockets-json the "
                             "selection is cluster-aware (one representative per "
                             "pocket cluster); without it the legacy sort-by-evidence "
                             "top-N is used.")
    parser.add_argument(
        "--pockets-json", type=Path, default=None,
        help="Path to ``template_pockets.json`` from extract_template_pockets. "
             "When supplied, Track 2 picks one representative per pocket cluster "
             "(see select_cluster_representative_templates) instead of "
             "filtered_hits.tsv top-N. ``template_pocket_clusters.json`` is read "
             "automatically from the same directory."
    )
    parser.add_argument(
        "--strong-clusters-only", action="store_true",
        help="Only emit Track 2 for clusters with n_unique_pdb >= 2 "
             "(multi-PDB consensus). Useful when the template pool has many "
             "single-PDB clusters that don't add receptor diversity."
    )
    parser.add_argument("--box-size", type=float, default=22.5)
    parser.add_argument(
        "--cofold-ref-cif", type=Path, default=None,
        help="Path to the cofolding reference CIF (e.g. the best aligned "
             "model). When provided, every template CIF is USalign'd onto "
             "this reference and rewritten in cofold frame *before* "
             "receptor.pdb / template_ligand SDF / docking inputs are "
             "generated. Without this, Track 2 outputs land in template "
             "frame and downstream RMSD evaluation against the cofold-frame "
             "crystal pose returns garbage (~23 Å mean error)."
    )
    parser.add_argument(
        "--cofold-lig-anchor", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Cofold-predicted ligand centroid (x y z). When set, "
             "templates with multiple ligand instances (homo-tetramers, "
             "etc.) pick the instance closest to this anchor as the "
             "docking-box centre. Without this, the first-chain "
             "instance is used (much better than the previous "
             "average-of-all-instances behaviour, but still arbitrary "
             "for multimers)."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse hits
    hits = parse_filtered_hits(args.hits_tsv)
    if not hits:
        print("No template hits with ligands found.")
        return 0

    # 2. Extract ligand SMILES from input
    ligands = []
    if args.input_yaml and args.input_yaml.exists():
        ligands = extract_smiles_from_yaml(args.input_yaml)
    if not ligands:
        print("No ligand SMILES found in input.")
        return 1

    # 3. Choose templates — cluster-aware when pockets json is available,
    # falls back to sort-by-evidence top-N otherwise. Cluster-aware spreads
    # Track 2 across distinct binding modes (one representative per cluster)
    # instead of doing 3 alternate chains/conformations of the same site.
    selected: list[dict]
    if args.pockets_json and args.pockets_json.exists():
        hits_by_pdb = {h["pdb_id"].strip().lower(): h for h in hits}
        selected = select_cluster_representative_templates(
            args.pockets_json, hits_by_pdb,
            max_clusters=args.max_templates,
            require_strong=args.strong_clusters_only,
        )
        if selected:
            print(f"Cluster-aware Track 2: {len(selected)} representative templates "
                  f"(one per pocket cluster, max={args.max_templates}, "
                  f"strong_only={args.strong_clusters_only})")
        else:
            print("Cluster-aware selection returned 0 — falling back to sort top-N")
    else:
        selected = []

    if not selected:
        selected = hits[:args.max_templates]
        print(f"Found {len(hits)} template hits with ligands, "
              f"using top {len(selected)} by sort order")

    all_templates = []
    for i, hit in enumerate(selected):
        pdb_id = hit["pdb_id"]
        ligand_codes = hit.get("ligand_codes", "").split(";")

        print(f"\n{'='*60}")
        print(f"  Template {i+1}: {pdb_id} (pident={hit.get('pident', '?')}%)")
        print(f"  Ligands: {', '.join(ligand_codes)}")
        print(f"{'='*60}")

        # Find and extract CIF
        cif_path = find_template_cif(pdb_id, args.rcsb_dir)
        if not cif_path:
            print(f"  CIF not found for {pdb_id}, skipping.")
            continue

        template_dir = args.output_dir / f"template_{pdb_id}"
        template_dir.mkdir(parents=True, exist_ok=True)

        cif = extract_cif(cif_path, template_dir)
        print(f"  Extracted: {cif.name}")

        # Move the template CIF into cofold frame BEFORE we extract the
        # receptor PDB / ligand SDF / dock against it. This is the only
        # transform — every downstream artefact (receptor.pdb, .pdbqt,
        # template_ligand_*.sdf, docked.pdbqt, docked.dlg, lig-align SDF)
        # then lives in cofold frame and matches the crystal-pose RMSD
        # evaluator's expectations.
        if args.cofold_ref_cif is not None and args.cofold_ref_cif.exists():
            align = _usalign_template_to_cofold(cif, args.cofold_ref_cif)
            if align is None:
                print(f"  WARNING: USalign({pdb_id} → cofold) failed; "
                      "Track 2 outputs will be in template frame")
            else:
                R, t, tm_score, _aligned_rmsd = align
                if tm_score < 0.4:
                    print(f"  WARNING: low USalign TM={tm_score:.3f} for {pdb_id}; "
                          "transform may be unreliable")
                if _transform_cif_in_place(cif, R, t):
                    print(f"  Applied template→cofold transform "
                          f"(USalign TM={tm_score:.3f})")
                else:
                    print("  WARNING: failed to rewrite CIF in cofold frame")

        # Extract ligand center for docking box. When the cofold ligand
        # anchor is provided, the closest instance wins (= same binding
        # site the cofold model picked) — critical for multimers where
        # the symmetric copies sit on opposite faces of the protein and
        # an "average" centroid is structural-centre noise.
        center = None
        template_ligand_ccd = None
        for ccd_code in ligand_codes:
            ccd_code = ccd_code.strip()
            if not ccd_code:
                continue
            center = extract_ligand_center(
                cif, ccd_code, prefer_near=args.cofold_lig_anchor
            )
            if center:
                template_ligand_ccd = ccd_code
                break

        if not center:
            print("  Could not extract ligand center, skipping.")
            continue

        box_size = [args.box_size] * 3
        print(f"  Ligand {template_ligand_ccd} center: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")

        # Extract template ligand SDF (bound pose for lig-align)
        template_ligand_sdf = None
        template_lig_sdf_path = template_dir / f"template_ligand_{template_ligand_ccd}.sdf"
        extracted = extract_template_ligand_sdf(cif, template_ligand_ccd, template_lig_sdf_path)
        if extracted:
            template_ligand_sdf = str(extracted)
            print(f"  Template ligand SDF: {extracted.name}")
        else:
            print(f"  WARNING: Could not extract template ligand SDF for {template_ligand_ccd}")

        # Prepare receptor — strip the bound target ligand (we dock against
        # that site so ligand atoms shouldn't be in the receptor PDB) but
        # keep metals and cofactors. ``ligand_codes`` is the candidate CCD
        # list for this template hit; ``template_ligand_ccd`` is the one
        # we picked for the box centre. Both go into ``target_ccds`` so any
        # alternate copy of the same ligand on a different chain is also
        # stripped.
        target_ccds = {c.strip() for c in ligand_codes if c.strip()}
        if template_ligand_ccd:
            target_ccds.add(template_ligand_ccd)
        receptor_pdb = cif_to_receptor_pdb(
            cif, template_dir / "receptor.pdb",
            target_ccds=target_ccds,
        )
        try:
            protonated_pdb, receptor_pdbqt = prepare_receptor_pdbqt(receptor_pdb, template_dir)
        except RuntimeError as exc:
            # Edge case: huge biological assemblies (e.g. 5BP4 expanded to
            # ~260 k atoms) crash pdb2pqr, and the previous silent fallback
            # produced an unusable PDBQT. Drop this template cleanly.
            print(f"  WARNING: receptor PDBQT prep failed for {pdb_id}: {exc}")
            print(f"  Skipping Track 2 docking for template {pdb_id}.")
            continue
        print(f"  Receptor PDB: {receptor_pdb.name}")
        print(f"  Receptor PDBQT: {receptor_pdbqt.name}")

        # Prepare target ligand
        for lig_id, smiles in ligands:
            sdf_path, pdbqt_path = prepare_ligand_files(smiles, lig_id, template_dir)
            print(f"  Ligand {lig_id}: {sdf_path.name}, {pdbqt_path.name}")

        # Write docking prep summary (same format as cofolding-based prep)
        summary = {
            "receptor_pdb": str(protonated_pdb),
            "receptor_pdb_raw": str(receptor_pdb),
            "receptor_pdbqt": str(receptor_pdbqt),
            "ligands": [
                {
                    "id": lig_id,
                    "smiles": smiles,
                    "sdf": str(template_dir / f"ligand_{lig_id}.sdf"),
                    "pdbqt": str(template_dir / f"ligand_{lig_id}.pdbqt"),
                }
                for lig_id, smiles in ligands
            ],
            "box_center": center,
            "box_size": box_size,
            "box_method": "template_ligand",
            "template_pdb_id": pdb_id,
            "template_ligand_ccd": template_ligand_ccd,
            "template_ligand_sdf": template_ligand_sdf,
            "template_pident": float(hit.get("pident", 0)),
            "best_mcs_coverage": float(hit.get("best_mcs_coverage", 0)),
            "best_tanimoto": float(hit.get("best_tanimoto", 0)),
        }
        summary_path = template_dir / "docking_prep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  Summary: {summary_path}")

        all_templates.append(summary)

    # Write master summary
    master_summary = args.output_dir / "template_docking_summary.json"
    master_summary.write_text(json.dumps(all_templates, indent=2) + "\n")
    print(f"\n{len(all_templates)} templates prepared → {master_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
