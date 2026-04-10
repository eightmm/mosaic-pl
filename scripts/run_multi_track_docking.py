#!/usr/bin/env python3
"""Multi-track docking orchestrator.

After Track 1 (cofolding-based docking) completes, this script checks
template search results for MCS >= threshold and runs:
  - Track 2: Vina + AutoDock-GPU + Protenix-Dock on template structures
  - Track 3: lig-align (MCS-guided pose generation from template ligand)

Usage:
    python run_multi_track_docking.py \
        --run-dir runs/target \
        --input-yaml runs/target/inputs/boltz_input.yaml \
        --hits-tsv runs/target/outputs/template_search_sequence/filtered_hits.tsv \
        --rcsb-dir ~/DB/RCSB/raw/mmCIF_data \
        --rcsb-db ~/DB/RCSB/processed/rcsb_index.db \
        --mcs-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def check_mcs_hits(hits_tsv: Path, threshold: float) -> list[dict]:
    """Return template hits with best_mcs_coverage >= threshold."""
    if not hits_tsv.exists():
        return []
    hits = []
    with open(hits_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mcs = float(row.get("best_mcs_coverage", 0))
            num_lig = int(row.get("num_ligands", 0))
            if mcs >= threshold and num_lig > 0:
                hits.append(row)
    return hits


def run_template_docking_prep(
    hits_tsv: Path,
    rcsb_dir: Path,
    input_yaml: Path,
    output_dir: Path,
    rcsb_db: Path | None = None,
    max_templates: int = 3,
) -> list[dict]:
    """Run prepare_template_docking.py and return template summaries."""
    script = Path(__file__).resolve().parent / "prepare_template_docking.py"
    cmd = [
        sys.executable, str(script),
        "--hits-tsv", str(hits_tsv),
        "--rcsb-dir", str(rcsb_dir),
        "--input-yaml", str(input_yaml),
        "--output-dir", str(output_dir),
        "--max-templates", str(max_templates),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  Template docking prep failed: {result.stderr}", file=sys.stderr)
        return []

    summary_path = output_dir / "template_docking_summary.json"
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text())


def run_vina_on_template(template: dict, output_dir: Path, seed: int = 42) -> dict | None:
    """Run AutoDock Vina on a template-prepared structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = template.get("receptor_pdbqt")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdbqt or not ligands:
        return None

    ligand_pdbqt = ligands[0].get("pdbqt")
    if not ligand_pdbqt or not Path(ligand_pdbqt).exists():
        return None

    try:
        from vina import Vina

        out_path = output_dir / "docked.pdbqt"
        v = Vina(sf_name="vina", seed=seed)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=32, n_poses=10)
        v.write_poses(str(out_path), n_poses=10, overwrite=True)

        energies = v.energies()
        best_score = float(energies[0][0]) if energies is not None and len(energies) > 0 else None
        print(f"    Vina best score: {best_score}")

        return {
            "tool": "vina",
            "output": str(out_path),
            "best_score": best_score,
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    Vina failed: {e}")
        return None


def run_autodock_gpu_on_template(
    template: dict, output_dir: Path, seed: int = 42
) -> dict | None:
    """Run AutoDock-GPU on a template-prepared structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    receptor_pdbqt = template.get("receptor_pdbqt")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdbqt or not ligands:
        return None

    ligand_pdbqt = ligands[0].get("pdbqt")
    if not ligand_pdbqt or not Path(ligand_pdbqt).exists():
        return None

    adg_binary = repo_root / ".local" / "bin" / "autodock_gpu_128wi"
    autogrid_binary = repo_root / ".local" / "bin" / "autogrid4"
    if not adg_binary.exists() or not autogrid_binary.exists():
        print("    AutoDock-GPU/autogrid4 not found, skipping.")
        return None

    grid_dir = output_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Generate GPF
    npts = [max(1, int(s / 0.375)) for s in size]
    gpf_path = grid_dir / "receptor.gpf"
    fld_path = grid_dir / "receptor.maps.fld"
    gpf_lines = [
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"gridfld {fld_path.name}",
        "spacing 0.375",
        "receptor_types A C HD N NA OA SA",
        "ligand_types A C HD N NA OA SA",
        f"receptor {receptor_pdbqt}",
        f"gridcenter {center[0]} {center[1]} {center[2]}",
        "smooth 0.5",
        "map receptor.A.map", "map receptor.C.map", "map receptor.HD.map",
        "map receptor.N.map", "map receptor.NA.map", "map receptor.OA.map",
        "map receptor.SA.map", "elecmap receptor.e.map",
        "dsolvmap receptor.d.map", "dielectric -0.1465",
    ]
    gpf_path.write_text("\n".join(gpf_lines) + "\n")

    try:
        # Run autogrid4
        subprocess.run(
            [str(autogrid_binary), "-p", str(gpf_path), "-l", str(grid_dir / "autogrid.log")],
            check=True, capture_output=True, text=True, cwd=str(grid_dir), timeout=300,
        )
        # Run autodock_gpu
        dlg_path = output_dir / "docked"
        subprocess.run(
            [
                str(adg_binary),
                "--ffile", str(fld_path),
                "--lfile", str(ligand_pdbqt),
                "--resnam", str(dlg_path),
                "--nrun", "20",
                "--nev", "2500000",
                "--seed", str(seed),
                "--heuristics", "1",
                "--autostop", "1",
            ],
            check=True, capture_output=True, text=True, timeout=600,
        )
        dlg_file = output_dir / "docked.dlg"
        print(f"    AutoDock-GPU done: {dlg_file}")
        return {
            "tool": "autodock_gpu",
            "output": str(dlg_file),
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    AutoDock-GPU failed: {e}")
        return None


def run_protenix_dock_on_template(
    template: dict, output_dir: Path, seed: int = 42
) -> dict | None:
    """Run Protenix-Dock on a template-prepared structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    receptor_pdb = template.get("receptor_pdb")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdb or not ligands:
        return None

    ligand_sdf = ligands[0].get("sdf")
    if not ligand_sdf or not Path(ligand_sdf).exists():
        return None

    pxdock_python = repo_root / ".venvs" / "protenix-dock" / "bin" / "python"
    if not pxdock_python.exists():
        print("    Protenix-Dock venv not found, skipping.")
        return None

    try:
        cmd = [
            str(pxdock_python), "-m", "protenix_dock",
            "--receptor", str(receptor_pdb),
            "--ligand", str(ligand_sdf),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--output_dir", str(output_dir),
            "--seed", str(seed),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print(f"    Protenix-Dock done: {output_dir}")
        return {
            "tool": "protenix_dock",
            "output": str(output_dir),
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    Protenix-Dock failed: {e}")
        return None


def run_lig_align_on_template(
    template: dict, target_smiles: str, output_dir: Path
) -> dict | None:
    """Run lig-align: MCS-guided pose generation from template ligand."""
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdb = template.get("receptor_pdb_raw") or template.get("receptor_pdb")
    ref_ligand_sdf = template.get("template_ligand_sdf")

    if not receptor_pdb or not Path(receptor_pdb).exists():
        print("    lig-align: receptor PDB not found, skipping.")
        return None
    if not ref_ligand_sdf or not Path(ref_ligand_sdf).exists():
        print("    lig-align: template ligand SDF not found, skipping.")
        return None

    try:
        from lig_align import run_pipeline

        result = run_pipeline(
            protein_pdb=str(receptor_pdb),
            ref_ligand=str(ref_ligand_sdf),
            query_ligand=target_smiles,
            output_dir=str(output_dir),
            num_confs=1000,
            mcs_mode="auto",
            optimize=True,
            weight_preset="vina",
            top_k=10,
            verbose=True,
        )
        print(f"    lig-align: {result.get('num_poses', 0)} poses, "
              f"best_score={result.get('best_score', 'N/A')}")
        return {
            "tool": "lig_align",
            "output": result.get("output_file"),
            "num_poses": result.get("num_poses"),
            "best_score": result.get("best_score"),
            "mcs_size": result.get("mcs_size"),
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    lig-align failed: {e}")
        return None


def extract_target_smiles(input_yaml: Path) -> list[tuple[str, str]]:
    """Extract (ligand_id, SMILES) from unified input YAML."""
    results = []
    text = input_yaml.read_text()
    current_id, in_ligand = None, False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ligand:") or stripped == "ligand:":
            in_ligand = True
            current_id = None
            continue
        if in_ligand:
            if stripped.startswith("id:"):
                current_id = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("smiles:"):
                smiles = stripped.split(":", 1)[1].strip().strip("'\"")
                results.append((current_id or "L", smiles))
                in_ligand = False
            elif stripped.startswith("- ") or (stripped and not stripped.startswith((" ", "#"))):
                in_ligand = False
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-track docking: template docking + lig-align.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-yaml", type=Path, required=True)
    parser.add_argument("--hits-tsv", type=Path, help="Filtered hits TSV (auto-detected if omitted).")
    parser.add_argument("--rcsb-dir", type=Path, default=Path.home() / "DB/RCSB/raw/mmCIF_data")
    parser.add_argument("--rcsb-db", type=Path, default=Path.home() / "DB/RCSB/processed/rcsb_index.db")
    parser.add_argument("--mcs-threshold", type=float, default=0.5)
    parser.add_argument("--max-templates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()

    # 1. Find filtered hits TSV
    hits_tsv = args.hits_tsv
    if hits_tsv is None:
        hits_tsv = run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
    if not hits_tsv.exists():
        # Try running template filter first
        mmseqs_tsv = run_dir / "outputs" / "template_search_sequence" / "mmseqs_hits.tsv"
        if not mmseqs_tsv.exists():
            print("No template search results found, skipping multi-track docking.")
            return 0

        print("Running template filter on mmseqs hits...")
        ligands = extract_target_smiles(args.input_yaml)
        target_smiles = ligands[0][1] if ligands else None

        from casp17.template_filter import filter_hits_with_ligands
        filter_hits_with_ligands(
            hits_tsv=mmseqs_tsv,
            db_path=args.rcsb_db,
            target_smiles=target_smiles,
            output_path=hits_tsv,
        )

    # 2. Check MCS threshold
    qualifying_hits = check_mcs_hits(hits_tsv, args.mcs_threshold)
    if not qualifying_hits:
        print(f"No template hits with MCS >= {args.mcs_threshold}, "
              "skipping Track 2 + Track 3.")
        return 0

    print(f"\n{'='*60}")
    print(f"  MULTI-TRACK DOCKING: {len(qualifying_hits)} templates with "
          f"MCS >= {args.mcs_threshold}")
    print(f"{'='*60}")

    # 3. Extract target SMILES
    ligands = extract_target_smiles(args.input_yaml)
    if not ligands:
        print("No ligand SMILES found in input, skipping.")
        return 1
    target_smiles = ligands[0][1]

    # 4. Prepare template docking inputs
    template_dock_dir = run_dir / "inputs" / "template_docking"
    templates = run_template_docking_prep(
        hits_tsv=hits_tsv,
        rcsb_dir=args.rcsb_dir,
        input_yaml=args.input_yaml,
        output_dir=template_dock_dir,
        rcsb_db=args.rcsb_db,
        max_templates=args.max_templates,
    )
    if not templates:
        print("Template docking prep produced no templates.")
        return 0

    # 5. Run Track 2 + Track 3 for each template
    all_results: list[dict] = []
    for i, template in enumerate(templates):
        pdb_id = template.get("template_pdb_id", f"template_{i}")
        pident = template.get("template_pident", 0)
        mcs = template.get("best_mcs_coverage", 0)

        print(f"\n{'='*60}")
        print(f"  Template {i+1}/{len(templates)}: {pdb_id} "
              f"(pident={pident:.1f}%, MCS={mcs:.2f})")
        print(f"{'='*60}")

        template_out = run_dir / "outputs" / "template_docking" / pdb_id

        # Track 2: Docking tools on template structure
        print("\n  --- Track 2: Template-based docking ---")

        vina_result = run_vina_on_template(
            template, template_out / "vina", seed=args.seed,
        )
        if vina_result:
            all_results.append(vina_result)

        adg_result = run_autodock_gpu_on_template(
            template, template_out / "autodock_gpu", seed=args.seed,
        )
        if adg_result:
            all_results.append(adg_result)

        pxdock_result = run_protenix_dock_on_template(
            template, template_out / "protenix_dock", seed=args.seed,
        )
        if pxdock_result:
            all_results.append(pxdock_result)

        # Track 3: lig-align
        print("\n  --- Track 3: lig-align (MCS-guided) ---")
        lig_align_result = run_lig_align_on_template(
            template, target_smiles, template_out / "lig_align",
        )
        if lig_align_result:
            all_results.append(lig_align_result)

    # 6. Write summary
    summary = {
        "mcs_threshold": args.mcs_threshold,
        "num_templates": len(templates),
        "num_results": len(all_results),
        "results": all_results,
    }
    summary_path = run_dir / "outputs" / "template_docking" / "multi_track_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"  Multi-track docking complete: {len(all_results)} results")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
