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


def check_template_hits(hits_tsv: Path) -> list[dict]:
    """Return template hits with at least one bound ligand.

    Track 2 (template-based box docking) only needs the template's
    receptor pocket geometry — the template ligand's atom graph doesn't
    have to match the query. Any template with a bound ligand gives a
    usable box. The MCS threshold only gates Track 3 (lig-MCS-align),
    which actually overlays the template ligand atoms onto the query.
    """
    if not hits_tsv.exists():
        return []
    hits = []
    with open(hits_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            num_lig = int(row.get("num_ligands", 0))
            if num_lig > 0:
                hits.append(row)
    return hits


def run_template_docking_prep(
    hits_tsv: Path,
    rcsb_dir: Path,
    input_yaml: Path,
    output_dir: Path,
    rcsb_db: Path | None = None,
    max_templates: int = 10,
    cofold_ref_cif: Path | None = None,
    cofold_lig_anchor: list[float] | None = None,
    pockets_json: Path | None = None,
) -> list[dict]:
    """Run prepare_template_docking.py and return template summaries.

    When ``pockets_json`` is provided (the ``template_pockets.json`` written
    by ``extract_template_pockets``), prepare_template_docking selects one
    representative template per pocket cluster instead of the legacy
    sort-top-N. Default ``max_templates=10`` matches
    ``TEMPLATE_CONSENSUS_TOP_K`` so Track 2 docks the same K cluster sites
    Track 1's vina_template_consensus_* sources cover.
    """
    script = Path(__file__).resolve().parent / "prepare_template_docking.py"
    cmd = [
        sys.executable, str(script),
        "--hits-tsv", str(hits_tsv),
        "--rcsb-dir", str(rcsb_dir),
        "--input-yaml", str(input_yaml),
        "--output-dir", str(output_dir),
        "--max-templates", str(max_templates),
    ]
    if pockets_json is not None and pockets_json.exists():
        cmd += ["--pockets-json", str(pockets_json)]
    if cofold_ref_cif is not None:
        cmd += ["--cofold-ref-cif", str(cofold_ref_cif)]
    if cofold_lig_anchor is not None:
        cmd += ["--cofold-lig-anchor",
                f"{cofold_lig_anchor[0]}",
                f"{cofold_lig_anchor[1]}",
                f"{cofold_lig_anchor[2]}"]
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


def run_vina_on_template(
    template: dict, output_dir: Path, seed: int = 42,
    ligand: dict | None = None,
) -> dict | None:
    """Run AutoDock Vina on a template-prepared structure for one ligand.

    The ``ligand`` arg is one entry from ``template["ligands"]`` (the
    multi-ligand list ``prepare_template_docking`` writes). When ``None``
    falls back to the first ligand for backward compat with single-ligand
    runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_pdbqt = template.get("receptor_pdbqt")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdbqt or not ligands:
        return None

    if ligand is None:
        ligand = ligands[0]
    ligand_pdbqt = ligand.get("pdbqt")
    lig_id = ligand.get("id", "L")
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
            "ligand_id": lig_id,
            "output": str(out_path),
            "best_score": best_score,
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    Vina failed: {e}")
        return None


def run_autodock_gpu_on_template(
    template: dict, output_dir: Path, seed: int = 42,
    ligand: dict | None = None,
) -> dict | None:
    """Run AutoDock-GPU on a template-prepared structure for one ligand."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    receptor_pdbqt = template.get("receptor_pdbqt")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdbqt or not ligands:
        return None

    if ligand is None:
        ligand = ligands[0]
    ligand_pdbqt = ligand.get("pdbqt")
    lig_id = ligand.get("id", "L")
    if not ligand_pdbqt or not Path(ligand_pdbqt).exists():
        return None

    adg_binary = repo_root / ".local" / "bin" / "autodock_gpu_128wi"
    autogrid_binary = repo_root / ".local" / "bin" / "autogrid4"
    if not adg_binary.exists() or not autogrid_binary.exists():
        print("    AutoDock-GPU/autogrid4 not found, skipping.")
        return None

    grid_dir = output_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Generate GPF — receptor / ligand atom types parsed from the actual
    # PDBQTs so nucleic acid receptors (P from phosphate), retained
    # metals, and ligands with halogens/Si etc. all reach the grid maps.
    def _parse_pdbqt_types(path: str) -> list[str]:
        types: list[str] = []
        seen: set[str] = set()
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")):
                        tok = line[77:79].strip() if len(line) >= 79 else line.split()[-1].strip()
                        if tok and tok not in seen:
                            seen.add(tok)
                            types.append(tok)
        except OSError:
            pass
        return types

    base_types = ["A", "C", "HD", "N", "NA", "OA", "SA"]
    rec_actual = _parse_pdbqt_types(receptor_pdbqt)
    lig_actual = _parse_pdbqt_types(ligand_pdbqt)
    if not lig_actual:
        lig_actual = list(base_types)
    rec_types = list(dict.fromkeys(base_types + rec_actual + lig_actual))
    lig_types = list(dict.fromkeys(base_types + lig_actual))

    npts = [max(1, int(s / 0.375)) for s in size]
    gpf_path = grid_dir / "receptor.gpf"
    fld_path = grid_dir / "receptor.maps.fld"
    gpf_lines = [
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"gridfld {fld_path.name}",
        "spacing 0.375",
        f"receptor_types {' '.join(rec_types)}",
        f"ligand_types {' '.join(lig_types)}",
        f"receptor {receptor_pdbqt}",
        f"gridcenter {center[0]} {center[1]} {center[2]}",
        "smooth 0.5",
    ]
    # One map line per ligand atom type; elecmap + dsolvmap always.
    for t in lig_types:
        gpf_lines.append(f"map receptor.{t}.map")
    gpf_lines += [
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
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
            "ligand_id": lig_id,
            "output": str(dlg_file),
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    AutoDock-GPU failed: {e}")
        return None


def run_protenix_dock_on_template(
    template: dict, output_dir: Path, seed: int = 42,
    ligand: dict | None = None,
) -> dict | None:
    """Run Protenix-Dock on a template-prepared structure for one ligand."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    receptor_pdb = template.get("receptor_pdb")
    ligands = template.get("ligands", [])
    center = template.get("box_center", [0, 0, 0])
    size = template.get("box_size", [22.5, 22.5, 22.5])

    if not receptor_pdb or not ligands:
        return None

    if ligand is None:
        ligand = ligands[0]
    ligand_sdf = ligand.get("sdf")
    lig_id = ligand.get("id", "L")
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
            "ligand_id": lig_id,
            "output": str(output_dir),
            "template_pdb_id": template.get("template_pdb_id"),
        }
    except Exception as e:
        print(f"    Protenix-Dock failed: {e}")
        return None


_MAIN_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"


def run_lig_align_on_template(
    template: dict, target_smiles: str, output_dir: Path,
    ligand_id: str = "L",
) -> dict | None:
    """Run lig-align: MCS-guided pose generation from template ligand.

    ``lig_align`` is only installed in the main ``.venv`` (it needs a
    different torch/rdkit stack than the protenix-dock venv this script
    runs in). We shell out to that interpreter via a small inline driver
    so the import happens in the correct environment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdb = template.get("receptor_pdb_raw") or template.get("receptor_pdb")
    ref_ligand_sdf = template.get("template_ligand_sdf")

    if not receptor_pdb or not Path(receptor_pdb).exists():
        print("    lig-align: receptor PDB not found, skipping.")
        return None
    if not ref_ligand_sdf or not Path(ref_ligand_sdf).exists():
        print("    lig-align: template ligand SDF not found, skipping.")
        return None
    if not _MAIN_VENV_PYTHON.exists():
        print(f"    lig-align: main venv python not found at {_MAIN_VENV_PYTHON}, skipping.")
        return None

    payload = {
        "protein_pdb": str(receptor_pdb),
        "ref_ligand": str(ref_ligand_sdf),
        "query_ligand": target_smiles,
        "output_dir": str(output_dir),
    }
    # num_confs + optimize=False chosen so a 3-template run stays inside a
    # 10-min per-target budget on CPU. Production lig_align with optimize=True
    # was observed to exceed 90min on some pockets — well outside what the
    # wrapper's SLURM 12h cap can afford across Track 2 (vina + adg) and this
    # stage combined.
    driver = (
        "import json, sys\n"
        "from lig_align import run_pipeline\n"
        "args = json.loads(sys.stdin.read())\n"
        "result = run_pipeline(\n"
        "    protein_pdb=args['protein_pdb'],\n"
        "    ref_ligand=args['ref_ligand'],\n"
        "    query_ligand=args['query_ligand'],\n"
        "    output_dir=args['output_dir'],\n"
        "    num_confs=500,\n"
        "    mcs_mode='auto',\n"
        "    optimize=False,\n"
        "    weight_preset='vina',\n"
        "    top_k=5,\n"
        "    verbose=False,\n"
        ")\n"
        "print('__LIG_ALIGN_RESULT__' + json.dumps({\n"
        "    'output_file': result.get('output_file'),\n"
        "    'num_poses': result.get('num_poses'),\n"
        "    'best_score': result.get('best_score'),\n"
        "    'mcs_size': result.get('mcs_size'),\n"
        "}))\n"
    )
    try:
        proc = subprocess.run(
            [str(_MAIN_VENV_PYTHON), "-c", driver],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("    lig-align: timed out after 900s")
        return None
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:] if proc.stderr else []
        print(f"    lig-align failed (exit {proc.returncode}): {' | '.join(tail)}")
        return None
    marker = "__LIG_ALIGN_RESULT__"
    line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        print("    lig-align: driver produced no result marker")
        return None
    try:
        result = json.loads(line[len(marker):])
    except json.JSONDecodeError as e:
        print(f"    lig-align: failed to parse driver output: {e}")
        return None
    print(f"    lig-align: {result.get('num_poses', 0)} poses, "
          f"best_score={result.get('best_score', 'N/A')}")
    return {
        "tool": "lig_align",
        "ligand_id": ligand_id,
        "output": result.get("output_file"),
        "num_poses": result.get("num_poses"),
        "best_score": result.get("best_score"),
        "mcs_size": result.get("mcs_size"),
        "template_pdb_id": template.get("template_pdb_id"),
    }


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
    parser.add_argument("--max-templates", type=int, default=10,
                        help="Max Track 2 templates. Default 10 matches "
                             "TEMPLATE_CONSENSUS_TOP_K so Track 2 covers "
                             "the same cluster set Track 1's "
                             "vina_template_consensus_* uses.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-pxdock",
        action="store_true",
        help="Skip Protenix-Dock in Track 2 (keeps Vina + ADG + lig-MCS-align).",
    )
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
    qualifying_hits = check_template_hits(hits_tsv)
    if not qualifying_hits:
        print("No template hits with bound ligands, skipping Track 2 + Track 3.")
        return 0

    print(f"\n{'='*60}")
    print(f"  MULTI-TRACK DOCKING: {len(qualifying_hits)} templates available")
    print(f"  (Track 2: all templates; Track 3 lig-MCS-align: MCS >= "
          f"{args.mcs_threshold} only)")
    print(f"{'='*60}")

    # 3. Extract target SMILES
    ligands = extract_target_smiles(args.input_yaml)
    if not ligands:
        print("No ligand SMILES found in input, skipping.")
        return 1
    target_smiles = ligands[0][1]

    # 4. Prepare template docking inputs
    template_dock_dir = run_dir / "inputs" / "template_docking"
    # Resolve cofold reference CIF — Track 1 docking-prep already picked
    # one best model and recorded it in docking_prep_summary.cofolding_structure.
    # Honour that pick so Track 2 lands in the same coordinate frame Track 1
    # uses; otherwise the eval transform (cofold→crystal) misplaces every
    # Track 2 pose by ~20 Å.
    cofold_ref_cif: Path | None = None
    cofold_lig_anchor: list[float] | None = None
    prep_summary = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if prep_summary.exists():
        try:
            data = json.loads(prep_summary.read_text())
            cof_path = data.get("cofolding_structure")
            if cof_path and Path(cof_path).exists():
                cofold_ref_cif = Path(cof_path)
            # Track 1's box_method == "cofolding_*" gives us the cofold
            # ligand centroid; on multimers that's the chain we want
            # Track 2 to dock against too. cofolding_1 is the largest
            # cluster (most placements) so it's the canonical anchor;
            # if missing (low-confidence target), fall back to whatever
            # cofolding_N is registered.
            bs_preds = data.get("binding_site_predictions") or {}
            cofold_keys = sorted(k for k in bs_preds.keys() if k.startswith("cofolding_"))
            bs = bs_preds[cofold_keys[0]] if cofold_keys else {}
            anchor = bs.get("center") or (
                data.get("box_center")
                if str(data.get("box_method", "")).startswith("cofolding")
                else None
            )
            if anchor and len(anchor) == 3:
                cofold_lig_anchor = [float(v) for v in anchor]
        except Exception:
            cofold_ref_cif = None
            cofold_lig_anchor = None
    if cofold_ref_cif is None:
        # Fallback: first aligned cofold cif we can find.
        candidates = sorted((run_dir / "outputs").rglob("*_aligned.cif"))
        if candidates:
            cofold_ref_cif = candidates[0]
    if cofold_ref_cif is not None:
        print(f"  cofold reference: {cofold_ref_cif}")
        if cofold_lig_anchor is not None:
            print(f"  cofold ligand anchor (chain pick): {cofold_lig_anchor}")
    else:
        print("  WARNING: no cofold reference found; Track 2 outputs will "
              "stay in template frame (eval will be wrong)")

    # Cluster-aware Track 2: when extract_template_pockets has produced a
    # pockets json, prepare_template_docking picks one representative
    # template per pocket cluster (matches Track 1's
    # vina_template_consensus_* coverage). Without it, falls back to
    # filtered_hits.tsv sort top-N.
    pockets_json = run_dir / "outputs" / "template_pockets" / "template_pockets.json"
    if not pockets_json.exists():
        pockets_json = None

    templates = run_template_docking_prep(
        hits_tsv=hits_tsv,
        rcsb_dir=args.rcsb_dir,
        input_yaml=args.input_yaml,
        output_dir=template_dock_dir,
        rcsb_db=args.rcsb_db,
        max_templates=args.max_templates,
        cofold_ref_cif=cofold_ref_cif,
        cofold_lig_anchor=cofold_lig_anchor,
        pockets_json=pockets_json,
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

        # Multi-ligand expansion: iterate every dockable ligand the template
        # was prepped with (prepare_template_docking writes one entry per
        # ligand from the input YAML's smiles-bearing entries; ccd-only
        # entries like ions are skipped at prep time so they never reach
        # this loop).
        template_ligands = template.get("ligands", []) or []
        if not template_ligands:
            print("    Template has no dockable ligands, skipping all tracks.")
            continue

        # Track 2: Docking tools on template structure (per-ligand)
        print("\n  --- Track 2: Template-based docking ---")
        for tlig in template_ligands:
            lig_id = tlig.get("id", "L")
            lig_dir_suffix = f"ligand_{lig_id}"
            print(f"    [ligand {lig_id}]")

            vina_result = run_vina_on_template(
                template, template_out / "vina" / lig_dir_suffix,
                seed=args.seed, ligand=tlig,
            )
            if vina_result:
                all_results.append(vina_result)

            adg_result = run_autodock_gpu_on_template(
                template, template_out / "autodock_gpu" / lig_dir_suffix,
                seed=args.seed, ligand=tlig,
            )
            if adg_result:
                all_results.append(adg_result)

            if args.skip_pxdock:
                print(f"      Protenix-Dock: skipped (--skip-pxdock).")
            else:
                pxdock_result = run_protenix_dock_on_template(
                    template, template_out / "protenix_dock" / lig_dir_suffix,
                    seed=args.seed, ligand=tlig,
                )
                if pxdock_result:
                    all_results.append(pxdock_result)

        # Track 3: lig-MCS-align (per-ligand) — gated by MCS threshold
        # because it performs an actual atom-mapped overlay of the template
        # ligand. Below threshold the MCS is too small to produce a
        # meaningful alignment; skip rather than emit a low-quality pose.
        # Note: MCS is computed from filtered_hits.tsv which currently
        # tracks the primary target ligand only — per-ligand MCS would
        # need filtered_hits annotations per (template, target_ligand)
        # pair. For multi-ligand targets all ligands inherit the same
        # MCS gate decision until that lands.
        if mcs >= args.mcs_threshold:
            print(f"\n  --- Track 3: lig-MCS-align (MCS={mcs:.2f} >= {args.mcs_threshold}) ---")
            for tlig in template_ligands:
                lig_id = tlig.get("id", "L")
                lig_smiles = tlig.get("smiles") or target_smiles
                if not lig_smiles:
                    continue
                lig_align_result = run_lig_align_on_template(
                    template, lig_smiles,
                    template_out / "lig_align" / f"ligand_{lig_id}",
                    ligand_id=lig_id,
                )
                if lig_align_result:
                    all_results.append(lig_align_result)
        else:
            print(f"\n  --- Track 3: skipped (MCS={mcs:.2f} < {args.mcs_threshold}) ---")

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
