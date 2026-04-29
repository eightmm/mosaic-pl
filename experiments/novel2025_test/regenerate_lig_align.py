"""Regenerate lig_align Track-3 outputs for all runs that have empty
``template_docking/<pdb>/lig_align/`` directories.

The original wrapper invoked ``run_multi_track_docking.py`` from the
``protenix-dock`` venv, which doesn't ship with ``lig_align`` — so the
``from lig_align import run_pipeline`` silently raised ImportError and
zero targets got Track-3 poses. We fixed ``run_multi_track_docking.py``
to shell out to the main venv for lig_align; this script walks existing
run dirs and triggers only the lig_align portion for any template that
didn't produce an SDF, leaving vina/adg outputs untouched.

Usage:
    # single target (for sanity)
    python regenerate_lig_align.py --run-dir experiments/runs/10sl_input

    # all run dirs listed in /tmp/rerun_runs.txt
    python regenerate_lig_align.py --runlist /tmp/rerun_runs.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))


def extract_primary_smiles(input_yaml: Path) -> str | None:
    """Pull the first ``smiles:`` value from a Boltz input YAML (same
    heuristic as ``run_multi_track_docking.extract_target_smiles`` but
    returns just the primary string instead of a (id, smi) list)."""
    if not input_yaml.exists():
        return None
    for line in input_yaml.read_text().splitlines():
        line = line.strip()
        if line.startswith("smiles:"):
            return line.split("smiles:", 1)[1].strip().strip("'\"")
    return None


def regenerate_one(run_dir: Path) -> dict:
    """Run lig_align for every template that doesn't already have an SDF."""
    from run_multi_track_docking import run_lig_align_on_template  # type: ignore

    template_root = run_dir / "outputs" / "template_docking"
    if not template_root.exists():
        return {"run_dir": run_dir.name, "status": "skip_no_templates"}

    input_yaml = run_dir / "inputs" / "boltz_input.yaml"
    smiles = extract_primary_smiles(input_yaml)
    if not smiles:
        return {"run_dir": run_dir.name, "status": "skip_no_smiles"}

    out_results: list[dict] = []
    for tpl_dir in sorted(template_root.iterdir()):
        if not tpl_dir.is_dir():
            continue
        lig_align_out = tpl_dir / "lig_align"
        if lig_align_out.exists() and any(lig_align_out.glob("*.sdf")):
            out_results.append({"template": tpl_dir.name, "status": "already_has_sdf"})
            continue
        prep = run_dir / "inputs" / "template_docking" / f"template_{tpl_dir.name}" / "docking_prep_summary.json"
        if not prep.exists():
            out_results.append({"template": tpl_dir.name, "status": "no_prep_summary"})
            continue
        try:
            tpl_dict = json.loads(prep.read_text())
        except Exception as e:
            out_results.append({"template": tpl_dir.name, "status": f"prep_parse_fail: {e}"})
            continue
        print(f"  → lig_align: {run_dir.name}/{tpl_dir.name}", flush=True)
        result = run_lig_align_on_template(tpl_dict, smiles, lig_align_out)
        out_results.append({
            "template": tpl_dir.name,
            "status": "ok" if result else "failed",
            "num_poses": (result or {}).get("num_poses"),
            "mcs_size": (result or {}).get("mcs_size"),
        })
    return {"run_dir": run_dir.name, "status": "processed", "templates": out_results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runlist", type=Path, default=None,
                        help="File listing run_dir basenames (under experiments/runs/), one per line")
    parser.add_argument("--idx", type=int, default=None,
                        help="When using --runlist, process only this index (0-based). Intended for SLURM arrays.")
    parser.add_argument("--runs-root", type=Path,
                        default=_REPO / "experiments" / "runs")
    args = parser.parse_args()

    if args.run_dir is not None:
        result = regenerate_one(args.run_dir)
        print(json.dumps(result, indent=2))
        return 0

    if args.runlist is None:
        parser.error("Provide --run-dir or --runlist")

    names = [l.strip() for l in args.runlist.read_text().splitlines() if l.strip()]
    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(names):
            print(f"idx {args.idx} out of range (0..{len(names)-1})", file=sys.stderr)
            return 1
        names = [names[args.idx]]

    for nm in names:
        rd = args.runs_root / nm
        if not rd.exists():
            print(f"skip missing: {rd}")
            continue
        try:
            result = regenerate_one(rd)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"run_dir": nm, "status": f"crash: {e!r}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
