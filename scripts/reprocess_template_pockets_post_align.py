#!/usr/bin/env python3
"""Re-extract + re-cluster + re-prep template pockets for runs that
generated their pockets BEFORE the script_builder bridge-order fix.

Background: ``_emit_template_bridges`` used to fire before
``align_cofolding_outputs``, so ``extract_template_pockets`` aligned
each template against the *unaligned* cofold cif. The resulting
``template_consensus_*`` centroids ended up in a different frame than
the docking receptor (which prepare_docking_inputs builds from the
aligned cif), so docking against those centroids was geometrically
broken — boxes landed near other-protomer pockets or outside the
protein entirely.

Already-completed runs have aligned cifs on disk (Stage 2.5 ran), so
re-running just the three downstream steps is enough to repair the
frame mismatch:

  1. ``extract_template_pockets.py --reference-cif <aligned cif>``
  2. ``cluster_template_pockets.py``
  3. ``prepare_docking_inputs.py`` to refresh
     ``binding_site_predictions[template_consensus_*]`` in
     ``docking_prep_summary.json``.

(Re-running cofold + docking themselves is out of scope here — that
would be a much heavier re-do.)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

EXTRACT_SCRIPT = _REPO / "scripts" / "extract_template_pockets.py"
CLUSTER_SCRIPT = _REPO / "scripts" / "cluster_template_pockets.py"
PREP_SCRIPT = _REPO / "scripts" / "prepare_docking_inputs.py"
HUB_PY = _REPO / ".venv" / "bin" / "python"
DOCK_PY = _REPO / ".venvs" / "protenix-dock" / "bin" / "python"
RCSB_DIR = Path("/home/jaemin/DB/RCSB/raw/mmCIF_data")


def _pick_aligned_cif(run: Path) -> Path | None:
    """Match prepare_docking_inputs's preference: best-pLDDT model's
    ``*_aligned.cif``. We use the same priority order to keep frames in
    sync (any aligned cif is in the common frame, but picking via the
    same rule avoids confusion in logs)."""
    for model in ("alphafold3", "boltz2x", "boltz2", "protenix"):
        d = run / "outputs" / model
        if not d.exists():
            continue
        for cif in sorted(d.rglob("*_aligned.cif")):
            return cif
    return None


def _reprocess_one(target_dir: Path) -> dict:
    target = target_dir.name  # e.g. 7hqq_input
    run = target_dir / target
    summary_path = run / "inputs" / "docking" / "docking_prep_summary.json"
    if not summary_path.exists():
        return {"target": target, "status": "no_docking_prep"}

    aligned = _pick_aligned_cif(run)
    if aligned is None:
        return {"target": target, "status": "no_aligned_cif"}

    # 1. Re-extract — reference-cif now pinned to an aligned cif.
    p = subprocess.run(
        [str(HUB_PY), str(EXTRACT_SCRIPT),
         "--run-dir", str(run),
         "--rcsb-dir", str(RCSB_DIR),
         "--reference-cif", str(aligned)],
        capture_output=True, text=True, timeout=2400,
    )
    if p.returncode != 0:
        return {"target": target, "status": "extract_failed",
                "stderr": p.stderr[-400:]}

    # 2. Re-cluster (reads template_pockets.json's reference_cif field).
    pockets_json = run / "outputs" / "template_pockets" / "template_pockets.json"
    if not pockets_json.exists():
        return {"target": target, "status": "no_pockets_json"}
    p = subprocess.run(
        [str(HUB_PY), str(CLUSTER_SCRIPT),
         "--pockets-json", str(pockets_json)],
        capture_output=True, text=True, timeout=300,
    )
    if p.returncode != 0:
        return {"target": target, "status": "cluster_failed",
                "stderr": p.stderr[-400:]}

    # 3. Re-prep — refreshes binding_site_predictions[template_consensus_*].
    # Use the dock venv (matches what the wrapper uses) so meeko/RDKit
    # versions agree with the existing receptor/ligand artifacts.
    input_yaml = run / "inputs" / "boltz_input.yaml"
    p = subprocess.run(
        [str(DOCK_PY), str(PREP_SCRIPT),
         "--input-yaml", str(input_yaml),
         "--run-dir", str(run),
         "--output-dir", str(run / "inputs" / "docking"),
         "--model", "auto"],
        capture_output=True, text=True, timeout=900,
    )
    if p.returncode != 0:
        return {"target": target, "status": "prep_failed",
                "stderr": p.stderr[-400:]}

    # Sanity: read back template_consensus count.
    summary = json.loads(summary_path.read_text())
    n_tc = sum(1 for k in summary.get("binding_site_predictions", {})
               if k.startswith("template_consensus_"))
    return {"target": target, "status": "ok", "n_template_consensus": n_tc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("experiments/msa_e2e_test/runs"))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N targets (0 = all).")
    args = ap.parse_args()

    if args.targets:
        target_dirs = [args.runs_dir / f"{t}_input" for t in args.targets]
    else:
        target_dirs = sorted(d for d in args.runs_dir.iterdir()
                             if d.is_dir() and d.name.endswith("_input"))
    if args.limit:
        target_dirs = target_dirs[: args.limit]

    print(f"[reprocess] {len(target_dirs)} target(s), workers={args.workers}")

    counts: dict[str, int] = {}
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_reprocess_one, d): d.name for d in target_dirs}
        for i, fut in enumerate(as_completed(futs), 1):
            name = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"target": name, "status": "exception", "error": str(e)}
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if res["status"] != "ok":
                failures.append(res)
                print(f"  [{i}/{len(target_dirs)}] {name}: FAIL ({res['status']})")
            elif i % 25 == 0 or i == len(target_dirs):
                print(f"  [{i}/{len(target_dirs)}] {name}: ok "
                      f"(template_consensus={res.get('n_template_consensus')})")

    print()
    print("=== reprocess summary ===")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if failures[:5]:
        print()
        print("first failures:")
        for f in failures[:5]:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
