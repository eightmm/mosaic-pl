#!/usr/bin/env python3
"""Re-run only the template_consensus_* docking variants (+ Track 2/3 +
post-analysis + submission) for a list of targets whose
``template_pocket_clusters.json`` was rewritten by
``reprocess_template_pockets_post_align.py``.

Why not a full pipeline rerun:
    Cofold / Stage 2.5 alignment / receptor prep / cofolding/swinsite/p2rank
    docking variants were never frame-broken. Cofold alone is ~36 min/target;
    redoing it for 154 targets is wasted GPU-hours. The only outputs that
    docked against wrong-frame box centers were ``vina_template_consensus_*``
    + ``autodock_gpu_template_consensus_*`` (Track 1) and the multi-track
    Track 2/3 (because cluster representatives were chosen from the
    wrong-frame cluster json). Those are the only artefacts this driver
    rebuilds.

What this driver does, per target:
    1. Cleanup (delete artefacts that were derived from wrong-frame data):
       ``outputs/{vina,autodock_gpu}_template_consensus_*``
       ``outputs/template_docking``
       ``outputs/analysis``
       ``submissions/<target>.lg``
    2. Re-dock template_consensus_{1..10} via Vina + ADG × 5 seeds, reusing
       the per-target runner scripts already on disk
       (``scripts/run_{vina,autodock_gpu}_template_consensus_*.py``). They
       read the FRESH ``docking_prep_summary.json`` so the new box centers
       land in the aligned frame.
    3. Re-run ``run_multi_track_docking.py`` (Track 2 cluster-rep selection
       now reads the new clusters, Track 3 re-derives MCS gating on those).
    4. Re-run ``run_post_analysis.py`` so BA-Pred / RMSD-Pred TSVs reflect
       the new poses (existing cofold/swinsite/p2rank/PxDock TSVs get
       re-staged as well — cheap, ~1-2 min/target).
    5. Re-aggregate scores + regenerate the LG submission.

Outputs are dropped at the SAME paths as the original wrapper, so post-
analysis / submission code doesn't need to know we re-ran. Anything we
don't touch (cofold cifs, vina/adg cofolding/swinsite/p2rank, PxDock,
ion placement, alignment_summary.json, template_pockets/) is preserved.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCK_PY = REPO / ".venvs" / "protenix-dock" / "bin" / "python"
PRED_PY = REPO / ".venvs" / "pred" / "bin" / "python"
HUB_PY = REPO / ".venv" / "bin" / "python"

DOCK_SEEDS = [42, 101, 202, 303, 404]

# Match the original wrapper's submission knobs so the regenerated .lg
# file is identical to what the SLURM job would have produced.
SUBMISSION_AUTHOR = "6095-5696-9732"
SUBMISSION_METHOD = (
    "Boltz-2x + Multi-track ensemble + lig-align (5 seeds x 5 samples) "
    "[time-split 2025]"
)
SUBMISSION_PARENT = "N/A"
SUBMISSION_DIR_NAME = "submissions"


def _build_sbatch(target: str, run_dir: Path, runs_root: Path,
                  partition: str = "6000ada", time_limit: str = "06:00:00",
                  log_dir: Path | None = None) -> Path:
    """Generate a per-target sbatch script that does cleanup → redock →
    multi-track → post-analysis → submission. Returns the script path."""
    run_dir = run_dir.resolve()
    runs_root = runs_root.resolve()
    scripts_dir = run_dir / "scripts"
    log_dir = (log_dir or runs_root.parent / "logs" / "redock").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = scripts_dir / "redock_template_consensus.sbatch.sh"
    # Match the original wrapper exactly: it writes to ``runs/submissions/``
    # (sibling of the per-target run dirs), NOT ``../submissions/``. Mismatch
    # would scatter redock outputs to a different dir than main-batch ones.
    submission_path = (runs_root / SUBMISSION_DIR_NAME / f"{target}_LCDD.lg").resolve()
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the per-source × per-seed redock invocations from the existing
    # variant runner scripts. We don't regenerate those .py files — prep's
    # JSON is the single source of truth and the runner scripts read it
    # at runtime.
    redock_lines: list[str] = []
    for src_idx in range(1, 11):
        src = f"template_consensus_{src_idx}"
        vina_script = scripts_dir / f"run_vina_{src}.py"
        adg_script = scripts_dir / f"run_autodock_gpu_{src}.py"
        if vina_script.exists():
            for seed in DOCK_SEEDS:
                out = run_dir / "outputs" / f"vina_{src}" / f"seed_{seed}"
                redock_lines.append(
                    f"timeout 900 env DOCK_SEED={seed} "
                    f"DOCK_OUT_DIR={shlex.quote(str(out))} "
                    f"{shlex.quote(str(DOCK_PY))} "
                    f"{shlex.quote(str(vina_script))} "
                    f"|| echo '  (vina {src} seed {seed} failed, continuing)'"
                )
        if adg_script.exists():
            for seed in DOCK_SEEDS:
                out = run_dir / "outputs" / f"autodock_gpu_{src}" / f"seed_{seed}"
                redock_lines.append(
                    f"timeout 900 env DOCK_SEED={seed} "
                    f"DOCK_OUT_DIR={shlex.quote(str(out))} "
                    f"{shlex.quote(str(DOCK_PY))} "
                    f"{shlex.quote(str(adg_script))} "
                    f"|| echo '  (adg {src} seed {seed} failed, continuing)'"
                )

    cleanup_targets = [
        run_dir / "outputs" / "template_docking",
        run_dir / "outputs" / "analysis",
        # vina_template_consensus_* and autodock_gpu_template_consensus_*
        # caught via shell glob inside the script.
    ]
    cleanup_block = "\n".join(
        f"rm -rf {shlex.quote(str(p))}" for p in cleanup_targets
    )

    cmd_multi = (
        f"{shlex.quote(str(DOCK_PY))} "
        f"{shlex.quote(str(REPO / 'scripts' / 'run_multi_track_docking.py'))} "
        f"--run-dir {shlex.quote(str(run_dir))} "
        f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
        f"--rcsb-dir /home/jaemin/DB/RCSB/raw/mmCIF_data "
        f"--rcsb-db /home/jaemin/DB/RCSB/processed/rcsb_index.db "
        f"--mcs-threshold 0.5 --skip-pxdock"
    )
    cmd_post = (
        f"{shlex.quote(str(PRED_PY))} "
        f"{shlex.quote(str(REPO / 'scripts' / 'run_post_analysis.py'))} "
        f"--run-dir {shlex.quote(str(run_dir))} --device cuda"
    )
    cmd_submission = (
        f"{shlex.quote(str(HUB_PY))} "
        f"{shlex.quote(str(REPO / 'scripts' / 'make_casp_submission.py'))} "
        f"--run-dir {shlex.quote(str(run_dir))} "
        f"--target-id {target}_input "
        f"--author {SUBMISSION_AUTHOR} "
        f"--method {shlex.quote(SUBMISSION_METHOD)} "
        f"--parent {SUBMISSION_PARENT} "
        f"--output {shlex.quote(str(submission_path))} "
        f"--include-affinity"
    )

    out_dir = run_dir / "outputs"
    redock_block = ("\n".join(redock_lines) if redock_lines
                    else "echo '  (no template_consensus runner scripts found, skipping)'")
    sbatch_body = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={target}_redock\n"
        f"#SBATCH --partition={partition}\n"
        "#SBATCH --gres=gpu:1\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --output={log_dir}/{target}_redock_%j.out\n"
        f"#SBATCH --error={log_dir}/{target}_redock_%j.err\n"
        "\n"
        "set -uo pipefail\n"
        f"cd {shlex.quote(str(REPO))}\n"
        "module load cuda/12.8 2>/dev/null || true\n"
        f"export PATH={shlex.quote(str(REPO / '.local' / 'bin'))}:$PATH\n"
        "\n"
        f'echo "=== redock template_consensus for {target} '
        '(job $SLURM_JOB_ID on $SLURM_NODELIST) ==="\n'
        "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true\n"
        "\n"
        'echo "--- 1/5: cleanup wrong-frame artefacts ---"\n'
        f"for d in {shlex.quote(str(out_dir))}/vina_template_consensus_* "
        f"{shlex.quote(str(out_dir))}/autodock_gpu_template_consensus_*; do\n"
        '    [ -d "$d" ] && rm -rf "$d"\n'
        "done\n"
        f"{cleanup_block}\n"
        "\n"
        'echo "--- 2/5: redock template_consensus variants '
        '(vina + adg, 5 seeds x 10 sources) ---"\n'
        f"{redock_block}\n"
        "\n"
        'echo "--- 3/5: multi-track docking (Track 2 + Track 3) ---"\n'
        f"{cmd_multi} || echo '  (multi-track docking failed, continuing)'\n"
        "\n"
        'echo "--- 4/5: post-analysis (BA-Pred + RMSD-Pred) ---"\n'
        f"{cmd_post} || echo '  (post-analysis failed, continuing)'\n"
        "\n"
        'echo "--- 5/5: regenerate CASP LG submission ---"\n'
        f"{cmd_submission} || echo '  (submission generation failed)'\n"
        "\n"
        f'echo "=== redock done for {target} ==="\n'
    )
    sbatch_path.write_text(sbatch_body)
    sbatch_path.chmod(0o755)
    return sbatch_path


def _slurm_running_or_pending(user: str) -> int:
    """Return the count of running + pending SLURM jobs for ``user`` so we
    can self-throttle under MaxSubmit limits without hammering the
    scheduler."""
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-t", "PD,R"],
            capture_output=True, text=True, timeout=15,
        )
        return len(out.stdout.strip().splitlines()) if out.stdout.strip() else 0
    except Exception:
        return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets-file", type=Path,
                   default=Path("/tmp/reprocess_targets.txt"),
                   help="Newline-separated list of target ids to redock.")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Explicit target ids (overrides --targets-file).")
    p.add_argument("--runs-dir", type=Path,
                   default=Path("experiments/msa_e2e_test/runs"))
    p.add_argument("--partition", default="6000ada")
    p.add_argument("--time-limit", default="06:00:00")
    p.add_argument("--max-jobs-in-queue", type=int, default=400,
                   help="Pause submission when running+pending exceeds this. "
                        "Conservative under the 500 MaxSubmit ceiling.")
    p.add_argument("--throttle-sleep", type=int, default=60)
    p.add_argument("--user", default="jaemin")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate sbatch scripts but don't submit them.")
    p.add_argument("--limit", type=int, default=0,
                   help="Submit at most N targets (0 = all).")
    args = p.parse_args()

    if args.targets:
        targets = list(args.targets)
    elif args.targets_file.exists():
        targets = [t.strip() for t in args.targets_file.read_text().splitlines()
                   if t.strip()]
    else:
        print(f"error: provide --targets or write to {args.targets_file}",
              file=sys.stderr)
        return 2
    if args.limit:
        targets = targets[: args.limit]
    print(f"[redock] {len(targets)} target(s) queued")

    submitted = 0
    skipped = 0
    failed: list[str] = []
    for i, t in enumerate(targets, 1):
        run_dir = args.runs_dir / f"{t}_input" / f"{t}_input"
        if not (run_dir / "inputs/docking/docking_prep_summary.json").exists():
            print(f"  [{i}/{len(targets)}] {t}: SKIP (no docking_prep_summary)")
            skipped += 1
            continue
        sbatch = _build_sbatch(t, run_dir, args.runs_dir,
                               partition=args.partition,
                               time_limit=args.time_limit)
        if args.dry_run:
            print(f"  [{i}/{len(targets)}] {t}: dry-run sbatch={sbatch}")
            continue
        # Throttle to stay under MaxSubmit.
        while _slurm_running_or_pending(args.user) >= args.max_jobs_in_queue:
            print(f"  queue at limit ({args.max_jobs_in_queue}), sleeping "
                  f"{args.throttle_sleep}s...")
            time.sleep(args.throttle_sleep)
        r = subprocess.run(
            ["sbatch", str(sbatch)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  [{i}/{len(targets)}] {t}: sbatch FAILED — {r.stderr.strip()}")
            failed.append(t)
            continue
        submitted += 1
        if i % 25 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] {t}: submitted ({r.stdout.strip()})")

    print()
    print("=== redock dispatch summary ===")
    print(f"  submitted:  {submitted}")
    print(f"  skipped:    {skipped}")
    print(f"  failed:     {len(failed)}")
    if failed:
        print(f"  failures:   {failed[:5]}{'...' if len(failed) > 5 else ''}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
