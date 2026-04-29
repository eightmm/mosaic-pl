#!/usr/bin/env bash
#SBATCH --job-name=rerun_t2d
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=6000ada
#
# Array dispatcher for the template→docking rerun. Submit with explicit
# --array (mirrors run_array.sbatch.sh):
#
#   sbatch --array=0-498%30 experiments/novel2025_test/rerun_template_to_dock.sbatch.sh
#
# Each task picks the Nth pipeline yaml (sorted) and invokes the per-target
# driver. Skips already-rerun targets by checking a marker file written at
# the end of the driver — set RERUN_FORCE=1 in the environment to override.

set -euo pipefail
cd /home/jaemin/project/CASP17

PIPELINE_DIR=experiments/novel2025_test/pipeline
DRIVER=experiments/novel2025_test/rerun_template_to_dock.sh

mapfile -t TASKS < <(ls "$PIPELINE_DIR"/*_input.yaml | sort)
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
    echo "task $idx out of range (total=$N) — exit 0"
    exit 0
fi

yaml="${TASKS[$idx]}"
target="$(basename "$yaml" .yaml)"

# Marker-based skip: each successful driver run touches this file at the
# very end. If present + RERUN_FORCE not set, skip without re-running. The
# marker lives next to the LG submission so a manual `rm` resets one
# target at a time.
marker="experiments/runs/${target}/_rerun_t2d.done"
if [[ -f "$marker" && -z "${RERUN_FORCE:-}" ]]; then
    echo "[$idx/$N] $target: marker present — skip (set RERUN_FORCE=1 to override)"
    exit 0
fi

echo "[$idx/$N] $target: rerun start"
if bash "$DRIVER" "$target"; then
    touch "$marker"
    echo "[$idx/$N] $target: rerun ok"
else
    rc=$?
    echo "[$idx/$N] $target: rerun FAILED rc=$rc"
    exit $rc
fi
