#!/usr/bin/env bash
#SBATCH --job-name=rerun_t2d_cpu
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d_cpu-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d_cpu-%A_%a.err
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_only
#
# CPU-only array dispatcher for the template→docking rerun. Pairs with
# rerun_t2d_gpu.sbatch.sh.
#
#   sbatch --array=0-498%50 experiments/novel2025_test/rerun_t2d_cpu.sbatch.sh
#
# Each task picks the Nth pipeline yaml (sorted) and runs the CPU half:
# prepare-wrapper → foldseek → align → filter → pocket extract/cluster
# → prepare_docking (with binding-site cache reuse) → vina variants.
# Marker `_rerun_t2d_cpu.done` is written when the half finishes; the
# GPU array dispatcher reads that marker.
#
# Skip rule: if `_rerun_t2d_cpu.done` already exists, the task exits 0
# (set RERUN_FORCE=1 to override). Useful when only some targets need
# replay because they failed.

set -euo pipefail
cd /home/jaemin/project/CASP17

PIPELINE_DIR=experiments/novel2025_test/pipeline
DRIVER=experiments/novel2025_test/rerun_t2d_cpu.sh

mapfile -t TASKS < <(ls "$PIPELINE_DIR"/*_input.yaml | sort)
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
    echo "task $idx out of range (total=$N) — exit 0"
    exit 0
fi

yaml="${TASKS[$idx]}"
target="$(basename "$yaml" .yaml)"
marker="experiments/runs/${target}/_rerun_t2d_cpu.done"

if [[ -f "$marker" && -z "${RERUN_FORCE:-}" ]]; then
    echo "[$idx/$N] $target: cpu marker present — skip"
    exit 0
fi

echo "[$idx/$N] $target: cpu half start"
bash "$DRIVER" "$target"
echo "[$idx/$N] $target: cpu half ok"
