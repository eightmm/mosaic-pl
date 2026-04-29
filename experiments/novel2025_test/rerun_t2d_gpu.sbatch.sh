#!/usr/bin/env bash
#SBATCH --job-name=rerun_t2d_gpu
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d_gpu-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_t2d_gpu-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=6000ada
#
# GPU array dispatcher for the template→docking rerun. Run AFTER the CPU
# array completes (or alongside it once a few CPU markers exist). Each
# task waits for `_rerun_t2d_cpu.done` of its target; if missing, exits
# 0 so you can re-submit the array later.
#
#   sbatch --array=0-498%30 experiments/novel2025_test/rerun_t2d_gpu.sbatch.sh
#
# Two practical patterns:
#   1. Submit GPU after CPU finishes (simplest):
#        sbatch --array=... cpu.sbatch.sh
#        # wait for it
#        sbatch --array=... gpu.sbatch.sh
#   2. Submit both at once with afterany dependency on the CPU job-id:
#        JOB=$(sbatch --array=0-498%50 cpu.sbatch.sh | awk '{print $4}')
#        sbatch --dependency=aftercorr:$JOB --array=0-498%30 gpu.sbatch.sh
#      With aftercorr each GPU task starts as its CPU peer finishes.

set -euo pipefail
cd /home/jaemin/project/CASP17

PIPELINE_DIR=experiments/novel2025_test/pipeline
DRIVER=experiments/novel2025_test/rerun_t2d_gpu.sh

mapfile -t TASKS < <(ls "$PIPELINE_DIR"/*_input.yaml | sort)
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
    echo "task $idx out of range (total=$N) — exit 0"
    exit 0
fi

yaml="${TASKS[$idx]}"
target="$(basename "$yaml" .yaml)"
cpu_marker="experiments/runs/${target}/_rerun_t2d_cpu.done"
gpu_marker="experiments/runs/${target}/_rerun_t2d_gpu.done"

if [[ -f "$gpu_marker" && -z "${RERUN_FORCE:-}" ]]; then
    echo "[$idx/$N] $target: gpu marker present — skip"
    exit 0
fi
if [[ ! -f "$cpu_marker" ]]; then
    echo "[$idx/$N] $target: cpu half not done — skip (re-submit gpu array later)"
    exit 0
fi

echo "[$idx/$N] $target: gpu half start"
bash "$DRIVER" "$target"
echo "[$idx/$N] $target: gpu half ok"
