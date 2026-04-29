#!/usr/bin/env bash
#SBATCH --job-name=novel2025
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/slurm-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/slurm-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=6000ada
# --array is set at submit time (`sbatch --array=0-498%30 run_array.sbatch.sh`)
#
# Array-task dispatcher: each task picks the Nth wrapper from TASK_LIST
# (in sorted order), skipping any whose LG submission already exists so
# the array is idempotent across restarts.

set -euo pipefail
cd /home/jaemin/project/CASP17

PIPELINE_DIR=experiments/novel2025_test/pipeline
RUNS_ROOT=experiments/runs
SUBMIT_DIR=experiments/submissions

# Build the deterministic task list (same order every time).
mapfile -t TASKS < <(ls "$PIPELINE_DIR"/*_input.yaml | sort)
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
  echo "task $idx out of range (total=$N) — nothing to do"
  exit 0
fi

yaml="${TASKS[$idx]}"
base="$(basename "$yaml" .yaml)"       # e.g. 22mj_input
wrapper="$RUNS_ROOT/${base}/scripts/run_wrapper.sbatch.sh"
lg="$SUBMIT_DIR/${base}.lg"

if [ -f "$lg" ]; then
  echo "[$idx/$N] $base: LG already exists — skipping"
  exit 0
fi
if [ ! -f "$wrapper" ]; then
  echo "[$idx/$N] $base: wrapper missing at $wrapper — skipping"
  exit 0
fi

echo "[$idx/$N] running $base (wrapper=$wrapper)"
# Execute the wrapper inline (SBATCH directives inside are no-ops when
# invoked via bash; we've already claimed the resources on this array task).
bash "$wrapper"
