#!/usr/bin/env bash
#SBATCH --job-name=rerun_la
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_la-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_la-%A_%a.err
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_only
#SBATCH --array=0-372%32

set -euo pipefail

REPO=/home/jaemin/project/CASP17
PYTHON="$REPO/.venvs/protenix-dock/bin/python"
SCRIPT="$REPO/experiments/novel2025_test/regenerate_lig_align.py"
RUNLIST="$REPO/experiments/novel2025_test/rerun_runs.txt"

"$PYTHON" "$SCRIPT" --runlist "$RUNLIST" --idx "$SLURM_ARRAY_TASK_ID"
