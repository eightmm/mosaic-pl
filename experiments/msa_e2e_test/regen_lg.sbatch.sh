#!/usr/bin/env bash
#SBATCH --job-name=regen_lg
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/regen_lg-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/regen_lg-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# Submit:
#   sbatch --array=0-498%16 experiments/msa_e2e_test/regen_lg.sbatch.sh

set -euo pipefail
cd /home/jaemin/project/CASP17

TGT_LIST=experiments/msa_e2e_test/regen_lg_targets.txt
mapfile -t TASKS < "$TGT_LIST"
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
    echo "task $idx out of range (total=$N) — exit 0"
    exit 0
fi

target="${TASKS[$idx]}"
echo "[$idx/$N] $target start"
bash experiments/msa_e2e_test/regen_lg.sh "$target"
echo "[$idx/$N] $target finished"
