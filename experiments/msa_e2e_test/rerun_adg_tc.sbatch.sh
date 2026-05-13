#!/usr/bin/env bash
#SBATCH --job-name=rerun_adg_tc
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_adg_tc-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_adg_tc-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# Submit:
#   sbatch --array=0-216%16 experiments/msa_e2e_test/rerun_adg_tc.sbatch.sh
#
# Target list (217 broken targets) at experiments/msa_e2e_test/rerun_adg_tc_targets.txt.

set -euo pipefail
cd /home/jaemin/project/CASP17

TGT_LIST=experiments/msa_e2e_test/rerun_adg_tc_targets.txt
mapfile -t TASKS < "$TGT_LIST"
N=${#TASKS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
    echo "task $idx out of range (total=$N) — exit 0"
    exit 0
fi

target="${TASKS[$idx]}"

module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done
export PATH=/home/jaemin/project/CASP17/.local/bin:$PATH

echo "[$idx/$N] $target start"
bash experiments/msa_e2e_test/rerun_adg_tc.sh "$target"
echo "[$idx/$N] $target finished"
