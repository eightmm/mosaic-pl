#!/usr/bin/env bash
#SBATCH --job-name=L1000_rerun
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10800
#SBATCH --partition=6000ada

set -euo pipefail
cd /home/jaemin/project/CASP17
module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done
export PATH=/home/jaemin/project/CASP17/.local/bin:$PATH

PRED_PY=/home/jaemin/project/CASP17/.venvs/pred/bin/python
HUB_PY=/home/jaemin/project/CASP17/.venv/bin/python
POST=/home/jaemin/project/CASP17/scripts/run_post_analysis.py
SUBM=/home/jaemin/project/CASP17/scripts/make_casp_submission.py

SUB_CFG_AUTHOR="0000-0000-0000"
SUB_CFG_METHOD="Boltz-2x + Multi-track ensemble + lig-align (5 seeds x 5 samples) [protein=boltz2x, pose=vina+protenix_dock+cofold, top5]"

for tid in L1001 L1002 L1003 L1004 L1005 L1006 L1007 L1008 L1009 L1010 L1011 L1012 L1013 L1014 L1015 L1016 L1017; do
  run_dir=/home/jaemin/project/CASP17/experiments/runs/${tid}_input
  if [ ! -d "$run_dir" ]; then
    echo "skip $tid (no run dir)"; continue
  fi
  echo ""
  echo "================================================================"
  echo " RE-POST-ANALYSIS: $tid"
  echo "================================================================"
  # Purge previous cofold tsvs so they get regenerated cleanly; keep docking ones.
  rm -f "$run_dir"/outputs/analysis/ba_pred_cofold_*.tsv
  rm -f "$run_dir"/outputs/analysis/rmsd_pred_cofold_*.tsv
  rm -f "$run_dir"/outputs/analysis/poses/cofold_*.sdf

  "$PRED_PY" "$POST" --run-dir "$run_dir" --device cuda \
    || { echo "  (post-analysis failed for $tid)"; continue; }

  echo ""
  echo "--- LG SUBMISSION: $tid ---"
  out_lg=/home/jaemin/project/CASP17/experiments/submissions/${tid}_input.lg
  "$HUB_PY" "$SUBM" \
    --run-dir "$run_dir" \
    --target-id "${tid}_input" \
    --ligand-name "${tid}_input" \
    --ligand-number 1 \
    --author "$SUB_CFG_AUTHOR" \
    --method "$SUB_CFG_METHOD" \
    --parent "N/A" \
    --include-affinity \
    --output "$out_lg" \
    || echo "  (submission failed for $tid)"
done

echo ""
echo "DONE: all 17 reprocessed"
