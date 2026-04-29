#!/bin/bash
set -euo pipefail
cd /home/jaemin/project/CASP17
module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done
run_dir=/home/jaemin/project/CASP17/experiments/runs/L1001_input
rm -f "$run_dir"/outputs/analysis/ba_pred_cofold_*.tsv
rm -f "$run_dir"/outputs/analysis/rmsd_pred_cofold_*.tsv
rm -f "$run_dir"/outputs/analysis/poses/cofold_*.sdf
/home/jaemin/project/CASP17/.venvs/pred/bin/python /home/jaemin/project/CASP17/scripts/run_post_analysis.py --run-dir "$run_dir" --device cuda
