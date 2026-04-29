#!/usr/bin/env bash
# GPU half of the template→docking rerun. Companion to rerun_t2d_cpu.sh.
# Picks up where the CPU driver left off (marker `_rerun_t2d_cpu.done`)
# and runs only the GPU-bound stages:
#
#   1.  ADG variants (CUDA binary)
#   2.  Multi-track docking --skip-pxdock
#   3.  Ion placement
#   4.  Post-analysis (BA-Pred + RMSD-Pred, GNNs on dgl 2.4)
#   5.  Compute submission scores + LG submission
#   6.  touch  _rerun_t2d_gpu.done
#
# Usage:  bash rerun_t2d_gpu.sh <target_basename>

set -euo pipefail
cd /home/jaemin/project/CASP17

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <target_basename>" >&2
    exit 2
fi

TARGET="$1"
RUN_DIR="experiments/runs/${TARGET}"
INPUT_YAML="experiments/novel2025_test/pipeline/${TARGET}.yaml"
SUBMIT_LG="experiments/submissions/${TARGET}.lg"

RCSB_DIR="${HOME}/DB/RCSB/raw/mmCIF_data"
RCSB_DB="${HOME}/DB/RCSB/processed/rcsb_index.db"

if [[ ! -f "$INPUT_YAML" ]]; then
    echo "[$TARGET] missing input yaml — skip"; exit 0
fi
if [[ ! -f "$RUN_DIR/_rerun_t2d_cpu.done" ]]; then
    echo "[$TARGET] CPU half not finished (marker missing) — skip"
    exit 0
fi

# CUDA + venv paths (mirror run_wrapper.sbatch.sh)
module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done
export PATH=/home/jaemin/project/CASP17/.local/bin:$PATH

echo "============================================================"
echo "[$TARGET] rerun-t2d-GPU starting"
echo "============================================================"

# 1. ADG variants (CUDA)
SEEDS=(42 101 202 303 404)
DOCK_TIMEOUT=900

echo "[$TARGET] ADG variants"
export PATH=/home/jaemin/project/CASP17/.venvs/protenix-dock/bin:$PATH
for script in "$RUN_DIR/scripts/run_autodock_gpu_"*.py; do
    [[ -f "$script" ]] || continue
    name="$(basename "$script" .py)"
    src="${name#run_}"
    out_root="$RUN_DIR/outputs/$src"
    mkdir -p "$out_root"
    for seed in "${SEEDS[@]}"; do
        out_seed="$out_root/seed_${seed}"
        mkdir -p "$out_seed"
        timeout "$DOCK_TIMEOUT" env DOCK_SEED="$seed" DOCK_OUT_DIR="$out_seed" \
            .venvs/protenix-dock/bin/python "$script" \
            && echo "  $src seed=$seed ok" \
            || echo "  $src seed=$seed FAIL"
    done
done

# 2. Multi-track docking (skip PxDock)
echo "[$TARGET] multi-track --skip-pxdock"
.venvs/protenix-dock/bin/python scripts/run_multi_track_docking.py \
    --run-dir "$RUN_DIR" \
    --input-yaml "$INPUT_YAML" \
    --rcsb-dir "$RCSB_DIR" \
    --rcsb-db "$RCSB_DB" \
    --mcs-threshold 0.5 \
    --skip-pxdock \
    || echo "[$TARGET] multi-track failed (continuing)"

# 3. Ion placement (cheap, no-op if no ion entity)
.venv/bin/python scripts/collect_template_ions.py \
    --run-dir "$RUN_DIR" \
    --input-yaml "$INPUT_YAML" \
    --rcsb-dir "$RCSB_DIR" \
    --rcsb-db "$RCSB_DB" \
    || echo "[$TARGET] ion placement skipped/failed (continuing)"

# 4. Post-analysis (GPU)
echo "[$TARGET] post-analysis"
.venvs/pred/bin/python scripts/run_post_analysis.py \
    --run-dir "$RUN_DIR" \
    --device cuda \
    || { echo "[$TARGET] post-analysis failed — abort"; exit 1; }

# 5. submission scores + LG
echo "[$TARGET] submission scores"
.venv/bin/python scripts/compute_submission_scores.py --run-dir "$RUN_DIR" \
    || echo "[$TARGET] submission_scores failed (continuing)"

echo "[$TARGET] LG submission"
.venv/bin/python scripts/make_casp_submission.py \
    --run-dir "$RUN_DIR" \
    --target-id "$TARGET" \
    --author 0000-0000-0000 \
    --method 'Boltz-2x + Multi-track ensemble + lig-align (5 seeds × 5 samples) [time-split 2025, cofold-cluster top-3]' \
    --parent N/A \
    --output "$SUBMIT_LG" \
    --include-affinity \
    || { echo "[$TARGET] LG submission failed"; exit 1; }

# 6. marker
touch "$RUN_DIR/_rerun_t2d_gpu.done"
echo "[$TARGET] GPU half done — marker $RUN_DIR/_rerun_t2d_gpu.done"
