#!/usr/bin/env bash
# CPU-only half of the template→docking rerun. Companion to
# rerun_t2d_gpu.sh. Together they replace rerun_template_to_dock.sh
# when you want to use cpu_only + 6000ada partitions in parallel
# instead of one big 6000ada job per target.
#
# This driver covers everything that does not need a GPU:
#   1.  prepare-wrapper (regenerate per-variant runner scripts)
#   2.  foldseek (CPU binary)
#   3.  align cofolding outputs (gemmi)
#   4.  union template filter
#   5.  extract template pockets (USalign)
#   6.  cluster template pockets
#   7.  cleanup stale Vina/ADG outputs
#   8.  prepare_docking_inputs --reuse-binding-site-cache
#       (skips SwinSite GPU call when cache from prior run is on disk;
#        P2Rank is CPU/JDK so always safe.)
#   9.  Vina variants (Python API, CPU)
#  10.  touch  _rerun_t2d_cpu.done   (marker for the GPU job)
#
# Usage:  bash rerun_t2d_cpu.sh <target_basename>

set -euo pipefail
cd /home/jaemin/project/CASP17

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <target_basename>" >&2
    exit 2
fi

TARGET="$1"
RUN_DIR="experiments/runs/${TARGET}"
INPUT_YAML="experiments/novel2025_test/pipeline/${TARGET}.yaml"
CONFIG="experiments/novel2025_test/novel2025_config.yaml"

RCSB_DIR="${HOME}/DB/RCSB/raw/mmCIF_data"
RCSB_DB="${HOME}/DB/RCSB/processed/rcsb_index.db"

if [[ ! -f "$INPUT_YAML" ]]; then
    echo "[$TARGET] missing input yaml — skip"; exit 0
fi
if [[ ! -d "$RUN_DIR/outputs/boltz2" ]]; then
    echo "[$TARGET] no cofold output — skip"; exit 0
fi

# CPU-only path — no module load cuda; protenix-dock venv has no GPU deps
# at import time so it's fine for the Vina-only Python API. Keep PATH so
# .local/bin (foldseek, USalign, prank) is reachable.
export PATH=/home/jaemin/project/CASP17/.local/bin:$PATH

echo "============================================================"
echo "[$TARGET] rerun-t2d-CPU starting"
echo "============================================================"

# 1. prepare-wrapper
echo "[$TARGET] prepare-wrapper"
uv run casp17-pl prepare-wrapper \
    --input "$INPUT_YAML" \
    --config "$CONFIG" \
    --output-root experiments/runs \
    --backend slurm >/dev/null \
    || { echo "[$TARGET] prepare-wrapper failed — abort"; exit 1; }

# 2. foldseek if missing
FOLDSEEK_TSV="$RUN_DIR/outputs/template_search_structure/foldseek_hits.tsv"
if [[ ! -s "$FOLDSEEK_TSV" ]]; then
    if [[ -f "$RUN_DIR/scripts/run_template_search_structure.sbatch.sh" ]]; then
        echo "[$TARGET] foldseek"
        bash "$RUN_DIR/scripts/run_template_search_structure.sbatch.sh" \
            || echo "[$TARGET] foldseek failed (continuing)"
    fi
else
    echo "[$TARGET] foldseek_hits.tsv present — skip"
fi

# 3. frame alignment
echo "[$TARGET] align"
.venv/bin/python scripts/align_cofolding_outputs.py --run-dir "$RUN_DIR" \
    || echo "[$TARGET] align failed (continuing)"

# 4. union template filter
echo "[$TARGET] template filter"
filter_args=(
    --rcsb-db "$RCSB_DB"
    --input-yaml "$INPUT_YAML"
    --output-tsv "$RUN_DIR/outputs/template_search_sequence/filtered_hits.tsv"
    --hits-tsv "$RUN_DIR/outputs/template_search_sequence/mmseqs_hits.tsv"
    --max-deposition-date 2025-01-01
)
[[ -s "$FOLDSEEK_TSV" ]] && filter_args+=(--foldseek-tsv "$FOLDSEEK_TSV")
.venv/bin/python scripts/run_template_filter.py "${filter_args[@]}" \
    || echo "[$TARGET] filter failed (continuing)"

# 5. pocket extract (skip if cached)
POCKETS_JSON="$RUN_DIR/outputs/template_pockets/template_pockets.json"
if [[ ! -s "$POCKETS_JSON" ]]; then
    echo "[$TARGET] pocket extract (--max-templates 1000)"
    .venv/bin/python scripts/extract_template_pockets.py \
        --run-dir "$RUN_DIR" \
        --rcsb-dir "$RCSB_DIR" \
        --max-templates 1000 \
        || echo "[$TARGET] pocket extract failed (continuing)"
else
    echo "[$TARGET] template_pockets.json present — skip"
fi

# 6. pocket cluster
if [[ -s "$POCKETS_JSON" ]]; then
    echo "[$TARGET] pocket cluster"
    .venv/bin/python scripts/cluster_template_pockets.py \
        --pockets-json "$POCKETS_JSON" \
        || echo "[$TARGET] cluster failed (continuing)"
fi

# 7. cleanup stale docking outputs
echo "[$TARGET] cleanup stale vina/adg/template_docking/analysis"
rm -rf "$RUN_DIR/outputs/vina_"*
rm -rf "$RUN_DIR/outputs/autodock_gpu_"*
rm -rf "$RUN_DIR/outputs/template_docking"
rm -rf "$RUN_DIR/outputs/analysis"
rm -f  "$RUN_DIR/outputs/submission_scores.json"

# 8. prepare_docking_inputs with cache reuse (no GPU)
echo "[$TARGET] prepare_docking_inputs (--reuse-binding-site-cache)"
.venvs/protenix-dock/bin/python scripts/prepare_docking_inputs.py \
    --input-yaml "$INPUT_YAML" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/inputs/docking" \
    --model auto \
    --reuse-binding-site-cache \
    || { echo "[$TARGET] prepare-docking failed — abort"; exit 1; }

# 9. Vina variants (Python API, CPU)
SEEDS=(42 101 202 303 404)
DOCK_TIMEOUT=900

echo "[$TARGET] Vina variants"
for script in "$RUN_DIR/scripts/run_vina_"*.py; do
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

# 10. marker
touch "$RUN_DIR/_rerun_t2d_cpu.done"
echo "[$TARGET] CPU half done — marker $RUN_DIR/_rerun_t2d_cpu.done"
