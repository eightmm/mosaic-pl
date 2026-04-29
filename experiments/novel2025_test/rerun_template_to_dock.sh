#!/usr/bin/env bash
# Rerun template→docking stages for one novel2025 target.
#
#   * Reuse cofolding outputs (boltz2/boltz2x/protenix/alphafold3) as-is.
#   * Re-do everything else with the updated docking-prep code:
#       - cofold cluster top-3 (cofolding_1..3) replaces the single cofolding
#         centroid; expects all aligned cifs to share one frame
#         (verified end-to-end against align_cofolding_outputs.rglob).
#       - template_consensus_1..10 sources (mmseqs ∪ foldseek union pockets,
#         single-link 5 Å cluster, n_members ≥ 2 floor).
#       - --max-templates 1000 for pocket extraction (down from default 2000;
#         USalign min-tmscore 0.4 gate already drops the tail so pool >500
#         contributes mostly skips, halving wall time).
#   * Skip PxDock (single-shot, slow ~5–30 min, not affected by new sources
#     in a way that justifies the cost). Track 1 PxDock outputs from prior
#     runs stay in place.
#   * ADG variants are still re-run despite ~1 % native rate observed on
#     novel2025 (commit 4d3ee27); we want the new cofold/consensus boxes
#     measured for ADG too even though it underperforms.
#
# Usage:
#   bash rerun_template_to_dock.sh <target_basename>          # e.g. 10sl_input
#
# Idempotency:
#   * align_cofolding_outputs / template-filter / prepare_docking_inputs
#     overwrite their outputs every run — safe to re-execute.
#   * foldseek and pocket-extraction skip when their primary output is
#     already present (foldseek_hits.tsv / template_pockets.json).
#   * stale Vina / ADG / template_docking / analysis outputs are removed
#     before docking, so post-analysis never picks up old poses with
#     defunct source names.

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
SUBMIT_LG="experiments/submissions/${TARGET}.lg"

RCSB_DIR="${HOME}/DB/RCSB/raw/mmCIF_data"
RCSB_DB="${HOME}/DB/RCSB/processed/rcsb_index.db"

if [[ ! -f "$INPUT_YAML" ]]; then
    echo "[$TARGET] missing input yaml: $INPUT_YAML — skip"
    exit 0
fi
if [[ ! -d "$RUN_DIR/outputs/boltz2" ]]; then
    echo "[$TARGET] no cofold output (boltz2 dir missing) — skip"
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
echo "[$TARGET] rerun-template-to-dock starting"
echo "============================================================"

# ------------------------------------------------------------ 1. Re-prepare wrapper FIRST
# Old-format runs (pre-union-template) lack run_template_search_structure.sbatch.sh
# and the new vina_cofolding_1/2/3 + template_consensus_* variant runners.
# prepare-wrapper writes only to experiments/runs/<target>/scripts/ (no
# outputs touched), so it's safe to call before downstream stages.
echo "[$TARGET] re-prepare wrapper (regenerates scripts + variant runners)"
uv run casp17-pl prepare-wrapper \
    --input "$INPUT_YAML" \
    --config "$CONFIG" \
    --output-root experiments/runs \
    --backend slurm >/dev/null \
    || { echo "[$TARGET] prepare-wrapper failed — abort"; exit 1; }

# ------------------------------------------------------------ 2. Foldseek (now the runner script exists)
FOLDSEEK_TSV="$RUN_DIR/outputs/template_search_structure/foldseek_hits.tsv"
if [[ ! -s "$FOLDSEEK_TSV" ]]; then
    if [[ -f "$RUN_DIR/scripts/run_template_search_structure.sbatch.sh" ]]; then
        echo "[$TARGET] foldseek_hits.tsv missing — running foldseek"
        bash "$RUN_DIR/scripts/run_template_search_structure.sbatch.sh" \
            || echo "[$TARGET] foldseek failed (continuing without)"
    else
        echo "[$TARGET] no foldseek runner script — config disables structure search"
    fi
else
    echo "[$TARGET] foldseek_hits.tsv present — skip"
fi

# ------------------------------------------------------------ 3. Frame alignment (always; idempotent overwrite)
echo "[$TARGET] aligning cofolding outputs"
.venv/bin/python scripts/align_cofolding_outputs.py --run-dir "$RUN_DIR" \
    || echo "[$TARGET] alignment failed (continuing)"

# ------------------------------------------------------------ 4. Union template filter (always)
echo "[$TARGET] union template filter"
filter_args=(
    --rcsb-db "$RCSB_DB"
    --input-yaml "$INPUT_YAML"
    --output-tsv "$RUN_DIR/outputs/template_search_sequence/filtered_hits.tsv"
    --hits-tsv "$RUN_DIR/outputs/template_search_sequence/mmseqs_hits.tsv"
    --max-deposition-date 2025-01-01
)
if [[ -s "$FOLDSEEK_TSV" ]]; then
    filter_args+=(--foldseek-tsv "$FOLDSEEK_TSV")
fi
.venv/bin/python scripts/run_template_filter.py "${filter_args[@]}" \
    || echo "[$TARGET] filter failed (continuing)"

# ------------------------------------------------------------ 5. Pocket extraction (skip if cached)
POCKETS_JSON="$RUN_DIR/outputs/template_pockets/template_pockets.json"
if [[ ! -s "$POCKETS_JSON" ]]; then
    echo "[$TARGET] extracting template pockets (--max-templates 1000)"
    .venv/bin/python scripts/extract_template_pockets.py \
        --run-dir "$RUN_DIR" \
        --rcsb-dir "$RCSB_DIR" \
        --max-templates 1000 \
        || echo "[$TARGET] pocket extraction failed (continuing)"
else
    echo "[$TARGET] template_pockets.json present — skip extraction"
fi

# ------------------------------------------------------------ 6. Pocket clustering (always)
if [[ -s "$POCKETS_JSON" ]]; then
    echo "[$TARGET] clustering template pockets"
    .venv/bin/python scripts/cluster_template_pockets.py \
        --pockets-json "$POCKETS_JSON" \
        || echo "[$TARGET] clustering failed (continuing)"
fi

# ------------------------------------------------------------ 7. Cleanup stale docking outputs
# Old runs may have vina_cofolding (pre-cluster naming) that no longer
# has a matching variant. Keep outputs/protenix_dock untouched (PxDock skip).
echo "[$TARGET] cleanup stale docking outputs"
rm -rf "$RUN_DIR/outputs/vina_"*
rm -rf "$RUN_DIR/outputs/autodock_gpu_"*
rm -rf "$RUN_DIR/outputs/template_docking"
rm -rf "$RUN_DIR/outputs/analysis"
rm -f  "$RUN_DIR/outputs/submission_scores.json"

# ------------------------------------------------------------ 8. Docking prep (NEW: cofold cluster top-3 + template_consensus_1..10)
echo "[$TARGET] preparing docking inputs"
.venvs/protenix-dock/bin/python scripts/prepare_docking_inputs.py \
    --input-yaml "$INPUT_YAML" \
    --run-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR/inputs/docking" \
    --model auto \
    || { echo "[$TARGET] docking prep failed — abort"; exit 1; }

# ------------------------------------------------------------ 9. Run Vina + ADG variants directly (skip PxDock; skip cofold scripts)
SEEDS=(42 101 202 303 404)
DOCK_TIMEOUT=900

echo "[$TARGET] running Vina + ADG variants"
for script in "$RUN_DIR/scripts/run_vina_"*.py "$RUN_DIR/scripts/run_autodock_gpu_"*.py; do
    [[ -f "$script" ]] || continue
    name="$(basename "$script" .py)"        # run_vina_cofolding_1
    src="${name#run_}"                       # vina_cofolding_1
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

# ------------------------------------------------------------ 10. Multi-track docking (skip PxDock)
echo "[$TARGET] multi-track docking (--skip-pxdock)"
.venvs/protenix-dock/bin/python scripts/run_multi_track_docking.py \
    --run-dir "$RUN_DIR" \
    --input-yaml "$INPUT_YAML" \
    --rcsb-dir "$RCSB_DIR" \
    --rcsb-db "$RCSB_DB" \
    --mcs-threshold 0.5 \
    --skip-pxdock \
    || echo "[$TARGET] multi-track failed (continuing)"

# ------------------------------------------------------------ 11. Ion placement (cheap; skip if no ion entity)
.venv/bin/python scripts/collect_template_ions.py \
    --run-dir "$RUN_DIR" \
    --input-yaml "$INPUT_YAML" \
    --rcsb-dir "$RCSB_DIR" \
    --rcsb-db "$RCSB_DB" \
    || echo "[$TARGET] ion placement failed or no ion entity (continuing)"

# ------------------------------------------------------------ 12. Post-analysis (BA-Pred + RMSD-Pred)
echo "[$TARGET] post-analysis"
.venvs/pred/bin/python scripts/run_post_analysis.py \
    --run-dir "$RUN_DIR" \
    --device cuda \
    || { echo "[$TARGET] post-analysis failed — abort"; exit 1; }

# ------------------------------------------------------------ 13. Submission scores + LG
echo "[$TARGET] computing submission scores"
.venv/bin/python scripts/compute_submission_scores.py --run-dir "$RUN_DIR" \
    || echo "[$TARGET] submission_scores compute failed (continuing)"

echo "[$TARGET] writing CASP17 LG submission"
.venv/bin/python scripts/make_casp_submission.py \
    --run-dir "$RUN_DIR" \
    --target-id "$TARGET" \
    --author 0000-0000-0000 \
    --method 'Boltz-2x + Multi-track ensemble + lig-align (5 seeds × 5 samples) [time-split 2025, cofold-cluster top-3]' \
    --parent N/A \
    --output "$SUBMIT_LG" \
    --include-affinity \
    || { echo "[$TARGET] LG submission failed"; exit 1; }

echo "============================================================"
echo "[$TARGET] rerun-template-to-dock done"
echo "============================================================"
