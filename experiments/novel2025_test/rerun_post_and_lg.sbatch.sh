#!/usr/bin/env bash
#SBATCH --job-name=rerun_pla
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/rerun_pla-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/rerun_pla-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=6000ada
#SBATCH --array=0-372%16

set -euo pipefail

REPO=/home/jaemin/project/CASP17
RUNLIST="$REPO/experiments/novel2025_test/rerun_runs.txt"

mapfile -t RUNS < "$RUNLIST"
RUN_NAME="${RUNS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$REPO/experiments/runs/$RUN_NAME"
TARGET_ID="$RUN_NAME"
LG_OUT="$REPO/experiments/submissions/${RUN_NAME}.lg"

echo "=== $SLURM_ARRAY_TASK_ID: $RUN_NAME ==="
echo "  run_dir: $RUN_DIR"

# Re-run post-analysis to pick up Track 2 vina/adg template poses
"$REPO/.venvs/pred/bin/python" "$REPO/scripts/run_post_analysis.py" \
    --run-dir "$RUN_DIR" \
    --device cuda \
    || echo "  (post-analysis failed, continuing)"

# Regenerate LG submission with the new pose pool
"$REPO/.venv/bin/python" "$REPO/scripts/make_casp_submission.py" \
    --run-dir "$RUN_DIR" \
    --target-id "$TARGET_ID" \
    --ligand-name "$TARGET_ID" \
    --author "0000-0000-0000" \
    --method "Boltz-2x + Multi-track ensemble + lig-align (5 seeds x 5 samples) [time-split 2025, Track2 fix]" \
    --parent "N/A" \
    --output "$LG_OUT" \
    --include-affinity \
    || echo "  (submission generation failed)"

echo "=== done: $RUN_NAME ==="
