#!/usr/bin/env bash
# Batch driver for novel2025 targets:
#   1) prepare-wrapper for every *_input.yaml under pipeline/ (skip if already exists)
#   2) sbatch-submit work, either as individual jobs (legacy, throttled) or
#      as a single SLURM job array (default, bypasses MaxSubmit limits).
#
# Usage:
#   bash batch_prepare_and_submit.sh prepare              # prep wrappers only
#   bash batch_prepare_and_submit.sh submit               # submit via job array (default)
#   bash batch_prepare_and_submit.sh prepare submit       # do both
#   bash batch_prepare_and_submit.sh submit --mode linear # legacy one-sbatch-per-target
#   bash batch_prepare_and_submit.sh submit --concurrency 30  # array throttle (%N)
#   bash batch_prepare_and_submit.sh submit --limit 50    # cap array tasks
#
# Idempotent — rerunning skips targets whose wrappers/submissions already
# exist. The array dispatcher also checks ``submissions/<id>.lg`` at task
# launch time so restarts don't redo completed work.

set -euo pipefail
cd /home/jaemin/project/CASP17

PIPELINE_DIR=experiments/novel2025_test/pipeline
CFG=experiments/novel2025_test/novel2025_config.yaml
RUNS_ROOT=experiments/runs
SUBMIT_DIR=experiments/submissions
ARRAY_SCRIPT=experiments/novel2025_test/run_array.sbatch.sh
LIMIT=0
SUBMIT_MODE=array       # array | linear
CONCURRENCY=30          # array %N throttle

MODE_PREPARE=0
MODE_SUBMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    prepare) MODE_PREPARE=1 ;;
    submit)  MODE_SUBMIT=1 ;;
    --limit) shift; LIMIT="$1" ;;
    --mode) shift; SUBMIT_MODE="$1" ;;
    --concurrency) shift; CONCURRENCY="$1" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
if [ $MODE_PREPARE -eq 0 ] && [ $MODE_SUBMIT -eq 0 ]; then
  MODE_PREPARE=1
  MODE_SUBMIT=1
fi

mapfile -t yamls < <(ls "$PIPELINE_DIR"/*_input.yaml | sort)
echo "total YAMLs: ${#yamls[@]}"

# --- prepare wrappers ---
if [ $MODE_PREPARE -eq 1 ]; then
  prepared=0
  skipped=0
  failed=0
  for y in "${yamls[@]}"; do
    base=$(basename "$y" .yaml)
    script="$RUNS_ROOT/${base}/scripts/run_wrapper.sbatch.sh"
    if [ -f "$script" ]; then
      skipped=$((skipped+1))
      continue
    fi
    if uv run casp17-pl prepare-wrapper \
         --input "$y" --config "$CFG" \
         --output-root "$RUNS_ROOT" --backend slurm \
         >/dev/null 2>&1; then
      prepared=$((prepared+1))
    else
      failed=$((failed+1))
      echo "  prepare failed: $base"
    fi
  done
  echo "prepared=$prepared skipped=$skipped failed=$failed"
fi

# --- submit ---
if [ $MODE_SUBMIT -eq 1 ]; then
  if [ "$SUBMIT_MODE" = "array" ]; then
    n=${#yamls[@]}
    last=$((n - 1))
    if [ $LIMIT -gt 0 ] && [ $LIMIT -le $n ]; then
      last=$((LIMIT - 1))
    fi
    echo "submitting as job array: 0-${last}%${CONCURRENCY}  (total=${n}, script=${ARRAY_SCRIPT})"
    sbatch --array=0-${last}%${CONCURRENCY} "$ARRAY_SCRIPT"
  else
    submitted=0
    skipped=0
    for y in "${yamls[@]}"; do
      base=$(basename "$y" .yaml)
      script="$RUNS_ROOT/${base}/scripts/run_wrapper.sbatch.sh"
      lg="$SUBMIT_DIR/${base}.lg"
      if [ ! -f "$script" ]; then continue; fi
      if [ -f "$lg" ]; then
        skipped=$((skipped+1))
        continue
      fi
      if [ $LIMIT -gt 0 ] && [ $submitted -ge $LIMIT ]; then break; fi
      # SLURM MaxSubmitJobs=100: throttle if queue is full
      current_queued=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
      if [ "$current_queued" -ge 95 ]; then
        echo "  queue full ($current_queued jobs), waiting for slots..."
        while [ "$(squeue -u "$USER" -h 2>/dev/null | wc -l)" -ge 90 ]; do
          sleep 60
        done
        echo "  slots available, resuming submission"
      fi
      sbatch "$script"
      submitted=$((submitted+1))
    done
    echo "submitted=$submitted (pre-existing LG skipped=$skipped)"
  fi
fi
