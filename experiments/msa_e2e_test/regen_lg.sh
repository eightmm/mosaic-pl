#!/usr/bin/env bash
# Per-target LG regeneration after ADG TC rerun + multi-ligand-aware fix.
# Calls make_casp_submission.py which internally re-aggregates scores
# (compute_submission_scores.aggregate) from the current outputs/ tree, so
# the new ADG template_consensus_* poses + corrected ligand matching land
# in the LG output.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <target_basename>" >&2
    exit 2
fi

TARGET="$1"
RUN_DIR=/home/jaemin/project/CASP17/experiments/msa_e2e_test/runs/${TARGET}/${TARGET}
OUTPUT=/home/jaemin/project/CASP17/experiments/msa_e2e_test/submissions/${TARGET}.lg

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[${TARGET}] missing run dir ${RUN_DIR} — skip"
    exit 0
fi
if [[ ! -f "${RUN_DIR}/inputs/docking/docking_prep_summary.json" ]]; then
    echo "[${TARGET}] missing docking_prep_summary — skip"
    exit 0
fi

cd /home/jaemin/project/CASP17

echo "============================================================"
echo "[${TARGET}] LG regen start"
echo "============================================================"

.venv/bin/python scripts/make_casp_submission.py \
    --run-dir "${RUN_DIR}" \
    --target-id "${TARGET}" \
    --author 0000-0000-0000 \
    --method 'Boltz-2x + Multi-track ensemble + lig-align (5 seeds x 5 samples) [time-split 2025]' \
    --parent N/A \
    --output "${OUTPUT}" \
    --include-affinity

echo "[${TARGET}] LG regen done -> ${OUTPUT}"
