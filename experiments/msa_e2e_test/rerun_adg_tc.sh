#!/usr/bin/env bash
# Per-target driver: rerun ADG template_consensus_{1..10} × 5 seeds.
# Idempotent — skips any (variant, seed) that already has docking.dlg.
# Reads receptor/ligand paths from docking_prep_summary.json which we
# rewrote to absolute paths on 2026-05-11 (silent-failure root cause was
# autogrid resolving relative paths against cwd=lig_grid_dir).
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <target_basename>" >&2
    exit 2
fi

TARGET="$1"
BASE=/home/jaemin/project/CASP17/experiments/msa_e2e_test/runs/${TARGET}/${TARGET}
SCRIPTS="${BASE}/scripts"
PY=/home/jaemin/project/CASP17/.venvs/protenix-dock/bin/python

if [[ ! -d "${BASE}" ]]; then
    echo "[${TARGET}] missing run dir: ${BASE}" >&2
    exit 1
fi

echo "============================================================"
echo "[${TARGET}] rerun ADG template_consensus_{1..10} × seeds"
echo "============================================================"

for tc in 1 2 3 4 5 6 7 8 9 10; do
    SCRIPT="${SCRIPTS}/run_autodock_gpu_template_consensus_${tc}.py"
    if [[ ! -f "${SCRIPT}" ]]; then
        echo "  TC${tc}: no runner script, skipping"
        continue
    fi
    for seed in 42 101 202 303 404; do
        OUT="${BASE}/outputs/autodock_gpu_template_consensus_${tc}/seed_${seed}"
        # Check for any existing docking.dlg (per-ligand subdir)
        if compgen -G "${OUT}/ligand_*/docking.dlg" > /dev/null 2>&1; then
            echo "  TC${tc} seed=${seed}: docking.dlg present, skip"
            continue
        fi
        # Clean stale partial (only grid/, no docking.dlg) to ensure fresh GPF
        if [[ -d "${OUT}" ]]; then
            rm -rf "${OUT}"
        fi
        mkdir -p "${OUT}"
        echo "  TC${tc} seed=${seed}: start"
        timeout 900 env DOCK_SEED="${seed}" DOCK_OUT_DIR="${OUT}" "${PY}" "${SCRIPT}" \
            || echo "    (seed ${seed} timed out or failed, continuing)"
    done
done

echo "[${TARGET}] done"
