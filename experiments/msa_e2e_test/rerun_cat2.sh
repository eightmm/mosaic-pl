#!/usr/bin/env bash
# Cat 2 — 5 targets whose post-analysis dir got wiped (raw vina+adg present).
# Re-run from post-analysis stage onward (skip cofold + docking).
set -euo pipefail

TARGETS=(8pvw_input 8rpa_input 8rx6_input 8s7n_input 8w9s_input)
cd /home/jaemin/project/CASP17

for t in "${TARGETS[@]}"; do
    echo "============================================================"
    echo "[$t] Cat 2 post-analysis rerun"
    echo "============================================================"
    RUN_DIR=experiments/msa_e2e_test/runs/${t}/${t}
    SUBMIT_LG=experiments/msa_e2e_test/submissions/${t}.lg

    if [[ ! -d "$RUN_DIR" ]]; then
        echo "[$t] missing run dir — skip"; continue
    fi
    if [[ ! -f "$RUN_DIR/inputs/docking/docking_prep_summary.json" ]]; then
        echo "[$t] missing prep_summary — skip (use full wrapper instead)"; continue
    fi

    # Post-analysis needs GPU for BA-Pred/RMSD-Pred — skip if no GPU.
    # If running on master, use --device cpu fallback (slower but works).
    DEV="${POSTANAL_DEVICE:-cpu}"
    .venvs/pred/bin/python scripts/run_post_analysis.py --run-dir "$RUN_DIR" --device "$DEV" \
        || { echo "[$t] post-analysis failed — continuing"; continue; }
    .venv/bin/python scripts/compute_submission_scores.py --run-dir "$RUN_DIR" \
        || { echo "[$t] compute scores failed — continuing"; continue; }
    .venv/bin/python scripts/make_casp_submission.py \
        --run-dir "$RUN_DIR" --target-id "$t" --author 0000-0000-0000 \
        --method 'Cat 2 post-analysis rerun' --parent N/A \
        --output "$SUBMIT_LG" --include-affinity \
        || { echo "[$t] LG failed — continuing"; continue; }
    echo "[$t] DONE"
done
