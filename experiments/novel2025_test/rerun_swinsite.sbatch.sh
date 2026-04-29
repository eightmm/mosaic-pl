#!/usr/bin/env bash
#SBATCH --job-name=swinsite_recover
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/slurm-%A_%a.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/slurm-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --partition=6000ada
#
# Per-target recovery of the SwinSite-guided docking variants that were
# silently skipped across 472 completed runs due to the old ``pocket_*.pdb``
# parser glob never matching the actual ``pocket0_score_0.7219.pdb`` file
# names. Because SwinSite's GPU prediction already ran and its grid PDBs
# are on disk, this script re-parses them in place (no GPU prediction
# re-run), then runs the missing vina_swinsite + adg_swinsite seed sweeps,
# post-analysis, and LG regeneration.
#
# TASK_LIST is built by rerun_swinsite_submit.sh from the list of targets
# that (a) have swinsite grid PDBs on disk and (b) don't already have
# swinsite in docking_prep_summary.json.

set -uo pipefail
cd /home/jaemin/project/CASP17

module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"; done
export PATH=/home/jaemin/project/CASP17/.local/bin:$PATH

TASK_LIST=/home/jaemin/project/CASP17/experiments/novel2025_test/_eval_work/swinsite_rerun_targets.txt
mapfile -t TARGETS < "$TASK_LIST"
N=${#TARGETS[@]}

idx="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$idx" -ge "$N" ]; then
  echo "task $idx out of range (total=$N) — nothing to do"
  exit 0
fi

tgt="${TARGETS[$idx]}"
R=/home/jaemin/project/CASP17/experiments/runs/${tgt}_input
echo "[$idx/$N] swinsite recovery for $tgt"
echo "run_dir=$R"

if [ ! -d "$R/inputs/docking" ]; then
  echo "  no docking inputs dir, skipping"
  exit 0
fi

# 1. Reparse SwinSite grid*.pdb and update docking_prep_summary.json
echo ""
echo "----------------------------------------------------------------"
echo "  STEP 1: Reparse SwinSite output"
echo "----------------------------------------------------------------"
/home/jaemin/project/CASP17/.venv/bin/python - "$R" <<'PYEOF'
import json, re, sys
from pathlib import Path

R = Path(sys.argv[1])
results_dir = R / "inputs" / "docking" / "swinsite" / "results" / "input" / "receptor"
summary_path = R / "inputs" / "docking" / "docking_prep_summary.json"

if not results_dir.exists():
    print("  no swinsite results dir, skipping")
    sys.exit(0)
if not summary_path.exists():
    print("  no docking_prep_summary.json, skipping")
    sys.exit(0)

score_re = re.compile(r"_score_([0-9.]+)\.pdb$")
def _score(p):
    m = score_re.search(p.name)
    try:
        return float(m.group(1)) if m else -1.0
    except ValueError:
        return -1.0

grids = sorted(results_dir.glob("grid*_score_*.pdb"), key=_score, reverse=True)
if not grids:
    print("  no grid*_score_*.pdb files found, skipping")
    sys.exit(0)
best = grids[0]
coords = []
for line in best.read_text().splitlines():
    if line.startswith(("ATOM", "HETATM")):
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
if not coords:
    print(f"  grid {best.name} empty, skipping")
    sys.exit(0)

n = len(coords)
cx = sum(c[0] for c in coords) / n
cy = sum(c[1] for c in coords) / n
cz = sum(c[2] for c in coords) / n
center = [cx, cy, cz]

summary = json.loads(summary_path.read_text())
bs = summary.setdefault("binding_site_predictions", {})
bs["swinsite"] = {"center": center, "size": [22.5, 22.5, 22.5]}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"  swinsite: center=[{cx:.1f}, {cy:.1f}, {cz:.1f}], atoms={n}, score={_score(best):.3f}")
PYEOF

# 2. Run vina_swinsite (5 seeds) + adg_swinsite (5 seeds) with per-seed timeout
echo ""
echo "----------------------------------------------------------------"
echo "  STEP 2: VINA_SWINSITE (5 seeds)"
echo "----------------------------------------------------------------"
for seed in 42 101 202 303 404; do
  mkdir -p "$R/outputs/vina_swinsite/seed_${seed}"
  timeout 900 env DOCK_SEED=${seed} DOCK_OUT_DIR="$R/outputs/vina_swinsite/seed_${seed}" \
    /home/jaemin/project/CASP17/.venvs/protenix-dock/bin/python "$R/scripts/run_vina_swinsite.py" \
    || echo "  (vina_swinsite seed ${seed} timed out or failed, continuing)"
done

echo ""
echo "----------------------------------------------------------------"
echo "  STEP 3: ADG_SWINSITE (5 seeds)"
echo "----------------------------------------------------------------"
for seed in 42 101 202 303 404; do
  mkdir -p "$R/outputs/autodock_gpu_swinsite/seed_${seed}"
  timeout 900 env DOCK_SEED=${seed} DOCK_OUT_DIR="$R/outputs/autodock_gpu_swinsite/seed_${seed}" \
    /home/jaemin/project/CASP17/.venvs/protenix-dock/bin/python "$R/scripts/run_autodock_gpu_swinsite.py" \
    || echo "  (adg_swinsite seed ${seed} timed out or failed, continuing)"
done

# 4. Re-run post-analysis to pick up the new pose SDFs
echo ""
echo "----------------------------------------------------------------"
echo "  STEP 4: POST-ANALYSIS (BA-Pred + RMSD-Pred)"
echo "----------------------------------------------------------------"
/home/jaemin/project/CASP17/.venvs/pred/bin/python /home/jaemin/project/CASP17/scripts/run_post_analysis.py \
  --run-dir "$R" --device cuda || echo "  (post-analysis failed, continuing)"

# 5. Regenerate LG with the now-enlarged pose pool
echo ""
echo "----------------------------------------------------------------"
echo "  STEP 5: CASP17 LG SUBMISSION"
echo "----------------------------------------------------------------"
mkdir -p /home/jaemin/project/CASP17/experiments/submissions
/home/jaemin/project/CASP17/.venv/bin/python /home/jaemin/project/CASP17/scripts/make_casp_submission.py \
  --run-dir "$R" \
  --target-id "${tgt}_input" \
  --author 0000-0000-0000 \
  --method 'Boltz-2x + Multi-track ensemble + lig-align (5 seeds x 5 samples) [time-split 2025] + SwinSite variants' \
  --parent N/A \
  --output /home/jaemin/project/CASP17/experiments/submissions/${tgt}_input.lg \
  --include-affinity || echo "  (submission generation failed)"

echo ""
echo "[$idx/$N] $tgt swinsite recovery done"
