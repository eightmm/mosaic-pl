#!/usr/bin/env bash
#SBATCH --job-name=surfdock-time
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.err
#SBATCH --partition=heavy
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# End-to-end SurfDock timing on a single protein-ligand case. Mirrors the
# upstream screen_pipeline.sh sequence (target processing → surface mesh →
# CSV → ESM-2 pocket embedding → diffusion inference) so the integration
# can be lifted into ``adapters.prepare_surfdock``'s ``run_surfdock.sh``
# template once we know it works.

set -euo pipefail
cd /home/jaemin/project/CASP17

module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/surfdock/lib/python*/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done

CASE=${1:-10sl_input}
SURFDOCK_DIR=/home/jaemin/project/CASP17/external/SurfDock
PYTHON_BIN=/home/jaemin/project/CASP17/.venvs/surfdock/bin/python
RUN_DIR=/home/jaemin/project/CASP17/experiments/novel2025_runs/runs/${CASE}/${CASE}
SUMMARY=$RUN_DIR/inputs/docking/docking_prep_summary.json

OUT_DIR=$RUN_DIR/outputs/surfdock_timing
DATA_DIR=$OUT_DIR/data           # SurfDock-shaped target dir
SURFACE_DIR=$OUT_DIR/surface
EMB_DIR=$OUT_DIR/embed
INFER_OUT=$OUT_DIR/inference
ESM_GIT_DIR=$SURFDOCK_DIR/esm    # facebookresearch/esm clone
PRECOMP_DIR=$OUT_DIR/precomputed_arrays

# SurfDock's so3.py / torus.py read this env var; an empty dir is enough —
# the modules build the cache on first import.
export precomputed_arrays=$PRECOMP_DIR
mkdir -p "$DATA_DIR" "$SURFACE_DIR" "$EMB_DIR" "$INFER_OUT" "$PRECOMP_DIR"

if [ ! -f "$SUMMARY" ]; then
  echo "FAIL: summary missing at $SUMMARY"; exit 1
fi

# Copy receptor + ligand into SurfDock's per-target directory shape:
# data/<target>/<target>_protein.pdb, data/<target>/<target>_ligand_for_Screen.sdf
TARGET=${CASE%_input}
TGT_DIR=$DATA_DIR/$TARGET
mkdir -p "$TGT_DIR"
RECEPTOR_PDB=$("$PYTHON_BIN" -c "import json; print(json.load(open('$SUMMARY'))['receptor_pdb'])")
LIGAND_SDF=$("$PYTHON_BIN" -c "import json; d=json.load(open('$SUMMARY')); print((d.get('ligands') or [{}])[0].get('sdf',''))")
[ -z "$RECEPTOR_PDB" ] || [ -z "$LIGAND_SDF" ] && { echo "FAIL: receptor or ligand path missing"; exit 1; }
cp "$RECEPTOR_PDB" "$TGT_DIR/${TARGET}_protein.pdb"
cp "$LIGAND_SDF"   "$TGT_DIR/${TARGET}_ligand_for_Screen.sdf"
SCREEN_LIB=$TGT_DIR/${TARGET}_ligand_for_Screen.sdf

echo "case=$CASE target=$TARGET receptor=$RECEPTOR_PDB ligand=$LIGAND_SDF"

# ── Step 1: target surface mesh ──────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 1 — target surface mesh"
echo "================================================================"
T1=$SECONDS
( cd "$SURFACE_DIR" && "$PYTHON_BIN" \
    "$SURFDOCK_DIR/comp_surface/prepare_target/computeTargetMesh_test_samples.py" \
    --data_dir "$DATA_DIR" \
    --out_dir  "$SURFACE_DIR" ) || echo "(surface step failed, continuing)"
echo "  step1 elapsed: $((SECONDS - T1))s"

# ── Step 2: input CSV ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 2 — construct input CSV"
echo "================================================================"
INPUT_CSV=$OUT_DIR/input.csv
T2=$SECONDS
"$PYTHON_BIN" "$SURFDOCK_DIR/inference_utils/construct_csv_input.py" \
    --data_dir "$DATA_DIR" \
    --surface_out_dir "$SURFACE_DIR" \
    --output_csv_file "$INPUT_CSV" \
    --Screen_ligand_library_file "$SCREEN_LIB" || echo "(csv step failed, continuing)"
echo "  step2 elapsed: $((SECONDS - T2))s"

# ── Step 3: ESM-2 pocket embedding (4 sub-steps) ─────────────────────────
echo ""
echo "================================================================"
echo "  Step 3 — ESM-2 pocket embedding"
echo "================================================================"
T3=$SECONDS
SEQ_FASTA=$EMB_DIR/sequences.fasta
FULL_EMB_DIR=$EMB_DIR/full
POCKET_EMB_DIR=$EMB_DIR/pocket
POCKET_EMB_PT=$EMB_DIR/pocket_single.pt
mkdir -p "$FULL_EMB_DIR" "$POCKET_EMB_DIR"
"$PYTHON_BIN" "$SURFDOCK_DIR/datasets/esm_embedding_preparation.py" \
    --out_file "$SEQ_FASTA" \
    --protein_ligand_csv "$INPUT_CSV" || echo "(esm prep failed)"
"$PYTHON_BIN" "$ESM_GIT_DIR/scripts/extract.py" \
    "esm2_t33_650M_UR50D" "$SEQ_FASTA" "$FULL_EMB_DIR" \
    --repr_layers 33 --include "per_tok" --truncation_seq_length 4096 \
    || echo "(esm extract failed)"
"$PYTHON_BIN" "$SURFDOCK_DIR/datasets/get_pocket_embedding.py" \
    --protein_pocket_csv "$INPUT_CSV" \
    --embeddings_dir "$FULL_EMB_DIR" \
    --pocket_emb_save_dir "$POCKET_EMB_DIR" || echo "(pocket emb failed)"
"$PYTHON_BIN" "$SURFDOCK_DIR/datasets/esm_pocket_embeddings_to_pt.py" \
    --esm_embeddings_path "$POCKET_EMB_DIR" \
    --output_path "$POCKET_EMB_PT" || echo "(pocket pt failed)"
echo "  step3 elapsed: $((SECONDS - T3))s"

# ── Step 4: SurfDock diffusion inference ─────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 4 — SurfDock diffusion inference"
echo "================================================================"
T4=$SECONDS
"$PYTHON_BIN" "$SURFDOCK_DIR/inference_accelerate.py" \
    --data_csv "$INPUT_CSV" \
    --model_dir "$SURFDOCK_DIR/model_weights/docking" \
    --ckpt best_ema_inference_epoch_model.pt \
    --confidence_model_dir "$SURFDOCK_DIR/model_weights/posepredict" \
    --confidence_ckpt best_model.pt \
    --esm_embeddings_path "$POCKET_EMB_PT" \
    --out_dir "$INFER_OUT" \
    --inference_steps 20 \
    --batch_size 40 \
    --batch_size_molecule 1 \
    --samples_per_complex 40 \
    --save_docking_result \
    --save_docking_result_number 40 \
    --head_index 0 \
    --tail_index 10000 \
    --inference_mode Screen \
    --mdn_dist_threshold_test 3 \
    --run_name surfdock_timing \
    --project surfdock_timing \
    --wandb_dir "$OUT_DIR/wandb" || echo "(inference failed)"
echo "  step4 elapsed: $((SECONDS - T4))s"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  TIMING SUMMARY ($CASE)"
echo "================================================================"
echo "  total wall time: ${SECONDS}s"
echo ""
echo "Inference output files:"
find "$INFER_OUT" -name "*.sdf" 2>/dev/null | head
echo "Output pose count: $(find "$INFER_OUT" -name "*.sdf" 2>/dev/null | wc -l)"
