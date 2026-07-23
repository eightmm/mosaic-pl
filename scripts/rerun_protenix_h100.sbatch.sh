#!/usr/bin/env bash
#SBATCH --job-name=protenix-rerun
#SBATCH --output=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.out
#SBATCH --error=/home/jaemin/project/CASP17/experiments/logs/slurm-%j.err
#SBATCH --partition=test
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:50:00
#
# R2317 / R2318 Protenix retry pinned to H100 — the 6000pro_maxq cards in
# the heavy partition do not have a CUDA kernel image compatible with the
# Protenix v1.0.0 build (RuntimeError: no kernel image is available for
# execution on the device). After this script lands a fresh set of cifs
# under outputs/protenix, run align + LG rebuild to fold the new samples
# into the submission pool.

set -euo pipefail
cd /home/jaemin/project/CASP17

module load cuda/12.8 2>/dev/null || true
export LD_LIBRARY_PATH="/home/jaemin/project/CASP17/.local/lib:/usr/lib/x86_64-linux-gnu:${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"
for _nv_lib in /home/jaemin/project/CASP17/.venvs/protenix/lib/python*/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"
done
export PATH=/home/jaemin/project/CASP17/.local/bin:/home/jaemin/project/CASP17/.venvs/protenix/bin:$PATH

echo "Starting Protenix retry on $(hostname)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

for TARGET in R2317 R2318; do
  RUN_DIR=/home/jaemin/project/CASP17/experiments/CASP17/$TARGET
  INPUT=$RUN_DIR/inputs/protenix_input.json
  echo ""
  echo "================================================================"
  echo "  $TARGET — Protenix on H100 (5 seeds)"
  echo "================================================================"
  for seed in 101 202 303 404 505; do
    OUT=$RUN_DIR/outputs/protenix/seed_$seed
    rm -rf "$OUT"
    mkdir -p "$OUT"
    echo "  seed $seed"
    /home/jaemin/project/CASP17/.venvs/protenix/bin/protenix pred \
      --input "$INPUT" \
      --out_dir "$OUT" \
      --seeds "$seed" \
      --cycle 10 --step 200 --sample 5 \
      --model_name protenix_base_default_v1.0.0 \
      --dtype bf16 \
      --use_msa true --use_default_params false \
      --trimul_kernel cuequivariance --triatt_kernel cuequivariance \
      --enable_cache true --enable_fusion true --enable_tf32 true \
      --msa_server_mode none \
      --use_template true --use_rna_msa true \
      --use_seeds_in_json false --need_atom_confidence false \
      --use_tfg_guidance false \
      --kalign_binary_path /home/jaemin/project/CASP17/.local/bin/kalign
  done
done

echo ""
echo "All Protenix reruns finished in ${SECONDS}s"
