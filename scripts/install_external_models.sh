#!/usr/bin/env bash
# =============================================================================
# CASP17 Protein-Ligand Hub — Full Installation Script
#
# Prerequisites:
#   - uv (Python package manager): https://docs.astral.sh/uv/
#   - git, make, g++, autoconf, automake, libtool
#   - libboost-all-dev (sudo apt-get install -y libboost-all-dev)
#   - CUDA toolkit (module or system) — only for AutoDock-GPU build
#
# Usage:
#   bash scripts/install_external_models.sh          # install everything
#   bash scripts/install_external_models.sh --verify  # verify installations
#
# After install, build AutoDock-GPU on a GPU node:
#   srun --partition=<gpu_partition> --gres=gpu:1 bash scripts/build_autodock_gpu.sh
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

banner() { echo ""; echo "================================================================"; echo "  $1"; echo "================================================================"; }
ok()     { echo "  ✓ $1"; }
fail()   { echo "  ✗ $1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--verify" ]]; then
  banner "Verifying installations"

  # Cofolding models
  .venvs/boltz/bin/python -c "import boltz; print('ok')" && ok "Boltz" || fail "Boltz"
  .venvs/protenix/bin/python -c "import protenix; print('ok')" && ok "Protenix" || fail "Protenix"
  .venvs/alphafold3/bin/python -c "import alphafold3; print('ok')" && ok "AlphaFold3" || fail "AlphaFold3"

  # Docking tools
  .venvs/protenix-dock/bin/python -c "from pxdock import ProtenixDock; print('ok')" && ok "Protenix-Dock" || fail "Protenix-Dock"
  .venvs/protenix-dock/bin/python -c "from vina import Vina; print('ok')" && ok "Vina (Python API)" || fail "Vina"
  .local/bin/autogrid4 --version 2>/dev/null | head -1 && ok "autogrid4" || fail "autogrid4"
  [ -f .local/bin/autodock_gpu_128wi ] && ok "AutoDock-GPU binary" || echo "  ⚠ AutoDock-GPU not built yet (run: srun ... bash scripts/build_autodock_gpu.sh)"

  # Search tools
  .local/bin/mmseqs version 2>/dev/null && ok "MMseqs2" || fail "MMseqs2"
  .local/bin/foldseek version 2>/dev/null && ok "Foldseek" || fail "Foldseek"

  # Binding site prediction
  .local/bin/prank -version 2>/dev/null | head -1 && ok "P2Rank" || fail "P2Rank"

  # Structure alignment
  .local/bin/USalign -h 2>/dev/null | head -1 && ok "US-align" || fail "US-align"

  # Helper libraries (docking prep)
  .venvs/protenix-dock/bin/python -c "import meeko, gemmi, pdb2pqr; from rdkit import Chem; print('ok')" \
    && ok "Helper libs (meeko, gemmi, pdb2pqr, RDKit)" || fail "Helper libs"

  # Post-processing
  .venvs/pred/bin/python -c "import bapred, rmsdpred, dgl; print('ok')" \
    && ok "BA-Pred + RMSD-Pred" || fail "BA-Pred / RMSD-Pred"
  (cd "${ROOT_DIR}/external/swinsite" && "${ROOT_DIR}/.venvs/pred/bin/python" -c "from SwinUnet import SwinSite; print('ok')") \
    && ok "SwinSite" || fail "SwinSite"

  # Template-guided docking
  .venv/bin/python -c "from lig_align import run_pipeline; print('ok')" \
    && ok "lig-align (MCS-guided)" || fail "lig-align"

  # Model weights
  [ -f external/alphafold3/models/af3.bin ] && ok "AF3 weights" || echo "  ⚠ AF3 weights not found"
  [ -f external/Protenix/models/protenix_base_default_v1.0.0.pt ] && ok "Protenix weights" || echo "  ⚠ Protenix weights not in external/Protenix/models/"

  echo ""; echo "All verifications passed."; exit 0
fi

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
banner "Checking prerequisites"

command -v uv    >/dev/null || fail "uv not found. Install: https://docs.astral.sh/uv/"
command -v git   >/dev/null || fail "git not found."
command -v make  >/dev/null || fail "make not found."
command -v g++   >/dev/null || fail "g++ not found."
command -v autoreconf >/dev/null || fail "autoreconf not found. Install: sudo apt-get install -y autoconf automake libtool"
[ -f /usr/include/boost/version.hpp ] || fail "Boost not found. Install: sudo apt-get install -y libboost-all-dev"
ok "All prerequisites found"

# ---------------------------------------------------------------------------
# Clone repositories
# ---------------------------------------------------------------------------
banner "Cloning external repositories"

clone_if_missing() {
  local repo_url="$1" target_dir="$2"
  if [[ ! -d "${target_dir}/.git" ]]; then
    git clone "${repo_url}" "${target_dir}"
    ok "Cloned ${target_dir}"
  else
    ok "Already exists: ${target_dir}"
  fi
}

clone_if_missing "https://github.com/jwohlwend/boltz.git"              "external/boltz"
clone_if_missing "https://github.com/bytedance/Protenix.git"           "external/Protenix"
clone_if_missing "https://github.com/google-deepmind/alphafold3.git"   "external/alphafold3"
clone_if_missing "https://github.com/bytedance/Protenix-Dock.git"      "external/Protenix-Dock"
clone_if_missing "https://github.com/ccsb-scripps/AutoDock-GPU.git"    "external/AutoDock-GPU"
clone_if_missing "https://github.com/ccsb-scripps/AutoGrid.git"        "external/AutoGrid"
clone_if_missing "https://github.com/eightmm/BA-Pred.git"             "external/BA-Pred"
clone_if_missing "https://github.com/eightmm/RMSD-Pred.git"           "external/RMSD-Pred"
clone_if_missing "https://github.com/ding-oh/swinsite.git"            "external/swinsite"
clone_if_missing "https://github.com/eightmm/lig-mcs-align.git"      "external/lig-mcs-align"

# ---------------------------------------------------------------------------
# Create virtual environments
# ---------------------------------------------------------------------------
banner "Creating virtual environments"

uv venv .venvs/boltz          --python 3.12
uv venv .venvs/protenix       --python 3.12
uv venv .venvs/alphafold3     --python 3.12
uv venv .venvs/protenix-dock  --python 3.11
uv venv .venvs/pred           --python 3.12
ok "All venvs created"

# ---------------------------------------------------------------------------
# [1/5] Boltz2
# ---------------------------------------------------------------------------
banner "[1/5] Installing Boltz2"

uv pip install --python .venvs/boltz/bin/python -e external/boltz
uv pip install --python .venvs/boltz/bin/python cuequivariance-torch cuequivariance-ops-torch-cu12
ok "Boltz2 installed"

# ---------------------------------------------------------------------------
# [2/5] Protenix
# ---------------------------------------------------------------------------
banner "[2/5] Installing Protenix"

uv pip install --python .venvs/protenix/bin/python -e external/Protenix
uv pip install --python .venvs/protenix/bin/python ninja
uv pip install --python .venvs/protenix/bin/python cuequivariance-torch cuequivariance-ops-torch-cu12
ok "Protenix installed"

# ---------------------------------------------------------------------------
# [3/5] AlphaFold3
# ---------------------------------------------------------------------------
banner "[3/5] Installing AlphaFold3"

uv pip install --python .venvs/alphafold3/bin/python -e external/alphafold3
.venvs/alphafold3/bin/build_data
ok "AlphaFold3 installed"

# ---------------------------------------------------------------------------
# [4/5] Protenix-Dock + Vina (shared venv)
# ---------------------------------------------------------------------------
banner "[4/5] Installing Protenix-Dock + Vina"

# Install deps first (vina as pre-built wheel to avoid Boost build)
uv pip install --python .venvs/protenix-dock/bin/python cmake "vina==1.2.7" \
  "byteff @ git+https://github.com/bytedance/byteff.git@90ef54a7c38e58f162adc23f3e8f457aaf922d46" \
  "AutoDockTools_py3 @ git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3@da0c87cac6a8a5b4cb6d1670f48e05d63f008dbf" \
  "pandas>=1.3.5" "func-timeout~=4.3.5" "parmed~=4.3.0" "pdb4amber~=1.4.1" \
  "MDAnalysis" "PyYaml" "pdb2pqr~=3.6.1" "meeko~=0.3.3" "gemmi"

# Build pxdock C++ engine (needs cmake + Boost in PATH)
PATH=".venvs/protenix-dock/bin:$PATH" \
  uv pip install --python .venvs/protenix-dock/bin/python \
    --no-deps --no-build-isolation external/Protenix-Dock
# Copy boost libs for compute nodes (may not have libboost installed)
mkdir -p .local/lib
cp -n /usr/lib/x86_64-linux-gnu/libboost_python3*.so* .local/lib/ 2>/dev/null || true
ok "Protenix-Dock + Vina installed (boost libs copied to .local/lib)"

# ---------------------------------------------------------------------------
# [5/5] Search & Docking Binaries
# ---------------------------------------------------------------------------
banner "[5/5] Installing search & docking binaries"

mkdir -p .local/bin

# MMseqs2 (sequence search)
if [ ! -f .local/bin/mmseqs ]; then
  curl -L https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar -xz -C /tmp/
  cp /tmp/mmseqs/bin/mmseqs .local/bin/
  ok "MMseqs2 installed"
else
  ok "MMseqs2 already installed"
fi

# Foldseek (structure search)
if [ ! -f .local/bin/foldseek ]; then
  curl -L https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz | tar -xz -C /tmp/
  cp /tmp/foldseek/bin/foldseek .local/bin/
  ok "Foldseek installed"
else
  ok "Foldseek already installed"
fi

# AutoGrid4 (required by AutoDock-GPU)
pushd external/AutoGrid >/dev/null
autoreconf -i 2>/dev/null
./configure --prefix="${ROOT_DIR}/.local" --quiet
# AutoGrid build needs csh for paramdat2h.csh — replace with inline bash
(
  echo 'const char *param_string_4_0[MAX_LINES] = {'
  grep -Ev '^#|^$' ad4_shared/AD4_parameters.dat | sed 's/\(.*\)$/"\1\\n", /'
  echo ' };'
  echo 'const char *param_string_4_1[MAX_LINES] = {'
  grep -Ev '^#|^$' ad4_shared/AD4.1_bound.dat | sed 's/\(.*\)$/"\1\\n", /'
  echo ' };'
  echo '// EOF'
) > default_parameters.h
make -j8 >/dev/null 2>&1 && make install >/dev/null 2>&1
popd >/dev/null
ok "autogrid4 built: .local/bin/autogrid4"

# JDK (required by P2Rank)
if [ ! -f .local/jdk/bin/java ]; then
  mkdir -p .local/jdk
  curl -L "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.6%2B7/OpenJDK21U-jre_x64_linux_hotspot_21.0.6_7.tar.gz" \
    | tar -xz -C .local/jdk --strip-components=1
  ok "JDK 21 installed"
else
  ok "JDK already installed"
fi

# P2Rank (binding site prediction)
if [ ! -d .local/p2rank_2.5 ]; then
  curl -L "https://github.com/rdk/p2rank/releases/download/2.5/p2rank_2.5.tar.gz" \
    | tar -xz -C .local/
  ok "P2Rank 2.5 installed"
else
  ok "P2Rank already installed"
fi
# Create prank wrapper that sets JAVA_HOME
cat > .local/bin/prank << 'PRANK_WRAPPER'
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAVA_HOME="$SCRIPT_DIR/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
P2RANK_HOME="$SCRIPT_DIR/p2rank_2.5"
exec java -Xmx2G -cp "$P2RANK_HOME/bin/p2rank.jar:$P2RANK_HOME/bin/lib/*" cz.siret.prank.program.Main "$@"
PRANK_WRAPPER
chmod +x .local/bin/prank
ok "P2Rank wrapper created"

# US-align / TMalign (structure alignment)
if [ ! -f .local/bin/USalign ]; then
  curl -L "https://zhanggroup.org/US-align/bin/module/USalign.cpp" -o /tmp/USalign.cpp
  g++ -O3 -o .local/bin/USalign /tmp/USalign.cpp
  ok "US-align installed"
else
  ok "US-align already installed"
fi
echo ""
echo "  NOTE: AutoDock-GPU requires CUDA and must be built on a GPU node:"
echo "    srun --partition=<gpu_partition> --gres=gpu:1 bash scripts/build_autodock_gpu.sh"

# ---------------------------------------------------------------------------
# [6/6] BA-Pred + RMSD-Pred (post-processing)
# ---------------------------------------------------------------------------
banner "[6/6] Installing BA-Pred + RMSD-Pred"

uv pip install --python .venvs/pred/bin/python \
  "torch==2.4.0" packaging gemmi
uv pip install --python .venvs/pred/bin/python \
  "dgl==2.4.0" -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
uv pip install --python .venvs/pred/bin/python \
  -e external/BA-Pred -e external/RMSD-Pred
uv pip install --python .venvs/pred/bin/python \
  openbabel-wheel einops scikit-image h5py timm
ok "BA-Pred + RMSD-Pred + SwinSite installed"

# ---------------------------------------------------------------------------
# Hub project itself
# ---------------------------------------------------------------------------
banner "Installing casp17-pl-hub"

uv sync --dev
uv pip install --python .venv/bin/python -e external/lig-mcs-align
ok "casp17-pl-hub + lig-align installed"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "Installation Complete"
cat <<'EOF'
  Installed:
    • Boltz2           (.venvs/boltz)
    • Protenix (.venvs/protenix)
    • AlphaFold3       (.venvs/alphafold3)
    • Protenix-Dock    (.venvs/protenix-dock)
    • Vina             (.venvs/protenix-dock)
    • autogrid4        (.local/bin/autogrid4)

  Still needed:
    • AutoDock-GPU — build on a GPU node:
        srun --partition=<gpu_partition> --gres=gpu:1 bash scripts/build_autodock_gpu.sh
    • AlphaFold3 model weights — download separately

  Verify with:
    bash scripts/install_external_models.sh --verify
EOF
