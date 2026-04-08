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

  .venvs/boltz/bin/python -c "import boltz; print('boltz-import-ok')" && ok "Boltz" || fail "Boltz"
  .venvs/protenix/bin/python -c "import protenix; print('protenix-import-ok')" && ok "Protenix" || fail "Protenix"
  .venvs/alphafold3/bin/python -c "import alphafold3; print('alphafold3-import-ok')" && ok "AlphaFold3" || fail "AlphaFold3"
  .venvs/protenix-dock/bin/python -c "from pxdock import ProtenixDock; print('pxdock-ok')" && ok "Protenix-Dock" || fail "Protenix-Dock"
  .venvs/protenix-dock/bin/python -c "import vina; print('vina-ok')" && ok "Vina (Python)" || fail "Vina"
  .local/bin/autogrid4 --version 2>/dev/null | head -1 && ok "autogrid4" || fail "autogrid4"
  [ -f .local/bin/autodock_gpu_128wi ] && ok "AutoDock-GPU binary" || echo "  ⚠ AutoDock-GPU not built yet (run: srun ... bash scripts/build_autodock_gpu.sh)"

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

# ---------------------------------------------------------------------------
# Create virtual environments
# ---------------------------------------------------------------------------
banner "Creating virtual environments"

uv venv .venvs/boltz          --python 3.12
uv venv .venvs/protenix       --python 3.12
uv venv .venvs/alphafold3     --python 3.12
uv venv .venvs/protenix-dock  --python 3.11
ok "All venvs created"

# ---------------------------------------------------------------------------
# [1/5] Boltz2
# ---------------------------------------------------------------------------
banner "[1/5] Installing Boltz2"

uv pip install --python .venvs/boltz/bin/python -e external/boltz
uv pip install --python .venvs/boltz/bin/python cuequivariance-torch cuequivariance-ops-torch-cu12
ok "Boltz2 installed"

# ---------------------------------------------------------------------------
# [2/5] Protenix v2
# ---------------------------------------------------------------------------
banner "[2/5] Installing Protenix v2"

uv pip install --python .venvs/protenix/bin/python -e external/Protenix
uv pip install --python .venvs/protenix/bin/python ninja
uv pip install --python .venvs/protenix/bin/python cuequivariance-torch cuequivariance-ops-torch-cu12
ok "Protenix v2 installed"

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
ok "Protenix-Dock + Vina installed"

# ---------------------------------------------------------------------------
# [5/5] AutoGrid4 + AutoDock-GPU
# ---------------------------------------------------------------------------
banner "[5/5] Building AutoGrid4"

mkdir -p .local/bin
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

echo ""
echo "  NOTE: AutoDock-GPU requires CUDA and must be built on a GPU node:"
echo "    srun --partition=<gpu_partition> --gres=gpu:1 bash scripts/build_autodock_gpu.sh"

# ---------------------------------------------------------------------------
# Hub project itself
# ---------------------------------------------------------------------------
banner "Installing casp17-pl-hub"

uv sync --dev
ok "casp17-pl-hub installed"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "Installation Complete"
cat <<'EOF'
  Installed:
    • Boltz2           (.venvs/boltz)
    • Protenix v2      (.venvs/protenix)
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
