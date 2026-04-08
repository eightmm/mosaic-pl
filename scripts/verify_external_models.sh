#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

.venvs/boltz/bin/python -c "import boltz; print('boltz-import-ok')"
timeout 20s .venvs/boltz/bin/boltz predict --help >/dev/null

.venvs/protenix/bin/python -c "import protenix; print('protenix-import-ok')"
.venvs/protenix/bin/python -c "from protenix.model.layer_norm.layer_norm import FusedLayerNorm; print('protenix-layernorm-fallback-ok')"
timeout 20s .venvs/protenix/bin/python -c "import runner.batch_inference; print('protenix-runner-import-ok')"

.venvs/alphafold3/bin/python -c "import alphafold3; from alphafold3.cpp import cif_dict; print('alphafold3-import-ok')"
test -f external/alphafold3/src/alphafold3/constants/converters/ccd.pickle
test -f external/alphafold3/src/alphafold3/constants/converters/chemical_component_sets.pickle

.venvs/protenix-dock/bin/python -c "from pxdock import ProtenixDock; print('protenix-dock-import-ok')"

cat <<'EOF'
Verification complete.

Notes:
- Boltz CLI help responds on the master node.
- Protenix imports on the master node with the local LayerNorm fallback patch.
- AlphaFold3 imports and its generated converter data files are present.
- Full inference for all three models should be run on SLURM compute nodes with GPUs.
EOF
