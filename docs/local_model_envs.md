# Local Model Environments

This repository keeps third-party model code under `external/` and isolated local Python environments under `.venvs/`.

## Cluster note

- This is a SLURM environment.
- The master or login node does not have a GPU.
- GPUs are available only on allocated compute nodes.
- Installation and import checks can run on the master node.
- CUDA runtime checks, model inference, and performance validation must run on a compute node.

## Local environments

- `external/Protenix` with `.venvs/protenix`
- `external/boltz` with `.venvs/boltz`
- `external/alphafold3` with `.venvs/alphafold3`

## Current status

- `Boltz`: installed and importable; `boltz predict --help` responds on the master node.
- `Protenix`: installed and importable; local clone patched so master-node imports do not force CUDA LayerNorm compilation.
- `AlphaFold3`: installed and importable; `build_data` completed and generated the required converter pickle files.

## Protenix local patch

`external/Protenix/protenix/model/layer_norm/layer_norm.py` was patched locally so that:

- on GPU nodes with CUDA available, it keeps using the fused CUDA path
- on the master node without CUDA, it falls back to PyTorch `layer_norm`

This avoids failing import-time extension compilation on the master node.

## Activation

```bash
source .venvs/protenix/bin/activate
source .venvs/boltz/bin/activate
source .venvs/alphafold3/bin/activate
```

## Install commands

```bash
git clone https://github.com/bytedance/Protenix.git external/Protenix
git clone https://github.com/jwohlwend/boltz.git external/boltz
git clone https://github.com/google-deepmind/alphafold3.git external/alphafold3

uv venv .venvs/protenix --python 3.12
uv venv .venvs/boltz --python 3.12
uv venv .venvs/alphafold3 --python 3.12

uv pip install --python .venvs/protenix/bin/python -e external/Protenix
uv pip install --python .venvs/boltz/bin/python -e external/boltz
uv pip install --python .venvs/alphafold3/bin/python -e external/alphafold3
```

## Compute-node reminder

Example pattern for GPU-backed runs:

```bash
srun --gres=gpu:1 --pty bash
source .venvs/protenix/bin/activate
```

Adapt the `srun` resource request to the actual cluster policy.

## Helpers

- `scripts/install_external_models.sh`
- `scripts/verify_external_models.sh`
