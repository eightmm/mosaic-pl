"""Shell script generation for CASP17 pipeline runs."""

from __future__ import annotations

import shlex
from pathlib import Path

from casp17_pl_hub.adapters import PreparedModelRun
from casp17_pl_hub.configs import RunnerConfig


def build_shell_script(
    job_name: str,
    config: RunnerConfig,
    model_runs: list[PreparedModelRun],
    backend: str,
    run_dir: Path,
    stage_label: str,
) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    lines = ["#!/usr/bin/env bash"]
    if backend == "slurm":
        lines.extend(build_sbatch_header(job_name, config))
    lines.extend(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            "",
            "# This repository runs on a SLURM cluster.",
            "# The master/login node has no GPU; actual inference should run on compute nodes.",
            "module load cuda/12.8 2>/dev/null || true",
            'export LD_LIBRARY_PATH="${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"',
            f'for _nv_lib in {shlex.quote(str(repo_root))}/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do '
            'export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"; done',
            f"export PATH={shlex.quote(str(repo_root / '.local' / 'bin'))}:$PATH",
            f'echo "Starting CASP17 {stage_label} stage"',
            f'echo "run_dir={run_dir}"',
            "",
        ]
    )
    total = len(model_runs)
    model_names = [m.model_name for m in model_runs]
    has_boltz = "boltz" in model_names
    boltz_output_dir = ""
    bridge_script = repo_root / "scripts" / "bridge_boltz_msa_to_af3.py"

    for idx, model_run in enumerate(model_runs, 1):
        name_upper = model_run.model_name.upper()
        var_name = model_run.model_name.replace("-", "_")
        venv_bin = Path(model_run.command[0]).resolve().parent

        # Insert bridge step: Boltz MSA → AF3 input patching
        if model_run.model_name == "alphafold3" and has_boltz and boltz_output_dir:
            lines.extend([
                f'echo ""',
                f'echo "----------------------------------------------------------------"',
                f'echo "  BRIDGE: Boltz MSA -> AF3"',
                f'echo "----------------------------------------------------------------"',
                f"python3 {shlex.quote(str(bridge_script))} "
                f"--boltz-output-dir {shlex.quote(boltz_output_dir)} "
                f"--af3-input-json {shlex.quote(str(model_run.input_path))}",
                "",
            ])

        lines.extend([
            f'echo ""',
            f'echo "================================================================"',
            f'echo "  [{idx}/{total}] {name_upper}"',
            f'echo "================================================================"',
            f"export PATH={shlex.quote(str(venv_bin))}:$PATH",
            f"_start_{var_name}=$SECONDS",
            _render_command(model_run.command),
            f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
            f'echo "  [{idx}/{total}] {name_upper} done in ${{_elapsed_{var_name}}}s"',
            f'echo "================================================================"',
            "",
        ])

        if model_run.model_name == "boltz":
            boltz_output_dir = str(model_run.output_dir)
    lines.extend([
        'echo ""',
        'echo "All models finished in ${SECONDS}s"',
    ])
    return "\n".join(lines) + "\n"


def build_wrapper_shell_script(
    job_name: str,
    config: RunnerConfig,
    backend: str,
    stage_scripts: list[tuple[str, Path]],
) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    lines = ["#!/usr/bin/env bash"]
    if backend == "slurm":
        lines.extend(build_sbatch_header(job_name, config))
    lines.extend(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            "",
            "# Stage wrapper for the larger CASP17 protein-ligand workflow.",
            "# Stages are modular and can be combined per target.",
            "module load cuda/12.8 2>/dev/null || true",
            'export LD_LIBRARY_PATH="${CUDA_HOME:-/appl/cuda/12.8}/targets/x86_64-linux/lib:${CUDA_HOME:-/appl/cuda/12.8}/lib64:${LD_LIBRARY_PATH:-}"',
            f'for _nv_lib in {shlex.quote(str(repo_root))}/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do '
            'export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"; done',
            f"export PATH={shlex.quote(str(repo_root / '.local' / 'bin'))}:$PATH",
            'echo "Starting CASP17 wrapper pipeline"',
        ]
    )
    prep_script = repo_root / "scripts" / "prepare_docking_inputs.py"
    dock_python = repo_root / ".venvs" / "protenix-dock" / "bin" / "python"
    prev_stage = None
    for stage_name, script_path in stage_scripts:
        # Insert docking prep bridge between cofolding and docking
        if stage_name == "docking" and prev_stage == "cofolding":
            run_dir = script_path.parent.parent
            lines.extend([
                f'echo ""',
                f'echo "----------------------------------------------------------------"',
                f'echo "  BRIDGE: Preparing docking inputs"',
                f'echo "----------------------------------------------------------------"',
                f"{shlex.quote(str(dock_python))} {shlex.quote(str(prep_script))} "
                f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
                f"--run-dir {shlex.quote(str(run_dir))} "
                f"--output-dir {shlex.quote(str(run_dir / 'inputs' / 'docking'))} "
                f"--model auto",
                "",
            ])
        lines.extend(
            [
                f'echo "stage={stage_name}"',
                f"bash {shlex.quote(str(script_path))}",
                "",
            ]
        )
        prev_stage = stage_name
    return "\n".join(lines) + "\n"


def build_sbatch_header(job_name: str, config: RunnerConfig) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "experiments" / "logs"
    slurm = config.slurm
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={logs_dir / 'slurm-%j.out'}",
        f"#SBATCH --error={logs_dir / 'slurm-%j.err'}",
        f"#SBATCH --gres=gpu:{slurm.gpus}",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --mem={slurm.mem}",
        f"#SBATCH --time={slurm.time}",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    for extra_arg in slurm.extra_sbatch_args:
        lines.append(f"#SBATCH {extra_arg}")
    lines.append("")
    return lines


def _render_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
