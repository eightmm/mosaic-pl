"""Shell script generation for CASP17 pipeline runs."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from casp17.adapters import PreparedModelRun
from casp17.configs import RunnerConfig


# Models that support multi-seed via command-line arg replacement
_COFOLDING_SEED_ARG = {
    "boltz2": "--seed",
    "boltz2x": "--seed",
    "protenix": "--seeds",
}

# Docking models that support multi-seed (PxDock excluded — too expensive)
_DOCKING_MULTI_SEED = {"vina", "autodock-gpu"}


def _multi_seed_cofolding_commands(
    model_run: PreparedModelRun,
    seeds: list[int],
) -> list[tuple[int, str, str]]:
    """Generate (seed, command_str, output_dir) for each cofolding seed.

    Replaces --seed/--seeds and --out_dir in the rendered command.
    """
    seed_arg = _COFOLDING_SEED_ARG.get(model_run.model_name)
    if not seed_arg or len(seeds) <= 1:
        return [(seeds[0] if seeds else 42, _render_command(model_run.command), str(model_run.output_dir))]

    base_cmd = _render_command(model_run.command)
    base_out = str(model_run.output_dir)
    results = []
    for seed in seeds:
        cmd = re.sub(
            rf"({re.escape(seed_arg)})\s+\d+",
            rf"\1 {seed}",
            base_cmd,
        )
        seed_out = f"{base_out}/seed_{seed}"
        cmd = cmd.replace(shlex.quote(base_out), shlex.quote(seed_out))
        results.append((seed, cmd, seed_out))
    return results


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
            f'export LD_LIBRARY_PATH="{repo_root / ".local" / "lib"}:/usr/lib/x86_64-linux-gnu:${{CUDA_HOME:-/appl/cuda/12.8}}/targets/x86_64-linux/lib:${{CUDA_HOME:-/appl/cuda/12.8}}/lib64:${{LD_LIBRARY_PATH:-}}"',
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
    has_boltz = any(n.startswith("boltz") for n in model_names)
    boltz_output_dir = ""
    bridge_script = repo_root / "scripts" / "bridge_boltz_msa_to_af3.py"

    cofolding_seeds = config.cofolding_seeds if hasattr(config, "cofolding_seeds") else [42]
    docking_seeds = config.docking_seeds if hasattr(config, "docking_seeds") else [42]

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

        lines.append(f"export PATH={shlex.quote(str(venv_bin))}:$PATH")

        # Determine if this model uses multi-seed
        is_cofolding_multi = model_run.model_name in _COFOLDING_SEED_ARG and len(cofolding_seeds) > 1
        is_docking_multi = model_run.model_name in _DOCKING_MULTI_SEED and len(docking_seeds) > 1

        if is_cofolding_multi:
            seed_runs = _multi_seed_cofolding_commands(model_run, cofolding_seeds)
            lines.extend([
                f'echo ""',
                f'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper} ({len(seed_runs)} seeds)"',
                f'echo "================================================================"',
                f"_start_{var_name}=$SECONDS",
            ])
            for si, (seed, cmd, out_dir) in enumerate(seed_runs, 1):
                lines.extend([
                    f'echo "  seed {seed} ({si}/{len(seed_runs)})"',
                    f"mkdir -p {shlex.quote(out_dir)}",
                    cmd,
                ])
            lines.extend([
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} ({len(seed_runs)} seeds) done in ${{_elapsed_{var_name}}}s"',
                f'echo "================================================================"',
                "",
            ])
        elif is_docking_multi:
            base_cmd = _render_command(model_run.command)
            base_out = str(model_run.output_dir)
            lines.extend([
                f'echo ""',
                f'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper} ({len(docking_seeds)} seeds)"',
                f'echo "================================================================"',
                f"_start_{var_name}=$SECONDS",
            ])
            for si, seed in enumerate(docking_seeds, 1):
                seed_out = f"{base_out}/seed_{seed}"
                lines.extend([
                    f'echo "  seed {seed} ({si}/{len(docking_seeds)})"',
                    f"mkdir -p {shlex.quote(seed_out)}",
                    f"DOCK_SEED={seed} DOCK_OUT_DIR={shlex.quote(seed_out)} {base_cmd}",
                ])
            lines.extend([
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} ({len(docking_seeds)} seeds) done in ${{_elapsed_{var_name}}}s"',
                f'echo "================================================================"',
                "",
            ])
        else:
            lines.extend([
                f'echo ""',
                f'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper}"',
                f'echo "================================================================"',
                f"_start_{var_name}=$SECONDS",
                _render_command(model_run.command),
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} done in ${{_elapsed_{var_name}}}s"',
                f'echo "================================================================"',
                "",
            ])

        if model_run.model_name.startswith("boltz") and not boltz_output_dir:
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
            f'export LD_LIBRARY_PATH="{repo_root / ".local" / "lib"}:/usr/lib/x86_64-linux-gnu:${{CUDA_HOME:-/appl/cuda/12.8}}/targets/x86_64-linux/lib:${{CUDA_HOME:-/appl/cuda/12.8}}/lib64:${{LD_LIBRARY_PATH:-}}"',
            f'for _nv_lib in {shlex.quote(str(repo_root))}/.venvs/*/lib/python*/site-packages/nvidia/*/lib; do '
            'export LD_LIBRARY_PATH="$_nv_lib:$LD_LIBRARY_PATH"; done',
            f"export PATH={shlex.quote(str(repo_root / '.local' / 'bin'))}:$PATH",
            'echo "Starting CASP17 wrapper pipeline"',
        ]
    )
    prep_script = repo_root / "scripts" / "prepare_docking_inputs.py"
    filter_script = repo_root / "scripts" / "run_template_filter.py"
    multi_track_script = repo_root / "scripts" / "run_multi_track_docking.py"
    ion_script = repo_root / "scripts" / "collect_template_ions.py"
    dock_python = repo_root / ".venvs" / "protenix-dock" / "bin" / "python"
    hub_python = repo_root / ".venv" / "bin" / "python"
    stage_names = [s for s, _ in stage_scripts]
    has_template_search = "template-search-sequence" in stage_names
    has_docking = "docking" in stage_names
    has_cofolding = "cofolding" in stage_names
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
        # Insert template filter after template-search-sequence
        if stage_name == "template-search-sequence":
            run_dir = script_path.parent.parent
            ts_cfg = config.template_search_sequence
            rcsb_db = Path(ts_cfg.rcsb_db_path).expanduser()
            mmseqs_tsv = run_dir / "outputs" / "template_search_sequence" / "mmseqs_hits.tsv"
            filtered_tsv = run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
            input_yaml = run_dir / "inputs" / "boltz_input.yaml"
            lines.extend([
                f'echo ""',
                f'echo "----------------------------------------------------------------"',
                f'echo "  BRIDGE: Filtering template hits (ligand + MCS)"',
                f'echo "----------------------------------------------------------------"',
                f"{shlex.quote(str(hub_python))} {shlex.quote(str(filter_script))} "
                f"--hits-tsv {shlex.quote(str(mmseqs_tsv))} "
                f"--rcsb-db {shlex.quote(str(rcsb_db))} "
                f"--input-yaml {shlex.quote(str(input_yaml))} "
                f"--output-tsv {shlex.quote(str(filtered_tsv))}",
                "",
            ])
        prev_stage = stage_name

    # Multi-track docking: run after all stages if template search + docking both present
    if has_template_search and has_docking:
        run_dir = stage_scripts[0][1].parent.parent
        ts_cfg = config.template_search_sequence
        rcsb_dir = Path(ts_cfg.rcsb_dir).expanduser()
        rcsb_db = Path(ts_cfg.rcsb_db_path).expanduser()
        lines.extend([
            f'echo ""',
            f'echo "================================================================"',
            f'echo "  MULTI-TRACK DOCKING (Track 2 + Track 3)"',
            f'echo "================================================================"',
            f"{shlex.quote(str(dock_python))} {shlex.quote(str(multi_track_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
            f"--rcsb-dir {shlex.quote(str(rcsb_dir))} "
            f"--rcsb-db {shlex.quote(str(rcsb_db))} "
            f"--mcs-threshold {ts_cfg.mcs_threshold}",
            "",
        ])

    # Ion/metal placement: run after cofolding if template search is present
    # Auto-skips if input YAML has no ion/metal CCD entities
    if has_template_search and has_cofolding:
        run_dir = stage_scripts[0][1].parent.parent
        ts_cfg = config.template_search_sequence
        rcsb_dir = Path(ts_cfg.rcsb_dir).expanduser()
        rcsb_db = Path(ts_cfg.rcsb_db_path).expanduser()
        lines.extend([
            f'echo ""',
            f'echo "================================================================"',
            f'echo "  ION/METAL PLACEMENT (template alignment)"',
            f'echo "================================================================"',
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(ion_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
            f"--rcsb-dir {shlex.quote(str(rcsb_dir))} "
            f"--rcsb-db {shlex.quote(str(rcsb_db))} "
            f"|| echo '  (no ion entities or no templates found, skipping)'",
            "",
        ])

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
