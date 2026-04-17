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

    Replaces --seed/--seeds and --out_dir in the rendered command. Caller
    guarantees ``model_run.model_name in _COFOLDING_SEED_ARG`` and
    ``len(seeds) > 1``.
    """
    seed_arg = _COFOLDING_SEED_ARG[model_run.model_name]
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
            '# MSA cache variable — set by the first Boltz seed, reused by',
            '# subsequent seeds + Boltz2x + Protenix to avoid redundant server fetches.',
            '_boltz_msa_cache=""',
            "",
        ]
    )
    total = len(model_runs)
    model_names = [m.model_name for m in model_runs]
    has_boltz = any(n.startswith("boltz") for n in model_names)
    boltz_output_dir = ""
    bridge_script = repo_root / "scripts" / "bridge_boltz_msa_to_af3.py"

    cofolding_seeds = config.cofolding_seeds
    docking_seeds = config.docking_seeds

    for idx, model_run in enumerate(model_runs, 1):
        name_upper = model_run.model_name.upper()
        var_name = model_run.model_name.replace("-", "_")
        venv_bin = Path(model_run.command[0]).resolve().parent

        # Insert bridge step: Boltz MSA → AF3 input patching
        if model_run.model_name == "alphafold3" and has_boltz and boltz_output_dir:
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Boltz MSA -> AF3"',
                'echo "----------------------------------------------------------------"',
                f"python3 {shlex.quote(str(bridge_script))} "
                f"--boltz-output-dir {shlex.quote(boltz_output_dir)} "
                f"--af3-input-json {shlex.quote(str(model_run.input_path))}",
                "",
            ])

        # Insert bridge step: Boltz MSA → Protenix JSON patching
        # Reuses the cached .a3m from Boltz so Protenix skips its own MSA
        # server fetch. The adapter already wires unpairedMsaPath when the
        # JSON contains it, and --use_msa true is set via config.
        if model_run.model_name == "protenix" and has_boltz:
            ptx_json = shlex.quote(str(model_run.input_path))
            # The python snippet reads the JSON path and a3m path from its
            # own argv so no unsafe string interpolation is needed. Using a
            # heredoc-style argument avoids the ``-c "..."`` escaping hell.
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Boltz MSA -> Protenix"',
                'echo "----------------------------------------------------------------"',
                'if [ -n "$_boltz_msa_cache" ] && [ -d "$_boltz_msa_cache" ]; then',
                '  _a3m=$(find "$_boltz_msa_cache" -name "uniref.a3m" 2>/dev/null | head -1)',
                '  if [ -n "$_a3m" ]; then',
                '    echo "  injecting MSA: $_a3m"',
                "    python3 - \"$_a3m\" " + ptx_json + " <<'PYEOF'",
                "import json, sys",
                "a3m_path, ptx_json_path = sys.argv[1], sys.argv[2]",
                "with open(ptx_json_path) as fh:",
                "    d = json.load(fh)",
                "items = d if isinstance(d, list) else [d]",
                "for item in items:",
                "    for seq in item.get('sequences', []):",
                "        if 'proteinChain' in seq:",
                "            seq['proteinChain']['unpairedMsaPath'] = a3m_path",
                "            seq['proteinChain']['pairedMsa'] = ''",
                "with open(ptx_json_path, 'w') as fh:",
                "    json.dump(d, fh, indent=2)",
                "PYEOF",
                '  else',
                '    echo "  no uniref.a3m found in cache, Protenix will use its own MSA server"',
                '  fi',
                'else',
                '  echo "  no Boltz MSA cache available, Protenix will use its own MSA server"',
                'fi',
                "",
            ])

        lines.append(f"export PATH={shlex.quote(str(venv_bin))}:$PATH")

        # Determine if this model uses multi-seed
        is_cofolding_multi = model_run.model_name in _COFOLDING_SEED_ARG and len(cofolding_seeds) > 1
        is_docking_multi = model_run.model_name in _DOCKING_MULTI_SEED and len(docking_seeds) > 1

        if is_cofolding_multi:
            seed_runs = _multi_seed_cofolding_commands(model_run, cofolding_seeds)
            is_boltz = model_run.model_name.startswith("boltz")
            lines.extend([
                'echo ""',
                'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper} ({len(seed_runs)} seeds)"',
                'echo "================================================================"',
                f"_start_{var_name}=$SECONDS",
            ])
            for si, (seed, cmd, out_dir) in enumerate(seed_runs, 1):
                # For Boltz models: after the first seed, pre-populate the MSA
                # directory for subsequent seeds so they skip the server fetch.
                # MSA is sequence-deterministic and independent of the diffusion
                # seed, so one fetch covers all seeds + Boltz2x.
                if is_boltz and si == 1:
                    lines.extend([
                        f'echo "  seed {seed} ({si}/{len(seed_runs)})"',
                        f"mkdir -p {shlex.quote(out_dir)}",
                        '# If MSA was already cached by a prior Boltz model (e.g. boltz2→boltz2x), reuse it',
                        'if [ -n "$_boltz_msa_cache" ] && [ -d "$_boltz_msa_cache" ]; then',
                        f'  _dest_parent={shlex.quote(out_dir)}/boltz_results_boltz_input',
                        '  mkdir -p "$_dest_parent"',
                        '  cp -r "$_boltz_msa_cache" "$_dest_parent/msa" 2>/dev/null || true',
                        'fi',
                        cmd,
                        '# Cache MSA from first seed for reuse by remaining seeds',
                        f'_boltz_msa_cache=$(find {shlex.quote(out_dir)} -type d -name "msa" -path "*/boltz_results_*/msa" 2>/dev/null | head -1)',
                    ])
                elif is_boltz and si > 1:
                    lines.extend([
                        f'echo "  seed {seed} ({si}/{len(seed_runs)})"',
                        f"mkdir -p {shlex.quote(out_dir)}",
                        '# Reuse cached MSA (skip server fetch)',
                        'if [ -n "$_boltz_msa_cache" ] && [ -d "$_boltz_msa_cache" ]; then',
                        f'  _dest_parent={shlex.quote(out_dir)}/boltz_results_boltz_input',
                        '  mkdir -p "$_dest_parent"',
                        '  cp -r "$_boltz_msa_cache" "$_dest_parent/msa" 2>/dev/null || true',
                        'fi',
                        cmd,
                    ])
                else:
                    lines.extend([
                        f'echo "  seed {seed} ({si}/{len(seed_runs)})"',
                        f"mkdir -p {shlex.quote(out_dir)}",
                        cmd,
                    ])
            lines.extend([
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} ({len(seed_runs)} seeds) done in ${{_elapsed_{var_name}}}s"',
                'echo "================================================================"',
                "",
            ])
        elif is_docking_multi:
            base_cmd = _render_command(model_run.command)
            base_out = str(model_run.output_dir)
            lines.extend([
                'echo ""',
                'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper} ({len(docking_seeds)} seeds)"',
                'echo "================================================================"',
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
                'echo "================================================================"',
                "",
            ])
        else:
            lines.extend([
                'echo ""',
                'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper}"',
                'echo "================================================================"',
                f"_start_{var_name}=$SECONDS",
                _render_command(model_run.command),
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} done in ${{_elapsed_{var_name}}}s"',
                'echo "================================================================"',
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
    align_script = repo_root / "scripts" / "align_cofolding_outputs.py"
    prep_script = repo_root / "scripts" / "prepare_docking_inputs.py"
    filter_script = repo_root / "scripts" / "run_template_filter.py"
    multi_track_script = repo_root / "scripts" / "run_multi_track_docking.py"
    ion_script = repo_root / "scripts" / "collect_template_ions.py"
    post_analysis_script = repo_root / "scripts" / "run_post_analysis.py"
    submission_script = repo_root / "scripts" / "make_casp_submission.py"
    dock_python = repo_root / ".venvs" / "protenix-dock" / "bin" / "python"
    pred_python = repo_root / ".venvs" / "pred" / "bin" / "python"
    hub_python = repo_root / ".venv" / "bin" / "python"
    stage_names = [s for s, _ in stage_scripts]
    has_template_search = "template-search-sequence" in stage_names
    has_docking = "docking" in stage_names
    has_cofolding = "cofolding" in stage_names
    prev_stage = None
    for stage_name, script_path in stage_scripts:
        # Insert alignment + docking prep bridges between cofolding and docking
        if stage_name == "docking" and prev_stage == "cofolding":
            run_dir = script_path.parent.parent
            # Align all cofolding outputs to a common frame
            lines.extend([
                'echo ""',
                'echo "================================================================"',
                'echo "  BRIDGE: Aligning cofolding outputs to common frame"',
                'echo "================================================================"',
                f"{shlex.quote(str(hub_python))} {shlex.quote(str(align_script))} "
                f"--run-dir {shlex.quote(str(run_dir))}",
                "",
            ])
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Preparing docking inputs"',
                'echo "----------------------------------------------------------------"',
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
            filter_cmd = (
                f"{shlex.quote(str(hub_python))} {shlex.quote(str(filter_script))} "
                f"--hits-tsv {shlex.quote(str(mmseqs_tsv))} "
                f"--rcsb-db {shlex.quote(str(rcsb_db))} "
                f"--input-yaml {shlex.quote(str(input_yaml))} "
                f"--output-tsv {shlex.quote(str(filtered_tsv))}"
            )
            if ts_cfg.max_deposition_date:
                filter_cmd += f" --max-deposition-date {shlex.quote(ts_cfg.max_deposition_date)}"
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Filtering template hits (ligand + MCS)"',
                'echo "----------------------------------------------------------------"',
                filter_cmd,
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
            'echo ""',
            'echo "================================================================"',
            'echo "  MULTI-TRACK DOCKING (Track 2 + Track 3)"',
            'echo "================================================================"',
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
            'echo ""',
            'echo "================================================================"',
            'echo "  ION/METAL PLACEMENT (template alignment)"',
            'echo "================================================================"',
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(ion_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
            f"--rcsb-dir {shlex.quote(str(rcsb_dir))} "
            f"--rcsb-db {shlex.quote(str(rcsb_db))} "
            f"|| echo '  (no ion entities or no templates found, skipping)'",
            "",
        ])

    # Post-analysis: BA-Pred + RMSD-Pred on all docking results
    if has_docking and config.post_analysis.enabled:
        run_dir = stage_scripts[0][1].parent.parent
        lines.extend([
            'echo ""',
            'echo "================================================================"',
            'echo "  POST-ANALYSIS (BA-Pred + RMSD-Pred)"',
            'echo "================================================================"',
            f"{shlex.quote(str(pred_python))} {shlex.quote(str(post_analysis_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--device {shlex.quote(config.post_analysis.device)} "
            f"|| echo '  (post-analysis failed, continuing)'",
            "",
        ])

    # CASP17 LG submission: generate .lg file from aggregated scores
    if has_docking and config.submission.enabled:
        run_dir = stage_scripts[0][1].parent.parent
        sub_cfg = config.submission
        submission_output = run_dir.parent.parent / "submissions" / f"{job_name}.lg"
        cmd = (
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(submission_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--target-id {shlex.quote(job_name)} "
            f"--ligand-name {shlex.quote(job_name)} "  # fallback to target id
            f"--ligand-number {sub_cfg.ligand_number} "
            f"--author {shlex.quote(sub_cfg.author)} "
            f"--method {shlex.quote(sub_cfg.method)} "
            f"--parent {shlex.quote(sub_cfg.parent)} "
            f"--output {shlex.quote(str(submission_output))}"
        )
        if sub_cfg.include_affinity:
            cmd += " --include-affinity"
        lines.extend([
            'echo ""',
            'echo "================================================================"',
            'echo "  CASP17 LG SUBMISSION"',
            'echo "================================================================"',
            f"mkdir -p {shlex.quote(str(submission_output.parent))}",
            f"{cmd} || echo '  (submission generation failed)'",
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
