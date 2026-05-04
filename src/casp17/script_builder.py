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

# Docking models that support multi-seed (PxDock excluded — too expensive).
# After the binding-site variant refactor, model_names are
# ``vina_cofolding``/``vina_swinsite``/``vina_p2rank`` etc., so we match by
# prefix rather than exact set membership.
_DOCKING_MULTI_SEED_PREFIXES = ("vina", "autodock-gpu")


def _is_docking_multi_seed(model_name: str) -> bool:
    return model_name.startswith(_DOCKING_MULTI_SEED_PREFIXES) and model_name != "protenix-dock"


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
        is_docking_multi = _is_docking_multi_seed(model_run.model_name) and len(docking_seeds) > 1

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
            # Per-seed timeout: Vina's autodock engine can occasionally fail
            # to converge on flexible / polyisoprenoid-like ligands and hang
            # in ``Performing docking`` for hours (observed eating full 12h
            # wall-time on 9emd/9emo/9emt). Cap each seed at 15 min so one
            # bad seed doesn't kill the whole docking stage; surviving seeds
            # still produce usable poses.
            for si, seed in enumerate(docking_seeds, 1):
                seed_out = f"{base_out}/seed_{seed}"
                lines.extend([
                    f'echo "  seed {seed} ({si}/{len(docking_seeds)})"',
                    f"mkdir -p {shlex.quote(seed_out)}",
                    f"timeout 900 env DOCK_SEED={seed} DOCK_OUT_DIR={shlex.quote(seed_out)} {base_cmd}"
                    f" || echo '  (seed {seed} timed out or failed, continuing)'",
                ])
            lines.extend([
                f'_elapsed_{var_name}=$(( SECONDS - _start_{var_name} ))',
                f'echo "  [{idx}/{total}] {name_upper} ({len(docking_seeds)} seeds) done in ${{_elapsed_{var_name}}}s"',
                'echo "================================================================"',
                "",
            ])
        else:
            rendered = _render_command(model_run.command)
            # PxDock has no per-seed structure but can hang in conformer
            # generation on large/flexible ligands (observed: 9cv8 spent
            # 5h wall-time stuck there). Cap at 25 min; failure is caught
            # by the outer ``|| echo`` fallback so downstream stages still run.
            if model_run.model_name == "protenix-dock":
                rendered = (
                    f"timeout 1500 {rendered}"
                    f" || echo '  (protenix-dock timed out or failed, continuing)'"
                )
            # Re-create the stage output directory at runtime as well as
            # prep time. ``adapters.prepare_template_search_*`` already
            # makes the dir at prep, but if the user wipes ``outputs/`` and
            # reruns the same wrapper (common during fix-and-retest cycles),
            # mmseqs/foldseek then fail with "Cannot create temporary
            # directory" because ``--tmp-dir outputs/.../tmp`` expects the
            # parent. ``mkdir -p`` is a no-op when the dir already exists.
            lines.extend([
                'echo ""',
                'echo "================================================================"',
                f'echo "  [{idx}/{total}] {name_upper}"',
                'echo "================================================================"',
                f"mkdir -p {shlex.quote(str(model_run.output_dir))}",
                f"_start_{var_name}=$SECONDS",
                rendered,
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
            # Stamp the assigned GPU/node up-front so post-hoc log review
            # (e.g. diagnosing CUDA kernel mismatches on a specific node)
            # can tell which GPU ran this job without sacct round-trips.
            'echo "--- GPU allocation ---"',
            'echo "SLURM_NODELIST=${SLURM_NODELIST:-<none>}  SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"',
            'nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"',
            'echo "----------------------"',
        ]
    )
    align_script = repo_root / "scripts" / "align_cofolding_outputs.py"
    prep_script = repo_root / "scripts" / "prepare_docking_inputs.py"
    filter_script = repo_root / "scripts" / "run_template_filter.py"
    pockets_script = repo_root / "scripts" / "extract_template_pockets.py"
    cluster_script = repo_root / "scripts" / "cluster_template_pockets.py"
    msa_distribute_script = repo_root / "scripts" / "bridge_distribute_msa_templates.py"
    multi_track_script = repo_root / "scripts" / "run_multi_track_docking.py"
    ion_script = repo_root / "scripts" / "collect_template_ions.py"
    post_analysis_script = repo_root / "scripts" / "run_post_analysis.py"
    submission_script = repo_root / "scripts" / "make_casp_submission.py"
    dock_python = repo_root / ".venvs" / "protenix-dock" / "bin" / "python"
    pred_python = repo_root / ".venvs" / "pred" / "bin" / "python"
    hub_python = repo_root / ".venv" / "bin" / "python"
    stage_names = [s for s, _ in stage_scripts]
    has_template_search_seq = "template-search-sequence" in stage_names
    has_template_search_struct = "template-search-structure" in stage_names
    has_template_search = has_template_search_seq or has_template_search_struct
    has_docking = "docking" in stage_names
    has_cofolding = "cofolding" in stage_names

    def _emit_template_bridges(run_dir: Path) -> None:
        """Filter (mmseqs+foldseek union) → pocket extraction → spatial cluster.
        Runs as one block so downstream consumers (docking-prep, multi-track,
        ion placement) all see the same canonical artifacts.
        """
        ts_cfg = config.template_search_sequence
        rcsb_db = Path(ts_cfg.rcsb_db_path).expanduser()
        rcsb_dir = Path(ts_cfg.rcsb_dir).expanduser()
        mmseqs_tsv = run_dir / "outputs" / "template_search_sequence" / "mmseqs_hits.tsv"
        foldseek_tsv = run_dir / "outputs" / "template_search_structure" / "foldseek_hits.tsv"
        filtered_tsv = run_dir / "outputs" / "template_search_sequence" / "filtered_hits.tsv"
        input_yaml = run_dir / "inputs" / "boltz_input.yaml"
        pockets_json = run_dir / "outputs" / "template_pockets" / "template_pockets.json"

        filter_cmd = (
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(filter_script))} "
            f"--rcsb-db {shlex.quote(str(rcsb_db))} "
            f"--input-yaml {shlex.quote(str(input_yaml))} "
            f"--output-tsv {shlex.quote(str(filtered_tsv))}"
        )
        if has_template_search_seq:
            filter_cmd += f" --hits-tsv {shlex.quote(str(mmseqs_tsv))}"
        if has_template_search_struct:
            filter_cmd += f" --foldseek-tsv {shlex.quote(str(foldseek_tsv))}"
            qtm_min = config.template_search_structure.qtmscore_min
            if qtm_min > 0.0:
                filter_cmd += f" --foldseek-qtmscore-min {qtm_min}"
        if ts_cfg.max_deposition_date:
            filter_cmd += f" --max-deposition-date {shlex.quote(ts_cfg.max_deposition_date)}"
        lines.extend([
            'echo ""',
            'echo "================================================================"',
            'echo "  BRIDGE: Filtering template hits (mmseqs + foldseek union)"',
            'echo "================================================================"',
            f"{filter_cmd} || echo '  (template filter failed, continuing)'",
            "",
        ])

        # Pocket extraction + clustering need cofolding to have produced a
        # reference cif. When cofolding isn't a stage we still run the filter
        # so multi-track docking can use the hits, but skip the pocket steps.
        if has_cofolding:
            pockets_cmd = (
                f"{shlex.quote(str(hub_python))} {shlex.quote(str(pockets_script))} "
                f"--run-dir {shlex.quote(str(run_dir))} "
                f"--rcsb-dir {shlex.quote(str(rcsb_dir))}"
            )
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Extracting template pocket centers"',
                'echo "----------------------------------------------------------------"',
                f"{pockets_cmd} || echo '  (pocket extraction failed, continuing)'",
                "",
            ])
            cluster_cmd = (
                f"{shlex.quote(str(hub_python))} {shlex.quote(str(cluster_script))} "
                f"--pockets-json {shlex.quote(str(pockets_json))}"
            )
            lines.extend([
                'echo ""',
                'echo "----------------------------------------------------------------"',
                'echo "  BRIDGE: Clustering template pockets (top-K consensus)"',
                'echo "----------------------------------------------------------------"',
                f"{cluster_cmd} || echo '  (pocket clustering failed, continuing)'",
                "",
            ])

    def _emit_msa_pipeline_bridge(run_dir: Path) -> None:
        """Run AF3's data pipeline (jackhmmer + hmmsearch + templates),
        then distribute the MSA + template list to Boltz / Protenix /
        AF3 inputs so all three cofolding tools share the exact same
        homologs and templates instead of each fetching its own.

        This is the single CPU-only stage that replaces the
        ``Boltz fetches → bridge into the others`` chain. Cost is
        ~30-60 min/target on the SLURM 12h budget, swallowed by the
        existing wrapper job.
        """
        msa_cfg = config.msa_pipeline
        af3_python = repo_root / config.alphafold3.python_bin
        af3_script = repo_root / config.alphafold3.script
        af3_venv = af3_python.parent.parent
        nv_lib_glob = str(af3_venv / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
        msa_input_json = run_dir / "inputs" / "alphafold3_input.json"
        msa_output_dir = run_dir / "outputs" / "msa_pipeline"
        msa_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_parts = [
            shlex.quote(str(af3_python)),
            shlex.quote(str(af3_script)),
            f"--json_path={shlex.quote(str(msa_input_json))}",
            f"--output_dir={shlex.quote(str(msa_output_dir))}",
            "--run_data_pipeline=true",
            "--run_inference=false",
            f"--db_dir={shlex.quote(msa_cfg.db_dir)}",
            f"--jackhmmer_n_cpu={msa_cfg.n_cpu}",
            f"--nhmmer_n_cpu={msa_cfg.n_cpu}",
        ]
        # Optional binary path overrides. Default is whatever ``shutil.which``
        # picks up from PATH (we add ``.local/bin`` at the top of the wrapper),
        # so leave None ⇒ omit flag.
        for opt, val in (
            ("--jackhmmer_binary_path", msa_cfg.jackhmmer_binary_path),
            ("--hmmsearch_binary_path", msa_cfg.hmmsearch_binary_path),
            ("--hmmbuild_binary_path", msa_cfg.hmmbuild_binary_path),
            ("--nhmmer_binary_path", msa_cfg.nhmmer_binary_path),
            ("--hmmalign_binary_path", msa_cfg.hmmalign_binary_path),
        ):
            if val:
                cmd_parts.append(f"{opt}={shlex.quote(val)}")
        if msa_cfg.max_template_date:
            cmd_parts.append(f"--max_template_date={shlex.quote(msa_cfg.max_template_date)}")

        lines.extend([
            'echo ""',
            'echo "================================================================"',
            'echo "  BRIDGE: AF3 data pipeline (jackhmmer + hmmsearch + templates)"',
            'echo "================================================================"',
            f'for _nv_lib in {nv_lib_glob}; do export LD_LIBRARY_PATH="$_nv_lib:${{LD_LIBRARY_PATH:-}}"; done',
            f"{' '.join(cmd_parts)} || echo '  (msa pipeline failed, continuing — Boltz/Protenix/AF3 will use whatever fallback they have)'",
            "",
            'echo "----------------------------------------------------------------"',
            'echo "  BRIDGE: Distributing MSA + templates to Boltz / Protenix / AF3"',
            'echo "----------------------------------------------------------------"',
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(msa_distribute_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--max-templates {msa_cfg.max_templates}"
            f" || echo '  (distribute failed, continuing)'",
            "",
        ])

    template_bridges_done = False
    msa_bridges_done = False
    cofold_done = False
    for stage_name, script_path in stage_scripts:
        # Pre-cofolding bridge: AF3 data pipeline + distribute MSA/templates.
        # Runs once before whichever cofolding stage fires first; both Boltz
        # and Protenix consume the patched inputs from disk.
        if (
            stage_name == "cofolding"
            and config.msa_pipeline.enabled
            and not msa_bridges_done
        ):
            _emit_msa_pipeline_bridge(script_path.parent.parent)
            msa_bridges_done = True
        # Pre-docking bridges. Order matters: template artifacts (filter +
        # pockets + cluster) must come BEFORE docking-prep so the prep step
        # can register template_consensus_* sources for vina/adg/pxdock.
        if stage_name == "docking":
            run_dir = script_path.parent.parent
            if has_template_search and not template_bridges_done:
                _emit_template_bridges(run_dir)
                template_bridges_done = True
            if cofold_done:
                lines.extend([
                    'echo ""',
                    'echo "================================================================"',
                    'echo "  BRIDGE: Aligning cofolding outputs to common frame"',
                    'echo "================================================================"',
                    f"{shlex.quote(str(hub_python))} {shlex.quote(str(align_script))} "
                    f"--run-dir {shlex.quote(str(run_dir))}"
                    f" || echo '  (cofolding alignment failed, continuing)'",
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
                    f"--model auto"
                    f" || echo '  (docking prep failed, continuing)'",
                    "",
                ])
        lines.extend(
            [
                f'echo "stage={stage_name}"',
                f"bash {shlex.quote(str(script_path))} "
                f"|| echo '  (stage={stage_name} failed, continuing)'",
                "",
            ]
        )
        if stage_name == "cofolding":
            cofold_done = True

    # Template-only pipeline (no docking stage): emit bridges post-loop so
    # multi-track docking + ion placement still find filtered_hits.tsv.
    if has_template_search and not template_bridges_done:
        run_dir = stage_scripts[0][1].parent.parent
        _emit_template_bridges(run_dir)
        template_bridges_done = True

    # Multi-track docking: run after all stages if template search + docking both present
    if has_template_search and has_docking:
        run_dir = stage_scripts[0][1].parent.parent
        ts_cfg = config.template_search_sequence
        rcsb_dir = Path(ts_cfg.rcsb_dir).expanduser()
        rcsb_db = Path(ts_cfg.rcsb_db_path).expanduser()
        multi_track_cmd = (
            f"{shlex.quote(str(dock_python))} {shlex.quote(str(multi_track_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--input-yaml {shlex.quote(str(run_dir / 'inputs' / 'boltz_input.yaml'))} "
            f"--rcsb-dir {shlex.quote(str(rcsb_dir))} "
            f"--rcsb-db {shlex.quote(str(rcsb_db))} "
            f"--mcs-threshold {ts_cfg.mcs_threshold}"
        )
        if not config.protenix_dock.enabled:
            multi_track_cmd += " --skip-pxdock"
        lines.extend([
            'echo ""',
            'echo "================================================================"',
            'echo "  MULTI-TRACK DOCKING (Track 2 + Track 3)"',
            'echo "================================================================"',
            f"{multi_track_cmd} || echo '  (multi-track docking failed, continuing)'",
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
        # Ligand numbers + names are derived from docking_prep_summary.json
        # inside make_casp_submission.py. The --ligand-name flag overrides the
        # MDL "name" label — default "LIG" matches CASP convention.
        cmd = (
            f"{shlex.quote(str(hub_python))} {shlex.quote(str(submission_script))} "
            f"--run-dir {shlex.quote(str(run_dir))} "
            f"--target-id {shlex.quote(job_name)} "
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
