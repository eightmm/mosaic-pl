"""Pipeline orchestration: prepare runs, generate manifests and shell scripts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from casp17.adapters import (
    PreparedModelRun,
    prepare_alphafold3,
    prepare_autodock_gpu,
    prepare_boltz,
    prepare_protenix,
    prepare_protenix_dock,
    prepare_template_search_sequence,
    prepare_template_search_structure,
    prepare_vina,
)
from casp17.configs import RunnerConfig
from casp17.io_utils import dump_json, dump_structured_file
from casp17.models import CommonInput
from casp17.script_builder import build_shell_script, build_wrapper_shell_script
from casp17.validation import _model_is_enabled, _resolve_stages, _validate_stage_dependencies


@dataclass(slots=True)
class PreparedRun:
    run_dir: Path
    manifest_path: Path
    shell_script: Path
    model_runs: list[PreparedModelRun]


def prepare_run(common: CommonInput, config: RunnerConfig, output_root: Path, backend: str) -> PreparedRun:
    config.validate(pipeline="structure")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_runs = [
        *prepare_boltz(common, config, run_dir),
        *[m for m in [
            prepare_protenix(common, config, run_dir),
            prepare_alphafold3(common, config, run_dir),
        ] if m is not None],
    ]
    if not model_runs:
        raise ValueError("No model is enabled in the runner config.")

    shell_script = run_dir / "scripts" / (
        "run_structure.sbatch.sh" if backend == "slurm" else "run_structure.sh"
    )
    return _finalize_prepared_run(
        common.name,
        config,
        model_runs,
        backend,
        run_dir,
        shell_script,
        run_dir / "run_manifest.json",
        "co-folding",
    )


def prepare_template_search_sequence_run(
    common: CommonInput, config: RunnerConfig, output_root: Path, backend: str
) -> PreparedRun:
    config.validate(pipeline="template_search_sequence")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_run = prepare_template_search_sequence(common, config, run_dir)
    if model_run is None:
        raise ValueError("Sequence-based template search is not enabled in the runner config.")

    shell_script = run_dir / "scripts" / (
        "run_template_search_sequence.sbatch.sh"
        if backend == "slurm"
        else "run_template_search_sequence.sh"
    )
    return _finalize_prepared_run(
        common.name,
        config,
        [model_run],
        backend,
        run_dir,
        shell_script,
        run_dir / "template_search_sequence_manifest.json",
        "template-search-sequence",
    )


def prepare_template_search_structure_run(
    common: CommonInput, config: RunnerConfig, output_root: Path, backend: str
) -> PreparedRun:
    config.validate(pipeline="template_search_structure")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_run = prepare_template_search_structure(common, config, run_dir)
    if model_run is None:
        raise ValueError("Structure-based template search is not enabled in the runner config.")

    shell_script = run_dir / "scripts" / (
        "run_template_search_structure.sbatch.sh"
        if backend == "slurm"
        else "run_template_search_structure.sh"
    )
    return _finalize_prepared_run(
        common.name,
        config,
        [model_run],
        backend,
        run_dir,
        shell_script,
        run_dir / "template_search_structure_manifest.json",
        "template-search-structure",
    )


def prepare_vina_run(common: CommonInput, config: RunnerConfig, output_root: Path, backend: str) -> PreparedRun:
    config.validate(pipeline="vina")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_run = prepare_vina(common, config, run_dir)
    if model_run is None:
        raise ValueError("AutoDock Vina is not enabled in the runner config.")

    shell_script = run_dir / "scripts" / ("run_vina.sbatch.sh" if backend == "slurm" else "run_vina.sh")
    return _finalize_prepared_run(
        common.name,
        config,
        [model_run],
        backend,
        run_dir,
        shell_script,
        run_dir / "vina_manifest.json",
        "docking",
    )


def prepare_protenix_dock_run(
    common: CommonInput, config: RunnerConfig, output_root: Path, backend: str
) -> PreparedRun:
    config.validate(pipeline="protenix_dock")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_run = prepare_protenix_dock(common, config, run_dir)
    if model_run is None:
        raise ValueError("Protenix-Dock is not enabled in the runner config.")

    shell_script = run_dir / "scripts" / (
        "run_protenix_dock.sbatch.sh" if backend == "slurm" else "run_protenix_dock.sh"
    )
    return _finalize_prepared_run(
        common.name,
        config,
        [model_run],
        backend,
        run_dir,
        shell_script,
        run_dir / "protenix_dock_manifest.json",
        "docking (protenix-dock)",
    )


def prepare_docking_run(
    common: CommonInput, config: RunnerConfig, output_root: Path, backend: str
) -> PreparedRun:
    config.validate(pipeline="docking")
    output_root = output_root.resolve()
    run_dir = output_root / common.name
    _ensure_run_dirs(run_dir)

    model_runs = [
        model_run
        for model_run in [
            prepare_vina(common, config, run_dir),
            prepare_autodock_gpu(common, config, run_dir),
            prepare_protenix_dock(common, config, run_dir),
        ]
        if model_run is not None
    ]
    if not model_runs:
        raise ValueError("No docking tool is enabled in the runner config.")

    shell_script = run_dir / "scripts" / (
        "run_docking.sbatch.sh" if backend == "slurm" else "run_docking.sh"
    )
    return _finalize_prepared_run(
        common.name,
        config,
        model_runs,
        backend,
        run_dir,
        shell_script,
        run_dir / "docking_manifest.json",
        "docking",
    )


def prepare_wrapper_run(common: CommonInput, config: RunnerConfig, output_root: Path, backend: str) -> PreparedRun:
    return prepare_stage_wrapper_run(common, config, output_root, backend, stages=None)


def prepare_stage_wrapper_run(
    common: CommonInput,
    config: RunnerConfig,
    output_root: Path,
    backend: str,
    stages: list[str] | None,
) -> PreparedRun:
    resolved_stages = _resolve_stages(config, stages)
    _validate_stage_dependencies(config, resolved_stages)
    prepared_by_stage: list[tuple[str, PreparedRun]] = []
    for stage in resolved_stages:
        if stage == "template-search-sequence":
            prepared_by_stage.append(
                (stage, prepare_template_search_sequence_run(common, config, output_root, backend))
            )
        elif stage == "template-search-structure":
            prepared_by_stage.append(
                (stage, prepare_template_search_structure_run(common, config, output_root, backend))
            )
        elif stage == "cofolding":
            prepared_by_stage.append((stage, prepare_run(common, config, output_root, backend)))
        elif stage == "docking":
            prepared_by_stage.append((stage, prepare_docking_run(common, config, output_root, backend)))

    wrapper_run = prepared_by_stage[0][1]
    shell_script = wrapper_run.run_dir / "scripts" / (
        "run_wrapper.sbatch.sh" if backend == "slurm" else "run_wrapper.sh"
    )
    shell_script.write_text(
        build_wrapper_shell_script(
            common.name,
            config,
            backend,
            [(stage, prepared.shell_script) for stage, prepared in prepared_by_stage],
        )
    )
    shell_script.chmod(0o755)
    manifest_path = wrapper_run.run_dir / "wrapper_manifest.json"
    dump_json(
        {
            "name": common.name,
            "backend": backend,
            "run_dir": str(wrapper_run.run_dir),
            "shell_script": str(shell_script),
            "stages": [
                {
                    "stage": stage,
                    "script": str(prepared.shell_script),
                    "manifest": str(prepared.manifest_path),
                }
                for stage, prepared in prepared_by_stage
            ],
            "models": [
                model_run.to_dict()
                for _, prepared in prepared_by_stage
                for model_run in prepared.model_runs
            ],
        },
        manifest_path,
    )
    return PreparedRun(
        run_dir=wrapper_run.run_dir,
        manifest_path=manifest_path,
        shell_script=shell_script,
        model_runs=[model_run for _, prepared in prepared_by_stage for model_run in prepared.model_runs],
    )


def execute_prepared_run(prepared: PreparedRun, backend: str) -> subprocess.CompletedProcess[str]:
    if backend == "local":
        return subprocess.run(
            ["bash", str(prepared.shell_script)],
            check=True,
            text=True,
            capture_output=True,
        )
    if backend == "slurm":
        return subprocess.run(
            ["sbatch", str(prepared.shell_script)],
            check=True,
            text=True,
            capture_output=True,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _ensure_run_dirs(run_dir: Path) -> None:
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "scripts").mkdir(parents=True, exist_ok=True)


def _finalize_prepared_run(
    job_name: str,
    config: RunnerConfig,
    model_runs: list[PreparedModelRun],
    backend: str,
    run_dir: Path,
    shell_script: Path,
    manifest_path: Path,
    stage_label: str,
) -> PreparedRun:
    shell_script.write_text(
        build_shell_script(job_name, config, model_runs, backend, run_dir, stage_label)
    )
    shell_script.chmod(0o755)
    dump_json(
        {
            "name": job_name,
            "backend": backend,
            "run_dir": str(run_dir),
            "shell_script": str(shell_script),
            "models": [model_run.to_dict() for model_run in model_runs],
        },
        manifest_path,
    )
    return PreparedRun(run_dir=run_dir, manifest_path=manifest_path, shell_script=shell_script, model_runs=model_runs)


def write_example_input(path: Path) -> None:
    dump_structured_file(
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
                        "msa": "empty",
                    }
                },
                {
                    "ligand": {
                        "id": "L",
                        "smiles": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
                    }
                },
            ],
            "seed": 101,
            "properties": [{"affinity": {"binder": "L"}}],
        },
        path,
    )


def write_example_config(path: Path, preset: str = "balanced") -> None:
    config = RunnerConfig.from_dict({"preset": preset}).to_dict()
    config["boltz"]["msa_server_url"] = "https://api.colabfold.com"
    config["boltz"]["msa_pairing_strategy"] = "greedy"
    config["template_search_sequence"]["database_path"] = "/path/to/rcsb/mmseqs/protein_db"
    config["template_search_structure"]["database_path"] = "/path/to/rcsb/foldseek/structure_db"
    config["template_search_structure"]["query_structure_path"] = "/path/to/query_structure.cif"
    config["template_search_structure"]["query_from_cofolding"] = False
    config["template_search_structure"]["query_model_priority"] = ["alphafold3", "boltz", "protenix"]
    config["alphafold3"]["model_dir"] = "/path/to/alphafold3/models"
    config["alphafold3"]["db_dirs"] = ["/path/to/alphafold3/databases"]
    config["alphafold3"]["jackhmmer_n_cpu"] = 8
    config["alphafold3"]["nhmmer_n_cpu"] = 8
    config["alphafold3"]["gpu_device"] = 0
    config["alphafold3"]["flash_attention_implementation"] = "triton"
    config["vina"]["receptor_pdbqt"] = "/path/to/receptor.pdbqt"
    config["vina"]["ligand_pdbqt"] = "/path/to/ligand.pdbqt"
    config["vina"]["center_x"] = 0.0
    config["vina"]["center_y"] = 0.0
    config["vina"]["center_z"] = 0.0
    config["vina"]["size_x"] = 20.0
    config["vina"]["size_y"] = 20.0
    config["vina"]["size_z"] = 20.0
    config["protenix_dock"]["receptor_pdb"] = "/path/to/receptor.pdb"
    config["protenix_dock"]["ligand_sdf"] = "/path/to/ligand.sdf"
    config["protenix_dock"]["center_x"] = 0.0
    config["protenix_dock"]["center_y"] = 0.0
    config["protenix_dock"]["center_z"] = 0.0
    config["protenix_dock"]["size_x"] = 20.0
    config["protenix_dock"]["size_y"] = 20.0
    config["protenix_dock"]["size_z"] = 20.0
    config["autodock_gpu"]["receptor_pdbqt"] = "/path/to/receptor.pdbqt"
    config["autodock_gpu"]["ligand_pdbqt"] = "/path/to/ligand.pdbqt"
    config["autodock_gpu"]["center_x"] = 0.0
    config["autodock_gpu"]["center_y"] = 0.0
    config["autodock_gpu"]["center_z"] = 0.0
    config["autodock_gpu"]["size_x"] = 20.0
    config["autodock_gpu"]["size_y"] = 20.0
    config["autodock_gpu"]["size_z"] = 20.0
    config["slurm"]["partition"] = "gpu"
    dump_structured_file(config, path)


def manifest_as_text(prepared: PreparedRun) -> str:
    manifest = json.loads(prepared.manifest_path.read_text())
    return json.dumps(manifest, indent=2)
