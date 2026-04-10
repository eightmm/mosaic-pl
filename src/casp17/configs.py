"""Configuration dataclasses and preset definitions for CASP17 pipeline models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MODEL_NAMES = (
    "boltz",
    "protenix",
    "alphafold3",
    "vina",
    "autodock_gpu",
    "protenix_dock",
    "template_search_sequence",
    "template_search_structure",
)
PRESET_NAMES = ("fast", "balanced", "quality")
PRESET_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "fast": {
        "boltz": {
            "recycling_steps": 1,
            "sampling_steps": 75,
            "diffusion_samples": 1,
            "max_parallel_samples": 8,
            "sampling_steps_affinity": 75,
            "diffusion_samples_affinity": 2,
        },
        "protenix": {
            "cycle": 4,
            "step": 75,
            "sample": 1,
        },
        "alphafold3": {
            "num_recycles": 3,
            "num_diffusion_samples": 1,
        },
        "vina": {
            "cpu": 4,
            "exhaustiveness": 8,
            "num_modes": 9,
            "energy_range": 3.0,
        },
        "protenix_dock": {
            "cache_map_spacing": 0.375,
        },
    },
    "balanced": {
        "boltz": {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 5,
            "sampling_steps_affinity": 200,
            "diffusion_samples_affinity": 5,
        },
        "protenix": {
            "cycle": 10,
            "step": 200,
            "sample": 5,
        },
        "alphafold3": {
            "num_recycles": 10,
            "num_diffusion_samples": 5,
        },
        "vina": {
            "cpu": 8,
            "exhaustiveness": 16,
            "num_modes": 20,
            "energy_range": 4.0,
        },
        "protenix_dock": {
            "cache_map_spacing": 0.375,
        },
    },
    "quality": {
        "boltz": {
            "recycling_steps": 6,
            "sampling_steps": 400,
            "diffusion_samples": 5,
            "max_parallel_samples": 2,
            "sampling_steps_affinity": 400,
            "diffusion_samples_affinity": 10,
            "write_full_pae": True,
            "write_full_pde": True,
        },
        "protenix": {
            "cycle": 20,
            "step": 400,
            "sample": 8,
        },
        "alphafold3": {
            "num_recycles": 20,
            "num_diffusion_samples": 10,
            "save_embeddings": True,
        },
        "vina": {
            "cpu": 8,
            "exhaustiveness": 32,
            "num_modes": 20,
            "energy_range": 6.0,
        },
        "protenix_dock": {
            "cache_map_spacing": 0.375,
        },
    },
}


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


def _normalize_preset_name(value: Any) -> str:
    preset = str(value).strip().lower()
    if preset not in PRESET_NAMES:
        raise ValueError(f"preset must be one of {', '.join(PRESET_NAMES)}.")
    return preset


def _merge_preset_section(
    preset: str, section_name: str, overrides: dict[str, Any] | None
) -> dict[str, Any]:
    merged = deepcopy(PRESET_OVERRIDES.get(preset, {}).get(section_name, {}))
    if overrides is None:
        return merged
    if not isinstance(overrides, dict):
        raise ValueError(f"{section_name} must be a mapping when provided.")
    merged.update(overrides)
    return merged


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _all_or_none(values: list[Any]) -> bool:
    return all(value is None for value in values) or all(value is not None for value in values)


@dataclass(slots=True)
class BoltzConfig:
    enabled: bool = True
    binary: str = ".venvs/boltz/bin/boltz"
    model: str = "boltz2"
    accelerator: str = "gpu"
    devices: int = 1
    cache: str | None = None
    checkpoint: str | None = None
    affinity_checkpoint: str | None = None
    use_msa_server: bool = False
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    max_parallel_samples: int | None = None
    step_scale: float | None = None
    output_format: str = "mmcif"
    num_workers: int = 2
    override: bool = False
    msa_server_url: str | None = None
    msa_pairing_strategy: str | None = None
    msa_server_username: str | None = None
    msa_server_password: str | None = None
    api_key_header: str | None = None
    api_key_value: str | None = None
    use_potentials: bool = False
    method: str | None = None
    preprocessing_threads: int | None = None
    affinity_mw_correction: bool = False
    sampling_steps_affinity: int = 200
    diffusion_samples_affinity: int = 5
    max_msa_seqs: int = 8192
    subsample_msa: bool = True
    num_subsampled_msa: int = 1024
    no_kernels: bool = False
    write_full_pae: bool = False
    write_full_pde: bool = False
    write_embeddings: bool = False
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoltzConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), True),
            binary=str(data.get("binary", ".venvs/boltz/bin/boltz")),
            model=str(data.get("model", "boltz2")),
            accelerator=str(data.get("accelerator", "gpu")),
            devices=int(data.get("devices", 1)),
            cache=_optional_string(data.get("cache"), "boltz.cache"),
            checkpoint=_optional_string(data.get("checkpoint"), "boltz.checkpoint"),
            affinity_checkpoint=_optional_string(
                data.get("affinity_checkpoint"), "boltz.affinity_checkpoint"
            ),
            use_msa_server=_to_bool(data.get("use_msa_server"), False),
            recycling_steps=int(data.get("recycling_steps", 3)),
            sampling_steps=int(data.get("sampling_steps", 200)),
            diffusion_samples=int(data.get("diffusion_samples", 1)),
            max_parallel_samples=(
                int(data["max_parallel_samples"])
                if data.get("max_parallel_samples") is not None
                else None
            ),
            step_scale=(
                float(data["step_scale"]) if data.get("step_scale") is not None else None
            ),
            output_format=str(data.get("output_format", "mmcif")),
            num_workers=int(data.get("num_workers", 2)),
            override=_to_bool(data.get("override"), False),
            msa_server_url=_optional_string(data.get("msa_server_url"), "boltz.msa_server_url"),
            msa_pairing_strategy=_optional_string(
                data.get("msa_pairing_strategy"), "boltz.msa_pairing_strategy"
            ),
            msa_server_username=_optional_string(
                data.get("msa_server_username"), "boltz.msa_server_username"
            ),
            msa_server_password=_optional_string(
                data.get("msa_server_password"), "boltz.msa_server_password"
            ),
            api_key_header=_optional_string(data.get("api_key_header"), "boltz.api_key_header"),
            api_key_value=_optional_string(data.get("api_key_value"), "boltz.api_key_value"),
            use_potentials=_to_bool(data.get("use_potentials"), False),
            method=_optional_string(data.get("method"), "boltz.method"),
            preprocessing_threads=(
                int(data["preprocessing_threads"])
                if data.get("preprocessing_threads") is not None
                else None
            ),
            affinity_mw_correction=_to_bool(data.get("affinity_mw_correction"), False),
            sampling_steps_affinity=int(data.get("sampling_steps_affinity", 200)),
            diffusion_samples_affinity=int(data.get("diffusion_samples_affinity", 5)),
            max_msa_seqs=int(data.get("max_msa_seqs", 8192)),
            subsample_msa=_to_bool(data.get("subsample_msa"), True),
            num_subsampled_msa=int(data.get("num_subsampled_msa", 1024)),
            no_kernels=_to_bool(data.get("no_kernels"), False),
            write_full_pae=_to_bool(data.get("write_full_pae"), False),
            write_full_pde=_to_bool(data.get("write_full_pde"), False),
            write_embeddings=_to_bool(data.get("write_embeddings"), False),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class ProtenixConfig:
    enabled: bool = True
    binary: str = ".venvs/protenix/bin/protenix"
    model_name: str = "protenix-v2"
    cycle: int = 10
    step: int = 200
    sample: int = 5
    dtype: str = "bf16"
    use_msa: bool = False
    use_default_params: bool = False
    trimul_kernel: str = "cuequivariance"
    triatt_kernel: str = "cuequivariance"
    enable_cache: bool = True
    enable_fusion: bool = True
    enable_tf32: bool = True
    msa_server_mode: str = "protenix"
    use_template: bool = False
    use_rna_msa: bool = False
    use_seeds_in_json: bool = False
    need_atom_confidence: bool = False
    use_tfg_guidance: bool = False
    kalign_binary_path: str | None = None
    hmmsearch_binary_path: str | None = None
    hmmbuild_binary_path: str | None = None
    seqres_database_path: str | None = None
    nhmmer_binary_path: str | None = None
    hmmalign_binary_path: str | None = None
    hmmbuild_rna_binary_path: str | None = None
    ntrna_database_path: str | None = None
    rfam_database_path: str | None = None
    rna_central_database_path: str | None = None
    nhmmer_n_cpu: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProtenixConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), True),
            binary=str(data.get("binary", ".venvs/protenix/bin/protenix")),
            model_name=str(data.get("model_name", "protenix-v2")),
            cycle=int(data.get("cycle", 10)),
            step=int(data.get("step", 200)),
            sample=int(data.get("sample", 5)),
            dtype=str(data.get("dtype", "bf16")),
            use_msa=_to_bool(data.get("use_msa"), False),
            use_default_params=_to_bool(data.get("use_default_params"), False),
            trimul_kernel=str(data.get("trimul_kernel", "cuequivariance")),
            triatt_kernel=str(data.get("triatt_kernel", "cuequivariance")),
            enable_cache=_to_bool(data.get("enable_cache"), True),
            enable_fusion=_to_bool(data.get("enable_fusion"), True),
            enable_tf32=_to_bool(data.get("enable_tf32"), True),
            msa_server_mode=str(data.get("msa_server_mode", "protenix")),
            use_template=_to_bool(data.get("use_template"), False),
            use_rna_msa=_to_bool(data.get("use_rna_msa"), False),
            use_seeds_in_json=_to_bool(data.get("use_seeds_in_json"), False),
            need_atom_confidence=_to_bool(data.get("need_atom_confidence"), False),
            use_tfg_guidance=_to_bool(data.get("use_tfg_guidance"), False),
            kalign_binary_path=_optional_string(
                data.get("kalign_binary_path"), "protenix.kalign_binary_path"
            ),
            hmmsearch_binary_path=_optional_string(
                data.get("hmmsearch_binary_path"), "protenix.hmmsearch_binary_path"
            ),
            hmmbuild_binary_path=_optional_string(
                data.get("hmmbuild_binary_path"), "protenix.hmmbuild_binary_path"
            ),
            seqres_database_path=_optional_string(
                data.get("seqres_database_path"), "protenix.seqres_database_path"
            ),
            nhmmer_binary_path=_optional_string(
                data.get("nhmmer_binary_path"), "protenix.nhmmer_binary_path"
            ),
            hmmalign_binary_path=_optional_string(
                data.get("hmmalign_binary_path"), "protenix.hmmalign_binary_path"
            ),
            hmmbuild_rna_binary_path=_optional_string(
                data.get("hmmbuild_rna_binary_path"), "protenix.hmmbuild_rna_binary_path"
            ),
            ntrna_database_path=_optional_string(
                data.get("ntrna_database_path"), "protenix.ntrna_database_path"
            ),
            rfam_database_path=_optional_string(
                data.get("rfam_database_path"), "protenix.rfam_database_path"
            ),
            rna_central_database_path=_optional_string(
                data.get("rna_central_database_path"), "protenix.rna_central_database_path"
            ),
            nhmmer_n_cpu=(
                int(data["nhmmer_n_cpu"]) if data.get("nhmmer_n_cpu") is not None else None
            ),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class AlphaFold3Config:
    enabled: bool = True
    python_bin: str = ".venvs/alphafold3/bin/python"
    script: str = "external/alphafold3/run_alphafold.py"
    model_dir: str | None = None
    db_dirs: list[str] = field(default_factory=list)
    run_data_pipeline: bool = False
    run_inference: bool = True
    jackhmmer_binary_path: str | None = None
    nhmmer_binary_path: str | None = None
    hmmalign_binary_path: str | None = None
    hmmsearch_binary_path: str | None = None
    hmmbuild_binary_path: str | None = None
    jackhmmer_n_cpu: int | None = None
    jackhmmer_max_parallel_shards: int | None = None
    nhmmer_n_cpu: int | None = None
    nhmmer_max_parallel_shards: int | None = None
    resolve_msa_overlaps: bool | None = None
    max_template_date: str | None = None
    conformer_max_iterations: int | None = None
    jax_compilation_cache_dir: str | None = None
    gpu_device: int | None = None
    buckets: list[int] = field(default_factory=list)
    flash_attention_implementation: str | None = None
    num_recycles: int | None = None
    num_diffusion_samples: int | None = None
    num_seeds: int | None = None
    save_embeddings: bool = False
    save_distogram: bool = False
    force_output_dir: bool = False
    compress_large_output_files: bool = False
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AlphaFold3Config":
        if data is None:
            return cls()
        db_dirs = [str(item) for item in data.get("db_dirs", [])]
        return cls(
            enabled=_to_bool(data.get("enabled"), True),
            python_bin=str(data.get("python_bin", ".venvs/alphafold3/bin/python")),
            script=str(data.get("script", "external/alphafold3/run_alphafold.py")),
            model_dir=_optional_string(data.get("model_dir"), "alphafold3.model_dir"),
            db_dirs=db_dirs,
            run_data_pipeline=_to_bool(data.get("run_data_pipeline"), False),
            run_inference=_to_bool(data.get("run_inference"), True),
            jackhmmer_binary_path=_optional_string(
                data.get("jackhmmer_binary_path"), "alphafold3.jackhmmer_binary_path"
            ),
            nhmmer_binary_path=_optional_string(
                data.get("nhmmer_binary_path"), "alphafold3.nhmmer_binary_path"
            ),
            hmmalign_binary_path=_optional_string(
                data.get("hmmalign_binary_path"), "alphafold3.hmmalign_binary_path"
            ),
            hmmsearch_binary_path=_optional_string(
                data.get("hmmsearch_binary_path"), "alphafold3.hmmsearch_binary_path"
            ),
            hmmbuild_binary_path=_optional_string(
                data.get("hmmbuild_binary_path"), "alphafold3.hmmbuild_binary_path"
            ),
            jackhmmer_n_cpu=(
                int(data["jackhmmer_n_cpu"])
                if data.get("jackhmmer_n_cpu") is not None
                else None
            ),
            jackhmmer_max_parallel_shards=(
                int(data["jackhmmer_max_parallel_shards"])
                if data.get("jackhmmer_max_parallel_shards") is not None
                else None
            ),
            nhmmer_n_cpu=(
                int(data["nhmmer_n_cpu"]) if data.get("nhmmer_n_cpu") is not None else None
            ),
            nhmmer_max_parallel_shards=(
                int(data["nhmmer_max_parallel_shards"])
                if data.get("nhmmer_max_parallel_shards") is not None
                else None
            ),
            resolve_msa_overlaps=(
                _to_bool(data["resolve_msa_overlaps"], False)
                if data.get("resolve_msa_overlaps") is not None
                else None
            ),
            max_template_date=_optional_string(
                data.get("max_template_date"), "alphafold3.max_template_date"
            ),
            conformer_max_iterations=(
                int(data["conformer_max_iterations"])
                if data.get("conformer_max_iterations") is not None
                else None
            ),
            jax_compilation_cache_dir=_optional_string(
                data.get("jax_compilation_cache_dir"),
                "alphafold3.jax_compilation_cache_dir",
            ),
            gpu_device=int(data["gpu_device"]) if data.get("gpu_device") is not None else None,
            buckets=[int(item) for item in data.get("buckets", [])],
            flash_attention_implementation=_optional_string(
                data.get("flash_attention_implementation"),
                "alphafold3.flash_attention_implementation",
            ),
            num_recycles=(
                int(data["num_recycles"]) if data.get("num_recycles") is not None else None
            ),
            num_diffusion_samples=(
                int(data["num_diffusion_samples"])
                if data.get("num_diffusion_samples") is not None
                else None
            ),
            num_seeds=int(data["num_seeds"]) if data.get("num_seeds") is not None else None,
            save_embeddings=_to_bool(data.get("save_embeddings"), False),
            save_distogram=_to_bool(data.get("save_distogram"), False),
            force_output_dir=_to_bool(data.get("force_output_dir"), False),
            compress_large_output_files=_to_bool(data.get("compress_large_output_files"), False),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class SlurmConfig:
    account: str | None = None
    partition: str | None = None
    gpus: int = 1
    cpus_per_task: int = 8
    mem: str = "64G"
    time: str = "04:00:00"
    extra_sbatch_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SlurmConfig":
        if data is None:
            return cls()
        return cls(
            account=_optional_string(data.get("account"), "slurm.account"),
            partition=_optional_string(data.get("partition"), "slurm.partition"),
            gpus=int(data.get("gpus", 1)),
            cpus_per_task=int(data.get("cpus_per_task", 8)),
            mem=str(data.get("mem", "64G")),
            time=str(data.get("time", "04:00:00")),
            extra_sbatch_args=[str(arg) for arg in data.get("extra_sbatch_args", [])],
        )


@dataclass(slots=True)
class VinaConfig:
    enabled: bool = False
    binary: str = ".venvs/protenix-dock/bin/vina"
    receptor_pdbqt: str | None = None
    ligand_pdbqt: str | None = None
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None
    cpu: int = 8
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: float = 4.0
    scoring: str | None = "vina"
    seed: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VinaConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), False),
            binary=str(data.get("binary", ".venvs/protenix-dock/bin/vina")),
            receptor_pdbqt=_optional_string(data.get("receptor_pdbqt"), "vina.receptor_pdbqt"),
            ligand_pdbqt=_optional_string(data.get("ligand_pdbqt"), "vina.ligand_pdbqt"),
            center_x=float(data["center_x"]) if data.get("center_x") is not None else None,
            center_y=float(data["center_y"]) if data.get("center_y") is not None else None,
            center_z=float(data["center_z"]) if data.get("center_z") is not None else None,
            size_x=float(data["size_x"]) if data.get("size_x") is not None else None,
            size_y=float(data["size_y"]) if data.get("size_y") is not None else None,
            size_z=float(data["size_z"]) if data.get("size_z") is not None else None,
            cpu=int(data.get("cpu", 8)),
            exhaustiveness=int(data.get("exhaustiveness", 16)),
            num_modes=int(data.get("num_modes", 20)),
            energy_range=float(data.get("energy_range", 4.0)),
            scoring=_optional_string(data.get("scoring"), "vina.scoring"),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class AutoDockGPUConfig:
    enabled: bool = False
    binary: str = ".local/bin/autodock_gpu_128wi"
    receptor_pdbqt: str | None = None
    ligand_pdbqt: str | None = None
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None
    nrun: int = 100
    nev: int = 2500000
    heuristics: int = 1
    autostop: bool = True
    seed: int | None = None
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AutoDockGPUConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), False),
            binary=str(data.get("binary", ".local/bin/autodock_gpu_128wi")),
            receptor_pdbqt=_optional_string(data.get("receptor_pdbqt"), "autodock_gpu.receptor_pdbqt"),
            ligand_pdbqt=_optional_string(data.get("ligand_pdbqt"), "autodock_gpu.ligand_pdbqt"),
            center_x=float(data["center_x"]) if data.get("center_x") is not None else None,
            center_y=float(data["center_y"]) if data.get("center_y") is not None else None,
            center_z=float(data["center_z"]) if data.get("center_z") is not None else None,
            size_x=float(data["size_x"]) if data.get("size_x") is not None else None,
            size_y=float(data["size_y"]) if data.get("size_y") is not None else None,
            size_z=float(data["size_z"]) if data.get("size_z") is not None else None,
            nrun=int(data.get("nrun", 100)),
            nev=int(data.get("nev", 2500000)),
            heuristics=int(data.get("heuristics", 1)),
            autostop=_to_bool(data.get("autostop"), True),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class ProtenixDockConfig:
    enabled: bool = False
    python_bin: str = ".venvs/protenix-dock/bin/python"
    receptor_pdb: str | None = None
    ligand_sdf: str | None = None
    center_x: float | None = None
    center_y: float | None = None
    center_z: float | None = None
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None
    cache_map_spacing: float = 0.375
    use_cache_maps: bool = True
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProtenixDockConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), False),
            python_bin=str(data.get("python_bin", ".venvs/protenix-dock/bin/python")),
            receptor_pdb=_optional_string(data.get("receptor_pdb"), "protenix_dock.receptor_pdb"),
            ligand_sdf=_optional_string(data.get("ligand_sdf"), "protenix_dock.ligand_sdf"),
            center_x=float(data["center_x"]) if data.get("center_x") is not None else None,
            center_y=float(data["center_y"]) if data.get("center_y") is not None else None,
            center_z=float(data["center_z"]) if data.get("center_z") is not None else None,
            size_x=float(data["size_x"]) if data.get("size_x") is not None else None,
            size_y=float(data["size_y"]) if data.get("size_y") is not None else None,
            size_z=float(data["size_z"]) if data.get("size_z") is not None else None,
            cache_map_spacing=float(data.get("cache_map_spacing", 0.375)),
            use_cache_maps=_to_bool(data.get("use_cache_maps"), True),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class TemplateSearchSequenceConfig:
    enabled: bool = False
    binary: str = "mmseqs"
    database_path: str | None = None
    min_seq_identity: float = 0.3
    min_coverage: float = 0.7
    sensitivity: float = 7.5
    max_hits: int = 200
    threads: int = 8
    extra_args: list[str] = field(default_factory=list)
    # Multi-track docking: template ligand filtering + template-guided docking
    rcsb_dir: str = "~/DB/RCSB/raw/mmCIF_data"
    rcsb_db_path: str = "~/DB/RCSB/processed/rcsb_index.db"
    mcs_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TemplateSearchSequenceConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), False),
            binary=str(data.get("binary", "mmseqs")),
            database_path=_optional_string(
                data.get("database_path"), "template_search_sequence.database_path"
            ),
            min_seq_identity=float(data.get("min_seq_identity", 0.3)),
            min_coverage=float(data.get("min_coverage", 0.7)),
            sensitivity=float(data.get("sensitivity", 7.5)),
            max_hits=int(data.get("max_hits", 200)),
            threads=int(data.get("threads", 8)),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
            rcsb_dir=str(data.get("rcsb_dir", "~/DB/RCSB/raw/mmCIF_data")),
            rcsb_db_path=str(data.get("rcsb_db_path", "~/DB/RCSB/processed/rcsb_index.db")),
            mcs_threshold=float(data.get("mcs_threshold", 0.5)),
        )


@dataclass(slots=True)
class TemplateSearchStructureConfig:
    enabled: bool = False
    binary: str = "foldseek"
    database_path: str | None = None
    query_structure_path: str | None = None
    query_from_cofolding: bool = False
    query_model_priority: list[str] = field(
        default_factory=lambda: ["alphafold3", "boltz", "protenix"]
    )
    sensitivity: float = 9.5
    max_hits: int = 200
    alignment_type: int = 1
    threads: int = 8
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TemplateSearchStructureConfig":
        if data is None:
            return cls()
        return cls(
            enabled=_to_bool(data.get("enabled"), False),
            binary=str(data.get("binary", "foldseek")),
            database_path=_optional_string(
                data.get("database_path"), "template_search_structure.database_path"
            ),
            query_structure_path=_optional_string(
                data.get("query_structure_path"),
                "template_search_structure.query_structure_path",
            ),
            query_from_cofolding=_to_bool(data.get("query_from_cofolding"), False),
            query_model_priority=[
                str(item) for item in data.get("query_model_priority", ["alphafold3", "boltz", "protenix"])
            ],
            sensitivity=float(data.get("sensitivity", 9.5)),
            max_hits=int(data.get("max_hits", 200)),
            alignment_type=int(data.get("alignment_type", 1)),
            threads=int(data.get("threads", 8)),
            extra_args=[str(arg) for arg in data.get("extra_args", [])],
        )


@dataclass(slots=True)
class RunnerConfig:
    preset: str = "balanced"
    boltz: BoltzConfig = field(default_factory=BoltzConfig)
    protenix: ProtenixConfig = field(default_factory=ProtenixConfig)
    alphafold3: AlphaFold3Config = field(default_factory=AlphaFold3Config)
    vina: VinaConfig = field(default_factory=VinaConfig)
    autodock_gpu: AutoDockGPUConfig = field(default_factory=AutoDockGPUConfig)
    protenix_dock: ProtenixDockConfig = field(default_factory=ProtenixDockConfig)
    template_search_sequence: TemplateSearchSequenceConfig = field(
        default_factory=TemplateSearchSequenceConfig
    )
    template_search_structure: TemplateSearchStructureConfig = field(
        default_factory=TemplateSearchStructureConfig
    )
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunnerConfig":
        preset = _normalize_preset_name(data.get("preset", "balanced"))
        return cls(
            preset=preset,
            boltz=BoltzConfig.from_dict(_merge_preset_section(preset, "boltz", data.get("boltz"))),
            protenix=ProtenixConfig.from_dict(
                _merge_preset_section(preset, "protenix", data.get("protenix"))
            ),
            alphafold3=AlphaFold3Config.from_dict(
                _merge_preset_section(preset, "alphafold3", data.get("alphafold3"))
            ),
            vina=VinaConfig.from_dict(_merge_preset_section(preset, "vina", data.get("vina"))),
            autodock_gpu=AutoDockGPUConfig.from_dict(
                _merge_preset_section(preset, "autodock_gpu", data.get("autodock_gpu"))
            ),
            protenix_dock=ProtenixDockConfig.from_dict(
                _merge_preset_section(preset, "protenix_dock", data.get("protenix_dock"))
            ),
            template_search_sequence=TemplateSearchSequenceConfig.from_dict(
                data.get("template_search_sequence")
            ),
            template_search_structure=TemplateSearchStructureConfig.from_dict(
                data.get("template_search_structure")
            ),
            slurm=SlurmConfig.from_dict(data.get("slurm")),
        )

    def validate(self, pipeline: str = "all") -> None:
        if pipeline not in {
            "all",
            "structure",
            "vina",
            "protenix_dock",
            "docking",
            "template_search_sequence",
            "template_search_structure",
        }:
            raise ValueError(
                "pipeline must be one of: all, structure, vina, template_search_sequence, template_search_structure."
            )
        if pipeline in {"all", "structure"} and self.alphafold3.enabled and self.alphafold3.run_inference and not self.alphafold3.model_dir:
            raise ValueError(
                "alphafold3.model_dir must be set when AlphaFold3 inference is enabled."
            )
        if pipeline in {"all", "structure"} and self.alphafold3.enabled and self.alphafold3.run_data_pipeline and not self.alphafold3.db_dirs:
            raise ValueError(
                "alphafold3.db_dirs must be set when AlphaFold3 data pipeline is enabled."
            )
        if pipeline in {"all", "vina", "docking"} and self.vina.enabled:
            # In wrapper mode, docking prep bridge auto-generates receptor/ligand files
            if pipeline != "docking" and not self.vina.receptor_pdbqt:
                raise ValueError("vina.receptor_pdbqt must be set when AutoDock Vina is enabled.")
            if pipeline != "docking" and not self.vina.ligand_pdbqt:
                raise ValueError("vina.ligand_pdbqt must be set when AutoDock Vina is enabled.")
            # In docking pipeline mode, bridge auto-generates box params
            if pipeline != "docking":
                if not _all_or_none(
                    [
                        self.vina.center_x,
                        self.vina.center_y,
                        self.vina.center_z,
                        self.vina.size_x,
                        self.vina.size_y,
                        self.vina.size_z,
                    ]
                ):
                    raise ValueError(
                        "vina center_* and size_* must either all be set or all be omitted."
                    )
                if self.vina.center_x is None:
                    raise ValueError(
                        "vina center_* and size_* must be set when AutoDock Vina is enabled."
                    )
        if pipeline in {"all", "protenix_dock", "docking"} and self.protenix_dock.enabled:
            if pipeline != "docking" and not self.protenix_dock.receptor_pdb:
                raise ValueError(
                    "protenix_dock.receptor_pdb must be set when Protenix-Dock is enabled."
                )
            if pipeline != "docking" and not self.protenix_dock.ligand_sdf:
                raise ValueError(
                    "protenix_dock.ligand_sdf must be set when Protenix-Dock is enabled."
                )
            if pipeline != "docking":
                if not _all_or_none(
                    [
                        self.protenix_dock.center_x,
                        self.protenix_dock.center_y,
                        self.protenix_dock.center_z,
                        self.protenix_dock.size_x,
                        self.protenix_dock.size_y,
                        self.protenix_dock.size_z,
                    ]
                ):
                    raise ValueError(
                        "protenix_dock center_* and size_* must either all be set or all be omitted."
                    )
                if self.protenix_dock.center_x is None:
                    raise ValueError(
                        "protenix_dock center_* and size_* must be set when Protenix-Dock is enabled."
                    )
        if pipeline in {"all", "template_search_sequence"} and self.template_search_sequence.enabled:
            if not self.template_search_sequence.database_path:
                raise ValueError(
                    "template_search_sequence.database_path must be set when sequence-based template search is enabled."
                )
        if pipeline in {"all", "template_search_structure"} and self.template_search_structure.enabled:
            if not self.template_search_structure.database_path:
                raise ValueError(
                    "template_search_structure.database_path must be set when structure-based template search is enabled."
                )
            if not self.template_search_structure.query_from_cofolding and not self.template_search_structure.query_structure_path:
                raise ValueError(
                    "template_search_structure.query_structure_path must be set when structure-based template search is enabled unless query_from_cofolding is true."
                )
            invalid_models = [
                model_name
                for model_name in self.template_search_structure.query_model_priority
                if model_name not in {"alphafold3", "boltz", "protenix"}
            ]
            if invalid_models:
                raise ValueError(
                    "template_search_structure.query_model_priority contains unsupported model names: "
                    + ", ".join(invalid_models)
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
