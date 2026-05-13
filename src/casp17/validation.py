"""Validation logic for CASP17 pipeline runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from casp17.configs import RunnerConfig
from casp17.models import CommonInput


@dataclass(slots=True)
class ValidationReport:
    errors: list[str]
    warnings: list[str]
    infos: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


STAGE_NAMES = ("template-search-sequence", "template-search-structure", "cofolding", "docking")


def validate_run(
    common: CommonInput,
    config: RunnerConfig,
    backend: str,
    repo_root: Path | None = None,
    stages: list[str] | None = None,
) -> ValidationReport:
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    resolved_stages = _resolve_stages(config, stages, allow_empty=True)
    errors: list[str] = []
    warnings: list[str] = []
    infos = [
        f"preset={config.preset}",
        f"backend={backend}",
        f"stages={','.join(resolved_stages) if resolved_stages else 'none'}",
        "This repository assumes a SLURM cluster where the master/login node has no GPU and inference should run on compute nodes.",
    ]

    if not resolved_stages:
        errors.append("No stage is enabled or requested. Choose at least one stage.")

    protein_count = sum(1 for entry in common.sequences if next(iter(entry)) == "protein")
    ligand_count = sum(1 for entry in common.sequences if next(iter(entry)) == "ligand")
    if protein_count == 0:
        warnings.append("The common input has no protein entity.")
    if ligand_count == 0:
        warnings.append("The common input has no ligand entity.")

    known_ids: set[str] = set()
    for entry in common.sequences:
        entity = next(iter(entry.values()))
        entity_id = entity.get("id")
        if isinstance(entity_id, list):
            known_ids.update(str(i) for i in entity_id)
        elif entity_id is not None:
            known_ids.add(str(entity_id))
    for constraint in common.constraints:
        if "bond" not in constraint:
            continue
        bond = constraint["bond"]
        for atom_key in ("atom1", "atom2"):
            atom = bond.get(atom_key)
            if isinstance(atom, list) and len(atom) >= 1:
                chain_id = str(atom[0])
                if chain_id not in known_ids:
                    errors.append(
                        f"Bond constraint references unknown chain id {chain_id!r} in {atom_key}."
                    )

    if backend == "slurm" and not config.slurm.partition:
        warnings.append("slurm.partition is empty; the job will use the scheduler default partition.")
    if backend == "local":
        warnings.append("local backend should only be used on an allocated compute node if GPU inference is required.")
    if "template-search-sequence" in resolved_stages and protein_count == 0:
        errors.append("Sequence-based template search requires at least one protein entity.")
    if "template-search-structure" in resolved_stages and protein_count == 0:
        errors.append("Structure-based template search requires at least one protein entity.")
    if "cofolding" in resolved_stages and config.slurm.gpus < 1 and any(
        _model_is_enabled(config, model_name) for model_name in ("boltz", "protenix", "alphafold3")
    ):
        warnings.append("slurm.gpus is set to 0 even though GPU structure models are enabled.")

    if "template-search-sequence" in resolved_stages:
        _validate_command_target(
            errors,
            repo_root,
            config.template_search_sequence.binary,
            "template_search_sequence.binary",
            enabled=True,
        )
        _validate_existing_path(
            errors,
            repo_root,
            config.template_search_sequence.database_path,
            "template_search_sequence.database_path",
        )
        try:
            config.validate(pipeline="template_search_sequence")
        except ValueError as exc:
            errors.append(str(exc))

    if "template-search-structure" in resolved_stages:
        _validate_command_target(
            errors,
            repo_root,
            config.template_search_structure.binary,
            "template_search_structure.binary",
            enabled=True,
        )
        _validate_existing_path(
            errors,
            repo_root,
            config.template_search_structure.database_path,
            "template_search_structure.database_path",
        )
        if not config.template_search_structure.query_from_cofolding:
            _validate_existing_path(
                errors,
                repo_root,
                config.template_search_structure.query_structure_path,
                "template_search_structure.query_structure_path",
            )
        try:
            config.validate(pipeline="template_search_structure")
        except ValueError as exc:
            errors.append(str(exc))
        if config.template_search_structure.query_from_cofolding:
            if "cofolding" not in resolved_stages:
                warnings.append(
                    "Structure-based template search is configured to reuse cofolding outputs, but cofolding is not in the selected stage list; existing outputs under runs/<target>/outputs/* will be expected."
                )
            elif resolved_stages.index("template-search-structure") < resolved_stages.index("cofolding"):
                errors.append(
                    "template-search-structure is configured with query_from_cofolding=true and must run after the cofolding stage in the selected stage order."
                )
        else:
            warnings.append(
                "Structure-based template search currently uses an explicit query structure path unless query_from_cofolding is enabled."
            )

    if "cofolding" in resolved_stages:
        if not any(_model_is_enabled(config, m) for m in ("boltz", "protenix", "alphafold3")):
            errors.append("cofolding stage is selected but no structure model (boltz, protenix, alphafold3) is enabled.")
        _validate_command_target(errors, repo_root, config.boltz.binary, "boltz.binary", enabled=config.boltz.enabled)
        _validate_command_target(errors, repo_root, config.protenix.binary, "protenix.binary", enabled=config.protenix.enabled)
        _validate_command_target(errors, repo_root, config.alphafold3.python_bin, "alphafold3.python_bin", enabled=config.alphafold3.enabled)
        _validate_file_path(errors, repo_root, config.alphafold3.script, "alphafold3.script", enabled=config.alphafold3.enabled)
        if config.alphafold3.enabled and config.alphafold3.run_inference:
            _validate_dir_path(errors, repo_root, config.alphafold3.model_dir, "alphafold3.model_dir")
        if config.alphafold3.enabled and config.alphafold3.run_data_pipeline:
            if not config.alphafold3.db_dirs:
                errors.append("alphafold3.db_dirs must be set when AlphaFold3 data pipeline is enabled.")
            for index, db_dir in enumerate(config.alphafold3.db_dirs, start=1):
                _validate_dir_path(errors, repo_root, db_dir, f"alphafold3.db_dirs[{index}]")
        try:
            config.validate(pipeline="structure")
        except ValueError as exc:
            errors.append(str(exc))

    if "docking" in resolved_stages:
        if config.vina.enabled:
            _validate_command_target(errors, repo_root, config.vina.binary, "vina.binary", enabled=True)
            _validate_file_path(errors, repo_root, config.vina.receptor_pdbqt, "vina.receptor_pdbqt", enabled=True)
            _validate_file_path(errors, repo_root, config.vina.ligand_pdbqt, "vina.ligand_pdbqt", enabled=True)
            if ligand_count == 0:
                errors.append("AutoDock Vina was enabled but the Boltz YAML input has no ligand entity.")
            elif ligand_count > 1:
                warnings.append(
                    "AutoDock Vina uses a single ligand_pdbqt path; multiple ligand entities in the Boltz YAML input are not automatically disambiguated."
                )
            try:
                config.validate(pipeline="vina")
            except ValueError as exc:
                errors.append(str(exc))
            warnings.append(
                "AutoDock Vina currently expects prebuilt receptor_pdbqt and ligand_pdbqt files; PDBQT generation is not automated from the Boltz YAML input yet."
            )
        if config.protenix_dock.enabled:
            _validate_command_target(
                errors, repo_root, config.protenix_dock.python_bin, "protenix_dock.python_bin", enabled=True
            )
            _validate_file_path(
                errors, repo_root, config.protenix_dock.receptor_pdb, "protenix_dock.receptor_pdb", enabled=True
            )
            _validate_file_path(
                errors, repo_root, config.protenix_dock.ligand_sdf, "protenix_dock.ligand_sdf", enabled=True
            )
            if ligand_count == 0:
                errors.append("Protenix-Dock was enabled but the Boltz YAML input has no ligand entity.")
            elif ligand_count > 1:
                warnings.append(
                    "Protenix-Dock uses a single ligand_sdf path; multiple ligand entities in the Boltz YAML input are not automatically disambiguated."
                )
            try:
                config.validate(pipeline="protenix_dock")
            except ValueError as exc:
                errors.append(str(exc))
            warnings.append(
                "Protenix-Dock currently expects prebuilt receptor_pdb and ligand_sdf files; file generation is not automated from the Boltz YAML input yet."
            )
        if config.autodock_gpu.enabled:
            _validate_command_target(
                errors, repo_root, config.autodock_gpu.binary, "autodock_gpu.binary", enabled=True
            )
        if (
            not config.vina.enabled
            and not config.protenix_dock.enabled
            and not config.autodock_gpu.enabled
            and not config.surfdock.enabled
        ):
            errors.append(
                "Docking stage is selected but no docking tool "
                "(vina, autodock_gpu, protenix_dock, surfdock) is enabled."
            )

    return ValidationReport(errors=errors, warnings=warnings, infos=infos)


def validation_report_as_text(report: ValidationReport) -> str:
    lines = [f"ok={'true' if report.ok else 'false'}"]
    for info in report.infos:
        lines.append(f"info: {info}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    for error in report.errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def _model_is_enabled(config: RunnerConfig, model_name: str) -> bool:
    return bool(getattr(config, model_name).enabled)


def _resolve_stages(
    config: RunnerConfig,
    stages: list[str] | None,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if stages:
        resolved = []
        for stage in stages:
            normalized = stage.strip().lower()
            if normalized not in STAGE_NAMES:
                raise ValueError(f"stage must be one of: {', '.join(STAGE_NAMES)}.")
            if normalized not in resolved:
                resolved.append(normalized)
        return resolved

    resolved: list[str] = []
    if config.template_search_sequence.enabled:
        resolved.append("template-search-sequence")
    if any(_model_is_enabled(config, model_name) for model_name in ("boltz", "protenix", "alphafold3")):
        resolved.append("cofolding")
    if config.template_search_structure.enabled:
        resolved.append("template-search-structure")
    if (
        config.vina.enabled
        or config.autodock_gpu.enabled
        or config.protenix_dock.enabled
        or config.surfdock.enabled
    ):
        resolved.append("docking")
    if not resolved and not allow_empty:
        raise ValueError("No stage is enabled or requested.")
    return resolved


def _validate_stage_dependencies(config: RunnerConfig, resolved_stages: list[str]) -> None:
    if (
        config.template_search_structure.enabled
        and config.template_search_structure.query_from_cofolding
        and "template-search-structure" in resolved_stages
        and "cofolding" in resolved_stages
        and resolved_stages.index("template-search-structure") < resolved_stages.index("cofolding")
    ):
        raise ValueError(
            "template-search-structure with query_from_cofolding=true must come after cofolding in the wrapper stage order."
        )


def _validate_command_target(
    errors: list[str],
    repo_root: Path,
    value: str,
    field_name: str,
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    if not value:
        errors.append(f"{field_name} must be set.")
        return
    if _looks_like_path(value):
        target = _resolve_repo_path(repo_root, value)
        if _is_placeholder_path(value):
            errors.append(f"{field_name} still uses a placeholder path: {value}")
        elif not target.exists():
            errors.append(f"{field_name} does not exist: {target}")
        return
    if shutil.which(value) is None:
        errors.append(f"{field_name} is not on PATH: {value}")


def _validate_file_path(
    errors: list[str],
    repo_root: Path,
    value: str | None,
    field_name: str,
    *,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    if not value:
        errors.append(f"{field_name} must be set.")
        return
    target = _resolve_repo_path(repo_root, value)
    if _is_placeholder_path(value):
        errors.append(f"{field_name} still uses a placeholder path: {value}")
    elif not target.is_file():
        errors.append(f"{field_name} does not exist: {target}")


def _validate_dir_path(errors: list[str], repo_root: Path, value: str | None, field_name: str) -> None:
    if not value:
        errors.append(f"{field_name} must be set.")
        return
    target = _resolve_repo_path(repo_root, value)
    if _is_placeholder_path(value):
        errors.append(f"{field_name} still uses a placeholder path: {value}")
    elif not target.is_dir():
        errors.append(f"{field_name} does not exist: {target}")


def _validate_existing_path(
    errors: list[str], repo_root: Path, value: str | None, field_name: str
) -> None:
    if not value:
        errors.append(f"{field_name} must be set.")
        return
    target = _resolve_repo_path(repo_root, value)
    if _is_placeholder_path(value):
        errors.append(f"{field_name} still uses a placeholder path: {value}")
    elif not target.exists():
        errors.append(f"{field_name} does not exist: {target}")


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path)


def _looks_like_path(value: str) -> bool:
    return "/" in value or value.startswith(".")


def _is_placeholder_path(value: str) -> bool:
    return value.startswith("/path/to/") or value.startswith("CHANGE_ME")
