"""Common input parsing and normalization for CASP17 pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from casp17_pl_hub.configs import RunnerConfig, _require_non_empty_string, _optional_string
from casp17_pl_hub.io_utils import load_structured_file


def _sanitize_job_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return sanitized or "casp17-run"


@dataclass(slots=True)
class CommonInput:
    name: str
    spec: dict[str, Any]
    seed: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_name: str = "casp17-run") -> "CommonInput":
        if "sequences" in data:
            spec = _normalize_boltz_yaml_spec(data)
        else:
            spec = _convert_legacy_input_to_boltz_spec(data)
        name = _sanitize_job_name(str(data.get("name") or default_name))
        seed = int(data.get("seed", 1))
        return cls(name=name, spec=spec, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seed": self.seed, **self.spec}

    @property
    def sequences(self) -> list[dict[str, Any]]:
        return list(self.spec.get("sequences", []))

    @property
    def constraints(self) -> list[dict[str, Any]]:
        return list(self.spec.get("constraints", []))

    @property
    def templates(self) -> list[dict[str, Any]]:
        return list(self.spec.get("templates", []))

    @property
    def properties(self) -> list[dict[str, Any]]:
        return list(self.spec.get("properties", []))


def _normalize_boltz_yaml_spec(data: dict[str, Any]) -> dict[str, Any]:
    sequences = data.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Boltz-style input must contain a non-empty sequences list.")

    normalized_sequences = []
    for idx, entry in enumerate(sequences):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"sequences[{idx}] must be a mapping with exactly one entity type.")
        entity_type, entity = next(iter(entry.items()))
        if entity_type not in {"protein", "dna", "rna", "ligand"}:
            raise ValueError(
                f"Unsupported Boltz entity type {entity_type!r}. Supported: protein, dna, rna, ligand."
            )
        if not isinstance(entity, dict):
            raise ValueError(f"sequences[{idx}].{entity_type} must be a mapping.")
        if "id" not in entity:
            raise ValueError(f"sequences[{idx}].{entity_type}.id is required.")
        if entity_type == "protein" and "sequence" not in entity:
            raise ValueError(f"sequences[{idx}].protein.sequence is required.")
        normalized_sequences.append({entity_type: dict(entity)})

    all_ids: list[str] = []
    for entry in normalized_sequences:
        _, ent = next(iter(entry.items()))
        entity_id = ent.get("id")
        if isinstance(entity_id, list):
            all_ids.extend(str(i) for i in entity_id)
        elif entity_id is not None:
            all_ids.append(str(entity_id))
    seen: set[str] = set()
    for eid in all_ids:
        if eid in seen:
            raise ValueError(f"Duplicate entity id {eid!r} found in sequences.")
        seen.add(eid)

    spec: dict[str, Any] = {"version": int(data.get("version", 1)), "sequences": normalized_sequences}
    for key in ("constraints", "templates", "properties"):
        value = data.get(key)
        if value:
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a list when provided.")
            spec[key] = value
    return spec


def _convert_legacy_input_to_boltz_spec(data: dict[str, Any]) -> dict[str, Any]:
    protein = data.get("protein", {})
    ligand = data.get("ligand", {})
    msa_mode = str(data.get("msa_mode", "single_sequence"))
    protein_block: dict[str, Any] = {
        "id": _require_non_empty_string(protein.get("id", "A"), "protein.id"),
        "sequence": _require_non_empty_string(protein.get("sequence"), "protein.sequence"),
    }
    if msa_mode == "external":
        msa_path = _optional_string(protein.get("msa_path"), "protein.msa_path")
        if not msa_path:
            raise ValueError("protein.msa_path is required when msa_mode is external.")
        protein_block["msa"] = msa_path
    elif msa_mode == "single_sequence":
        protein_block["msa"] = "empty"

    spec: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {"protein": protein_block},
            {
                "ligand": {
                    "id": _require_non_empty_string(ligand.get("id", "L"), "ligand.id"),
                    "smiles": _require_non_empty_string(ligand.get("smiles"), "ligand.smiles"),
                }
            },
        ],
    }
    if bool(data.get("predict_affinity", False)):
        spec["properties"] = [{"affinity": {"binder": ligand.get("id", "L")}}]
    return spec


def load_common_input(path: Path) -> CommonInput:
    return CommonInput.from_dict(load_structured_file(path), default_name=path.stem)


def load_runner_config(path: Path, validate: bool = True, pipeline: str = "all") -> RunnerConfig:
    config = RunnerConfig.from_dict(load_structured_file(path))
    if validate:
        config.validate(pipeline=pipeline)
    return config
