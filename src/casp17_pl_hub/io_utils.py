"""File I/O utilities for loading and saving structured data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def dump_json(data: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def dump_yaml_file(data: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def dump_structured_file(data: dict[str, Any] | list[Any], path: Path) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        dump_yaml_file(data, path)
        return
    dump_json(data, path)


def load_structured_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text())
    else:
        loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping/object at the top level.")
    return loaded
