from __future__ import annotations

from typing import Any


def dump_yaml(data: Any) -> str:
    return _dump_value(data, indent=0).rstrip() + "\n"


def _dump_value(value: Any, indent: int) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _dump_value(item, indent + 2).splitlines()
                lines.append(f"{prefix}- {rendered[0].lstrip()}")
                lines.extend(f"{prefix}  {line.lstrip()}" for line in rendered[1:])
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_format_scalar(value)}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "'", '"', "\\"]) or text.strip() != text:
        escaped = text.replace("\\", "\\\\").replace("'", "''")
        return f"'{escaped}'"
    return text
