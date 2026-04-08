#!/usr/bin/env python3
"""Bridge: convert Boltz MSA CSV files to A3M and patch AlphaFold3 input JSON.

Usage:
    python bridge_boltz_msa_to_af3.py \
        --boltz-output-dir runs/target/outputs/boltz \
        --af3-input-json runs/target/inputs/alphafold3_input.json

After Boltz runs with --use_msa_server, MSA files are saved as CSV in:
    boltz_results_*/msa/<chain_name>.csv

This script:
1. Finds all CSV MSA files in the Boltz output.
2. Converts each CSV to A3M format.
3. Patches the AF3 input JSON to set unpairedMsaPath for matching protein chains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def boltz_csv_to_a3m(csv_path: Path) -> str:
    """Convert a Boltz MSA CSV (key,sequence) to A3M format."""
    lines = csv_path.read_text().strip().splitlines()
    if not lines or not lines[0].startswith("key"):
        return ""
    a3m_lines: list[str] = []
    for idx, line in enumerate(lines[1:]):
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        key, seq = parts
        header = f">query" if idx == 0 else f">seq_{idx}_key_{key}"
        a3m_lines.append(header)
        a3m_lines.append(seq)
    return "\n".join(a3m_lines) + "\n"


def find_boltz_msa_csvs(boltz_output_dir: Path) -> dict[str, Path]:
    """Find MSA CSV files in Boltz output. Returns {chain_name: csv_path}."""
    results: dict[str, Path] = {}
    for results_dir in boltz_output_dir.glob("boltz_results_*/msa"):
        for csv_file in results_dir.glob("*.csv"):
            results[csv_file.stem] = csv_file
    return results


def patch_af3_input(af3_json_path: Path, a3m_paths: dict[str, Path]) -> None:
    """Patch AF3 input JSON with unpairedMsaPath for matching protein chains."""
    data = json.loads(af3_json_path.read_text())

    patched = 0
    for seq_entry in data.get("sequences", []):
        if "protein" not in seq_entry:
            continue
        protein = seq_entry["protein"]
        chain_id = protein.get("id")
        chain_ids = [chain_id] if isinstance(chain_id, str) else (chain_id or [])

        for cid in chain_ids:
            if cid in a3m_paths:
                protein["unpairedMsaPath"] = str(a3m_paths[cid])
                protein.pop("unpairedMsa", None)
                protein.pop("pairedMsa", None)
                patched += 1
                print(f"  Patched chain {cid} with MSA from {a3m_paths[cid]}")
                break

    af3_json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  Total {patched} chain(s) patched in {af3_json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Boltz MSA to AlphaFold3 input.")
    parser.add_argument("--boltz-output-dir", type=Path, required=True)
    parser.add_argument("--af3-input-json", type=Path, required=True)
    args = parser.parse_args()

    csv_files = find_boltz_msa_csvs(args.boltz_output_dir)
    if not csv_files:
        print("  No Boltz MSA CSV files found, skipping bridge.")
        return 0

    a3m_dir = args.af3_input_json.parent / "boltz_msa_a3m"
    a3m_dir.mkdir(parents=True, exist_ok=True)

    a3m_paths: dict[str, Path] = {}
    for chain_name, csv_path in csv_files.items():
        a3m_content = boltz_csv_to_a3m(csv_path)
        if not a3m_content:
            print(f"  Warning: empty MSA for chain {chain_name}, skipping.")
            continue
        a3m_path = a3m_dir / f"{chain_name}.a3m"
        a3m_path.write_text(a3m_content)
        a3m_paths[chain_name] = a3m_path
        print(f"  Converted {csv_path.name} -> {a3m_path}")

    if not a3m_paths:
        print("  No valid MSA conversions, skipping AF3 input patch.")
        return 0

    patch_af3_input(args.af3_input_json, a3m_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
