from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _get_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _check_status() -> dict[str, Any]:
    root = _get_root()
    venvs: dict[str, str] = {}
    for name, rel in [
        ("boltz", ".venvs/boltz/bin/boltz"),
        ("protenix", ".venvs/protenix/bin/protenix"),
        ("alphafold3", ".venvs/alphafold3/bin/python"),
    ]:
        venvs[name] = "ok" if (root / rel).exists() else "missing"

    slurm: dict[str, Any] = {"available": bool(shutil.which("sinfo"))}
    if slurm["available"]:
        try:
            result = subprocess.run(
                ["sinfo", "--noheader", "-o", "%P %a %l %D %G"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                slurm["partitions"] = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        except (subprocess.TimeoutExpired, OSError):
            pass

    tools = {t: shutil.which(t) or "" for t in ("mmseqs", "foldseek", "vina")}

    return {
        "root": str(root),
        "python": sys.version.split()[0],
        "venvs": venvs,
        "slurm": slurm,
        "tools": tools,
        "stages": ["template-search-sequence", "template-search-structure", "cofolding", "docking"],
    }


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit(
            "The optional MCP dependency is not installed. Run `uv sync --dev --extra mcp` first."
        ) from exc

    from casp17_pl_hub.configs import PRESET_NAMES, PRESET_OVERRIDES
    from casp17_pl_hub.models import load_common_input, load_runner_config
    from casp17_pl_hub.orchestrator import prepare_run
    from casp17_pl_hub.validation import validate_run

    server = FastMCP("casp17-protein-ligand")

    @server.tool()
    def status() -> dict[str, Any]:
        """Check workspace status: venvs, SLURM, tools availability."""
        return _check_status()

    @server.tool()
    def validate(
        input_path: str,
        config_path: str,
        backend: str = "slurm",
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate a pipeline configuration before submitting."""
        try:
            common = load_common_input(Path(input_path))
            config = load_runner_config(Path(config_path), validate=False)
            report = validate_run(common, config, backend, stages=stages)
            return {
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "infos": report.infos,
            }
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "errors": [str(exc)], "warnings": [], "infos": []}

    @server.tool()
    def prepare_cofolding(
        input_path: str,
        config_path: str,
        output_root: str = "experiments/runs",
        backend: str = "slurm",
    ) -> dict[str, Any]:
        """Prepare a cofolding run: generate inputs, scripts, and manifest."""
        try:
            common = load_common_input(Path(input_path))
            config = load_runner_config(Path(config_path))
            prepared = prepare_run(common, config, Path(output_root), backend)
            return {
                "run_dir": str(prepared.run_dir),
                "manifest": str(prepared.manifest_path),
                "script": str(prepared.shell_script),
                "models": [m.model_name for m in prepared.model_runs],
            }
        except (ValueError, FileNotFoundError) as exc:
            return {"error": str(exc)}

    @server.tool()
    def list_presets() -> dict[str, Any]:
        """List available hyperparameter presets and their overrides."""
        return {"presets": list(PRESET_NAMES), "overrides": PRESET_OVERRIDES}

    @server.tool()
    def add_template(
        input_path: str,
        template_file: str,
        chain_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Add a structural template (PDB/CIF) to a unified input YAML.

        Args:
            input_path: Path to the unified input YAML file.
            template_file: Path to template PDB or CIF file.
            chain_ids: Optional list of chain IDs to apply the template to.
                       If omitted, applies to all protein chains.
        """
        try:
            input_p = Path(input_path)
            template_p = Path(template_file)
            if not input_p.exists():
                return {"error": f"Input file not found: {input_path}"}
            if not template_p.exists():
                return {"error": f"Template file not found: {template_file}"}
            if template_p.suffix not in (".pdb", ".cif", ".mmcif"):
                return {"error": f"Unsupported template format: {template_p.suffix}. Use .pdb or .cif"}

            from casp17_pl_hub.io_utils import load_structured_file, dump_structured_file

            data = load_structured_file(input_p)
            templates = data.get("templates", [])
            entry: dict[str, Any] = {"path": str(template_p.resolve())}
            if chain_ids:
                entry["ids"] = chain_ids
            templates.append(entry)
            data["templates"] = templates
            dump_structured_file(data, input_p)

            return {
                "ok": True,
                "template_file": str(template_p.resolve()),
                "chain_ids": chain_ids or "all",
                "total_templates": len(templates),
            }
        except (ValueError, FileNotFoundError) as exc:
            return {"error": str(exc)}

    @server.tool()
    def create_input(
        sequence: str,
        name: str = "casp17-run",
        chain_id: str = "A",
        ligand_smiles: str | None = None,
        ligand_id: str = "L",
        template_file: str | None = None,
        seed: int = 42,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Create a unified input YAML from natural language parameters.

        Args:
            sequence: Protein amino acid sequence (one-letter codes).
            name: Run name.
            chain_id: Protein chain ID.
            ligand_smiles: Optional SMILES string for a ligand.
            ligand_id: Ligand chain ID.
            template_file: Optional path to a template PDB/CIF file.
            seed: Random seed.
            output_path: Where to write the YAML. Defaults to examples/<name>.yaml.
        """
        try:
            from casp17_pl_hub.io_utils import dump_structured_file

            sequences: list[dict[str, Any]] = [
                {"protein": {"id": chain_id, "sequence": sequence, "msa": "empty"}}
            ]
            if ligand_smiles:
                sequences.append({"ligand": {"id": ligand_id, "smiles": ligand_smiles}})

            data: dict[str, Any] = {
                "version": 1,
                "seed": seed,
                "sequences": sequences,
            }
            if template_file:
                template_p = Path(template_file)
                if not template_p.exists():
                    return {"error": f"Template file not found: {template_file}"}
                data["templates"] = [{"path": str(template_p.resolve()), "ids": [chain_id]}]

            out = Path(output_path) if output_path else Path(f"examples/{name}.yaml")
            out.parent.mkdir(parents=True, exist_ok=True)
            dump_structured_file(data, out)

            return {"ok": True, "path": str(out), "sequences": len(sequences), "has_template": bool(template_file)}
        except (ValueError, FileNotFoundError) as exc:
            return {"error": str(exc)}

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
