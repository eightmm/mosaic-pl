from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from casp17.configs import PRESET_NAMES
from casp17.models import load_common_input, load_runner_config
from casp17.orchestrator import (
    execute_prepared_run,
    prepare_protenix_dock_run,
    prepare_run,
    prepare_stage_wrapper_run,
    prepare_template_search_sequence_run,
    prepare_template_search_structure_run,
    prepare_vina_run,
    write_example_config,
    write_example_input,
)
from casp17.validation import STAGE_NAMES, validate_run, validation_report_as_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="casp17-pl",
        description="CASP17 protein-ligand workspace CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show workspace status.")
    status_parser.set_defaults(handler=cmd_status)

    example_input_parser = subparsers.add_parser(
        "write-example-input",
        help="Write an example Boltz-style YAML input for all model adapters.",
    )
    example_input_parser.add_argument(
        "--path",
        type=Path,
        default=Path("examples/unified_input.example.yaml"),
    )
    example_input_parser.set_defaults(handler=cmd_write_example_input)

    example_config_parser = subparsers.add_parser(
        "write-example-config",
        help="Write an example model runner config YAML.",
    )
    example_config_parser.add_argument(
        "--path",
        type=Path,
        default=Path("examples/runner_config.example.yaml"),
    )
    example_config_parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="balanced",
        help="Default hyperparameter preset used to populate the template.",
    )
    example_config_parser.set_defaults(handler=cmd_write_example_config)

    prepare_parser = subparsers.add_parser(
        "prepare-run",
        help="Generate the co-folding stage from a Boltz-style YAML input.",
    )
    prepare_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    prepare_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    prepare_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_parser.set_defaults(handler=cmd_prepare_run)

    prepare_cofolding_parser = subparsers.add_parser(
        "prepare-cofolding",
        help="Generate the co-folding stage from a Boltz-style YAML input.",
    )
    prepare_cofolding_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    prepare_cofolding_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    prepare_cofolding_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_cofolding_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_cofolding_parser.set_defaults(handler=cmd_prepare_run)

    prepare_template_search_sequence_parser = subparsers.add_parser(
        "prepare-template-search-sequence",
        help="Generate the sequence-based protein template-search stage.",
    )
    prepare_template_search_sequence_parser.add_argument(
        "--input", type=Path, required=True, help="Common input YAML."
    )
    prepare_template_search_sequence_parser.add_argument(
        "--config", type=Path, required=True, help="Runner config YAML."
    )
    prepare_template_search_sequence_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_template_search_sequence_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_template_search_sequence_parser.set_defaults(
        handler=cmd_prepare_template_search_sequence
    )

    prepare_template_search_structure_parser = subparsers.add_parser(
        "prepare-template-search-structure",
        help="Generate the structure-based protein template-search stage.",
    )
    prepare_template_search_structure_parser.add_argument(
        "--input", type=Path, required=True, help="Common input YAML."
    )
    prepare_template_search_structure_parser.add_argument(
        "--config", type=Path, required=True, help="Runner config YAML."
    )
    prepare_template_search_structure_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_template_search_structure_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_template_search_structure_parser.set_defaults(
        handler=cmd_prepare_template_search_structure
    )

    prepare_vina_parser = subparsers.add_parser(
        "prepare-vina",
        help="Generate the docking stage for AutoDock Vina from the shared YAML and runner config.",
    )
    prepare_vina_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    prepare_vina_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    prepare_vina_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_vina_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_vina_parser.set_defaults(handler=cmd_prepare_vina)

    prepare_protenix_dock_parser = subparsers.add_parser(
        "prepare-protenix-dock",
        help="Generate the docking stage for Protenix-Dock from the shared YAML and runner config.",
    )
    prepare_protenix_dock_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    prepare_protenix_dock_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    prepare_protenix_dock_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_protenix_dock_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_protenix_dock_parser.set_defaults(handler=cmd_prepare_protenix_dock)

    prepare_wrapper_parser = subparsers.add_parser(
        "prepare-wrapper",
        help="Generate a top-level wrapper that orchestrates the co-folding and optional docking stages.",
    )
    prepare_wrapper_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    prepare_wrapper_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    prepare_wrapper_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    prepare_wrapper_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend used to generate the shell script.",
    )
    prepare_wrapper_parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_NAMES,
        default=None,
        help="Optional stage subset to wrap. Default is automatic from enabled modules.",
    )
    prepare_wrapper_parser.set_defaults(handler=cmd_prepare_wrapper)

    validate_parser = subparsers.add_parser(
        "validate-run",
        help="Validate the staged pipeline config before submitting co-folding or docking.",
    )
    validate_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    validate_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    validate_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend to validate against.",
    )
    validate_parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_NAMES,
        default=None,
        help="Optional stage subset to validate. Default is automatic from enabled modules.",
    )
    validate_parser.set_defaults(handler=cmd_validate_run)

    run_parser = subparsers.add_parser(
        "run-all",
        help="Generate the co-folding stage and optionally execute it.",
    )
    run_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    run_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    run_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_parser.set_defaults(handler=cmd_run_all)

    run_template_search_sequence_parser = subparsers.add_parser(
        "run-template-search-sequence",
        help="Generate the sequence-based protein template-search stage and optionally execute it.",
    )
    run_template_search_sequence_parser.add_argument(
        "--input", type=Path, required=True, help="Common input YAML."
    )
    run_template_search_sequence_parser.add_argument(
        "--config", type=Path, required=True, help="Runner config YAML."
    )
    run_template_search_sequence_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_template_search_sequence_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_template_search_sequence_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_template_search_sequence_parser.set_defaults(handler=cmd_run_template_search_sequence)

    run_template_search_structure_parser = subparsers.add_parser(
        "run-template-search-structure",
        help="Generate the structure-based protein template-search stage and optionally execute it.",
    )
    run_template_search_structure_parser.add_argument(
        "--input", type=Path, required=True, help="Common input YAML."
    )
    run_template_search_structure_parser.add_argument(
        "--config", type=Path, required=True, help="Runner config YAML."
    )
    run_template_search_structure_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_template_search_structure_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_template_search_structure_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_template_search_structure_parser.set_defaults(
        handler=cmd_run_template_search_structure
    )

    run_vina_parser = subparsers.add_parser(
        "run-vina",
        help="Generate the docking stage and optionally execute it.",
    )
    run_vina_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    run_vina_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    run_vina_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_vina_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_vina_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_vina_parser.set_defaults(handler=cmd_run_vina)

    run_protenix_dock_parser = subparsers.add_parser(
        "run-protenix-dock",
        help="Generate the Protenix-Dock docking stage and optionally execute it.",
    )
    run_protenix_dock_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    run_protenix_dock_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    run_protenix_dock_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_protenix_dock_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_protenix_dock_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_protenix_dock_parser.set_defaults(handler=cmd_run_protenix_dock)

    run_wrapper_parser = subparsers.add_parser(
        "run-wrapper",
        help="Generate the top-level wrapper stage and optionally execute it.",
    )
    run_wrapper_parser.add_argument("--input", type=Path, required=True, help="Common input YAML.")
    run_wrapper_parser.add_argument("--config", type=Path, required=True, help="Runner config YAML.")
    run_wrapper_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory for generated run artifacts.",
    )
    run_wrapper_parser.add_argument(
        "--backend",
        choices=("local", "slurm"),
        default="slurm",
        help="Execution backend.",
    )
    run_wrapper_parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_NAMES,
        default=None,
        help="Optional stage subset to wrap. Default is automatic from enabled modules.",
    )
    run_wrapper_parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually execute the generated shell script. Without this flag, only files are prepared.",
    )
    run_wrapper_parser.set_defaults(handler=cmd_run_wrapper)
    return parser


def cmd_status(_: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    print("CASP17 protein-ligand hub")
    print(f"root={root}")
    print(f"python={sys.version.split()[0]}")

    venvs = {
        "boltz": ".venvs/boltz/bin/boltz",
        "protenix": ".venvs/protenix/bin/protenix",
        "alphafold3": ".venvs/alphafold3/bin/python",
        "protenix-dock": ".venvs/protenix-dock/bin/python",
    }
    for name, binary_rel in venvs.items():
        binary = root / binary_rel
        if binary.exists():
            print(f"{name}_venv=ok ({binary_rel})")
        else:
            print(f"{name}_venv=missing ({binary_rel})")

    if shutil.which("sinfo"):
        print("slurm=available")
        try:
            result = subprocess.run(
                ["sinfo", "--noheader", "-o", "%P %a %l %D %G"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f"  {line.strip()}")
        except (subprocess.TimeoutExpired, OSError):
            pass
    else:
        print("slurm=not_found")

    for tool in ("mmseqs", "foldseek", "vina"):
        loc = shutil.which(tool)
        print(f"{tool}={'found' if loc else 'not_found'}{f' ({loc})' if loc else ''}")

    print(f"stages={', '.join(STAGE_NAMES)}")
    return 0


def cmd_write_example_input(args: argparse.Namespace) -> int:
    write_example_input(args.path)
    print(f"wrote={args.path}")
    return 0


def cmd_write_example_config(args: argparse.Namespace) -> int:
    write_example_config(args.path, preset=args.preset)
    print(f"wrote={args.path}")
    return 0


def _print_prepared(prepared) -> None:
    print(f"run_dir={prepared.run_dir}")
    print(f"manifest={prepared.manifest_path}")
    print(f"script={prepared.shell_script}")
    for model_run in prepared.model_runs:
        print(f"model={model_run.model_name} input={model_run.input_path} output={model_run.output_dir}")
        for note in model_run.notes:
            print(f"  note: {note}")


def _maybe_submit(prepared, args: argparse.Namespace) -> int:
    _print_prepared(prepared)
    if not getattr(args, "submit", False):
        return 0
    result = execute_prepared_run(prepared, args.backend)
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)
    print("submitted=true")
    return 0


def cmd_prepare_run(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="structure")
    _print_prepared(prepare_run(common, config, args.output_root, args.backend))
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="structure")
    return _maybe_submit(prepare_run(common, config, args.output_root, args.backend), args)


def cmd_prepare_template_search_sequence(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="template_search_sequence")
    _print_prepared(prepare_template_search_sequence_run(common, config, args.output_root, args.backend))
    return 0


def cmd_run_template_search_sequence(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="template_search_sequence")
    return _maybe_submit(prepare_template_search_sequence_run(common, config, args.output_root, args.backend), args)


def cmd_prepare_template_search_structure(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="template_search_structure")
    _print_prepared(prepare_template_search_structure_run(common, config, args.output_root, args.backend))
    return 0


def cmd_run_template_search_structure(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="template_search_structure")
    return _maybe_submit(prepare_template_search_structure_run(common, config, args.output_root, args.backend), args)


def cmd_prepare_vina(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="vina")
    _print_prepared(prepare_vina_run(common, config, args.output_root, args.backend))
    return 0


def cmd_run_vina(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="vina")
    return _maybe_submit(prepare_vina_run(common, config, args.output_root, args.backend), args)


def cmd_prepare_protenix_dock(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="protenix_dock")
    _print_prepared(prepare_protenix_dock_run(common, config, args.output_root, args.backend))
    return 0


def cmd_run_protenix_dock(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, pipeline="protenix_dock")
    return _maybe_submit(prepare_protenix_dock_run(common, config, args.output_root, args.backend), args)


def cmd_prepare_wrapper(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, validate=False)
    _print_prepared(prepare_stage_wrapper_run(common, config, args.output_root, args.backend, args.stages))
    return 0


def cmd_run_wrapper(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, validate=False)
    return _maybe_submit(prepare_stage_wrapper_run(common, config, args.output_root, args.backend, args.stages), args)


def cmd_validate_run(args: argparse.Namespace) -> int:
    common = load_common_input(args.input)
    config = load_runner_config(args.config, validate=False)
    report = validate_run(common, config, args.backend, stages=args.stages)
    print(validation_report_as_text(report))
    return 0 if report.ok else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
