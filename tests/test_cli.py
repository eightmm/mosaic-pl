import json
from pathlib import Path

import yaml

from casp17_pl_hub.cli import build_parser, cmd_status
from casp17_pl_hub.configs import RunnerConfig
from casp17_pl_hub.models import CommonInput
from casp17_pl_hub.orchestrator import (
    prepare_docking_run,
    prepare_protenix_dock_run,
    prepare_run,
    prepare_stage_wrapper_run,
    prepare_template_search_sequence_run,
    prepare_template_search_structure_run,
    prepare_vina_run,
    write_example_config,
)
from casp17_pl_hub.validation import validate_run


def test_parser_has_status_command() -> None:
    args = build_parser().parse_args(["status"])
    assert args.command == "status"


def test_status_command_returns_zero(capsys) -> None:
    exit_code = cmd_status(None)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CASP17 protein-ligand hub" in captured.out


def test_parser_has_prepare_run_command() -> None:
    args = build_parser().parse_args(
        ["prepare-run", "--input", "input.yaml", "--config", "config.yaml"]
    )
    assert args.command == "prepare-run"
    assert args.backend == "slurm"


def test_prepare_run_generates_model_inputs(tmp_path: Path) -> None:
    common = CommonInput.from_dict(
        {
            "version": 1,
            "seed": 7,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
            "constraints": [
                {"bond": {"atom1": ["A", 1, "CA"], "atom2": ["L", 1, "C1"]}}
            ],
            "properties": [{"affinity": {"binder": "L"}}],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "boltz": {"devices": 2, "diffusion_samples": 3},
            "protenix": {"cycle": 8, "sample": 2},
            "alphafold3": {
                "model_dir": "/tmp/models",
                "run_data_pipeline": False,
                "run_inference": True,
                "num_recycles": 7,
                "num_diffusion_samples": 4,
            }
        }
    )
    prepared = prepare_run(common, config, tmp_path, "slurm")
    assert prepared.manifest_path.exists()
    manifest = json.loads(prepared.manifest_path.read_text())
    assert manifest["backend"] == "slurm"
    assert len(manifest["models"]) == 3
    assert "--devices" in manifest["models"][0]["command"]
    assert "2" in manifest["models"][0]["command"]
    assert "--cycle" in manifest["models"][1]["command"]
    af3_script = prepared.run_dir / "scripts" / "run_alphafold3.sh"
    assert af3_script.exists()
    assert "--num_recycles=7" in af3_script.read_text()
    assert (prepared.run_dir / "inputs" / "boltz_input.yaml").exists()
    assert (prepared.run_dir / "inputs" / "protenix_input.json").exists()
    assert (prepared.run_dir / "inputs" / "alphafold3_input.json").exists()
    assert manifest["models"][1]["notes"]
    assert manifest["models"][2]["notes"]
    assert prepared.shell_script.exists()


def test_write_example_config_populates_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "runner_config.yaml"
    write_example_config(config_path)

    config = yaml.safe_load(config_path.read_text())
    assert config["preset"] == "balanced"
    assert config["template_search_sequence"]["database_path"] == "/path/to/rcsb/mmseqs/protein_db"
    assert config["template_search_structure"]["database_path"] == "/path/to/rcsb/foldseek/structure_db"
    assert config["template_search_structure"]["query_structure_path"] == "/path/to/query_structure.cif"
    assert config["template_search_structure"]["query_from_cofolding"] is False
    assert config["template_search_structure"]["query_model_priority"] == [
        "alphafold3",
        "boltz",
        "protenix",
    ]
    assert config["boltz"]["max_parallel_samples"] == 5
    assert config["boltz"]["msa_server_url"] == "https://api.colabfold.com"
    assert config["protenix"]["cycle"] == 10
    assert config["protenix"]["sample"] == 5
    assert config["alphafold3"]["model_dir"] == "/path/to/alphafold3/models"
    assert config["alphafold3"]["db_dirs"] == ["/path/to/alphafold3/databases"]
    assert config["alphafold3"]["num_recycles"] == 10
    assert config["alphafold3"]["num_diffusion_samples"] == 5
    assert config["vina"]["receptor_pdbqt"] == "/path/to/receptor.pdbqt"
    assert config["vina"]["size_x"] == 20.0
    assert config["slurm"]["partition"] == "gpu"

    fast_config_path = tmp_path / "runner_config.fast.yaml"
    write_example_config(fast_config_path, preset="fast")
    fast_config = yaml.safe_load(fast_config_path.read_text())
    assert fast_config["preset"] == "fast"
    assert fast_config["boltz"]["max_parallel_samples"] == 8
    assert fast_config["alphafold3"]["num_recycles"] == 3


def test_prepare_template_search_sequence_stage(tmp_path: Path) -> None:
    db_path = tmp_path / "seqdb"
    db_path.write_text("db\n")
    common = CommonInput.from_dict(
        {
            "version": 1,
            "name": "target-1",
            "sequences": [
                {"protein": {"id": ["A", "B"], "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "template_search_sequence": {
                "enabled": True,
                "binary": "/bin/echo",
                "database_path": str(db_path),
                "min_seq_identity": 0.5,
                "max_hits": 50,
            },
            "alphafold3": {"enabled": False},
        }
    )
    prepared = prepare_template_search_sequence_run(common, config, tmp_path, "slurm")
    assert prepared.model_runs[0].model_name == "template-search-sequence"
    fasta_text = (prepared.run_dir / "inputs" / "template_search_sequence_queries.fasta").read_text()
    assert ">target-1|protein|A" in fasta_text
    assert ">target-1|protein|B" in fasta_text
    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["template-search-sequence"])
    assert report.ok


def test_prepare_template_search_structure_stage(tmp_path: Path) -> None:
    db_path = tmp_path / "structdb"
    db_path.write_text("db\n")
    query_path = tmp_path / "query.cif"
    query_path.write_text("data_query\n")
    common = CommonInput.from_dict(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "template_search_structure": {
                "enabled": True,
                "binary": "/bin/echo",
                "database_path": str(db_path),
                "query_structure_path": str(query_path),
            },
            "alphafold3": {"enabled": False},
        }
    )
    prepared = prepare_template_search_structure_run(common, config, tmp_path, "slurm")
    assert prepared.model_runs[0].model_name == "template-search-structure"
    manifest = json.loads(prepared.manifest_path.read_text())
    assert manifest["models"][0]["command"][0] == "bash"
    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["template-search-structure"])
    assert report.ok


def test_prepare_template_search_structure_from_cofolding(tmp_path: Path) -> None:
    db_path = tmp_path / "structdb"
    db_path.write_text("db\n")
    common = CommonInput.from_dict(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "template_search_structure": {
                "enabled": True,
                "binary": "/bin/echo",
                "database_path": str(db_path),
                "query_from_cofolding": True,
                "query_model_priority": ["boltz", "alphafold3"],
            },
            "boltz": {"enabled": False},
            "protenix": {"enabled": False},
            "alphafold3": {"enabled": False},
        }
    )
    prepared = prepare_template_search_structure_run(common, config, tmp_path, "slurm")
    script_text = (prepared.run_dir / "scripts" / "run_template_search_structure_inner.sh").read_text()
    assert "outputs/boltz" in script_text
    assert 'QUERY_FROM_COFOLDING=true' in script_text
    assert "break" not in script_text
    config_with_model = RunnerConfig.from_dict(
        {
            "template_search_structure": {
                "enabled": True,
                "binary": "/bin/echo",
                "database_path": str(db_path),
                "query_from_cofolding": True,
                "query_model_priority": ["boltz", "alphafold3"],
            },
            "boltz": {"enabled": True, "binary": "/bin/echo"},
            "protenix": {"enabled": False},
            "alphafold3": {"enabled": False},
        }
    )
    report = validate_run(
        common,
        config_with_model,
        "slurm",
        repo_root=tmp_path,
        stages=["cofolding", "template-search-structure"],
    )
    assert report.ok


def test_wrapper_rejects_structure_search_before_cofolding_when_reused(tmp_path: Path) -> None:
    db_path = tmp_path / "structdb"
    db_path.write_text("db\n")
    common = CommonInput.from_dict(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "template_search_structure": {
                "enabled": True,
                "binary": "/bin/echo",
                "database_path": str(db_path),
                "query_from_cofolding": True,
            },
            "alphafold3": {"enabled": False},
        }
    )
    try:
        prepare_stage_wrapper_run(
            common,
            config,
            tmp_path,
            "slurm",
            stages=["template-search-structure", "cofolding"],
        )
    except ValueError as exc:
        assert "must come after cofolding" in str(exc)
    else:
        raise AssertionError("expected wrapper stage-order validation to fail")


def test_prepare_vina_and_validate_docking_stage(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text("RECEPTOR\n")
    ligand.write_text("LIGAND\n")

    common = CommonInput.from_dict(
        {
            "version": 1,
            "seed": 11,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "alphafold3": {"enabled": False},
            "vina": {
                "enabled": True,
                "binary": "/bin/echo",
                "receptor_pdbqt": str(receptor),
                "ligand_pdbqt": str(ligand),
                "center_x": 0.0,
                "center_y": 1.0,
                "center_z": 2.0,
                "size_x": 20.0,
                "size_y": 21.0,
                "size_z": 22.0,
            },
        }
    )

    prepared = prepare_vina_run(common, config, tmp_path, "slurm")
    assert prepared.model_runs[0].model_name == "vina"
    assert (prepared.run_dir / "scripts" / "run_vina.py").exists()

    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["docking"])
    assert report.ok
    assert report.warnings


def test_prepare_wrapper_respects_selected_stages(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text("RECEPTOR\n")
    ligand.write_text("LIGAND\n")

    common = CommonInput.from_dict(
        {
            "version": 1,
            "seed": 5,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "alphafold3": {"enabled": False},
            "vina": {
                "enabled": True,
                "binary": "/bin/echo",
                "receptor_pdbqt": str(receptor),
                "ligand_pdbqt": str(ligand),
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
            },
        }
    )

    prepared = prepare_stage_wrapper_run(common, config, tmp_path, "slurm", stages=["docking"])
    manifest = json.loads(prepared.manifest_path.read_text())
    assert [stage["stage"] for stage in manifest["stages"]] == ["docking"]


def test_prepare_protenix_dock_and_validate(tmp_path: Path) -> None:
    receptor = tmp_path / "receptor.pdb"
    ligand = tmp_path / "ligand.sdf"
    receptor.write_text("ATOM\n")
    ligand.write_text("ligand\n")

    common = CommonInput.from_dict(
        {
            "version": 1,
            "seed": 42,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "alphafold3": {"enabled": False},
            "protenix_dock": {
                "enabled": True,
                "python_bin": "/bin/echo",
                "receptor_pdb": str(receptor),
                "ligand_sdf": str(ligand),
                "center_x": 1.0,
                "center_y": 2.0,
                "center_z": 3.0,
                "size_x": 15.0,
                "size_y": 16.0,
                "size_z": 17.0,
                "cache_map_spacing": 0.2,
            },
        }
    )

    prepared = prepare_protenix_dock_run(common, config, tmp_path, "slurm")
    assert prepared.model_runs[0].model_name == "protenix-dock"
    runner_script = prepared.run_dir / "scripts" / "run_protenix_dock.py"
    assert runner_script.exists()
    script_text = runner_script.read_text()
    assert "ProtenixDock" in script_text
    assert "1.0, 2.0, 3.0" in script_text
    assert "15.0, 16.0, 17.0" in script_text
    assert "0.2" in script_text

    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["docking"])
    assert report.ok


def test_prepare_docking_run_combines_vina_and_protenix_dock(tmp_path: Path) -> None:
    receptor_pdbqt = tmp_path / "receptor.pdbqt"
    ligand_pdbqt = tmp_path / "ligand.pdbqt"
    receptor_pdb = tmp_path / "receptor.pdb"
    ligand_sdf = tmp_path / "ligand.sdf"
    for f in [receptor_pdbqt, ligand_pdbqt, receptor_pdb, ligand_sdf]:
        f.write_text("data\n")

    common = CommonInput.from_dict(
        {
            "version": 1,
            "seed": 7,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "alphafold3": {"enabled": False},
            "vina": {
                "enabled": True,
                "binary": "/bin/echo",
                "receptor_pdbqt": str(receptor_pdbqt),
                "ligand_pdbqt": str(ligand_pdbqt),
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
            },
            "protenix_dock": {
                "enabled": True,
                "python_bin": "/bin/echo",
                "receptor_pdb": str(receptor_pdb),
                "ligand_sdf": str(ligand_sdf),
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "size_x": 20.0,
                "size_y": 20.0,
                "size_z": 20.0,
            },
        }
    )

    prepared = prepare_docking_run(common, config, tmp_path, "slurm")
    model_names = [m.model_name for m in prepared.model_runs]
    assert "vina" in model_names
    assert "protenix-dock" in model_names
    assert len(prepared.model_runs) == 2


def test_write_example_config_includes_protenix_dock(tmp_path: Path) -> None:
    config_path = tmp_path / "runner_config.yaml"
    write_example_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    assert config["protenix_dock"]["receptor_pdb"] == "/path/to/receptor.pdb"
    assert config["protenix_dock"]["ligand_sdf"] == "/path/to/ligand.sdf"
    assert config["protenix_dock"]["size_x"] == 20.0
    assert config["protenix_dock"]["cache_map_spacing"] == 0.175


def test_bool_string_coercion_in_from_dict() -> None:
    config = RunnerConfig.from_dict(
        {
            "boltz": {"enabled": "false"},
            "protenix": {"enabled": "false"},
            "alphafold3": {"enabled": "false", "run_inference": "false", "run_data_pipeline": "false"},
            "template_search_structure": {"enabled": "false", "query_from_cofolding": "false"},
            "protenix_dock": {"enabled": "false", "use_cache_maps": "false"},
            "vina": {"enabled": "false"},
        }
    )
    assert config.boltz.enabled is False
    assert config.protenix.enabled is False
    assert config.alphafold3.enabled is False
    assert config.alphafold3.run_inference is False
    assert config.alphafold3.run_data_pipeline is False
    assert config.template_search_structure.query_from_cofolding is False
    assert config.protenix_dock.use_cache_maps is False
    assert config.vina.enabled is False

    config2 = RunnerConfig.from_dict(
        {
            "boltz": {"enabled": "true"},
            "protenix": {"enabled": "true"},
        }
    )
    assert config2.boltz.enabled is True
    assert config2.protenix.enabled is True


def test_validate_run_rejects_cofolding_with_no_model(tmp_path: Path) -> None:
    common = CommonInput.from_dict(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "boltz": {"enabled": False},
            "protenix": {"enabled": False},
            "alphafold3": {"enabled": False},
        }
    )
    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["cofolding"])
    assert not report.ok
    assert any("no structure model" in e for e in report.errors)


def test_protein_sequence_required() -> None:
    import pytest

    with pytest.raises(ValueError, match="protein.sequence is required"):
        CommonInput.from_dict(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "msa": "empty"}},
                    {"ligand": {"id": "L", "smiles": "CCO"}},
                ],
            }
        )


def test_duplicate_entity_id_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="Duplicate entity id"):
        CommonInput.from_dict(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                    {"protein": {"id": "A", "sequence": "MBBB", "msa": "empty"}},
                    {"ligand": {"id": "L", "smiles": "CCO"}},
                ],
            }
        )


def test_validate_run_catches_invalid_bond_chain_id(tmp_path: Path) -> None:
    common = CommonInput.from_dict(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "MAAA", "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": "CCO"}},
            ],
            "constraints": [{"bond": {"atom1": ["Z", 1, "CA"], "atom2": ["L", 1, "C1"]}}],
        }
    )
    config = RunnerConfig.from_dict(
        {
            "boltz": {"enabled": True, "binary": "/bin/echo"},
            "protenix": {"enabled": False},
            "alphafold3": {"enabled": False},
        }
    )
    report = validate_run(common, config, "slurm", repo_root=tmp_path, stages=["cofolding"])
    assert not report.ok
    assert any("unknown chain id" in e.lower() for e in report.errors)
