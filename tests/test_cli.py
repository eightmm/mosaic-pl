import json
from pathlib import Path

import yaml

from casp17.cli import build_parser, cmd_status
from casp17.configs import RunnerConfig
from casp17.models import CommonInput
from casp17.orchestrator import (
    prepare_docking_run,
    prepare_protenix_dock_run,
    prepare_run,
    prepare_stage_wrapper_run,
    prepare_template_search_sequence_run,
    prepare_template_search_structure_run,
    prepare_vina_run,
    write_example_config,
)
from casp17.validation import validate_run


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
        "boltz2x",
        "boltz2",
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
    names = [m.model_name for m in prepared.model_runs]
    assert names == [
        "vina_cofolding_1",
        "vina_cofolding_2",
        "vina_cofolding_3",
        "vina_swinsite_1",
        "vina_swinsite_2",
        "vina_swinsite_3",
        "vina_p2rank_1",
        "vina_p2rank_2",
        "vina_p2rank_3",
        "vina_template_consensus_1",
        "vina_template_consensus_2",
        "vina_template_consensus_3",
        "vina_template_consensus_4",
        "vina_template_consensus_5",
        "vina_template_consensus_6",
        "vina_template_consensus_7",
        "vina_template_consensus_8",
        "vina_template_consensus_9",
        "vina_template_consensus_10",
    ]
    # One runner script per variant.
    for name in names:
        assert (prepared.run_dir / "scripts" / f"run_{name}.py").exists()

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
    # Vina fans out into one variant per binding-site source:
    #   - 3 cofold-cluster sources (top-K of cofolding ligand clusters
    #     across 4 models × 25 seeds, registered as cofolding_{1,2,3})
    #   - 3 SwinSite ML predictor pockets (top-3, ranked by score so
    #     multi-chain receptors expose chain-B/C equivalents at rank 2/3)
    #   - 3 P2Rank geometry predictor pockets (top-3, same rationale)
    #   - 10 template-consensus pocket centroids (top-K cluster centers
    #     from mmseqs+foldseek union)
    expected_cofold = {f"vina_cofolding_{i}" for i in range(1, 4)}
    expected_swinsite = {f"vina_swinsite_{i}" for i in range(1, 4)}
    expected_p2rank = {f"vina_p2rank_{i}" for i in range(1, 4)}
    expected_consensus = {f"vina_template_consensus_{i}" for i in range(1, 11)}
    assert expected_cofold.issubset(set(model_names))
    assert expected_swinsite.issubset(set(model_names))
    assert expected_p2rank.issubset(set(model_names))
    assert expected_consensus.issubset(set(model_names))
    assert "protenix-dock" in model_names
    # 19 vina variants + 1 protenix-dock; autodock_gpu disabled in this fixture.
    assert len(prepared.model_runs) == 20


def test_write_example_config_includes_protenix_dock(tmp_path: Path) -> None:
    config_path = tmp_path / "runner_config.yaml"
    write_example_config(config_path)
    config = yaml.safe_load(config_path.read_text())
    assert config["protenix_dock"]["receptor_pdb"] == "/path/to/receptor.pdb"
    assert config["protenix_dock"]["ligand_sdf"] == "/path/to/ligand.sdf"
    assert config["protenix_dock"]["size_x"] == 20.0
    assert config["protenix_dock"]["cache_map_spacing"] == 0.375


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


# --- Multi-track docking tests ---


def test_template_search_config_has_multi_track_fields() -> None:
    config = RunnerConfig.from_dict(
        {
            "template_search_sequence": {
                "enabled": True,
                "database_path": "/tmp/db",
                "rcsb_dir": "/data/rcsb",
                "rcsb_db_path": "/data/rcsb_index.db",
                "mcs_threshold": 0.6,
            }
        }
    )
    assert config.template_search_sequence.rcsb_dir == "/data/rcsb"
    assert config.template_search_sequence.rcsb_db_path == "/data/rcsb_index.db"
    assert config.template_search_sequence.mcs_threshold == 0.6


def test_template_search_config_multi_track_defaults() -> None:
    config = RunnerConfig.from_dict({})
    assert config.template_search_sequence.mcs_threshold == 0.5
    assert "RCSB" in config.template_search_sequence.rcsb_dir


def test_wrapper_includes_template_filter_bridge(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict(
        {
            "template_search_sequence": {
                "enabled": True,
                "database_path": "/tmp/db",
            },
        }
    )
    stage_scripts = [
        ("template-search-sequence", tmp_path / "scripts" / "run_tss.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)

    assert "run_template_filter.py" in script
    assert "filtered_hits.tsv" in script
    assert "BRIDGE: Filtering template hits" in script


def test_wrapper_includes_multi_track_docking(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict(
        {
            "template_search_sequence": {
                "enabled": True,
                "database_path": "/tmp/db",
                "mcs_threshold": 0.4,
            },
        }
    )
    stage_scripts = [
        ("template-search-sequence", tmp_path / "scripts" / "run_tss.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)

    assert "MULTI-TRACK DOCKING" in script
    assert "run_multi_track_docking.py" in script
    assert "--mcs-threshold 0.4" in script


def test_wrapper_no_multi_track_without_template_search(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict({})
    stage_scripts = [
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)

    assert "MULTI-TRACK" not in script
    assert "run_multi_track_docking.py" not in script


def test_check_template_hits_keeps_all_with_ligands(tmp_path: Path) -> None:
    """Track 2 (template box docking) doesn't need MCS atom matching — any
    template with a bound ligand yields a usable pocket box. Only Track 3
    lig-MCS-align is gated by MCS, and that gate lives per-template in the
    main loop, not in this helper."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_multi_track_docking import check_template_hits

    tsv = tmp_path / "filtered_hits.tsv"
    tsv.write_text(
        "query\ttarget\tpdb_id\tchain_id\tpident\tevalue\tnum_ligands\t"
        "best_tanimoto\tbest_mcs_coverage\tligand_codes\tligand_types\t"
        "ligand_smiles\tligand_tanimotos\tligand_mcs_coverages\n"
        "Q\t1abc_A\t1abc\tA\t85.0\t1e-50\t1\t0.8000\t0.7000\tATP\tsmall_molecule\tC\t0.8\t0.7\n"
        "Q\t2def_B\t2def\tB\t60.0\t1e-20\t1\t0.3000\t0.2000\tNAD\tcofactor\tCC\t0.3\t0.2\n"
        "Q\t3ghi_C\t3ghi\tC\t70.0\t1e-30\t0\t0.9000\t0.9000\t\t\t\t\t\n"
    )

    hits = check_template_hits(tsv)
    assert len(hits) == 2  # 3ghi excluded (num_ligands=0); 2def kept despite low MCS
    assert {h["pdb_id"] for h in hits} == {"1abc", "2def"}


def test_check_template_hits_missing_file(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_multi_track_docking import check_template_hits

    hits = check_template_hits(tmp_path / "nonexistent.tsv")
    assert hits == []


# --- Ion placement tests ---


def test_extract_ion_ccd_codes(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_template_ions import extract_ion_ccd_codes

    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        "      sequence: MAAA\n"
        "  - ligand:\n"
        "      id: L\n"
        "      smiles: CCO\n"
        "  - ligand:\n"
        "      id: M\n"
        "      ccd: ZN\n"
        "  - ligand:\n"
        "      id: N\n"
        "      ccd: MG\n"
    )
    ions = extract_ion_ccd_codes(yaml_file)
    assert ions == ["ZN", "MG"]


def test_extract_ion_ccd_codes_no_ions(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_template_ions import extract_ion_ccd_codes

    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - ligand:\n"
        "      id: L\n"
        "      smiles: CCO\n"
    )
    assert extract_ion_ccd_codes(yaml_file) == []


def test_extract_ion_ccd_codes_non_ion_ccd(tmp_path: Path) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_template_ions import extract_ion_ccd_codes

    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - ligand:\n"
        "      id: L\n"
        "      ccd: ATP\n"
    )
    # ATP is not an ion
    assert extract_ion_ccd_codes(yaml_file) == []


def test_cluster_ion_positions() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from collect_template_ions import cluster_positions, IonPosition

    positions = [
        IonPosition("1abc", 80.0, "A", "ZN", 10.0, 20.0, 30.0, 1.0, 100),
        IonPosition("2def", 75.0, "A", "ZN", 10.5, 20.3, 30.2, 1.2, 95),
        IonPosition("3ghi", 60.0, "B", "ZN", 50.0, 60.0, 70.0, 1.5, 80),
    ]
    clusters = cluster_positions(positions, threshold=2.0)
    assert len(clusters) == 2
    # Largest cluster first
    assert clusters[0]["num_templates"] == 2
    assert clusters[1]["num_templates"] == 1
    # Centroid of first cluster ~(10.25, 20.15, 30.1)
    c = clusters[0]["centroid"]
    assert 10.0 <= c[0] <= 11.0
    assert 20.0 <= c[1] <= 21.0


def test_wrapper_includes_ion_placement(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict(
        {"template_search_sequence": {"enabled": True, "database_path": "/tmp/db"}}
    )
    stage_scripts = [
        ("template-search-sequence", tmp_path / "scripts" / "run_tss.sh"),
        ("cofolding", tmp_path / "scripts" / "run_structure.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)
    assert "ION/METAL PLACEMENT" in script
    assert "collect_template_ions.py" in script


def test_wrapper_no_ion_placement_without_cofolding(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict(
        {"template_search_sequence": {"enabled": True, "database_path": "/tmp/db"}}
    )
    stage_scripts = [
        ("template-search-sequence", tmp_path / "scripts" / "run_tss.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)
    # No ion placement without cofolding (need reference structure)
    assert "ION/METAL PLACEMENT" not in script


# --- CASP17 LG submission tests ---


def _mk_ligand(number: int, name: str = "LIG", lscore: float | None = 0.82) -> dict:
    """Helper: one LIGAND entry in the new models schema."""
    mdl = (
        "     RDKit          3D\n\n"
        " 1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "   12.345   23.456   34.567 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END"
    )
    return {"ligand_number": number, "ligand_name": name,
            "ligand_mdl": mdl, "lscore": lscore}


def test_build_lg_submission_basic() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    protein_lines = [
        "ATOM      1  N   ILE A  21      16.852   8.985  28.369  1.00 85.00           N",
        "TER",
    ]
    result = build_lg_submission(
        target_id="L2001",
        author="0123-4567-8901",
        method="test method",
        models=[{
            "protein_pdb_lines": protein_lines,
            "parent": "1CGH",
            "ligands": [_mk_ligand(1, "761", 0.82)],
        }],
    )
    assert result.startswith("PFRMAT LG\n")
    assert "TARGET L2001" in result
    assert "AUTHOR 0123-4567-8901" in result
    assert "MODEL 1" in result
    assert "PARENT 1CGH" in result
    assert "LIGAND 1 761" in result
    assert "LSCORE 0.820" in result
    assert "M  END" in result
    assert result.strip().endswith("END")


def test_build_lg_submission_no_lscore() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    result = build_lg_submission(
        target_id="L2002",
        author="0000-0000-0000",
        method="test",
        models=[{
            "protein_pdb_lines": ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"],
            "ligands": [_mk_ligand(2, "380", lscore=None)],
        }],
    )
    assert "LIGAND 2 380" in result
    assert "LSCORE" not in result


def test_build_lg_submission_auto_appends_ter() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    result = build_lg_submission(
        target_id="T1",
        author="A",
        method="M",
        models=[{
            "protein_pdb_lines": ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00"],
            "ligands": [_mk_ligand(1, "X", lscore=None)],
        }],
    )
    assert "\nTER\n" in result


def test_build_lg_submission_with_affinity() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    result = build_lg_submission(
        target_id="L2001",
        author="0000-0000-0000",
        method="test",
        models=[{
            "protein_pdb_lines": ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"],
            "ligands": [_mk_ligand(1, "761", 0.85)],
            "affinity_nM": 12.5,
        }],
    )
    assert "LSCORE 0.850" in result
    assert "AFFNTY 12.500 aa" in result
    # AFFNTY must appear between the last M END and the MODEL's END
    lines = result.strip().splitlines()
    end_idx = lines.index("END")
    affnty_idx = next(i for i, l in enumerate(lines) if l.startswith("AFFNTY"))
    assert affnty_idx < end_idx


def test_build_lg_submission_without_affinity_still_works() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    result = build_lg_submission(
        target_id="L2001",
        author="0000-0000-0000",
        method="test",
        models=[{
            "protein_pdb_lines": ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"],
            "ligands": [_mk_ligand(1, "761", 0.85)],
        }],
    )
    assert "LSCORE 0.850" in result
    assert "AFFNTY" not in result
    assert result.strip().endswith("END")


def test_build_lg_submission_multi_ligand_per_model() -> None:
    """CASP17 spec: each MODEL carries every ligand of the target."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    result = build_lg_submission(
        target_id="T1214",
        author="0000-0000-0000",
        method="multi-ligand test",
        models=[{
            "protein_pdb_lines": ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"],
            "ligands": [_mk_ligand(1, "LIG", 0.82), _mk_ligand(2, "LIG", 0.65)],
            "affinity_nM": 45.0,
        }],
    )
    # Both LIGAND blocks present in same MODEL
    assert "LIGAND 1 LIG" in result
    assert "LIGAND 2 LIG" in result
    # Both LSCOREs present
    assert "LSCORE 0.820" in result
    assert "LSCORE 0.650" in result
    # Exactly one MODEL + one END in output (multi-ligand, single-MODEL case)
    lines = result.strip().splitlines()
    assert sum(1 for line in lines if line.startswith("MODEL ")) == 1
    assert sum(1 for line in lines if line == "END") == 1
    # AFFNTY appears once, after both LIGAND blocks
    m_end_positions = [i for i, line in enumerate(lines) if line == "M  END"]
    affnty_idx = next(i for i, line in enumerate(lines) if line.startswith("AFFNTY"))
    assert len(m_end_positions) == 2  # one per ligand MDL
    assert affnty_idx > m_end_positions[-1]


def test_build_lg_submission_five_alternate_models() -> None:
    """Up to 5 MODELs allowed; each MODEL is a complete snapshot."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    protein = ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"]
    models = [
        {"protein_pdb_lines": protein, "ligands": [_mk_ligand(1, "LIG", 0.90 - i * 0.05)]}
        for i in range(5)
    ]
    result = build_lg_submission(
        target_id="L2001", author="A", method="M", models=models,
    )
    lines = result.strip().splitlines()
    assert [line for line in lines if line.startswith("MODEL ")] == [
        "MODEL 1", "MODEL 2", "MODEL 3", "MODEL 4", "MODEL 5",
    ]
    # 5 ENDs (one per MODEL), 5 M ENDs (one per MDL)
    assert sum(1 for line in lines if line == "END") == 5
    assert sum(1 for line in lines if line == "M  END") == 5


def test_build_lg_submission_rejects_more_than_5_models() -> None:
    import sys
    import pytest
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_casp_submission import build_lg_submission

    protein = ["ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00", "TER"]
    models = [
        {"protein_pdb_lines": protein, "ligands": [_mk_ligand(1)]}
        for _ in range(6)
    ]
    with pytest.raises(ValueError, match="at most 5"):
        build_lg_submission(
            target_id="L", author="A", method="M", models=models,
        )


# --- Ensemble affinity tests ---


def test_pose_score_log_kd_conversion() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compute_submission_scores import PoseScore
    from pathlib import Path as P

    # pKd = 7 → Kd = 100 nM → log10(Kd nM) = 2
    p = PoseScore(source="vina", pose_file=P(""), pose_name="t", ba_pred_pkd=7.0)
    assert abs(p.log_kd_nM - 2.0) < 1e-6

    # pKd = 9 → Kd = 1 nM → log10(Kd nM) = 0
    p = PoseScore(source="vina", pose_file=P(""), pose_name="t", ba_pred_pkd=9.0)
    assert abs(p.log_kd_nM - 0.0) < 1e-6


def test_pose_score_lscore_from_rmsd_prob() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compute_submission_scores import PoseScore
    from pathlib import Path as P

    # Prob(RMSD > 2A) = 0.2 → LSCORE = 0.8 (good pose)
    p = PoseScore(source="vina", pose_file=P(""), pose_name="t", rmsd_gt_2a_prob=0.2)
    assert abs(p.lscore - 0.8) < 1e-6

    # Prob = 0.95 → LSCORE = 0.05 (bad pose)
    p = PoseScore(source="vina", pose_file=P(""), pose_name="t", rmsd_gt_2a_prob=0.95)
    assert abs(p.lscore - 0.05) < 1e-6


def test_boltz_affinity_log_kd_conversion() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compute_submission_scores import BoltzAffinity

    # log10(IC50 uM) = 2.62 → log10(Kd nM) = 5.62 (≈ 417 uM binding)
    b = BoltzAffinity(source="boltz2", affinity_value=2.62, binder_prob=0.5)
    assert abs(b.log_kd_nM - 5.62) < 1e-6


def test_ensemble_affinity_combines_sources() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compute_submission_scores import ensemble_affinity, PoseScore, BoltzAffinity
    from pathlib import Path as P

    poses = [
        PoseScore(source="vina", pose_file=P(""), pose_name="p1", ba_pred_pkd=7.0),  # log=2
        PoseScore(source="vina", pose_file=P(""), pose_name="p2", ba_pred_pkd=7.5),  # log=1.5
    ]
    boltz = [
        BoltzAffinity(source="b1", affinity_value=0.0, binder_prob=0.8),  # log=3
        BoltzAffinity(source="b2", affinity_value=1.0, binder_prob=0.3),  # filtered out
    ]
    log_kd, details = ensemble_affinity(poses, boltz)
    # BA median = 1.75, Boltz median = 3 (b2 filtered), avg = 2.375
    assert log_kd is not None
    assert abs(log_kd - 2.375) < 1e-6
    assert len(details["boltz_filtered_out"]) == 1


def test_ensemble_affinity_ba_only() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compute_submission_scores import ensemble_affinity, PoseScore
    from pathlib import Path as P

    poses = [PoseScore(source="v", pose_file=P(""), pose_name="p", ba_pred_pkd=8.0)]
    log_kd, details = ensemble_affinity(poses, [])
    # BA = log(Kd nM) = 1, no boltz → ensemble = 1
    assert log_kd == 1.0


# --- Wrapper post-analysis + submission tests ---


def test_wrapper_includes_post_analysis_by_default(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict({"template_search_sequence": {"enabled": True}})
    stage_scripts = [
        ("cofolding", tmp_path / "scripts" / "run_structure.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)
    assert "POST-ANALYSIS" in script
    assert "run_post_analysis.py" in script


def test_wrapper_post_analysis_can_be_disabled(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict({
        "post_analysis": {"enabled": False},
    })
    stage_scripts = [
        ("cofolding", tmp_path / "scripts" / "run_structure.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)
    assert "POST-ANALYSIS" not in script


def test_wrapper_includes_submission_when_enabled(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict({
        "submission": {
            "enabled": True,
            "author": "1234-5678-9012",
            "method": "Test ensemble",
            "include_affinity": True,
        },
    })
    stage_scripts = [
        ("cofolding", tmp_path / "scripts" / "run_structure.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("L2001", config, "local", stage_scripts)
    assert "CASP17 LG SUBMISSION" in script
    assert "make_casp_submission.py" in script
    assert "1234-5678-9012" in script
    assert "--include-affinity" in script


def test_wrapper_no_submission_by_default(tmp_path: Path) -> None:
    from casp17.script_builder import build_wrapper_shell_script

    config = RunnerConfig.from_dict({})
    stage_scripts = [
        ("cofolding", tmp_path / "scripts" / "run_structure.sh"),
        ("docking", tmp_path / "scripts" / "run_docking.sh"),
    ]
    script = build_wrapper_shell_script("T0001", config, "local", stage_scripts)
    # Submission disabled by default
    assert "CASP17 LG SUBMISSION" not in script


def test_post_analysis_config_defaults() -> None:
    config = RunnerConfig.from_dict({})
    assert config.post_analysis.enabled is True
    assert config.post_analysis.device == "cuda"


def test_submission_config_defaults() -> None:
    config = RunnerConfig.from_dict({})
    assert config.submission.enabled is False
    assert config.submission.author == "6095-5696-9732"
    assert config.submission.include_affinity is True
