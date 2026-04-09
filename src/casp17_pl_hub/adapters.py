from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from casp17_pl_hub.configs import RunnerConfig
from casp17_pl_hub.io_utils import dump_json, dump_yaml_file
from casp17_pl_hub.models import CommonInput


@dataclass(slots=True)
class PreparedModelRun:
    model_name: str
    input_path: Path
    output_dir: Path
    command: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "command": self.command,
            "notes": self.notes,
        }


def prepare_boltz(common: CommonInput, config: RunnerConfig, run_dir: Path) -> PreparedModelRun | None:
    if not config.boltz.enabled:
        return None

    input_path = run_dir / "inputs" / "boltz_input.yaml"
    output_dir = run_dir / "outputs" / "boltz"
    dump_yaml_file(common.spec, input_path)

    command = [
        config.boltz.binary,
        "predict",
        str(input_path),
        "--out_dir",
        str(output_dir),
        "--seed",
        str(common.seed),
        "--model",
        config.boltz.model,
        "--accelerator",
        config.boltz.accelerator,
        "--devices",
        str(config.boltz.devices),
        "--recycling_steps",
        str(config.boltz.recycling_steps),
        "--sampling_steps",
        str(config.boltz.sampling_steps),
        "--diffusion_samples",
        str(config.boltz.diffusion_samples),
        "--output_format",
        config.boltz.output_format,
        "--num_workers",
        str(config.boltz.num_workers),
        "--sampling_steps_affinity",
        str(config.boltz.sampling_steps_affinity),
        "--diffusion_samples_affinity",
        str(config.boltz.diffusion_samples_affinity),
        "--max_msa_seqs",
        str(config.boltz.max_msa_seqs),
        "--num_subsampled_msa",
        str(config.boltz.num_subsampled_msa),
    ]
    _append_option(command, "--cache", config.boltz.cache)
    _append_option(command, "--checkpoint", config.boltz.checkpoint)
    _append_option(command, "--affinity_checkpoint", config.boltz.affinity_checkpoint)
    _append_option(command, "--max_parallel_samples", config.boltz.max_parallel_samples)
    _append_option(command, "--step_scale", config.boltz.step_scale)
    if _needs_boltz_msa_server(common) or config.boltz.use_msa_server:
        command.append("--use_msa_server")
    _append_option(command, "--msa_server_url", config.boltz.msa_server_url)
    _append_option(command, "--msa_pairing_strategy", config.boltz.msa_pairing_strategy)
    _append_option(command, "--msa_server_username", config.boltz.msa_server_username)
    _append_option(command, "--msa_server_password", config.boltz.msa_server_password)
    _append_option(command, "--api_key_header", config.boltz.api_key_header)
    _append_option(command, "--api_key_value", config.boltz.api_key_value)
    _append_option(command, "--method", config.boltz.method)
    _append_option(command, "--preprocessing-threads", config.boltz.preprocessing_threads)
    if config.boltz.override:
        command.append("--override")
    if config.boltz.use_potentials:
        command.append("--use_potentials")
    if config.boltz.affinity_mw_correction:
        command.append("--affinity_mw_correction")
    if config.boltz.subsample_msa:
        command.append("--subsample_msa")
    if config.boltz.no_kernels:
        command.append("--no_kernels")
    if config.boltz.write_full_pae:
        command.append("--write_full_pae")
    if config.boltz.write_full_pde:
        command.append("--write_full_pde")
    if config.boltz.write_embeddings:
        command.append("--write_embeddings")
    command.extend(config.boltz.extra_args)
    return PreparedModelRun("boltz", input_path, output_dir, command)


def prepare_protenix(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.protenix.enabled:
        return None

    input_path = run_dir / "inputs" / "protenix_input.json"
    output_dir = run_dir / "outputs" / "protenix"
    chain_lookup = _build_chain_lookup(common)
    notes: list[str] = []
    sequences: list[dict[str, Any]] = []
    use_msa = False
    use_rna_msa = False

    for entry in common.sequences:
        entity_type, entity = _entity(entry)
        ids = _entity_ids(entity)
        count = len(ids)

        if entity_type == "protein":
            protein_chain: dict[str, Any] = {
                "sequence": entity["sequence"],
                "count": count,
                "id": ids,
            }
            modifications = [
                {"ptmPosition": int(mod["position"]), "ptmType": _to_protenix_ccd(mod["ccd"])}
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                protein_chain["modifications"] = modifications
            msa_value = entity.get("msa")
            if isinstance(msa_value, str) and msa_value and msa_value != "empty":
                if msa_value.endswith(".a3m"):
                    protein_chain["unpairedMsaPath"] = msa_value
                    protein_chain["pairedMsa"] = ""
                    use_msa = True
                else:
                    protein_chain["msa"] = {
                        "precomputed_msa_dir": msa_value,
                        "pairing_db": "uniref100",
                    }
                    use_msa = True
            sequences.append({"proteinChain": protein_chain})
            if entity.get("cyclic"):
                notes.append("Protenix does not receive the Boltz cyclic flag explicitly.")
        elif entity_type == "dna":
            dna_chain: dict[str, Any] = {
                "sequence": entity["sequence"],
                "count": count,
                "id": ids,
            }
            modifications = [
                {
                    "basePosition": int(mod["position"]),
                    "modificationType": _to_protenix_ccd(mod["ccd"]),
                }
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                dna_chain["modifications"] = modifications
            sequences.append({"dnaSequence": dna_chain})
            if entity.get("cyclic"):
                notes.append("Protenix does not receive the Boltz cyclic flag explicitly.")
        elif entity_type == "rna":
            rna_chain: dict[str, Any] = {
                "sequence": entity["sequence"],
                "count": count,
                "id": ids,
            }
            modifications = [
                {
                    "basePosition": int(mod["position"]),
                    "modificationType": _to_protenix_ccd(mod["ccd"]),
                }
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                rna_chain["modifications"] = modifications
            sequences.append({"rnaSequence": rna_chain})
            use_rna_msa = True
            if entity.get("cyclic"):
                notes.append("Protenix does not receive the Boltz cyclic flag explicitly.")
        elif entity_type == "ligand":
            ligand_value = entity.get("smiles")
            if ligand_value is None:
                ligand_value = f"CCD_{entity['ccd']}"
            sequences.append({"ligand": {"ligand": ligand_value, "count": count, "id": ids}})

    # Inject template paths into matching protein chains
    template_applied = False
    for tmpl in common.templates:
        tmpl_path = tmpl.get("path")
        tmpl_ids = tmpl.get("ids")
        if not tmpl_path:
            continue
        for seq_entry in sequences:
            if "proteinChain" not in seq_entry:
                continue
            chain = seq_entry["proteinChain"]
            chain_ids_in_entry = chain.get("id", [])
            if tmpl_ids and not any(cid in tmpl_ids for cid in chain_ids_in_entry):
                continue
            chain["templatesPath"] = tmpl_path
            template_applied = True
    if common.templates and not template_applied:
        notes.append("Templates specified but no matching protein chains found for Protenix.")

    payload = [{"name": common.name, "covalent_bonds": _protenix_bonds(common, chain_lookup, notes), "sequences": sequences}]
    dump_json(payload, input_path)
    for constraint in common.constraints:
        constraint_type = next(iter(constraint))
        if constraint_type in {"pocket", "contact"}:
            notes.append(f"Protenix adapter currently ignores Boltz {constraint_type} constraints.")
    if common.properties:
        notes.append("Protenix adapter currently ignores Boltz properties such as affinity.")

    command = [
        config.protenix.binary,
        "pred",
        "--input",
        str(input_path),
        "--out_dir",
        str(output_dir),
        "--seeds",
        str(common.seed),
        "--cycle",
        str(config.protenix.cycle),
        "--step",
        str(config.protenix.step),
        "--sample",
        str(config.protenix.sample),
        "--model_name",
        config.protenix.model_name,
        "--dtype",
        config.protenix.dtype,
        "--use_msa",
        str(config.protenix.use_msa or use_msa).lower(),
        "--use_default_params",
        str(config.protenix.use_default_params).lower(),
        "--trimul_kernel",
        config.protenix.trimul_kernel,
        "--triatt_kernel",
        config.protenix.triatt_kernel,
        "--enable_cache",
        str(config.protenix.enable_cache).lower(),
        "--enable_fusion",
        str(config.protenix.enable_fusion).lower(),
        "--enable_tf32",
        str(config.protenix.enable_tf32).lower(),
        "--msa_server_mode",
        config.protenix.msa_server_mode,
        "--use_template",
        str(config.protenix.use_template or template_applied).lower(),
        "--use_rna_msa",
        str(config.protenix.use_rna_msa or use_rna_msa).lower(),
        "--use_seeds_in_json",
        str(config.protenix.use_seeds_in_json).lower(),
        "--need_atom_confidence",
        str(config.protenix.need_atom_confidence).lower(),
        "--use_tfg_guidance",
        str(config.protenix.use_tfg_guidance).lower(),
    ]
    _append_option(command, "--kalign_binary_path", config.protenix.kalign_binary_path)
    _append_option(command, "--hmmsearch_binary_path", config.protenix.hmmsearch_binary_path)
    _append_option(command, "--hmmbuild_binary_path", config.protenix.hmmbuild_binary_path)
    _append_option(command, "--seqres_database_path", config.protenix.seqres_database_path)
    _append_option(command, "--nhmmer_binary_path", config.protenix.nhmmer_binary_path)
    _append_option(command, "--hmmalign_binary_path", config.protenix.hmmalign_binary_path)
    _append_option(command, "--hmmbuild_rna_binary_path", config.protenix.hmmbuild_rna_binary_path)
    _append_option(command, "--ntrna_database_path", config.protenix.ntrna_database_path)
    _append_option(command, "--rfam_database_path", config.protenix.rfam_database_path)
    _append_option(command, "--rna_central_database_path", config.protenix.rna_central_database_path)
    _append_option(command, "--nhmmer_n_cpu", config.protenix.nhmmer_n_cpu)
    command.extend(config.protenix.extra_args)
    return PreparedModelRun("protenix", input_path, output_dir, command, notes)


def prepare_alphafold3(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.alphafold3.enabled:
        return None

    input_path = run_dir / "inputs" / "alphafold3_input.json"
    output_dir = run_dir / "outputs" / "alphafold3"
    chain_lookup = _build_chain_lookup(common)
    notes: list[str] = []
    sequences: list[dict[str, Any]] = []

    for entry in common.sequences:
        entity_type, entity = _entity(entry)
        ids = _entity_ids(entity)
        af3_id: str | list[str] = ids[0] if len(ids) == 1 else ids

        if entity_type == "protein":
            protein_block: dict[str, Any] = {"id": af3_id, "sequence": entity["sequence"]}
            modifications = [
                {"ptmType": str(mod["ccd"]), "ptmPosition": int(mod["position"])}
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                protein_block["modifications"] = modifications
            msa_value = entity.get("msa")
            if msa_value == "empty":
                protein_block["unpairedMsa"] = ""
                protein_block["pairedMsa"] = ""
            elif isinstance(msa_value, str) and msa_value:
                if msa_value.endswith(".a3m"):
                    protein_block["unpairedMsaPath"] = msa_value
                    protein_block["pairedMsa"] = ""
                else:
                    notes.append(
                        f"AlphaFold3 adapter ignores non-A3M protein MSA path for chain(s) {ids}: {msa_value}"
                    )
            sequences.append({"protein": protein_block})
            if entity.get("cyclic"):
                notes.append("AlphaFold3 adapter currently ignores the Boltz cyclic flag.")
        elif entity_type == "dna":
            dna_block: dict[str, Any] = {"id": af3_id, "sequence": entity["sequence"]}
            modifications = [
                {"modificationType": str(mod["ccd"]), "basePosition": int(mod["position"])}
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                dna_block["modifications"] = modifications
            sequences.append({"dna": dna_block})
            if entity.get("cyclic"):
                notes.append("AlphaFold3 adapter currently ignores the Boltz cyclic flag.")
        elif entity_type == "rna":
            rna_block: dict[str, Any] = {"id": af3_id, "sequence": entity["sequence"]}
            modifications = [
                {"modificationType": str(mod["ccd"]), "basePosition": int(mod["position"])}
                for mod in entity.get("modifications", [])
            ]
            if modifications:
                rna_block["modifications"] = modifications
            sequences.append({"rna": rna_block})
            if entity.get("cyclic"):
                notes.append("AlphaFold3 adapter currently ignores the Boltz cyclic flag.")
        elif entity_type == "ligand":
            ligand_block: dict[str, Any] = {"id": af3_id}
            if "ccd" in entity:
                ligand_block["ccdCodes"] = [entity["ccd"]]
            else:
                ligand_block["smiles"] = entity["smiles"]
            sequences.append({"ligand": ligand_block})

    bonded_atom_pairs = _alphafold3_bonds(common, chain_lookup, notes)
    payload: dict[str, Any] = {
        "name": common.name,
        "modelSeeds": [common.seed],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 4,
    }
    if bonded_atom_pairs:
        payload["bondedAtomPairs"] = bonded_atom_pairs

    dump_json(payload, input_path)

    if common.templates:
        notes.append(
            "Boltz templates are not translated to AlphaFold3 templates because AlphaFold3 requires explicit query/template index mappings."
        )
    for constraint in common.constraints:
        constraint_type = next(iter(constraint))
        if constraint_type in {"pocket", "contact"}:
            notes.append(f"AlphaFold3 adapter currently ignores Boltz {constraint_type} constraints.")
    if common.properties:
        notes.append("AlphaFold3 adapter currently ignores Boltz properties such as affinity.")

    command = [
        config.alphafold3.python_bin,
        config.alphafold3.script,
        f"--json_path={input_path}",
        f"--output_dir={output_dir}",
        f"--run_data_pipeline={'true' if config.alphafold3.run_data_pipeline else 'false'}",
        f"--run_inference={'true' if config.alphafold3.run_inference else 'false'}",
    ]
    if config.alphafold3.model_dir:
        command.append(f"--model_dir={config.alphafold3.model_dir}")
    for db_dir in config.alphafold3.db_dirs:
        command.append(f"--db_dir={db_dir}")
    _append_option_eq(command, "--jackhmmer_binary_path", config.alphafold3.jackhmmer_binary_path)
    _append_option_eq(command, "--nhmmer_binary_path", config.alphafold3.nhmmer_binary_path)
    _append_option_eq(command, "--hmmalign_binary_path", config.alphafold3.hmmalign_binary_path)
    _append_option_eq(command, "--hmmsearch_binary_path", config.alphafold3.hmmsearch_binary_path)
    _append_option_eq(command, "--hmmbuild_binary_path", config.alphafold3.hmmbuild_binary_path)
    _append_option_eq(command, "--jackhmmer_n_cpu", config.alphafold3.jackhmmer_n_cpu)
    _append_option_eq(
        command, "--jackhmmer_max_parallel_shards", config.alphafold3.jackhmmer_max_parallel_shards
    )
    _append_option_eq(command, "--nhmmer_n_cpu", config.alphafold3.nhmmer_n_cpu)
    _append_option_eq(
        command, "--nhmmer_max_parallel_shards", config.alphafold3.nhmmer_max_parallel_shards
    )
    if config.alphafold3.resolve_msa_overlaps is not None:
        command.append(
            f"--resolve_msa_overlaps={'true' if config.alphafold3.resolve_msa_overlaps else 'false'}"
        )
    _append_option_eq(command, "--max_template_date", config.alphafold3.max_template_date)
    _append_option_eq(
        command, "--conformer_max_iterations", config.alphafold3.conformer_max_iterations
    )
    _append_option_eq(
        command, "--jax_compilation_cache_dir", config.alphafold3.jax_compilation_cache_dir
    )
    _append_option_eq(command, "--gpu_device", config.alphafold3.gpu_device)
    if config.alphafold3.buckets:
        command.append("--buckets=" + ",".join(str(item) for item in config.alphafold3.buckets))
    _append_option_eq(
        command,
        "--flash_attention_implementation",
        config.alphafold3.flash_attention_implementation,
    )
    _append_option_eq(command, "--num_recycles", config.alphafold3.num_recycles)
    _append_option_eq(
        command, "--num_diffusion_samples", config.alphafold3.num_diffusion_samples
    )
    _append_option_eq(command, "--num_seeds", config.alphafold3.num_seeds)
    if config.alphafold3.save_embeddings:
        command.append("--save_embeddings=true")
    if config.alphafold3.save_distogram:
        command.append("--save_distogram=true")
    if config.alphafold3.force_output_dir:
        command.append("--force_output_dir=true")
    if config.alphafold3.compress_large_output_files:
        command.append("--compress_large_output_files=true")
    command.extend(config.alphafold3.extra_args)
    return PreparedModelRun("alphafold3", input_path, output_dir, command, notes)


def prepare_template_search_sequence(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.template_search_sequence.enabled:
        return None

    input_path = run_dir / "inputs" / "template_search_sequence_queries.fasta"
    output_dir = run_dir / "outputs" / "template_search_sequence"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    notes: list[str] = []

    fasta_lines = _protein_query_fasta(common)
    if not fasta_lines:
        raise ValueError("Sequence-based template search requires at least one protein sequence.")
    input_path.write_text("\n".join(fasta_lines) + "\n")

    result_path = output_dir / "mmseqs_hits.tsv"
    command = [
        config.template_search_sequence.binary,
        "easy-search",
        str(input_path),
        str(config.template_search_sequence.database_path),
        str(result_path),
        str(tmp_dir),
        "--format-output",
        "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen",
        "--min-seq-id",
        str(config.template_search_sequence.min_seq_identity),
        "-c",
        str(config.template_search_sequence.min_coverage),
        "--cov-mode",
        "0",
        "-s",
        str(config.template_search_sequence.sensitivity),
        "--max-seqs",
        str(config.template_search_sequence.max_hits),
        "--threads",
        str(config.template_search_sequence.threads),
    ]
    command.extend(config.template_search_sequence.extra_args)
    notes.append(
        "Sequence-based template search is prepared against an external MMseqs2 database; build and update the RCSB sequence DB outside this repo."
    )
    return PreparedModelRun("template-search-sequence", input_path, output_dir, command, notes)


def prepare_template_search_structure(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.template_search_structure.enabled:
        return None

    input_path = run_dir / "inputs" / "template_search_structure_config.json"
    output_dir = run_dir / "outputs" / "template_search_structure"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    notes: list[str] = []
    query_resolver_script = run_dir / "scripts" / "run_template_search_structure_inner.sh"

    protein_count = sum(1 for entry in common.sequences if _entity(entry)[0] == "protein")
    if protein_count == 0:
        raise ValueError("Structure-based template search requires at least one protein entity.")

    dump_json(
        {
            "query_structure_path": config.template_search_structure.query_structure_path,
            "query_from_cofolding": config.template_search_structure.query_from_cofolding,
            "query_model_priority": config.template_search_structure.query_model_priority,
            "database_path": config.template_search_structure.database_path,
            "result_path": str(output_dir / "foldseek_hits.tsv"),
        },
        input_path,
    )

    result_path = output_dir / "foldseek_hits.tsv"
    resolver_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"QUERY_FROM_COFOLDING={'true' if config.template_search_structure.query_from_cofolding else 'false'}",
        f"EXPLICIT_QUERY={_shell_quote_or_empty(config.template_search_structure.query_structure_path)}",
        'QUERY_PATH=""',
        'if [[ "$QUERY_FROM_COFOLDING" == "true" ]]; then',
    ]
    for model_name in config.template_search_structure.query_model_priority:
        model_dir = run_dir / "outputs" / model_name
        resolver_lines.extend(
            [
                f"  if [[ -z \"$QUERY_PATH\" && -d { _shell_quote(str(model_dir)) } ]]; then",
                f"    QUERY_PATH=$(find { _shell_quote(str(model_dir)) } -type f \\( -name '*.cif' -o -name '*.mmcif' -o -name '*.pdb' \\) | sort | head -n 1 || true)",
                "  fi",
            ]
        )
    resolver_lines.extend(
        [
            "fi",
            'if [[ -z "$QUERY_PATH" && -n "$EXPLICIT_QUERY" ]]; then',
            '  QUERY_PATH="$EXPLICIT_QUERY"',
            "fi",
            'if [[ -z "$QUERY_PATH" ]]; then',
            '  echo "No query structure found for template-search-structure." >&2',
            "  exit 1",
            "fi",
            "",
            _render_shell_command(
                [
                    config.template_search_structure.binary,
                    "easy-search",
                    '"$QUERY_PATH"',
                    str(config.template_search_structure.database_path),
                    str(result_path),
                    str(tmp_dir),
                    "--format-output",
                    "query,target,evalue,bits,alntmscore,qtmscore,ttmscore,prob",
                    "--alignment-type",
                    str(config.template_search_structure.alignment_type),
                    "-s",
                    str(config.template_search_structure.sensitivity),
                    "--max-seqs",
                    str(config.template_search_structure.max_hits),
                    "--threads",
                    str(config.template_search_structure.threads),
                    *config.template_search_structure.extra_args,
                ],
                preserve_shell_vars=True,
            ),
            "",
        ]
    )
    query_resolver_script.write_text("\n".join(resolver_lines))
    query_resolver_script.chmod(0o755)
    command = ["bash", str(query_resolver_script)]
    notes.append(
        "Structure-based template search uses an external Foldseek database."
    )
    if config.template_search_structure.query_from_cofolding:
        notes.append(
            "Query structure will be resolved at runtime from cofolding outputs using query_model_priority."
        )
    elif config.template_search_structure.query_structure_path:
        notes.append("Query structure will be taken from template_search_structure.query_structure_path.")
    return PreparedModelRun("template-search-structure", input_path, output_dir, command, notes)


def _load_docking_prep_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load auto-generated docking prep summary if available."""
    summary_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    if summary_path.exists():
        import json
        return json.loads(summary_path.read_text())
    return None


def prepare_vina(common: CommonInput, config: RunnerConfig, run_dir: Path) -> PreparedModelRun | None:
    if not config.vina.enabled:
        return None

    input_path = run_dir / "inputs" / "vina_config.txt"
    output_dir = run_dir / "outputs" / "vina"
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # Auto-detect from docking prep bridge output
    prep = _load_docking_prep_summary(run_dir)
    receptor_pdbqt = config.vina.receptor_pdbqt
    ligand_pdbqt = config.vina.ligand_pdbqt
    center_x, center_y, center_z = config.vina.center_x, config.vina.center_y, config.vina.center_z
    size_x, size_y, size_z = config.vina.size_x, config.vina.size_y, config.vina.size_z

    if prep:
        receptor_pdbqt = receptor_pdbqt or prep["receptor_pdbqt"]
        if prep["ligands"]:
            ligand_pdbqt = ligand_pdbqt or prep["ligands"][0]["pdbqt"]
        bc = prep.get("box_center", [0, 0, 0])
        bs = prep.get("box_size", [20, 20, 20])
        center_x = center_x if center_x is not None else bc[0]
        center_y = center_y if center_y is not None else bc[1]
        center_z = center_z if center_z is not None else bc[2]
        size_x = size_x if size_x is not None else bs[0]
        size_y = size_y if size_y is not None else bs[1]
        size_z = size_z if size_z is not None else bs[2]
        notes.append("Vina inputs auto-detected from docking prep bridge output.")
    else:
        notes.append(
            "Vina consumes receptor_pdbqt and ligand_pdbqt from runner_config. "
            "Use the wrapper pipeline with cofolding+docking stages for auto-preparation."
        )

    out_path = output_dir / "docked.pdbqt"
    log_path = output_dir / "vina.log"
    config_lines = [
        f"receptor = {receptor_pdbqt}",
        f"ligand = {ligand_pdbqt}",
        f"out = {out_path}",
        f"log = {log_path}",
        f"center_x = {center_x}",
        f"center_y = {center_y}",
        f"center_z = {center_z}",
        f"size_x = {size_x}",
        f"size_y = {size_y}",
        f"size_z = {size_z}",
        f"cpu = {config.vina.cpu}",
        f"exhaustiveness = {config.vina.exhaustiveness}",
        f"num_modes = {config.vina.num_modes}",
        f"energy_range = {config.vina.energy_range}",
        f"seed = {config.vina.seed if config.vina.seed is not None else common.seed}",
    ]
    if config.vina.scoring:
        config_lines.append(f"scoring = {config.vina.scoring}")
    input_path.write_text("\n".join(config_lines) + "\n")

    command = [config.vina.binary, "--config", str(input_path), *config.vina.extra_args]
    return PreparedModelRun("vina", input_path, output_dir, command, notes)


def prepare_autodock_gpu(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.autodock_gpu.enabled:
        return None

    output_dir = run_dir / "outputs" / "autodock_gpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = run_dir / "inputs" / "autodock_gpu_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # Auto-detect from docking prep bridge output
    prep = _load_docking_prep_summary(run_dir)
    receptor_pdbqt = config.autodock_gpu.receptor_pdbqt
    ligand_pdbqt = config.autodock_gpu.ligand_pdbqt
    center_x, center_y, center_z = config.autodock_gpu.center_x, config.autodock_gpu.center_y, config.autodock_gpu.center_z
    size_x, size_y, size_z = config.autodock_gpu.size_x, config.autodock_gpu.size_y, config.autodock_gpu.size_z

    if prep:
        receptor_pdbqt = receptor_pdbqt or prep["receptor_pdbqt"]
        if prep["ligands"]:
            ligand_pdbqt = ligand_pdbqt or prep["ligands"][0]["pdbqt"]
        bc = prep.get("box_center", [0, 0, 0])
        bs = prep.get("box_size", [20, 20, 20])
        center_x = center_x if center_x is not None else bc[0]
        center_y = center_y if center_y is not None else bc[1]
        center_z = center_z if center_z is not None else bc[2]
        size_x = size_x if size_x is not None else bs[0]
        size_y = size_y if size_y is not None else bs[1]
        size_z = size_z if size_z is not None else bs[2]
        notes.append("AutoDock-GPU inputs auto-detected from docking prep bridge output.")
    else:
        notes.append(
            "AutoDock-GPU consumes receptor_pdbqt and ligand_pdbqt from runner_config. "
            "Use the wrapper pipeline with cofolding+docking stages for auto-preparation."
        )

    # Default box if not yet determined (bridge will override at runtime)
    center_x = center_x if center_x is not None else 0.0
    center_y = center_y if center_y is not None else 0.0
    center_z = center_z if center_z is not None else 0.0
    size_x = size_x if size_x is not None else 22.5
    size_y = size_y if size_y is not None else 22.5
    size_z = size_z if size_z is not None else 22.5

    # Generate autogrid4 GPF (Grid Parameter File) and run script
    npts_x = max(1, int(size_x / 0.375))
    npts_y = max(1, int(size_y / 0.375))
    npts_z = max(1, int(size_z / 0.375))

    # Write GPF for autogrid4 (will be run inline in the shell script)
    gpf_path = grid_dir / "receptor.gpf"
    fld_path = grid_dir / "receptor.maps.fld"
    gpf_lines = [
        f"npts {npts_x} {npts_y} {npts_z}",
        f"gridfld {fld_path.name}",
        f"spacing 0.375",
        f"receptor_types A C HD N NA OA SA",
        f"ligand_types A C HD N NA OA SA",
        f"receptor {receptor_pdbqt}",
        f"gridcenter {center_x} {center_y} {center_z}",
        f"smooth 0.5",
        "map receptor.A.map",
        "map receptor.C.map",
        "map receptor.HD.map",
        "map receptor.N.map",
        "map receptor.NA.map",
        "map receptor.OA.map",
        "map receptor.SA.map",
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
    ]
    gpf_path.write_text("\n".join(gpf_lines) + "\n")

    # Shell script that runs autogrid4 then autodock_gpu
    runner_script = run_dir / "scripts" / "run_autodock_gpu.sh"
    runner_script.parent.mkdir(parents=True, exist_ok=True)
    seed = config.autodock_gpu.seed if config.autodock_gpu.seed is not None else common.seed
    autostop_flag = "1" if config.autodock_gpu.autostop else "0"

    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {grid_dir}",
        f"echo 'Running autogrid4 for grid maps...'",
        f"autogrid4 -p {gpf_path.name} -l autogrid.log",
        "",
        f"echo 'Running AutoDock-GPU...'",
        f"{config.autodock_gpu.binary} \\",
        f"  --ffile {fld_path} \\",
        f"  --lfile {ligand_pdbqt} \\",
        f"  --nrun {config.autodock_gpu.nrun} \\",
        f"  --nev {config.autodock_gpu.nev} \\",
        f"  --heuristics {config.autodock_gpu.heuristics} \\",
        f"  --autostop {autostop_flag} \\",
        f"  --seed {seed} \\",
        f"  --resnam {output_dir / 'docking'} \\",
        f"  {' '.join(config.autodock_gpu.extra_args)}",
        "",
        f"echo 'AutoDock-GPU results in {output_dir}'",
    ]
    runner_script.write_text("\n".join(script_lines) + "\n")
    runner_script.chmod(0o755)

    input_path = gpf_path
    command = ["bash", str(runner_script)]
    notes.append("AutoDock-GPU requires autogrid4 in PATH for grid map generation.")
    return PreparedModelRun("autodock-gpu", input_path, output_dir, command, notes)


def prepare_protenix_dock(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> PreparedModelRun | None:
    if not config.protenix_dock.enabled:
        return None

    output_dir = run_dir / "outputs" / "protenix_dock"
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_script = run_dir / "scripts" / "run_protenix_dock.py"
    notes: list[str] = []

    ligand_count = sum(1 for entry in common.sequences if _entity(entry)[0] == "ligand")
    if ligand_count == 0:
        notes.append(
            "Protenix-Dock was enabled without any ligand entity in the Boltz YAML input."
        )

    # Auto-detect from docking prep bridge output
    prep = _load_docking_prep_summary(run_dir)
    receptor_pdb = config.protenix_dock.receptor_pdb
    ligand_sdf = config.protenix_dock.ligand_sdf
    center_x, center_y, center_z = config.protenix_dock.center_x, config.protenix_dock.center_y, config.protenix_dock.center_z
    size_x, size_y, size_z = config.protenix_dock.size_x, config.protenix_dock.size_y, config.protenix_dock.size_z

    if prep:
        receptor_pdb = receptor_pdb or prep["receptor_pdb"]
        if prep["ligands"]:
            ligand_sdf = ligand_sdf or prep["ligands"][0]["sdf"]
        bc = prep.get("box_center", [0, 0, 0])
        bs = prep.get("box_size", [20, 20, 20])
        center_x = center_x if center_x is not None else bc[0]
        center_y = center_y if center_y is not None else bc[1]
        center_z = center_z if center_z is not None else bc[2]
        size_x = size_x if size_x is not None else bs[0]
        size_y = size_y if size_y is not None else bs[1]
        size_z = size_z if size_z is not None else bs[2]
        notes.append("Protenix-Dock inputs auto-detected from docking prep bridge output.")
    else:
        notes.append(
            "Protenix-Dock consumes receptor_pdb and ligand_sdf from runner_config. "
            "Use the wrapper pipeline with cofolding+docking stages for auto-preparation."
        )

    result_path = output_dir / "docking_results.json"
    log_path = output_dir / "protenix_dock.log"
    script_lines = [
        "import json",
        "import sys",
        "",
        "from pxdock import ProtenixDock",
        "",
        f"receptor_pdb = {receptor_pdb!r}",
        f"ligand_sdf = {ligand_sdf!r}",
        f"box_center = [{center_x}, {center_y}, {center_z}]",
        f"box_size = [{size_x}, {size_y}, {size_z}]",
        f"output_path = {str(result_path)!r}",
        f"log_path = {str(log_path)!r}",
        f"cache_map_spacing = {config.protenix_dock.cache_map_spacing}",
        f"use_cache_maps = {config.protenix_dock.use_cache_maps}",
        "",
        'print(f"Protenix-Dock: receptor={receptor_pdb}")',
        'print(f"Protenix-Dock: ligand={ligand_sdf}")',
        'print(f"Protenix-Dock: box_center={box_center}, box_size={box_size}")',
        "",
        "dock = ProtenixDock(receptor_pdb)",
        "dock.set_box(box_center, box_size)",
        "",
        "if use_cache_maps:",
        "    cache_dir = dock.generate_cache_maps(spacing=cache_map_spacing)",
        '    print(f"Protenix-Dock: cache maps generated at {cache_dir}")',
        "",
        "results = dock.run_docking(ligand_sdf)",
        "",
        'with open(output_path, "w") as f:',
        "    json.dump(results, f, indent=2)",
        "",
        'print(f"Protenix-Dock: results written to {output_path}")',
    ]
    runner_script.parent.mkdir(parents=True, exist_ok=True)
    runner_script.write_text("\n".join(script_lines) + "\n")

    command = [config.protenix_dock.python_bin, str(runner_script)]
    command.extend(config.protenix_dock.extra_args)
    return PreparedModelRun("protenix-dock", runner_script, output_dir, command, notes)


def _entity(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    entity_type, entity = next(iter(entry.items()))
    return str(entity_type), dict(entity)


def _entity_ids(entity: dict[str, Any]) -> list[str]:
    entity_id = entity.get("id")
    if isinstance(entity_id, list):
        return [str(item) for item in entity_id]
    if isinstance(entity_id, str):
        return [entity_id]
    raise ValueError("Every Boltz YAML entity must define id as a string or list of strings.")


def _build_chain_lookup(common: CommonInput) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entity_index, entry in enumerate(common.sequences, start=1):
        entity_type, entity = _entity(entry)
        ids = _entity_ids(entity)
        for copy_index, chain_id in enumerate(ids, start=1):
            lookup[chain_id] = {
                "entity_index": entity_index,
                "copy_index": copy_index,
                "entity_type": entity_type,
                "ids": ids,
                "entity": entity,
            }
    return lookup


def _protenix_bonds(
    common: CommonInput, chain_lookup: dict[str, dict[str, Any]], notes: list[str]
) -> list[dict[str, Any]]:
    bonds: list[dict[str, Any]] = []
    for constraint in common.constraints:
        if "bond" not in constraint:
            continue
        atom1, atom2 = constraint["bond"]["atom1"], constraint["bond"]["atom2"]
        left = _map_chain_atom(atom1, chain_lookup)
        right = _map_chain_atom(atom2, chain_lookup)
        if _is_smiles_ligand(left) or _is_smiles_ligand(right):
            notes.append(
                f"Protenix dropped bond {atom1} <-> {atom2} because Boltz bond syntax does not provide stable atom mapping for SMILES ligands."
            )
            continue
        bonds.append(
            {
                "left_entity": left["entity_index"],
                "left_copy": left["copy_index"],
                "left_position": left["position"],
                "left_atom": left["atom_name"],
                "right_entity": right["entity_index"],
                "right_copy": right["copy_index"],
                "right_position": right["position"],
                "right_atom": right["atom_name"],
            }
        )
    return bonds


def _alphafold3_bonds(
    common: CommonInput, chain_lookup: dict[str, dict[str, Any]], notes: list[str]
) -> list[list[list[Any]]]:
    bonds: list[list[list[Any]]] = []
    for constraint in common.constraints:
        if "bond" not in constraint:
            continue
        atom1, atom2 = constraint["bond"]["atom1"], constraint["bond"]["atom2"]
        left = _map_chain_atom(atom1, chain_lookup)
        right = _map_chain_atom(atom2, chain_lookup)
        if _is_smiles_ligand(left) or _is_smiles_ligand(right):
            notes.append(
                f"AlphaFold3 dropped bond {atom1} <-> {atom2} because SMILES ligands cannot be addressed by atom name."
            )
            continue
        if left["entity_type"] != "ligand" and right["entity_type"] != "ligand":
            notes.append(
                f"AlphaFold3 dropped bond {atom1} <-> {atom2} because AF3 only supports manual bonds involving ligands."
            )
            continue
        bonds.append(
            [
                [left["chain_id"], left["position"], left["atom_name"]],
                [right["chain_id"], right["position"], right["atom_name"]],
            ]
        )
    return bonds


def _map_chain_atom(atom: list[Any], chain_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(atom, list) or len(atom) != 3:
        raise ValueError(f"Bond atom spec must be [CHAIN_ID, RES_IDX, ATOM_NAME], got {atom!r}")
    chain_id = str(atom[0])
    if chain_id not in chain_lookup:
        raise ValueError(f"Unknown chain id in bond constraint: {chain_id}")
    metadata = chain_lookup[chain_id]
    return {
        "chain_id": chain_id,
        "entity_index": metadata["entity_index"],
        "copy_index": metadata["copy_index"],
        "entity_type": metadata["entity_type"],
        "entity": metadata["entity"],
        "position": int(atom[1]),
        "atom_name": str(atom[2]),
    }


def _to_protenix_ccd(value: str) -> str:
    return value if value.startswith("CCD_") else f"CCD_{value}"


def _needs_boltz_msa_server(common: CommonInput) -> bool:
    for entry in common.sequences:
        entity_type, entity = _entity(entry)
        if entity_type == "protein" and "msa" not in entity:
            return True
    return False


def _is_smiles_ligand(atom: dict[str, Any]) -> bool:
    return atom["entity_type"] == "ligand" and "smiles" in atom["entity"]


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _append_option_eq(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.append(f"{flag}={value}")


def _protein_query_fasta(common: CommonInput) -> list[str]:
    lines: list[str] = []
    seen_headers: set[str] = set()
    for entry in common.sequences:
        entity_type, entity = _entity(entry)
        if entity_type != "protein":
            continue
        sequence = str(entity["sequence"])
        for chain_id in _entity_ids(entity):
            header = f"{common.name}|protein|{chain_id}"
            if header in seen_headers:
                continue
            seen_headers.add(header)
            lines.extend([f">{header}", sequence])
    return lines


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _shell_quote_or_empty(value: str | None) -> str:
    return '""' if not value else shlex.quote(value)


def _render_shell_command(command: list[str], preserve_shell_vars: bool = False) -> str:
    rendered: list[str] = []
    for part in command:
        if preserve_shell_vars and "$" in part:
            rendered.append(part)
        else:
            rendered.append(shlex.quote(part))
    return " ".join(rendered)
