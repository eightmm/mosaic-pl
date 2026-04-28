from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from casp17.configs import RunnerConfig
from casp17.io_utils import dump_json, dump_yaml_file
from casp17.models import CommonInput


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


# Boltz affinity module hard cap: larger ligands make prediction crash with
# ``Error: The ligand for affinity is too large``. Kept in sync with
# ``experiments/novel2025_test/build_inputs.py``.
_BOLTZ_AFFINITY_MAX_HEAVY_ATOMS = 128

# Elements Boltz rejects in free-form SMILES at input processing
# (``Molecule is excluded``). These ligands must use a ``ccd:`` entry
# instead, and cannot be used as an affinity binder even when available.
_BOLTZ_EXCLUDED_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "Sn", "Pb", "Bi",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})


def _boltz_affinity_compatible(smiles: str) -> bool:
    """Return True if ``smiles`` is scoreable by Boltz's affinity module.

    False when parsing fails, the molecule contains an excluded metal, or
    the heavy-atom count exceeds the affinity cap. Used to gate the
    auto-inject of the ``properties.affinity`` block in ``prepare_boltz``
    so structure prediction stays alive for heme / cobalamin / large
    glycolipid binders.
    """
    try:
        from rdkit import Chem
        from rdkit.RDLogger import DisableLog
        DisableLog("rdApp.*")
    except Exception:
        return True  # pessimistic default — keep existing behaviour
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return False
    heavy = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        if atom.GetSymbol() in _BOLTZ_EXCLUDED_METALS:
            return False
        heavy += 1
    return 1 < heavy <= _BOLTZ_AFFINITY_MAX_HEAVY_ATOMS


def _build_boltz_command(
    config: RunnerConfig, common: CommonInput, input_path: Path, output_dir: Path, *, use_potentials: bool
) -> list[str]:
    """Build a boltz predict command."""
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
    # If the unified MSA bridge is enabled, the bridge writes ``msa: <a3m>``
    # paths into the Boltz YAML and disables ``use_msa_server`` there. The
    # CLI flag is omitted accordingly so Boltz never falls back to ColabFold
    # silently when the local MSA file path is the one we want it to honour.
    use_msa_server = (_needs_boltz_msa_server(common) or config.boltz.use_msa_server) and not config.msa_pipeline.enabled
    if use_msa_server:
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
    if use_potentials:
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
    return command


def prepare_boltz(common: CommonInput, config: RunnerConfig, run_dir: Path) -> list[PreparedModelRun]:
    """Prepare Boltz runs. Returns both boltz2 and boltz2x if use_potentials is enabled."""
    if not config.boltz.enabled:
        return []

    # Auto-inject affinity properties if ligands present — but only when
    # the primary ligand is actually scoreable by Boltz's affinity module.
    # Skip when the user explicitly left properties empty and the primary
    # ligand is one of:
    #   - a ``ccd:`` entry (Boltz affinity needs SMILES input)
    #   - a SMILES with > 128 heavy atoms (Boltz affinity hard cap — crashes
    #     the whole structure prediction if present with an oversize binder)
    # Dropping the block keeps structure prediction alive for these edge
    # cases (heme cofactors, giant lipid-sugar conjugates, etc.).
    spec = dict(common.spec)
    has_ligand = any(_entity(e)[0] == "ligand" for e in common.sequences)
    if has_ligand and not common.properties:
        first_ligand = next(
            (_entity(e)[1] for e in common.sequences if _entity(e)[0] == "ligand"),
            None,
        )
        primary_smi = (first_ligand or {}).get("smiles") or ""
        if primary_smi and _boltz_affinity_compatible(primary_smi):
            ligand_ids = [_entity_ids(_entity(e)[1]) for e in common.sequences if _entity(e)[0] == "ligand"]
            if ligand_ids:
                spec["properties"] = [{"affinity": {"binder": ligand_ids[0][0]}}]

    input_path = run_dir / "inputs" / "boltz_input.yaml"
    dump_yaml_file(spec, input_path)

    runs: list[PreparedModelRun] = []

    # Boltz-2 (without potentials)
    output_dir_b2 = run_dir / "outputs" / "boltz2"
    cmd_b2 = _build_boltz_command(config, common, input_path, output_dir_b2, use_potentials=False)
    runs.append(PreparedModelRun("boltz2", input_path, output_dir_b2, cmd_b2))

    # Boltz-2x (with potentials)
    if config.boltz.use_potentials:
        output_dir_b2x = run_dir / "outputs" / "boltz2x"
        cmd_b2x = _build_boltz_command(config, common, input_path, output_dir_b2x, use_potentials=True)
        runs.append(PreparedModelRun("boltz2x", input_path, output_dir_b2x, cmd_b2x))

    return runs


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
        # Force on when the unified MSA bridge is wired — the bridge writes
        # ``unpairedMsaPath`` into the protein chain at runtime, but adapter
        # prep runs BEFORE the bridge, so the ``use_msa`` heuristic above
        # would otherwise see an empty JSON and emit ``--use_msa false``.
        str(config.protenix.use_msa or use_msa or config.msa_pipeline.enabled).lower(),
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
        # When the unified MSA bridge is on, override server mode to ``none``
        # so Protenix uses the patched ``unpairedMsaPath`` instead of fetching
        # a parallel MSA from the Protenix MSA server.
        ("none" if config.msa_pipeline.enabled else config.protenix.msa_server_mode),
        "--use_template",
        # Same reason as ``--use_msa``: the unified MSA bridge writes
        # ``templatesPath`` into the protein chain at runtime, after this
        # command line is baked in. Force on so Protenix actually reads
        # the patched ``templatesPath``.
        str(config.protenix.use_template or template_applied or config.msa_pipeline.enabled).lower(),
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

    # AF3 requires chain ids to be upper-case letters only (regex: [A-Z]+).
    # Our unified YAML uses conventions like ``L2``/``X2`` for additional
    # ligands/cofactors which AF3 rejects with
    # ``ValueError: IDs must be upper case letters``. Build a stable remap:
    # single-letter upper ids keep their name (so ``A``/``B``/``L`` survive);
    # anything else is replaced by the next unused letter in ``A..Z, AA..ZZ``.
    # The same remap is applied to bondedAtomPairs so covalent links still
    # resolve to the new ids.
    import re
    _AF3_ID_RE = re.compile(r"^[A-Z]+$")

    def _af3_alloc(used: set[str]) -> str:
        import string
        for c in string.ascii_uppercase:
            if c not in used:
                return c
        for a in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                cand = a + b
                if cand not in used:
                    return cand
        raise ValueError("AlphaFold3 adapter ran out of two-letter chain ids")

    used_af3_ids: set[str] = set()
    af3_id_remap: dict[str, str] = {}
    for entry in common.sequences:
        _, entity = _entity(entry)
        for cid in _entity_ids(entity):
            if _AF3_ID_RE.match(cid) and cid not in used_af3_ids:
                used_af3_ids.add(cid)
                af3_id_remap[cid] = cid
    for entry in common.sequences:
        _, entity = _entity(entry)
        for cid in _entity_ids(entity):
            if cid not in af3_id_remap:
                new = _af3_alloc(used_af3_ids)
                used_af3_ids.add(new)
                af3_id_remap[cid] = new
                notes.append(f"AlphaFold3 adapter renamed chain id '{cid}' → '{new}' (AF3 accepts letters only).")

    for entry in common.sequences:
        entity_type, entity = _entity(entry)
        ids = [af3_id_remap[c] for c in _entity_ids(entity)]
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
            # AF3 requires 'templates' field even when empty
            protein_block["templates"] = []
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
    # Translate bond chain ids through the AF3 remap so covalent links stay
    # resolvable after ``L2``/``X2`` → alphabet-letter renames.
    if bonded_atom_pairs and af3_id_remap:
        remapped_pairs = []
        for pair in bonded_atom_pairs:
            new_pair = []
            for atom in pair:
                cid = str(atom[0])
                new_pair.append([af3_id_remap.get(cid, cid), atom[1], atom[2]])
            remapped_pairs.append(new_pair)
        bonded_atom_pairs = remapped_pairs
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

    # AF3 uses JAX which needs explicit CUDA library paths
    repo_root = Path(__file__).resolve().parents[2]
    af3_runner = run_dir / "scripts" / "run_alphafold3.sh"
    af3_runner.parent.mkdir(parents=True, exist_ok=True)
    af3_python = str(repo_root / config.alphafold3.python_bin)
    af3_script = str(repo_root / config.alphafold3.script)
    af3_venv = Path(af3_python).parent.parent
    nv_lib_glob = str(af3_venv / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
    # When the unified MSA pipeline is enabled, the wrapper bridge already
    # ran AF3's data pipeline once and dropped the augmented JSON in place
    # of ``input_path``. Force inference-only here so AF3 doesn't redo
    # jackhmmer + hmmsearch when the populated MSA + templates are right
    # there in the input file.
    af3_run_data = config.alphafold3.run_data_pipeline and not config.msa_pipeline.enabled
    af3_cmd_args = [
        af3_python,
        af3_script,
        f"--json_path={input_path}",
        f"--output_dir={output_dir}",
        f"--run_data_pipeline={'true' if af3_run_data else 'false'}",
        f"--run_inference={'true' if config.alphafold3.run_inference else 'false'}",
    ]
    if config.alphafold3.model_dir:
        af3_cmd_args.append(f"--model_dir={repo_root / config.alphafold3.model_dir}")
    for db_dir in config.alphafold3.db_dirs:
        af3_cmd_args.append(f"--db_dir={db_dir}")
    _append_option_eq(af3_cmd_args, "--jackhmmer_binary_path", config.alphafold3.jackhmmer_binary_path)
    _append_option_eq(af3_cmd_args, "--nhmmer_binary_path", config.alphafold3.nhmmer_binary_path)
    _append_option_eq(af3_cmd_args, "--hmmalign_binary_path", config.alphafold3.hmmalign_binary_path)
    _append_option_eq(af3_cmd_args, "--hmmsearch_binary_path", config.alphafold3.hmmsearch_binary_path)
    _append_option_eq(af3_cmd_args, "--hmmbuild_binary_path", config.alphafold3.hmmbuild_binary_path)
    _append_option_eq(af3_cmd_args, "--jackhmmer_n_cpu", config.alphafold3.jackhmmer_n_cpu)
    _append_option_eq(
        af3_cmd_args, "--jackhmmer_max_parallel_shards", config.alphafold3.jackhmmer_max_parallel_shards
    )
    _append_option_eq(af3_cmd_args, "--nhmmer_n_cpu", config.alphafold3.nhmmer_n_cpu)
    _append_option_eq(
        af3_cmd_args, "--nhmmer_max_parallel_shards", config.alphafold3.nhmmer_max_parallel_shards
    )
    if config.alphafold3.resolve_msa_overlaps is not None:
        af3_cmd_args.append(
            f"--resolve_msa_overlaps={'true' if config.alphafold3.resolve_msa_overlaps else 'false'}"
        )
    _append_option_eq(af3_cmd_args, "--max_template_date", config.alphafold3.max_template_date)
    _append_option_eq(
        af3_cmd_args, "--conformer_max_iterations", config.alphafold3.conformer_max_iterations
    )
    _append_option_eq(
        af3_cmd_args, "--jax_compilation_cache_dir", config.alphafold3.jax_compilation_cache_dir
    )
    _append_option_eq(af3_cmd_args, "--gpu_device", config.alphafold3.gpu_device)
    if config.alphafold3.buckets:
        af3_cmd_args.append("--buckets=" + ",".join(str(item) for item in config.alphafold3.buckets))
    _append_option_eq(
        af3_cmd_args,
        "--flash_attention_implementation",
        config.alphafold3.flash_attention_implementation,
    )
    _append_option_eq(af3_cmd_args, "--num_recycles", config.alphafold3.num_recycles)
    _append_option_eq(
        af3_cmd_args, "--num_diffusion_samples", config.alphafold3.num_diffusion_samples
    )
    _append_option_eq(af3_cmd_args, "--num_seeds", config.alphafold3.num_seeds)
    if config.alphafold3.save_embeddings:
        af3_cmd_args.append("--save_embeddings=true")
    if config.alphafold3.save_distogram:
        af3_cmd_args.append("--save_distogram=true")
    if config.alphafold3.force_output_dir:
        af3_cmd_args.append("--force_output_dir=true")
    if config.alphafold3.compress_large_output_files:
        af3_cmd_args.append("--compress_large_output_files=true")
    af3_cmd_args.extend(config.alphafold3.extra_args)

    # Write a bash wrapper that sets LD_LIBRARY_PATH for JAX CUDA
    af3_runner.write_text(
        "#!/usr/bin/env bash\n"
        "# Auto-generated AF3 runner with CUDA library paths for JAX\n"
        f'for _nv_lib in {nv_lib_glob}; do export LD_LIBRARY_PATH="$_nv_lib:${{LD_LIBRARY_PATH:-}}"; done\n'
        f"exec {' '.join(shlex.quote(a) for a in af3_cmd_args)}\n"
    )
    af3_runner.chmod(0o755)

    command = ["bash", str(af3_runner)]
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


_DOCKING_BOX_SOURCES = (
    "cofolding",
    "swinsite",
    "p2rank",
    # Template-consensus pockets (top-K from spatial cluster of bound-ligand
    # centroids across all mmseqs+foldseek union hits). Each source maps to
    # one cluster centroid; runtime exits cleanly when prep_summary lacks
    # the entry, so targets with fewer credible clusters behave gracefully
    # (most consensus variants quick-exit). Set to 10 because realistic
    # protein structures rarely have more than that many distinct credible
    # binding pockets; ``prepare_docking_inputs`` filters out singleton
    # clusters (n_members<2) before assigning these slots so noise sites
    # never reach the docking variants.
    "template_consensus_1",
    "template_consensus_2",
    "template_consensus_3",
    "template_consensus_4",
    "template_consensus_5",
    "template_consensus_6",
    "template_consensus_7",
    "template_consensus_8",
    "template_consensus_9",
    "template_consensus_10",
)


def prepare_vina(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> list[PreparedModelRun]:
    """Emit one PreparedModelRun per binding-site source.

    Instead of dispatching docking from a single box center (the old
    priority-picked "best" prediction), we create three independent variants —
    ``vina_cofolding``, ``vina_swinsite``, ``vina_p2rank`` — each reading its
    designated center from ``docking_prep_summary.binding_site_predictions``
    at runtime. Variants whose predictor yielded no pocket exit cleanly so
    the pipeline does not fail; downstream post-analysis simply sees fewer
    pose files for that tool key.
    """
    if not config.vina.enabled:
        return []
    return [_prepare_vina_variant(common, config, run_dir, box_source)
            for box_source in _DOCKING_BOX_SOURCES]


def _prepare_vina_variant(
    common: CommonInput, config: RunnerConfig, run_dir: Path, box_source: str
) -> PreparedModelRun:
    model_name = f"vina_{box_source}"
    output_dir = run_dir / "outputs" / model_name
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

    seed = config.vina.seed if config.vina.seed is not None else common.seed

    summary_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    runner_script = run_dir / "scripts" / f"run_{model_name}.py"
    runner_script.parent.mkdir(parents=True, exist_ok=True)
    script_lines = [
        "import json, os, sys",
        "from pathlib import Path",
        "",
        f"BOX_SOURCE = {box_source!r}  # binding-site predictor this variant uses",
        f"receptor_pdbqt = {receptor_pdbqt!r}",
        f"fallback_ligand_pdbqt = {ligand_pdbqt!r}",
        f"center = [{center_x}, {center_y}, {center_z}]",
        f"size = [{size_x}, {size_y}, {size_z}]",
        f"default_out_dir = {str(output_dir)!r}",
        f"exhaustiveness = {config.vina.exhaustiveness}",
        f"n_poses = {config.vina.num_modes}",
        f"energy_range = {config.vina.energy_range}",
        f"seed = {seed}",
        "",
        "# Multi-seed override via environment variable",
        "seed = int(os.environ.get('DOCK_SEED', str(seed)))",
        "seed_out_dir = Path(os.environ.get('DOCK_OUT_DIR', default_out_dir))",
        "seed_out_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "# Runtime auto-detect from docking prep bridge",
        f"summary_path = Path({str(summary_path)!r})",
        "prep_ligands = []",
        "if summary_path.exists():",
        "    prep = json.loads(summary_path.read_text())",
        "    receptor_pdbqt = receptor_pdbqt or prep['receptor_pdbqt']",
        "    prep_ligands = [lig for lig in (prep.get('ligands') or []) if lig.get('pdbqt')]",
        "    # Prefer this variant's dedicated binding-site source over the",
        "    # default box_center. If the source predictor yielded no pocket,",
        "    # exit cleanly so the wrapper moves on to the next variant/tool.",
        "    bs_preds = prep.get('binding_site_predictions') or {}",
        "    source_pred = bs_preds.get(BOX_SOURCE)",
        "    if source_pred and source_pred.get('center'):",
        "        center = list(source_pred['center'])",
        "        if source_pred.get('size'):",
        "            size = list(source_pred['size'])",
        "    else:",
        "        print(f'Vina[{BOX_SOURCE}]: no {BOX_SOURCE} prediction in prep summary — skipping variant.')",
        "        sys.exit(0)",
        "    if size[0] is None:",
        "        size = prep.get('box_size', [20, 20, 20])",
        "",
        "# Dock each ligand of the target separately. Output goes under",
        "# ``{seed_out_dir}/ligand_{id}/docked.pdbqt`` so downstream post-analysis",
        "# can stage per-ligand poses without ambiguity.",
        "if not prep_ligands and fallback_ligand_pdbqt:",
        "    prep_ligands = [{'id': 'L', 'pdbqt': fallback_ligand_pdbqt}]",
        "if not prep_ligands:",
        "    print(f'Vina[{BOX_SOURCE}]: no ligand pdbqt available — nothing to dock.')",
        "    sys.exit(0)",
        "",
        "from vina import Vina",
        "print(f'Vina[{BOX_SOURCE}]: receptor={receptor_pdbqt}')",
        "print(f'Vina[{BOX_SOURCE}]: {len(prep_ligands)} ligand(s) to dock')",
        "print(f'Vina[{BOX_SOURCE}]: center={center}, size={size}, seed={seed}')",
        "",
        "for lig in prep_ligands:",
        "    lig_id = str(lig.get('id') or 'L')",
        "    lig_pdbqt = lig.get('pdbqt')",
        "    if not lig_pdbqt:",
        "        print(f'Vina[{BOX_SOURCE}]: ligand {lig_id} has no pdbqt, skipping.')",
        "        continue",
        "    lig_out_dir = seed_out_dir / f'ligand_{lig_id}'",
        "    lig_out_dir.mkdir(parents=True, exist_ok=True)",
        "    out_path = str(lig_out_dir / 'docked.pdbqt')",
        "    log_path = str(lig_out_dir / 'vina.log')",
        "    print(f'  -> {lig_id}: {lig_pdbqt}')",
        "    try:",
        "        v = Vina(sf_name='vina', seed=seed)",
        "        v.set_receptor(receptor_pdbqt)",
        "        v.set_ligand_from_file(lig_pdbqt)",
        "        v.compute_vina_maps(center=center, box_size=size)",
        "        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)",
        "        v.write_poses(out_path, n_poses=n_poses, overwrite=True)",
        "        with open(log_path, 'w') as f:",
        "            f.write(v.score().__repr__())",
        "        print(f'  -> {lig_id}: wrote {out_path}')",
        "    except Exception as exc:",
        "        print(f'  -> {lig_id}: Vina failed: {exc}')",
        "        continue",
    ]
    runner_script.write_text("\n".join(script_lines) + "\n")

    input_path = runner_script
    command = [".venvs/protenix-dock/bin/python", str(runner_script)]
    return PreparedModelRun(model_name, input_path, output_dir, command, notes)


def prepare_autodock_gpu(
    common: CommonInput, config: RunnerConfig, run_dir: Path
) -> list[PreparedModelRun]:
    """Emit one PreparedModelRun per binding-site source (see ``prepare_vina``)."""
    if not config.autodock_gpu.enabled:
        return []
    return [_prepare_autodock_gpu_variant(common, config, run_dir, box_source)
            for box_source in _DOCKING_BOX_SOURCES]


def _prepare_autodock_gpu_variant(
    common: CommonInput, config: RunnerConfig, run_dir: Path, box_source: str
) -> PreparedModelRun:
    repo_root = Path(__file__).resolve().parents[2]

    model_name = f"autodock-gpu_{box_source}"
    output_dir_name = f"autodock_gpu_{box_source}"  # underscore in dir, hyphen in model_name
    output_dir = run_dir / "outputs" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_dir = run_dir / "inputs" / f"autodock_gpu_grid_{box_source}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    # Pass explicit user config through to the runtime wrapper. None values
    # mean "auto-detect from docking_prep_summary.json at runtime". Any value
    # set here by the user overrides the prep summary at runtime.
    receptor_pdbqt = config.autodock_gpu.receptor_pdbqt
    ligand_pdbqt = config.autodock_gpu.ligand_pdbqt
    center_x, center_y, center_z = (
        config.autodock_gpu.center_x,
        config.autodock_gpu.center_y,
        config.autodock_gpu.center_z,
    )
    size_x, size_y, size_z = (
        config.autodock_gpu.size_x,
        config.autodock_gpu.size_y,
        config.autodock_gpu.size_z,
    )
    notes.append(
        "AutoDock-GPU resolves receptor/ligand/box from docking_prep_summary.json "
        "at runtime (after the docking prep bridge runs). Explicit runner_config "
        "values override the prep summary."
    )

    # GPF is generated at runtime inside the wrapper script (per-seed, with
    # actual ligand atom types parsed from ligand pdbqt and box center loaded
    # from the docking prep summary). No adapter-time GPF is written because
    # the docking prep bridge has not run yet at this point.

    # Shell script that runs autogrid4 then autodock_gpu
    runner_script = run_dir / "scripts" / f"run_{output_dir_name}.py"
    runner_script.parent.mkdir(parents=True, exist_ok=True)
    seed = config.autodock_gpu.seed if config.autodock_gpu.seed is not None else common.seed
    autostop_flag = "1" if config.autodock_gpu.autostop else "0"
    summary_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"

    script_lines = [
        "import json, subprocess, os, sys",
        "from pathlib import Path",
        "",
        f"BOX_SOURCE = {box_source!r}  # binding-site predictor this variant uses",
        f"receptor_pdbqt = {receptor_pdbqt!r}",
        f"fallback_ligand_pdbqt = {ligand_pdbqt!r}",
        f"user_center = [{center_x!r}, {center_y!r}, {center_z!r}]",
        f"user_size = [{size_x!r}, {size_y!r}, {size_z!r}]",
        "center = [0.0, 0.0, 0.0]",
        "size = [22.5, 22.5, 22.5]",
        f"default_out_dir = Path({str(output_dir)!r})",
        f"binary = {str(repo_root / config.autodock_gpu.binary)!r}",
        f"nrun = {config.autodock_gpu.nrun}",
        f"nev = {config.autodock_gpu.nev}",
        f"heuristics = {config.autodock_gpu.heuristics}",
        f"autostop = {autostop_flag}",
        f"seed = {seed}",
        "",
        "# Multi-seed override via environment variable",
        "seed = int(os.environ.get('DOCK_SEED', str(seed)))",
        "seed_out_dir = Path(os.environ.get('DOCK_OUT_DIR', str(default_out_dir)))",
        "seed_out_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "# Runtime auto-detect from docking prep bridge. Prefer runtime prep",
        "# summary over compile-time config because adapter runs before the",
        "# docking prep bridge creates the summary file.",
        f"summary_path = Path({str(summary_path)!r})",
        "prep_ligands = []",
        "if summary_path.exists():",
        "    prep = json.loads(summary_path.read_text())",
        "    receptor_pdbqt = receptor_pdbqt or prep['receptor_pdbqt']",
        "    prep_ligands = [lig for lig in (prep.get('ligands') or []) if lig.get('pdbqt')]",
        "    # Prefer this variant's dedicated binding-site source; skip if missing.",
        "    bs_preds = prep.get('binding_site_predictions') or {}",
        "    source_pred = bs_preds.get(BOX_SOURCE)",
        "    if source_pred and source_pred.get('center'):",
        "        center = list(source_pred['center'])",
        "        if source_pred.get('size'):",
        "            size = list(source_pred['size'])",
        "    else:",
        "        print(f'AutoDock-GPU[{BOX_SOURCE}]: no {BOX_SOURCE} prediction in prep summary — skipping variant.')",
        "        sys.exit(0)",
        "# Explicit user config takes precedence over prep summary",
        "for i, v in enumerate(user_center):",
        "    if v is not None:",
        "        center[i] = v",
        "for i, v in enumerate(user_size):",
        "    if v is not None:",
        "        size[i] = v",
        "",
        "if not prep_ligands and fallback_ligand_pdbqt:",
        "    prep_ligands = [{'id': 'L', 'pdbqt': fallback_ligand_pdbqt}]",
        "if not prep_ligands:",
        "    print(f'AutoDock-GPU[{BOX_SOURCE}]: no ligand pdbqt available — nothing to dock.')",
        "    sys.exit(0)",
        "",
        "# Dynamically derive ligand atom types from ligand pdbqt (last token of ATOM/HETATM lines)",
        "def _parse_lig_types(path):",
        "    types = []",
        "    seen = set()",
        "    with open(path) as fh:",
        "        for line in fh:",
        "            if line.startswith(('ATOM', 'HETATM')):",
        "                tok = line[77:79].strip() if len(line) >= 79 else line.split()[-1].strip()",
        "                if tok and tok not in seen:",
        "                    seen.add(tok)",
        "                    types.append(tok)",
        "    return types",
        "",
        f"autogrid_bin = {str(repo_root / '.local' / 'bin' / 'autogrid4')!r}",
        "",
        "print(f'AutoDock-GPU[{BOX_SOURCE}]: receptor={receptor_pdbqt}')",
        "print(f'AutoDock-GPU[{BOX_SOURCE}]: {len(prep_ligands)} ligand(s) to dock')",
        "",
        "# Dock each ligand separately. Each ligand gets its own grid (ligand",
        "# atom types vary) and its own output subdir so post-analysis can pick",
        "# up per-ligand DLG files.",
        "for lig in prep_ligands:",
        "    lig_id = str(lig.get('id') or 'L')",
        "    lig_pdbqt = lig.get('pdbqt')",
        "    if not lig_pdbqt:",
        "        print(f'  -> {lig_id}: missing pdbqt, skipping.')",
        "        continue",
        "    lig_out_dir = seed_out_dir / f'ligand_{lig_id}'",
        "    lig_out_dir.mkdir(parents=True, exist_ok=True)",
        "    lig_grid_dir = lig_out_dir / 'grid'",
        "    lig_grid_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "    lig_types = _parse_lig_types(lig_pdbqt)",
        "    if not lig_types:",
        "        lig_types = ['A', 'C', 'HD', 'N', 'NA', 'OA', 'SA']",
        "    _rec_base = ['A', 'C', 'HD', 'N', 'NA', 'OA', 'SA']",
        "    rec_types = list(dict.fromkeys(_rec_base + lig_types))",
        "",
        "    npts = [max(1, int(s / 0.375)) for s in size]",
        "    gpf = [",
        "        f'npts {npts[0]} {npts[1]} {npts[2]}',",
        "        f'gridfld receptor.maps.fld',",
        "        f'spacing 0.375',",
        "        f'receptor_types {chr(32).join(rec_types)}',",
        "        f'ligand_types {chr(32).join(lig_types)}',",
        "        f'receptor {receptor_pdbqt}',",
        "        f'gridcenter {center[0]} {center[1]} {center[2]}',",
        "        f'smooth 0.5',",
        "    ]",
        "    for t in lig_types:",
        "        gpf.append(f'map receptor.{t}.map')",
        "    gpf.extend([",
        "        'elecmap receptor.e.map', 'dsolvmap receptor.d.map',",
        "        'dielectric -0.1465',",
        "    ])",
        "    gpf_path = lig_grid_dir / 'receptor.gpf'",
        "    gpf_path.write_text('\\n'.join(gpf) + '\\n')",
        "",
        "    print(f'  -> {lig_id}: ligand={lig_pdbqt}, types={lig_types}')",
        "    try:",
        "        subprocess.run(",
        "            [autogrid_bin, '-p', 'receptor.gpf', '-l', 'autogrid.log'],",
        "            cwd=str(lig_grid_dir), check=True,",
        "        )",
        "        subprocess.run([",
        "            binary, '--ffile', str(lig_grid_dir / 'receptor.maps.fld'),",
        "            '--lfile', lig_pdbqt,",
        "            '--nrun', str(nrun), '--nev', str(nev),",
        "            '--heuristics', str(heuristics), '--autostop', str(autostop),",
        "            '--seed', str(seed),",
        "            '--resnam', str(lig_out_dir / 'docking'),",
        "        ], check=True)",
        "        print(f'  -> {lig_id}: results in {lig_out_dir}')",
        "    except Exception as exc:",
        "        print(f'  -> {lig_id}: AutoDock-GPU failed: {exc}')",
        "        continue",
    ]
    runner_script.write_text("\n".join(script_lines) + "\n")

    input_path = runner_script
    command = [".venvs/protenix-dock/bin/python", str(runner_script)]
    notes.append("AutoDock-GPU requires autogrid4 in PATH for grid map generation.")
    return PreparedModelRun(model_name, input_path, output_dir, command, notes)


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

    log_path = output_dir / "protenix_dock.log"
    summary_path = run_dir / "inputs" / "docking" / "docking_prep_summary.json"
    script_lines = [
        "import json, os, sys",
        "from pathlib import Path",
        "",
        "# Ensure protenix-dock venv bin is in PATH (for pdb4amber, tleap, etc.)",
        "os.environ['PATH'] = os.path.dirname(sys.executable) + ':' + os.environ.get('PATH', '')",
        "",
        "from pxdock import ProtenixDock",
        "",
        f"receptor_pdb = {receptor_pdb!r}",
        f"fallback_ligand_sdf = {ligand_sdf!r}",
        f"box_center = [{center_x}, {center_y}, {center_z}]",
        f"box_size = [{size_x}, {size_y}, {size_z}]",
        f"base_output_dir = Path({str(output_dir)!r})",
        f"log_path = {str(log_path)!r}",
        f"cache_map_spacing = {config.protenix_dock.cache_map_spacing}",
        f"use_cache_maps = {config.protenix_dock.use_cache_maps}",
        "",
        "# Runtime auto-detect from docking prep bridge",
        f"summary_path = Path({str(summary_path)!r})",
        "prep_ligands = []",
        "if summary_path.exists():",
        "    prep = json.loads(summary_path.read_text())",
        "    receptor_pdb = receptor_pdb or prep['receptor_pdb']",
        "    prep_ligands = [lig for lig in (prep.get('ligands') or []) if lig.get('sdf')]",
        "    if box_center[0] is None:",
        "        box_center = prep.get('box_center', [0, 0, 0])",
        "    if box_size[0] is None:",
        "        box_size = prep.get('box_size', [20, 20, 20])",
        "",
        "if not prep_ligands and fallback_ligand_sdf:",
        "    prep_ligands = [{'id': 'L', 'sdf': fallback_ligand_sdf}]",
        "if not prep_ligands:",
        "    print('Protenix-Dock: no ligand sdf available — nothing to dock.')",
        "    sys.exit(0)",
        "",
        "print(f'Protenix-Dock: receptor={receptor_pdb}')",
        "print(f'Protenix-Dock: box_center={box_center}, box_size={box_size}')",
        "print(f'Protenix-Dock: {len(prep_ligands)} ligand(s) to dock')",
        "",
        "dock = ProtenixDock(receptor_pdb)",
        "dock.set_box(box_center, box_size)",
        "",
        "if use_cache_maps:",
        "    cache_dir = dock.generate_cache_maps(spacing=cache_map_spacing)",
        "    print(f'Protenix-Dock: cache maps generated at {cache_dir}')",
        "",
        "# Dock each ligand separately. PxDock writes into its own per-ligand",
        "# output dir; a per-ligand docking_results.json summary lives at",
        "# ``{base_output_dir}/ligand_{id}/docking_results.json`` so downstream",
        "# post-analysis can discover poses by ligand id.",
        "for lig in prep_ligands:",
        "    lig_id = str(lig.get('id') or 'L')",
        "    lig_sdf = lig.get('sdf')",
        "    if not lig_sdf:",
        "        print(f'  -> {lig_id}: missing sdf, skipping.')",
        "        continue",
        "    lig_out_dir = base_output_dir / f'ligand_{lig_id}'",
        "    lig_out_dir.mkdir(parents=True, exist_ok=True)",
        "    print(f'  -> {lig_id}: ligand={lig_sdf}')",
        "    try:",
        "        results = dock.run_docking(lig_sdf, out_dir=str(lig_out_dir))",
        "        import glob",
        "        result_files = glob.glob(str(results) + '/**/*', recursive=True) if isinstance(results, str) else []",
        "        summary = {",
        "            'ligand_id': lig_id,",
        "            'docking_output': str(results),",
        "            'files': [os.path.basename(f) for f in result_files if os.path.isfile(f)],",
        "        }",
        "        (lig_out_dir / 'docking_results.json').write_text(json.dumps(summary, indent=2))",
        "        print(f'  -> {lig_id}: results in {lig_out_dir}')",
        "    except Exception as exc:",
        "        print(f'  -> {lig_id}: Protenix-Dock failed: {exc}')",
        "        continue",
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
