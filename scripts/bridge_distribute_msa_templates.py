#!/usr/bin/env python3
"""Distribute AF3-data-pipeline output to all three cofolding tools.

After AF3 runs in ``--run_data_pipeline=true --run_inference=false``
mode it writes ``{name}_data.json`` under
``<run_dir>/outputs/msa_pipeline/<sanitised>/``. The JSON is the
input JSON augmented per-protein with::

    sequences[*].protein:
      unpairedMsa: "<huge A3M string>"
      pairedMsa:   "<huge A3M string>"
      templates: [
        {mmcifPath: ".../1abc.cif",
         queryIndices: [...], templateIndices: [...]},
        ...
      ]

This script:

1. Extracts ``unpairedMsa`` and ``pairedMsa`` per chain → A3M files
   under ``<run_dir>/outputs/msa_pipeline/shared_msa/{chain_id}_{kind}.a3m``
2. Patches ``inputs/boltz_input.yaml`` so each protein chain points to
   its A3M and gets a ``templates: [{cif: PATH}, ...]`` block (top-level
   ``templates:`` block).
3. Patches ``inputs/protenix_input.json`` so each ``proteinChain`` gets
   ``unpairedMsaPath`` / ``pairedMsaPath`` plus ``templatesPath``
   pointing at a Protenix-format hmmsearch.a3m derived from the same
   chain's unpaired MSA.
4. Replaces ``inputs/alphafold3_input.json`` with the data-pipeline
   output JSON so AF3 inference reads the same MSA + templates set.

The result is that Boltz, Protenix, and AlphaFold3 all see the same
homologs and templates — no ColabFold round-trip and no per-tool
divergence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _find_data_json(msa_pipeline_dir: Path) -> Path | None:
    """Locate the AF3 data-pipeline output. AF3 writes it under
    ``<output_dir>/<sanitised_name>/<sanitised_name>_data.json``; the
    sanitised name is derived from the input JSON's ``name`` field, so
    we glob rather than guess."""
    candidates = sorted(msa_pipeline_dir.glob("*/*_data.json"))
    return candidates[0] if candidates else None


def _write_a3m(content: str | None, path: Path) -> bool:
    if not content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def _per_chain_msa_paths(
    msa_pipeline_dir: Path, chain_id: str, kind: str
) -> Path:
    """Path layout: ``shared_msa/<chain_id>_<kind>.a3m``."""
    return msa_pipeline_dir / "shared_msa" / f"{chain_id}_{kind}.a3m"


def _materialize_inline_mmcif(
    text: str, dump_dir: Path, fallback_stem: str
) -> str | None:
    """Persist an inline mmCIF blob to disk and return the file path.

    AF3's data pipeline emits ``templates[*].mmcif`` as a full mmCIF
    document inline (not a path). Boltz's templates schema wants a
    ``cif: PATH`` and Protenix's ``templatesPath`` similarly expects a
    file backing — so we dump the blob to a deterministic location
    derived from the cif's ``data_<id>`` block when present, else from
    a fallback stem.
    """
    if not text:
        return None
    pdb_id = fallback_stem
    for line in text.splitlines():
        if line.startswith("data_"):
            pdb_id = line[len("data_"):].strip() or fallback_stem
            break
    dump_dir.mkdir(parents=True, exist_ok=True)
    dst = dump_dir / f"{pdb_id}.cif"
    if not dst.exists():
        dst.write_text(text)
    return str(dst)


def _collect_template_cifs(
    af3_chain: dict[str, Any],
    max_templates: int,
    dump_dir: Path,
    chain_id: str,
) -> list[str]:
    """Return on-disk mmCIF paths for the chain's templates.

    AF3 ships templates two ways: either ``mmcifPath`` (a string path,
    used when the data pipeline already wrote the mmCIF to disk and only
    references it) or ``mmcif`` (the entire mmCIF document inline as a
    string). The latter is the default in current AF3 builds. Both are
    handled — inline blobs are materialised under ``dump_dir`` so the
    downstream Boltz / Protenix patches can hand off a real file path.
    """
    templates = af3_chain.get("templates") or []
    cif_paths: list[str] = []
    for idx, t in enumerate(templates[:max_templates]):
        path = t.get("mmcifPath")
        if path:
            cif_paths.append(str(path))
            continue
        inline = t.get("mmcif")
        if isinstance(inline, str) and inline.strip():
            fallback_stem = f"{chain_id}_template_{idx}"
            dumped = _materialize_inline_mmcif(inline, dump_dir, fallback_stem)
            if dumped:
                cif_paths.append(dumped)
    return cif_paths


def patch_boltz_yaml(
    yaml_path: Path,
    af3_data: dict[str, Any],
    msa_pipeline_dir: Path,
    max_templates: int,
) -> None:
    """Inject MSA + templates per chain into the Boltz YAML.

    Boltz wants ``msa: <a3m_path>`` per protein chain, plus an optional
    top-level ``templates:`` list of ``{cif: PATH}`` entries. We turn
    off ``use_msa_server`` since the explicit MSA path takes precedence
    (and we don't want a silent fallback to ColabFold).
    """
    import yaml as pyyaml

    if not yaml_path.exists():
        print(f"[bridge] boltz yaml not found at {yaml_path}; skipping")
        return

    spec = pyyaml.safe_load(yaml_path.read_text()) or {}

    # Map chain id → AF3 chain dict (AF3 input uses an array of single-key
    # entity wrappers — same shape as Boltz's ``sequences`` list).
    af3_chains = {}
    for entry in af3_data.get("sequences", []) or []:
        if isinstance(entry, dict) and "protein" in entry:
            chain = entry["protein"]
            cid = chain.get("id")
            if isinstance(cid, list):
                # Multi-chain expansion: same MSA / templates apply to every
                # listed chain id (Boltz / AF3 both broadcast).
                for c in cid:
                    af3_chains[str(c)] = chain
            elif cid:
                af3_chains[str(cid)] = chain

    # Per-chain MSA injection
    for entry in spec.get("sequences", []) or []:
        if not isinstance(entry, dict) or "protein" not in entry:
            continue
        prot = entry["protein"]
        cid = prot.get("id")
        ids = cid if isinstance(cid, list) else [cid]
        # Each Boltz chain gets the AF3 MSA from the matching chain id.
        # If the YAML chain id lives under a different letter than AF3's
        # (shouldn't happen with our adapter, but be defensive), the
        # patch silently falls back to "empty".
        primary = next((str(c) for c in ids if str(c) in af3_chains), None)
        if primary is None:
            print(f"[bridge] boltz chain {cid}: no matching AF3 chain — leaving msa empty")
            continue
        af3_chain = af3_chains[primary]
        unpaired = af3_chain.get("unpairedMsa")
        if unpaired:
            a3m_path = _per_chain_msa_paths(msa_pipeline_dir, primary, "unpaired")
            _write_a3m(unpaired, a3m_path)
            prot["msa"] = str(a3m_path)
        paired = af3_chain.get("pairedMsa")
        if paired:
            paired_path = _per_chain_msa_paths(msa_pipeline_dir, primary, "paired")
            _write_a3m(paired, paired_path)
            # Boltz schema: paired MSA via a CSV (same key for paired rows).
            # AF3 ships it as a separate A3M and Boltz YAML has no native
            # ``paired`` slot, so we leave the file on disk for callers
            # that want to inspect it but don't wire it back into Boltz —
            # Boltz handles pairing internally from its own A3M when
            # needed. This is a pragmatic limitation of the Boltz YAML
            # surface, not a quality loss for monomeric targets.

    # Top-level templates: collect unique mmcif paths across chains.
    # AF3's data pipeline embeds mmCIF inline; materialise blobs to a
    # shared cif dump so Boltz can address them by path.
    template_dump = msa_pipeline_dir / "shared_msa" / "templates"
    template_cifs: list[str] = []
    seen_paths: set[str] = set()
    for chain_id, chain in af3_chains.items():
        for cif in _collect_template_cifs(
            chain, max_templates, template_dump, chain_id
        ):
            if cif not in seen_paths:
                seen_paths.add(cif)
                template_cifs.append(cif)
    if template_cifs:
        spec["templates"] = [{"cif": cif} for cif in template_cifs]

    # Disable server-side MSA fetch — explicit path wins.
    spec["use_msa_server"] = False

    yaml_path.write_text(pyyaml.safe_dump(spec, sort_keys=False))
    print(
        f"[bridge] patched {yaml_path}: "
        f"{sum(1 for s in spec.get('sequences', []) if isinstance(s, dict) and 'protein' in s)} chains, "
        f"{len(template_cifs)} templates"
    )


def patch_protenix_json(
    json_path: Path,
    af3_data: dict[str, Any],
    msa_pipeline_dir: Path,
    max_templates: int,
) -> None:
    """Inject MSA paths + templatesPath into the Protenix input JSON.

    Protenix expects each protein chain to carry ``unpairedMsaPath``,
    optionally ``pairedMsaPath``, and ``templatesPath`` (path to an
    hmmsearch-format A3M). We reuse the same per-chain unpaired A3M
    that Boltz reads, and synthesise an hmmsearch.a3m by stacking the
    AF3 template list as A3M-style ``>pdb_id_chain`` entries. Protenix
    parses the headers to look up the templates via its own RCSB index.
    """
    if not json_path.exists():
        print(f"[bridge] protenix json not found at {json_path}; skipping")
        return

    data = json.loads(json_path.read_text())

    af3_chains: dict[str, dict[str, Any]] = {}
    for entry in af3_data.get("sequences", []) or []:
        if isinstance(entry, dict) and "protein" in entry:
            chain = entry["protein"]
            cid = chain.get("id")
            if isinstance(cid, list):
                for c in cid:
                    af3_chains[str(c)] = chain
            elif cid:
                af3_chains[str(cid)] = chain

    # Protenix input is a list[dict] (one per fold target). Iterate every
    # target's sequences[].proteinChain and patch.
    targets = data if isinstance(data, list) else [data]
    n_chains = 0
    n_templates = 0

    for target in targets:
        seqs = target.get("sequences") or []
        # Protenix proteinChain entries do not carry an explicit chain
        # id (the chain id is implicit in the input order, expanded by
        # ``count``). Walk in declaration order and pair with AF3 chains.
        af3_iter = iter(af3_chains.items())
        for entry in seqs:
            if not isinstance(entry, dict) or "proteinChain" not in entry:
                continue
            chain = entry["proteinChain"]
            try:
                af3_id, af3_chain = next(af3_iter)
            except StopIteration:
                break
            n_chains += 1
            unpaired_path = _per_chain_msa_paths(msa_pipeline_dir, af3_id, "unpaired")
            paired_path = _per_chain_msa_paths(msa_pipeline_dir, af3_id, "paired")
            if unpaired_path.exists():
                chain["unpairedMsaPath"] = str(unpaired_path)
            if paired_path.exists():
                chain["pairedMsaPath"] = str(paired_path)

            # Synthesise hmmsearch.a3m for Protenix from AF3's template list.
            # Protenix parses header lines to extract pdb_id; a minimal
            # form is one ">pdb_chain" header per template plus a single
            # placeholder sequence line referencing the query length.
            cifs = _collect_template_cifs(
                af3_chain,
                max_templates,
                msa_pipeline_dir / "shared_msa" / "templates",
                af3_id,
            )
            if cifs:
                hmmsearch_path = msa_pipeline_dir / "shared_msa" / f"{af3_id}_hmmsearch.a3m"
                hmmsearch_path.parent.mkdir(parents=True, exist_ok=True)
                lines: list[str] = []
                # Query line first (Protenix conventions follow standard
                # A3M where the first record is the query)
                query_seq = af3_chain.get("sequence", "")
                lines.append(f">{af3_id}")
                lines.append(query_seq)
                for cif in cifs:
                    pdb_stem = Path(cif).stem  # e.g. "1abc"
                    lines.append(f">{pdb_stem}_A")
                    lines.append(query_seq)  # placeholder: same length as query
                hmmsearch_path.write_text("\n".join(lines) + "\n")
                chain["templatesPath"] = str(hmmsearch_path)
                n_templates += len(cifs)

    json_path.write_text(json.dumps(data, indent=2))
    print(f"[bridge] patched {json_path}: {n_chains} chains, {n_templates} template entries")


def replace_alphafold3_json(
    af3_input_path: Path, af3_data_json: Path
) -> None:
    """The data-pipeline-produced JSON IS the input AF3 inference wants.
    Just copy it over the prep-time stub so the inference stage reads
    the populated MSA + templates instead of the empty placeholder."""
    if not af3_data_json.exists():
        print(f"[bridge] missing AF3 data json at {af3_data_json}; skipping replace")
        return
    af3_input_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(af3_data_json, af3_input_path)
    print(f"[bridge] replaced {af3_input_path} with data-pipeline output")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-templates", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir
    msa_pipeline_dir = run_dir / "outputs" / "msa_pipeline"
    if not msa_pipeline_dir.exists():
        print(f"[bridge] msa_pipeline dir missing at {msa_pipeline_dir}; nothing to distribute")
        return 0

    data_json_path = _find_data_json(msa_pipeline_dir)
    if data_json_path is None:
        print(f"[bridge] no *_data.json found under {msa_pipeline_dir}; skipping")
        return 0

    af3_data = json.loads(data_json_path.read_text())
    print(f"[bridge] using {data_json_path}")
    print(
        f"[bridge] target name: {af3_data.get('name', '?')} "
        f"with {len(af3_data.get('sequences', []) or [])} entities"
    )

    patch_boltz_yaml(
        run_dir / "inputs" / "boltz_input.yaml",
        af3_data,
        msa_pipeline_dir,
        args.max_templates,
    )
    patch_protenix_json(
        run_dir / "inputs" / "protenix_input.json",
        af3_data,
        msa_pipeline_dir,
        args.max_templates,
    )
    replace_alphafold3_json(
        run_dir / "inputs" / "alphafold3_input.json",
        data_json_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
