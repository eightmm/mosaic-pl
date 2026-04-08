# Template Search

Protein template search is split into two independent stages.

- `template-search-sequence`: sequence-based search with `mmseqs easy-search`
- `template-search-structure`: structure-based search with `foldseek easy-search`

## Inputs

Both stages are driven from the shared Boltz-style YAML input plus `runner_config.yaml`.

- Sequence search extracts protein chains from the YAML and writes `template_search_sequence_queries.fasta`.
- Structure search uses either:
  - `template_search_structure.query_structure_path`
  - or `template_search_structure.query_from_cofolding: true`, which resolves a structure at runtime from `outputs/alphafold3`, `outputs/boltz`, or `outputs/protenix` according to `query_model_priority`.

## Local Databases

This repo prepares commands and manifests, but it does not mirror RCSB for you.
Use local RCSB-derived inputs and build search DBs with:

```bash
bash scripts/build_template_search_dbs.sh \
  --sequence-fasta /path/to/rcsb_proteins.fasta \
  --structure-dir /path/to/rcsb_structures \
  --output-root /path/to/template_dbs
```

To refresh them later:

```bash
bash scripts/update_template_search_dbs.sh \
  --sequence-fasta /path/to/rcsb_proteins.fasta \
  --structure-dir /path/to/rcsb_structures \
  --output-root /path/to/template_dbs
```

The resulting paths are typically:

- sequence DB: `/path/to/template_dbs/mmseqs/protein_db`
- structure DB: `/path/to/template_dbs/foldseek/structure_db`

## Wrapper Usage

Example:

```bash
uv run casp17-pl prepare-wrapper \
  --input examples/unified_input.example.yaml \
  --config examples/runner_config.example.yaml \
  --backend slurm \
  --stages cofolding template-search-structure
```

When `query_from_cofolding: true`, `template-search-structure` must run after `cofolding` in the selected stage order.
