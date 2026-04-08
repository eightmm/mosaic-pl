#!/usr/bin/env bash
# =============================================================================
# Build MMseqs2 + Foldseek search databases from local RCSB data
#
# Usage:
#   bash scripts/build_search_dbs.sh
#   # or on SLURM:
#   srun --partition=cpu_only --time=03:00:00 bash scripts/build_search_dbs.sh
#
# Source data:
#   ~/DB/RCSB/processed/clustering/v2/all_protein_seqs.fasta  (85k sequences)
#   ~/DB/RCSB/processed/structures/extracted_structures_v2/    (68k CIFs)
#
# Output:
#   data/search_dbs/sequence/rcsb_seqDB*      (MMseqs2 preindexed)
#   data/search_dbs/structure/rcsb_structDB*   (Foldseek preindexed)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.local/bin:${PATH}"

SEQFASTA="${HOME}/DB/RCSB/processed/seqid_zones_work/target.fasta"  # 488k chains, full RCSB
STRUCT_DIR="${HOME}/DB/RCSB/raw/mmCIF_data"                        # 251k entries, full RCSB (gzipped CIFs)
DB_ROOT="${ROOT_DIR}/data/search_dbs"
THREADS=8

banner() { echo ""; echo "================================================================"; echo "  $1"; echo "================================================================"; }

# ---------------------------------------------------------------------------
# MMseqs2 sequence DB
# ---------------------------------------------------------------------------
banner "Building MMseqs2 sequence database"

SEQ_DB="${DB_ROOT}/sequence/rcsb_seqDB"
mkdir -p "${DB_ROOT}/sequence" "${DB_ROOT}/sequence/tmp"

if [ -f "${SEQ_DB}.dbtype" ]; then
  echo "  Sequence DB already exists, skipping createdb."
else
  echo "  Creating DB from ${SEQFASTA}..."
  mmseqs createdb "${SEQFASTA}" "${SEQ_DB}"
fi

if [ -f "${SEQ_DB}.idx" ]; then
  echo "  Sequence DB already indexed, skipping."
else
  echo "  Creating preindex (this speeds up all future searches)..."
  mmseqs createindex "${SEQ_DB}" "${DB_ROOT}/sequence/tmp" --threads "${THREADS}"
fi

echo "  Done. Sequence DB: ${SEQ_DB}"
echo "  Entries: $(mmseqs result2stats "${SEQ_DB}" "${SEQ_DB}" /dev/null --stat linecount 2>/dev/null || grep -c "^>" "${SEQFASTA}")"

# ---------------------------------------------------------------------------
# Foldseek structure DB
# ---------------------------------------------------------------------------
banner "Building Foldseek structure database"

STRUCT_DB="${DB_ROOT}/structure/rcsb_structDB"
mkdir -p "${DB_ROOT}/structure" "${DB_ROOT}/structure/tmp"

if [ -f "${STRUCT_DB}.dbtype" ]; then
  echo "  Structure DB already exists, skipping createdb."
else
  echo "  Creating DB from ${STRUCT_DIR} (this may take a while)..."
  foldseek createdb "${STRUCT_DIR}" "${STRUCT_DB}" --threads "${THREADS}"
fi

if [ -f "${STRUCT_DB}.idx" ]; then
  echo "  Structure DB already indexed, skipping."
else
  echo "  Creating preindex..."
  foldseek createindex "${STRUCT_DB}" "${DB_ROOT}/structure/tmp" --threads "${THREADS}"
fi

echo "  Done. Structure DB: ${STRUCT_DB}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "Search databases ready"
echo "  Sequence DB:  ${SEQ_DB}"
echo "  Structure DB: ${STRUCT_DB}"
echo ""
echo "  Configure in runner_config.yaml:"
echo "    template_search_sequence:"
echo "      enabled: true"
echo "      database_path: ${SEQ_DB}"
echo "    template_search_structure:"
echo "      enabled: true"
echo "      database_path: ${STRUCT_DB}"
echo ""
du -sh "${DB_ROOT}/sequence" "${DB_ROOT}/structure" 2>/dev/null
