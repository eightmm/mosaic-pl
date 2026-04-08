#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_template_search_dbs.sh \
    --sequence-fasta /path/to/rcsb_proteins.fasta \
    --structure-dir /path/to/rcsb_structures \
    --output-root /path/to/template_dbs \
    [--mmseqs-bin mmseqs] \
    [--foldseek-bin foldseek]

Build local MMseqs2 and Foldseek databases for protein template search.
The script expects a local RCSB-derived protein FASTA and a directory of mmCIF/PDB files.
EOF
}

MMSEQS_BIN="mmseqs"
FOLDSEEK_BIN="foldseek"
SEQUENCE_FASTA=""
STRUCTURE_DIR=""
OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence-fasta)
      SEQUENCE_FASTA="$2"
      shift 2
      ;;
    --structure-dir)
      STRUCTURE_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --mmseqs-bin)
      MMSEQS_BIN="$2"
      shift 2
      ;;
    --foldseek-bin)
      FOLDSEEK_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SEQUENCE_FASTA" || -z "$STRUCTURE_DIR" || -z "$OUTPUT_ROOT" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$SEQUENCE_FASTA" ]]; then
  echo "sequence FASTA not found: $SEQUENCE_FASTA" >&2
  exit 1
fi
if [[ ! -d "$STRUCTURE_DIR" ]]; then
  echo "structure directory not found: $STRUCTURE_DIR" >&2
  exit 1
fi

SEQ_DB_DIR="$OUTPUT_ROOT/mmseqs"
STRUCT_DB_DIR="$OUTPUT_ROOT/foldseek"
mkdir -p "$SEQ_DB_DIR" "$STRUCT_DB_DIR"

SEQ_DB="$SEQ_DB_DIR/protein_db"
STRUCT_DB="$STRUCT_DB_DIR/structure_db"

echo "Building MMseqs2 database"
"$MMSEQS_BIN" createdb "$SEQUENCE_FASTA" "$SEQ_DB"

echo "Building Foldseek database"
"$FOLDSEEK_BIN" createdb "$STRUCTURE_DIR" "$STRUCT_DB"

cat <<EOF
done=true
template_search_sequence.database_path=$SEQ_DB
template_search_structure.database_path=$STRUCT_DB
EOF
