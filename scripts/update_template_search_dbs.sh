#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  update_template_search_dbs.sh \
    --sequence-fasta /path/to/rcsb_proteins.fasta \
    --structure-dir /path/to/rcsb_structures \
    --output-root /path/to/template_dbs \
    [--mmseqs-bin mmseqs] \
    [--foldseek-bin foldseek]

Refresh local MMseqs2 and Foldseek template-search databases in place.
This rebuilds the databases from the provided local RCSB-derived inputs.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=("$@")

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

"$SCRIPT_DIR/build_template_search_dbs.sh" "${ARGS[@]}"
