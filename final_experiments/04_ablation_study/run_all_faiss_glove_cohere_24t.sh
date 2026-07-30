#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "$#" -ne 0 ]]; then
  echo "run_all_faiss_glove_cohere_24t.sh accepts no arguments; use the individual wrappers for partial runs."
  exit 2
fi

./run_faiss_glove_cohere_ablation_24t.sh
./run_faiss_glove_cohere_pseudogt_24t.sh
