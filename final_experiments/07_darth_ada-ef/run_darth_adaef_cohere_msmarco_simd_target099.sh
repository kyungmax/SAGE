#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "$#" -ne 0 ]]; then
  echo "run_darth_adaef_cohere_msmarco_simd_target099.sh accepts no arguments; use the individual wrappers for partial runs." >&2
  exit 2
fi

./run_darth_cohere_msmarco_simd_target099.sh
./run_adaef_cohere_msmarco_simd_target099.sh
