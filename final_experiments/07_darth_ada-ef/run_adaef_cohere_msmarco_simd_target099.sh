#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-24}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-24}"

python3 scripts/run_adaef_cohere_msmarco_simd_target099.py "$@"
