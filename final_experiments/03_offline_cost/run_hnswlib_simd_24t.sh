#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../..}" && pwd)"

export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export SAGE_HNSWLIB_EXTENSION_ROOT="${SAGE_HNSWLIB_EXTENSION_ROOT:-${REPO_ROOT}/hnswlib}"

exec "${SAGE_PYTHON:-python3}" "${SCRIPT_DIR}/scripts/run_offline_cost_24t_simd.py" \
  --backend hnswlib \
  --offline-num-threads 24 \
  --hnswlib-extension-root "${SAGE_HNSWLIB_EXTENSION_ROOT}" \
  "$@"
