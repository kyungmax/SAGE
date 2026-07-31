#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../..}" && pwd)"

export FAISS_OPT_LEVEL="${FAISS_OPT_LEVEL:-AVX512}"
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export FAISS_PYTHON_PATH="${FAISS_PYTHON_PATH:-${REPO_ROOT}/faiss/build_sage_avx512/faiss/python}"

exec "${SAGE_PYTHON:-python3}" "${SCRIPT_DIR}/scripts/run_offline_cost_24t_simd.py" \
  --backend faiss \
  --offline-num-threads 24 \
  --faiss-python-path "${FAISS_PYTHON_PATH}" \
  "$@"
