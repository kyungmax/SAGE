#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../..}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/msmarco_embedding_models_faiss_SIMD_on_24t_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${OUT_ROOT}/logs"
LOG="${LOG_DIR}/run_msmarco_embedding_models_faiss_24t.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

export FAISS_OPT_LEVEL="${FAISS_OPT_LEVEL:-AVX512}"
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export FAISS_PYTHON_PATH="${FAISS_PYTHON_PATH:-${REPO_ROOT}/faiss/build_sage_avx512/faiss/python}"

cd "${SCRIPT_DIR}"

echo "[START] $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[OUT_ROOT] ${OUT_ROOT}"
echo "[PYTHON] ${SAGE_PYTHON:-python3}"
echo "[FAISS_OPT_LEVEL] ${FAISS_OPT_LEVEL}"

"${SAGE_PYTHON:-python3}" "${SCRIPT_DIR}/scripts/run_msmarco_embedding_models_faiss_24t.py" \
  --out-root "${OUT_ROOT}" \
  --threads 24 \
  --faiss-python-path "${FAISS_PYTHON_PATH}" \
  "$@"

echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S %Z')"
