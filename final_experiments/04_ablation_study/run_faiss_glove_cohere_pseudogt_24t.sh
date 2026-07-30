#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${PSEUDOGT_OUT_DIR:-${SCRIPT_DIR}/probe_pseudo_gt_vs_exact_glove_cohere_faiss_p100_ef4096}"
LOG_DIR="${OUT_DIR}/logs"
LOG="${LOG_DIR}/run_faiss_glove_cohere_pseudogt_24t.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

export FAISS_OPT_LEVEL="${FAISS_OPT_LEVEL:-AVX512}"
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export FAISS_PYTHON_PATH="${FAISS_PYTHON_PATH:-/home/kyungmin/vectordb/faiss/build_hnsw_py312_avx512/faiss/python}"

cd "${SCRIPT_DIR}"

echo "[START] $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[OUT_DIR] ${OUT_DIR}"
echo "[PYTHON] ${SAGE_PYTHON:-python3}"
echo "[FAISS_OPT_LEVEL] ${FAISS_OPT_LEVEL}"

"${SAGE_PYTHON:-python3}" "${SCRIPT_DIR}/scripts/compare_probe_pseudo_gt_to_exact_faiss.py" \
  --output-dir "${OUT_DIR}" \
  --num-threads 24 \
  --index-build-threads 24 \
  --allow-system-faiss \
  "$@"

echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S %Z')"
