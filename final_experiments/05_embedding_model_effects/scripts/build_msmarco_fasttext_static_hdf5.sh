#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../../..}" && pwd)"
DATASET_DIR="${SAGE_DATA_DIR:-${REPO_ROOT}/datasets}"
PYTHON="${SAGE_PYTHON:-python3}"

COLLECTION_TSV="${COLLECTION_TSV:-${DATASET_DIR}/msmarco_passage_glove_static/raw/collection.tsv}"
QUERIES_TSV="${QUERIES_TSV:-${DATASET_DIR}/msmarco_passage_glove_static/raw/queries.dev.tsv}"
FASTTEXT_BIN="${FASTTEXT_BIN:-${DATASET_DIR}/msmarco_passage_fasttext_static/raw/cc.en.300.bin}"
FASTTEXT_LABEL="${FASTTEXT_LABEL:-fasttext-cc300d}"
FASTTEXT_NAME="${FASTTEXT_NAME:-cc.en.300.bin}"
TOKEN_VECTOR_CACHE_SIZE="${TOKEN_VECTOR_CACHE_SIZE:-200000}"
MEAN_OUT="${MEAN_OUT:-${DATASET_DIR}/msmarco-v1-fasttext-cc300d-full-ip.hdf5}"
IDF_CACHE="${IDF_CACHE:-${DATASET_DIR}/msmarco_passage_fasttext_static/fasttext-cc300d_msmarco_collection_token_idf.npz}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
LOG="${LOG:-${LOG_DIR}/build_msmarco_fasttext_static_hdf5_$(date '+%Y%m%d_%H%M%S').log}"

COMMON_ARGS=(
  --sample-size 0
  --sample-mode first
  --collection-tsv "${COLLECTION_TSV}"
  --queries-tsv "${QUERIES_TSV}"
  --fasttext-bin "${FASTTEXT_BIN}"
  --fasttext-label "${FASTTEXT_LABEL}"
  --fasttext-name "${FASTTEXT_NAME}"
  --token-vector-cache-size "${TOKEN_VECTOR_CACHE_SIZE}"
)

if [[ "${DOWNLOAD_ASSETS:-0}" == "1" ]]; then
  COMMON_ARGS+=(--download-assets)
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[START] $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[LOG] ${LOG}"
echo "[PYTHON] ${PYTHON}"
echo "[COLLECTION_TSV] ${COLLECTION_TSV}"
echo "[QUERIES_TSV] ${QUERIES_TSV}"
echo "[FASTTEXT_BIN] ${FASTTEXT_BIN}"
echo "[MEAN_OUT] ${MEAN_OUT}"

"${PYTHON}" "${SCRIPT_DIR}/prepare_msmarco_fasttext_static_hdf5.py" \
  "${COMMON_ARGS[@]}" \
  --pooling mean \
  --idf-cache-npz "${IDF_CACHE}" \
  --output-hdf5 "${MEAN_OUT}" \
  "$@"

echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S %Z')"
