#!/usr/bin/env bash
# Source this file from the repository root to use SAGE artifact defaults:
#   source ./sage_env.sh

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _SAGE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _SAGE_ENV_DIR="$(pwd)"
fi

export SAGE_PROJECT_ROOT="${SAGE_PROJECT_ROOT:-${SAGE_ROOT:-$_SAGE_ENV_DIR}}"
# Backward-compatible alias for older local commands.
export SAGE_ROOT="${SAGE_ROOT:-$SAGE_PROJECT_ROOT}"
export SAGE_DATA_DIR="${SAGE_DATA_DIR:-$SAGE_PROJECT_ROOT/datasets}"
export SAGE_INDEX_DIR="${SAGE_INDEX_DIR:-$SAGE_PROJECT_ROOT/index}"
export SAGE_FAISS_INDEX_ROOT="${SAGE_FAISS_INDEX_ROOT:-${FAISS_INDEX_ROOT:-$SAGE_INDEX_DIR/faiss_m32_efc500_main8_20260707/darth/index}}"
export FAISS_INDEX_ROOT="${FAISS_INDEX_ROOT:-$SAGE_FAISS_INDEX_ROOT}"
export FAISS_PYTHON_PATH="${FAISS_PYTHON_PATH:-$SAGE_PROJECT_ROOT/faiss/build_sage_avx512/faiss/python}"
export SAGE_HNSWLIB_EXTENSION_ROOT="${SAGE_HNSWLIB_EXTENSION_ROOT:-$SAGE_PROJECT_ROOT/hnswlib}"
export SAGE_PYTHON="${SAGE_PYTHON:-python3}"

unset _SAGE_ENV_DIR
