#!/usr/bin/env bash
# Source this file from the repository root to use SAGE artifact defaults:
#   source ./sage_env.sh

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _SAGE_ENV_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _SAGE_ENV_DIR="$(pwd)"
fi

# Select this checkout even after another SAGE version was sourced.
export SAGE_ROOT="$_SAGE_ENV_DIR"
if [[ -f "$SAGE_ROOT/sage.local.env" ]]; then
  source "$SAGE_ROOT/sage.local.env"
fi
export SAGE_OPT_LEVEL="${SAGE_OPT_LEVEL:-avx512}"
export PIP_OPT_LEVEL="${PIP_OPT_LEVEL:-$SAGE_OPT_LEVEL}"
export ABS_OPT_LEVEL="${ABS_OPT_LEVEL:-$SAGE_OPT_LEVEL}"
export SAGE_DATA_DIR="${SAGE_DATA_DIR:-$SAGE_ROOT/datasets}"
export SAGE_INDEX_DIR="${SAGE_INDEX_DIR:-$SAGE_ROOT/index}"
export SAGE_FAISS_INDEX_ROOT="${SAGE_FAISS_INDEX_ROOT:-${FAISS_INDEX_ROOT:-$SAGE_INDEX_DIR/faiss_m32_efc500_main8/index}}"
export FAISS_INDEX_ROOT="${FAISS_INDEX_ROOT:-$SAGE_FAISS_INDEX_ROOT}"
export FAISS_PYTHON_PATH="$SAGE_ROOT/faiss/build_sage_$SAGE_OPT_LEVEL/faiss/python"
export SAGE_HNSWLIB_EXTENSION_ROOT="$SAGE_ROOT/hnswlib"
export HNSW_PLAYGROUND_ROOT="$SAGE_ROOT"
export SAGE_FAISS_CONDA_ENV=""
export SAGE_PYTHON="${SAGE_PYTHON:-python3}"

unset _SAGE_ENV_DIR
