#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sage_env.sh"

mkdir -p "$SAGE_DATA_DIR"
mkdir -p "$SAGE_INDEX_DIR"
mkdir -p "$SAGE_FAISS_INDEX_ROOT"

cat <<EOF
SAGE artifact directories are ready.

  SAGE_PROJECT_ROOT=$SAGE_PROJECT_ROOT
  SAGE_DATA_DIR=$SAGE_DATA_DIR
  SAGE_INDEX_DIR=$SAGE_INDEX_DIR
  SAGE_FAISS_INDEX_ROOT=$SAGE_FAISS_INDEX_ROOT
  FAISS_PYTHON_PATH=$FAISS_PYTHON_PATH

Before running experiments in a new shell, load the same defaults with:

  source "$SAGE_PROJECT_ROOT/sage_env.sh"

Place HDF5 datasets under SAGE_DATA_DIR and prebuilt indexes under SAGE_INDEX_DIR.
EOF
