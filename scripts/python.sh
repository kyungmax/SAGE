#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_ROOT/sage_env.sh"
export PYTHONPATH="$FAISS_PYTHON_PATH:$SAGE_HNSWLIB_EXTENSION_ROOT:$SAGE_ROOT/rabitq/build_sage/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$SAGE_PYTHON" "$@"
