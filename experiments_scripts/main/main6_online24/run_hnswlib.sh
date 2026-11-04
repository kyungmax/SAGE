#!/usr/bin/env bash
set -euo pipefail
MAIN_RUNTIME_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SAGE_CHECKOUT="$(cd -P -- "$MAIN_RUNTIME_DIR/../.." && pwd)"
exec bash "$SAGE_CHECKOUT/scripts/python.sh" "$MAIN_RUNTIME_DIR/run_main6_online24.py" run-cell --cell hnswlib "$@"
