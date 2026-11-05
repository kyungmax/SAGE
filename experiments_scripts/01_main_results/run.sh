#!/usr/bin/env bash
set -euo pipefail
MAIN_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAGE_CHECKOUT="$(cd -P -- "$MAIN_DIR/../.." && pwd)"
case "${1:-}" in
  run-all|run-cell) ;;
  *) set -- run-all "$@" ;;
esac
exec bash "$SAGE_CHECKOUT/scripts/python.sh" "$MAIN_DIR/run_main6_online24.py" "$@"
