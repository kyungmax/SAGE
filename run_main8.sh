#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sage_env.sh"

if [[ $# -eq 0 ]]; then
  set -- run-all
fi

exec "$SAGE_PYTHON" "$SAGE_ROOT/final_experiments/01_main_results/run_main8_online24_20260707.py" "$@"
