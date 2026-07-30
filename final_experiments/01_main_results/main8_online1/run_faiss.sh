#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec "${SAGE_PYTHON:-python3}" run_main8_online1.py run-cell --cell faiss "$@"
