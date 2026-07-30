#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_faiss_simd_24t.sh" "$@"
"${SCRIPT_DIR}/run_hnswlib_simd_24t.sh" "$@"
