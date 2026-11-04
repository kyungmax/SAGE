#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/backend_runner"

if [[ ! -x "${BIN}" ]]; then
  echo "backend_runner is not built: ${BIN}" >&2
  echo "Run: ${ROOT_DIR}/scripts/build.sh" >&2
  exit 1
fi

exec "${BIN}" --experiments-root "${ROOT_DIR}/experiments" "$@"
