#!/usr/bin/env bash
set -euo pipefail
darth_port_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
darth_python="${DARTH_PYTHON:-${SAGE_PYTHON:-python3}}"
cmake -S "$darth_port_root" -B "$darth_port_root/build" \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$darth_python"
if (( $# == 0 )); then
  set -- benchmark_new
fi
cmake --build "$darth_port_root/build" --parallel "${DARTH_BUILD_JOBS:-4}" --target "$@"
