#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ADA_EF_ROOT="${SAGE_ADAEF_ROOT:-${SAGE_ROOT}/experiments_scripts/ada-ef}"
BUILD_DIR="${SAGE_ADAEF_SIMD_BUILD_DIR:-${ADA_EF_ROOT}/build-simd-avx512}"
JOBS="${JOBS:-$(nproc)}"
SIMD_CXX_FLAGS="${SIMD_CXX_FLAGS:--mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt}"
RELEASE_FLAGS="${CMAKE_CXX_FLAGS_RELEASE:--O3 -DNDEBUG ${SIMD_CXX_FLAGS}}"

if [[ ! -x "${ADA_EF_ROOT}/scripts/build.sh" ]]; then
  echo "Ada-EF build script not found: ${ADA_EF_ROOT}/scripts/build.sh" >&2
  exit 1
fi

export BUILD_DIR
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export JOBS

"${ADA_EF_ROOT}/scripts/build.sh" \
  -DCMAKE_CXX_FLAGS_RELEASE="${RELEASE_FLAGS}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Built ${BUILD_DIR}/backend_runner"
