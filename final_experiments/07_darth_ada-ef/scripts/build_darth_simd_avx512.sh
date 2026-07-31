#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SAGE_PROJECT_ROOT:-${SCRIPT_DIR}/../../..}" && pwd)"
DARTH_ROOT="${SAGE_DARTH_ROOT:-${PROJECT_ROOT}/baselines/darth/benchmarking-darth}"
BUILD_DIR="${SAGE_DARTH_SIMD_BUILD_ROOT:-${DARTH_ROOT}/build-simd-avx512}"
TMP_ROOT="${TMPDIR:-/tmp}"
LIGHTGBM_VERSION="${LIGHTGBM_VERSION:-4.5.0}"
LIGHTGBM_SDIST_ROOT="${LIGHTGBM_SDIST_ROOT:-${TMP_ROOT}/lightgbm-sdist-${LIGHTGBM_VERSION}}"
LIGHTGBM_SRC_DIR="${LIGHTGBM_SRC_DIR:-${LIGHTGBM_SDIST_ROOT}/lightgbm-${LIGHTGBM_VERSION}}"
LIGHTGBM_TARBALL="${LIGHTGBM_TARBALL:-${TMP_ROOT}/lightgbm-${LIGHTGBM_VERSION}.tar.gz}"
JOBS="${JOBS:-$(nproc)}"
SIMD_CXX_FLAGS="${SIMD_CXX_FLAGS:--mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt}"
RELEASE_FLAGS="${CMAKE_CXX_FLAGS_RELEASE:--O3 -DNDEBUG -DMM_MALLOC=1 ${SIMD_CXX_FLAGS}}"

if [[ ! -f "${DARTH_ROOT}/CMakeLists.txt" ]]; then
  echo "DARTH source root not found: ${DARTH_ROOT}" >&2
  exit 1
fi

LIGHTGBM_PY_ROOT="$(
python3 - <<'PY2'
import os
import lightgbm
print(os.path.dirname(lightgbm.__file__))
PY2
)"
LIGHTGBM_LIB="${LIGHTGBM_LIB:-${LIGHTGBM_PY_ROOT}/lib/lib_lightgbm.so}"
LIGHTGBM_INCLUDE_DIR="${LIGHTGBM_INCLUDE_DIR:-${LIGHTGBM_SRC_DIR}/include}"
LIGHTGBM_EXTRA_INCLUDE_DIRS="${LIGHTGBM_EXTRA_INCLUDE_DIRS:-${LIGHTGBM_SRC_DIR}/external_libs/fast_double_parser/include;${LIGHTGBM_SRC_DIR}/external_libs/fmt/include}"

if [[ ! -f "${LIGHTGBM_LIB}" ]]; then
  echo "LightGBM shared library not found at ${LIGHTGBM_LIB}" >&2
  exit 1
fi

if [[ ! -f "${LIGHTGBM_SRC_DIR}/include/LightGBM/boosting.h" ]]; then
  mkdir -p "${LIGHTGBM_SDIST_ROOT}"
  python3 -m pip download --no-deps --no-binary=:all: "lightgbm==${LIGHTGBM_VERSION}" -d "${TMP_ROOT}"
  tar -xf "${LIGHTGBM_TARBALL}" -C "${LIGHTGBM_SDIST_ROOT}"
fi

if [[ ! -f "${LIGHTGBM_SRC_DIR}/include/LightGBM/boosting.h" ]]; then
  echo "LightGBM headers not found under ${LIGHTGBM_SRC_DIR}/include" >&2
  exit 1
fi

cmake -S "${DARTH_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx512 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="${RELEASE_FLAGS}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLIGHTGBM_INCLUDE_DIR="${LIGHTGBM_INCLUDE_DIR}" \
  -DLIGHTGBM_EXTRA_INCLUDE_DIRS="${LIGHTGBM_EXTRA_INCLUDE_DIRS}" \
  -DLIGHTGBM_LIB="${LIGHTGBM_LIB}"

cmake --build "${BUILD_DIR}" --target hnsw_test -j "${JOBS}"

echo "Built ${BUILD_DIR}/hnsw-test/hnsw_test"
