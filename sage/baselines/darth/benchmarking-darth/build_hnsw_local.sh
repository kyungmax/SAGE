#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
TMP_ROOT="${TMPDIR:-/tmp}"
LIGHTGBM_VERSION="${LIGHTGBM_VERSION:-4.5.0}"
LIGHTGBM_SDIST_ROOT="${LIGHTGBM_SDIST_ROOT:-${TMP_ROOT}/lightgbm-sdist-${LIGHTGBM_VERSION}}"
LIGHTGBM_SRC_DIR="${LIGHTGBM_SRC_DIR:-${LIGHTGBM_SDIST_ROOT}/lightgbm-${LIGHTGBM_VERSION}}"
LIGHTGBM_TARBALL="${LIGHTGBM_TARBALL:-${TMP_ROOT}/lightgbm-${LIGHTGBM_VERSION}.tar.gz}"
JOBS="${JOBS:-4}"

LIGHTGBM_PY_ROOT="$(
python3 - <<'PY'
import os
import lightgbm

print(os.path.dirname(lightgbm.__file__))
PY
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

if [[ ! -f "${LIGHTGBM_SRC_DIR}/external_libs/fast_double_parser/include/fast_double_parser.h" ]]; then
  echo "LightGBM fast_double_parser headers not found under ${LIGHTGBM_SRC_DIR}/external_libs/fast_double_parser/include" >&2
  exit 1
fi

if [[ ! -f "${LIGHTGBM_SRC_DIR}/external_libs/fmt/include/fmt/format.h" ]]; then
  echo "LightGBM fmt headers not found under ${LIGHTGBM_SRC_DIR}/external_libs/fmt/include" >&2
  exit 1
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx512 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLIGHTGBM_INCLUDE_DIR="${LIGHTGBM_INCLUDE_DIR}" \
  -DLIGHTGBM_EXTRA_INCLUDE_DIRS="${LIGHTGBM_EXTRA_INCLUDE_DIRS}" \
  -DLIGHTGBM_LIB="${LIGHTGBM_LIB}"

cmake --build "${BUILD_DIR}" --target hnsw_test -j "${JOBS}"

echo "Built ${BUILD_DIR}/hnsw-test/hnsw_test"
