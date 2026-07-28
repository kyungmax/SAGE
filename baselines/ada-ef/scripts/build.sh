#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  for candidate in "${HOME}/anaconda3/envs/adaef" "${HOME}/miniconda3/envs/adaef"; do
    if [[ -d "${candidate}" ]]; then
      export CONDA_PREFIX="${candidate}"
      break
    fi
  done
fi

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
  if [[ -d "${CONDA_PREFIX}/include/eigen3" ]]; then
    cmake_args+=("-DEIGEN3_INCLUDE_DIR=${CONDA_PREFIX}/include/eigen3")
  fi
  if [[ -d "${CONDA_PREFIX}/include" ]]; then
    cmake_args+=("-DBOOST_INCLUDE_DIR=${CONDA_PREFIX}/include")
  fi
fi

cmake "${cmake_args[@]}" "$@"
cmake --build "${BUILD_DIR}" -j "${JOBS}"
