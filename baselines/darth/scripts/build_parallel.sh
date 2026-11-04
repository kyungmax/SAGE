#!/usr/bin/env bash
set -euo pipefail

# Example:
# LIGHTGBM_INCLUDE_DIR=/path/to/include LIGHTGBM_LIB=/path/to/lib_lightgbm.so \
#   bash scripts/build_parallel.sh
# Configure a separate directory; the upstream checkout contains build artifacts.
darth_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
darth_build_dir="${DARTH_BUILD_DIR:-${darth_source_dir}/build-parallel}"
darth_build_jobs="${DARTH_BUILD_JOBS:-4}"
darth_lightgbm_include="${LIGHTGBM_INCLUDE_DIR:-${darth_source_dir}/third_party/lightgbm/include}"
darth_lightgbm_lib=""
darth_lightgbm_probe_patterns=()
if [[ -n "${LIGHTGBM_LIB+x}" ]]; then
  # An explicitly supplied path is authoritative, including invalid paths.
  darth_lightgbm_lib="${LIGHTGBM_LIB}"
else
  darth_lightgbm_envs=()
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    darth_lightgbm_envs+=("${CONDA_PREFIX}")
  fi
  darth_lightgbm_envs+=("${HOME}/anaconda3/envs/hnsw")
  darth_restore_nullglob=false
  if ! shopt -q nullglob; then
    shopt -s nullglob
    darth_restore_nullglob=true
  fi
  for darth_lightgbm_env in "${darth_lightgbm_envs[@]}"; do
    darth_lightgbm_probe_patterns+=("${darth_lightgbm_env}/lib/python*/site-packages/lightgbm/lib/lib_lightgbm.so")
    darth_lightgbm_candidates=("${darth_lightgbm_env}"/lib/python*/site-packages/lightgbm/lib/lib_lightgbm.so)
    for darth_lightgbm_candidate in "${darth_lightgbm_candidates[@]}"; do
      if [[ -f "${darth_lightgbm_candidate}" ]]; then
        darth_lightgbm_lib="${darth_lightgbm_candidate}"
        break
      fi
    done
    if [[ -n "${darth_lightgbm_lib}" ]]; then
      break
    fi
  done
  if "${darth_restore_nullglob}"; then
    shopt -u nullglob
  fi
fi

if [[ ! "${darth_build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'DARTH_BUILD_JOBS must be a positive integer\n' >&2
  exit 2
fi
if [[ ! -f "${darth_lightgbm_include}/LightGBM/c_api.h" ]]; then
  printf 'LightGBM header not found: %s/LightGBM/c_api.h\nSet LIGHTGBM_INCLUDE_DIR.\n' "${darth_lightgbm_include}" >&2
  exit 2
fi
if [[ ! -f "${darth_lightgbm_lib}" ]]; then
  if [[ -n "${LIGHTGBM_LIB+x}" ]]; then
    printf 'Explicit LIGHTGBM_LIB does not name a file: %s\n' "${LIGHTGBM_LIB}" >&2
  else
    printf 'LightGBM library not found. Searched:\n' >&2
    printf '  %s\n' "${darth_lightgbm_probe_patterns[@]}" >&2
  fi
  printf 'Set LIGHTGBM_LIB to an existing lib_lightgbm.so.\n' >&2
  exit 2
fi

# Reject source-directory builds and directories holding tracked upstream output.
mkdir -p -- "${darth_build_dir}"
darth_build_dir="$(cd -- "${darth_build_dir}" && pwd -P)"
if [[ "${darth_build_dir}" == "${darth_source_dir}" ]]; then
  printf 'DARTH_BUILD_DIR must be separate from the source directory.\n' >&2
  exit 2
fi
if [[ "${darth_build_dir}" == "${darth_source_dir}/"* ]] &&
    [[ -n "$(git -C "${darth_source_dir}" ls-files -- "${darth_build_dir#"${darth_source_dir}/"}/")" ]]; then
  printf 'DARTH_BUILD_DIR contains tracked upstream files; select a fresh directory.\n' >&2
  exit 2
fi

cmake -S "${darth_source_dir}" -B "${darth_build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=generic \
  -DBUILD_SHARED_LIBS=ON \
  -DLIGHTGBM_INCLUDE_DIR="${darth_lightgbm_include}" \
  -DLIGHTGBM_LIB="${darth_lightgbm_lib}" \
  "$@"
cmake --build "${darth_build_dir}" --parallel "${darth_build_jobs}" \
  --target hnsw_test

printf 'Driver: %s/hnsw-test/hnsw_test\n' "${darth_build_dir}"
