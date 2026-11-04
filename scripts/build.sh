#!/usr/bin/env bash
# Build local packages without installing them into the active Python environment.
set -euo pipefail
SCRIPT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_ROOT/sage_env.sh"
BUILD_COMPONENT="${1:-all}"
if [[ "$BUILD_COMPONENT" == --help || "$BUILD_COMPONENT" == -h ]]; then
  printf 'Usage: scripts/build.sh [all|faiss|hnswlib|pip|abs|rabitq]\nRaBitQ is optional. all retains the existing four components.\n'
  exit 0
fi
case "$BUILD_COMPONENT" in all|faiss|hnswlib|pip|abs|rabitq) ;; *) exit 2 ;; esac
SAGE_BUILD_JOBS="${SAGE_BUILD_JOBS:-8}"
SAGE_BUILD_PYTHON="$("$SAGE_PYTHON" -c 'import sys; print(sys.executable)')"
if [[ "$BUILD_COMPONENT" == rabitq ]]; then
  cmake -S "$SAGE_ROOT/rabitq" -B "$SAGE_ROOT/rabitq/build_sage" \
    -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$SAGE_BUILD_PYTHON"
  cmake --build "$SAGE_ROOT/rabitq/build_sage" --parallel "$SAGE_BUILD_JOBS"
fi
if [[ "$BUILD_COMPONENT" == all || "$BUILD_COMPONENT" == faiss ]]; then
  case "$SAGE_OPT_LEVEL" in
    generic) SAGE_SWIG_TARGET=swigfaiss ;;
    avx2|avx512) SAGE_SWIG_TARGET="swigfaiss_$SAGE_OPT_LEVEL" ;;
    *) printf 'Supported SAGE_OPT_LEVEL: generic, avx2, avx512\n' >&2; exit 2 ;;
  esac
  SAGE_BUILD_DIR="$SAGE_ROOT/faiss/build_sage_$SAGE_OPT_LEVEL"
  cmake -S "$SAGE_ROOT/faiss" -B "$SAGE_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_MKL=OFF \
    -DFAISS_OPT_LEVEL="$SAGE_OPT_LEVEL" -DPython_EXECUTABLE="$SAGE_BUILD_PYTHON"
  cmake --build "$SAGE_BUILD_DIR" --target "$SAGE_SWIG_TARGET" --parallel "$SAGE_BUILD_JOBS"
  (cd "$SAGE_BUILD_DIR/faiss/python" && "$SAGE_BUILD_PYTHON" setup.py build)
fi
if [[ "$BUILD_COMPONENT" == all || "$BUILD_COMPONENT" == hnswlib ]]; then
  (cd "$SAGE_ROOT/hnswlib" && "$SAGE_BUILD_PYTHON" setup.py build_ext --inplace)
fi
if [[ "$BUILD_COMPONENT" == all || "$BUILD_COMPONENT" == pip ]]; then
  PIP_PYTHON="$SAGE_BUILD_PYTHON" PIP_OPT_LEVEL="$SAGE_OPT_LEVEL" PIP_BUILD_JOBS="$SAGE_BUILD_JOBS" \
    bash "$SAGE_ROOT/baselines/pip/scripts/build.sh"
fi
if [[ "$BUILD_COMPONENT" == all || "$BUILD_COMPONENT" == abs ]]; then
  ABS_PYTHON="$SAGE_BUILD_PYTHON" ABS_OPT_LEVEL="$SAGE_OPT_LEVEL" ABS_BUILD_JOBS="$SAGE_BUILD_JOBS" \
    bash "$SAGE_ROOT/baselines/abs/scripts/build.sh"
fi
