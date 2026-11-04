#!/usr/bin/env bash
set -euo pipefail
PIP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PIP_PYTHON=${PIP_PYTHON:-python3}
PIP_OPT_LEVEL=${PIP_OPT_LEVEL:-avx512}
PIP_BUILD_FLAVOR=paper095
if [[ "$PIP_OPT_LEVEL" != avx512 ]]; then PIP_BUILD_FLAVOR="paper095_$PIP_OPT_LEVEL"; fi
PIP_BUILD_DIR=${PIP_BUILD_DIR:-"$PIP_ROOT/build/$PIP_BUILD_FLAVOR"}
PIP_BUILD_JOBS=${PIP_BUILD_JOBS:-16}
case "$PIP_OPT_LEVEL" in
    generic) PIP_TARGET=swigfaiss ;;
    avx2|avx512) PIP_TARGET=swigfaiss_$PIP_OPT_LEVEL ;;
    *) printf 'Supported PIP_OPT_LEVEL values: generic, avx2, avx512\n' >&2; exit 1 ;;
esac
PIP_PYTHON_EXECUTABLE=$("$PIP_PYTHON" -c 'import sys; print(sys.executable)')
cmake -S "$PIP_ROOT/faiss_paper095" -B "$PIP_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_MKL=OFF \
    -DFAISS_OPT_LEVEL="$PIP_OPT_LEVEL" \
    -DPython_EXECUTABLE="$PIP_PYTHON_EXECUTABLE" "$@"
cmake --build "$PIP_BUILD_DIR" --target "$PIP_TARGET" --parallel "$PIP_BUILD_JOBS"
# setup.py stages a local importable package; it does not install anything.
(cd "$PIP_BUILD_DIR/faiss/python" && "$PIP_PYTHON_EXECUTABLE" setup.py build)
printf '\nLocal Python package: %s/faiss/python\n' "$PIP_BUILD_DIR"
