#!/usr/bin/env bash
# Build the vendored FAISS source and ABS adapter entirely inside this baseline.
set -euo pipefail
ABS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ABS_PYTHON=${ABS_PYTHON:-python3}
ABS_OPT_LEVEL=${ABS_OPT_LEVEL:-avx512}
ABS_BUILD_JOBS=${ABS_BUILD_JOBS:-16}
ABS_FAISS_BUILD_DIR="$ABS_ROOT/build/faiss/$ABS_OPT_LEVEL"
case "$ABS_OPT_LEVEL" in
    generic) ABS_TARGET=swigfaiss ;;
    avx2|avx512) ABS_TARGET=swigfaiss_$ABS_OPT_LEVEL ;;
    *) printf 'Supported ABS_OPT_LEVEL values: generic, avx2, avx512\n' >&2; exit 1 ;;
esac
ABS_PYTHON_EXECUTABLE=$("$ABS_PYTHON" -c 'import sys; print(sys.executable)')
cmake -S "$ABS_ROOT/faiss" -B "$ABS_FAISS_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_MKL=OFF \
    -DFAISS_OPT_LEVEL="$ABS_OPT_LEVEL" \
    -DPython_EXECUTABLE="$ABS_PYTHON_EXECUTABLE" "$@"
cmake --build "$ABS_FAISS_BUILD_DIR" --target "$ABS_TARGET" --parallel "$ABS_BUILD_JOBS"
# Stage a local package; do not install into the Python environment.
(cd "$ABS_FAISS_BUILD_DIR/faiss/python" && "$ABS_PYTHON_EXECUTABLE" setup.py build)
"$ABS_PYTHON_EXECUTABLE" "$ABS_ROOT/scripts/build.py" \
    --faiss-source "$ABS_ROOT/faiss" \
    --faiss-python-path "$ABS_FAISS_BUILD_DIR/faiss/python"
printf '\nABS FAISS package: %s/faiss/python\n' "$ABS_FAISS_BUILD_DIR"
