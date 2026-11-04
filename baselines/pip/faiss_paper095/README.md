# FAISS for PIP

This source copy contains the PIP native search changes and FAISS's
library and Python bindings. The enclosing baseline's `README.md` describes
its runner and build entry point.

To build this library directly, use a C++20 compiler, CMake >=3.24, OpenMP,
BLAS/LAPACK, SWIG, and Python with NumPy and setuptools:

```bash
cmake -S . -B build_local -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_ENABLE_PYTHON=ON -DFAISS_OPT_LEVEL=generic \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build_local --target swigfaiss --parallel 8
(cd build_local/faiss/python && python setup.py build)
PYTHONPATH="$PWD/build_local/faiss/python/build/lib" python -c 'import faiss'
```

`contrib/` is part of the Python package. Examples, benchmarks, upstream tests,
and CI files are omitted. See `LICENSE` and `THIRD_PARTY_NOTICES` for licensing.
