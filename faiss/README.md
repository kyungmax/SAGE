# FAISS with SAGE

This directory contains the FAISS library, Python bindings, and SAGE's native
HNSW search implementation. Examples, benchmarks, upstream tests, and CI files
are omitted. `contrib/` is retained because the Python package includes it.

From the `sage_icde` root, build the local module with:

```bash
bash scripts/build.sh faiss
bash scripts/python.sh -c 'import faiss; print(faiss.__file__)'
```

To build from this directory alone, use a C++20 compiler, CMake >=3.24, OpenMP,
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

The standalone module supplies native search. SAGE calibration and query policy
are provided by `../experiments_scripts/common/final_policy.py`; the FAISS index
adapter is `../experiments_scripts/faiss/faiss_sage_index.py`.

Upstream: <https://github.com/facebookresearch/faiss>. See `LICENSE` and
`THIRD_PARTY_NOTICES` for licensing.
