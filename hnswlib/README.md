# hnswlib with SAGE

This directory contains the header-only hnswlib library, Python bindings, and
SAGE's native HNSW search implementation. Examples, upstream tests, and CI files
are omitted.

From the `sage_icde` root, build the local module with:

```bash
bash scripts/build.sh hnswlib
bash scripts/python.sh -c 'import hnswlib; print(hnswlib.__file__)'
```

To build the Python module from this directory alone, use a C++ compiler with
OpenMP, Python headers, NumPy, pybind11, and setuptools:

```bash
python setup.py build_ext --inplace
python -c 'import hnswlib'
```

For C++ use, include `hnswlib/hnswlib.h`. CMake exports the header-only
`hnswlib::hnswlib` target and can install its headers:

```bash
cmake -S . -B build_headers -DCMAKE_INSTALL_PREFIX=/your/install/prefix
cmake --install build_headers
```

The standalone module supplies native search. SAGE calibration and query policy
are provided by `../experiments_scripts/hnswlib/final_policy.py`, which uses the
shared implementation in `../experiments_scripts/common/final_policy.py`.

Upstream: <https://github.com/nmslib/hnswlib>. See `LICENSE` for licensing.
