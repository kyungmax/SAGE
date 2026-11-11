# ABS

`scripts/native_abs.cpp` implements Algorithm 2 of arXiv:2505.15636v1 against
this directory's FAISS source. `scripts/abs_runtime.py` exposes `ABSSearch`.
Its `search(queries, k, gamma=...)` returns distances, labels and counters.
Gamma is Euclidean-distance slack; this interface does not use `efSearch`.

From the `sage_icde` root:

```bash
bash scripts/build.sh abs
bash scripts/python.sh baselines/abs/tests/test_abs.py -q
bash scripts/python.sh baselines/abs/tests/test_unit_or_zero.py -q
```

For batch measurement and gamma tuning, use the entry points documented in
[root execution guide](../../README.md#run).

Use L2 directly. Inner-product search requires unit-normalized database and
query vectors and `unit_normalized=True`. `scripts/unit_or_zero_runtime.py`
provides the separate unit-or-zero input adapter. Load ABS in its own Python
process so another FAISS binary is not already imported.

Building needs CMake, C++20, OpenMP, BLAS/LAPACK, SWIG, Python and NumPy.
`ABS_OPT_LEVEL=generic|avx2|avx512` selects the CPU build (default `avx512`),
and `ABS_PYTHON` / `ABS_BUILD_JOBS` select the interpreter and parallelism.
The build stages `build/faiss/<opt>/faiss/python`, then creates
`build/libpaper_abs.so` and `build/build.json`. Keep `build.json`: runtime checks
its library, FAISS and source hashes. Rebuild after changing these inputs or
moving the checkout.
