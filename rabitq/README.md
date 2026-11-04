# RaBitQ with SAGE

This directory contains RaBitQ-Library, its bundled headers, Python bindings,
and SAGE's native HNSW search implementation. Examples and upstream tests are
omitted.

From the `sage_icde` root, build the local module with:

```bash
bash scripts/build.sh rabitq
bash scripts/python.sh -c 'import rabitqlib; print(rabitqlib.__file__)'
```

To build from this directory alone, use CMake >=3.19, a C++17 compiler, OpenMP,
Python headers, and pybind11:

```bash
cmake -S . -B build_local -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build_local --parallel 8
PYTHONPATH="$PWD/build_local/python" python -c 'import rabitqlib'
```

The build uses `-march=native`; rebuild for a different CPU or Python ABI.
The standalone module supplies native search. SAGE calibration and query policy
are provided by `../experiments_scripts/rabitq/final_policy.py`, which uses
`../experiments_scripts/common/final_policy.py` and requires `SAGE_POLICY_API >= 2`.
Calibration needs the original training vectors and must be fitted to each index.

See `LICENSE` and the licenses bundled with the dependencies under `include/`.
