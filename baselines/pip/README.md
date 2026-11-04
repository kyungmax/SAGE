# PiP

`faiss_paper095/` is the buildable FAISS source with top-k set-overlap stopping.
The default overlap threshold is 0.95. `pip_runtime.py` exposes `load_faiss`,
`parameters` and `search_with_stats`; import it from `scripts/` in a separate
process from other FAISS builds.

From the `sage_icde` root:

```bash
bash scripts/build.sh pip
bash scripts/python.sh baselines/pip/scripts/benchmark.py --help
bash scripts/python.sh baselines/pip/tests/test_pip.py -q
```

The standalone benchmark accepts an HDF5 dataset with `train`, `test`, and
optional `neighbors`, plus an optional existing HNSW index. Without a dataset
it creates synthetic vectors. Pass a new output directory:

```bash
bash scripts/python.sh baselines/pip/scripts/benchmark.py \
  --dataset /path/dataset.hdf5 --index /path/index.faiss \
  --k 10 --ef 256 --thresholds 0.95 --patiences 64 \
  --threads 24 --output /path/new-run
```

Building needs CMake, C++20, OpenMP, BLAS/LAPACK, SWIG, Python and NumPy.
`PIP_OPT_LEVEL=generic|avx2|avx512` selects the CPU build (default `avx512`).
`PIP_PYTHON`, `PIP_BUILD_DIR` and `PIP_BUILD_JOBS` override the interpreter,
build directory and parallelism. The default local package is
`build/paper095/faiss/python`; no global installation is performed.
