# Baselines

This directory keeps source-only baseline code for DARTH and Ada-EF. Build outputs,
run outputs, caches, indexes, datasets, and generated model artifacts are not
committed here.

## Provenance

The DARTH code was copied from the live target-0.99 session running on
2026-07-28. Machine-specific source, dataset, index, and Python paths from that
session were intentionally replaced by repository-relative defaults and
environment overrides.

Copied DARTH source paths:

```text
baselines/darth/scripts/run_darth_target099_main4.py
baselines/darth/imported/final_implementation_reference/darth/scripts/
baselines/darth/benchmarking-darth/
```

The patched DARTH C++ tree in `baselines/darth/benchmarking-darth/` is the
metric-aware tree used by that session. Its previous `build-local/` directory was
intentionally omitted.

Ada-EF source was copied from the same session bundle:

```text
baselines/ada-ef/
```

Its previous `runs/` directory was intentionally omitted.

## Build DARTH

Install Python dependencies in the environment used for DARTH reruns:

```bash
python3 -m pip install -r baselines/darth/benchmarking-darth/requirements.txt
```

Build the DARTH HNSW test binary and local FAISS shared library:

```bash
cd final_experiments/07_darth_ada-ef
JOBS=24 ./scripts/build_darth_simd_avx512.sh
```

The expected binary after build is:

```text
baselines/darth/benchmarking-darth/build-simd-avx512/hnsw-test/hnsw_test
```

The build helper locates the installed Python `lightgbm` package and downloads a
matching LightGBM source distribution for headers when needed. Override
`LIGHTGBM_VERSION`, `LIGHTGBM_SRC_DIR`, `LIGHTGBM_LIB`, or
`LIGHTGBM_INCLUDE_DIR` if the local environment differs.

## Build Ada-EF

Ada-EF requires CMake, a C++17 compiler, OpenMP, HDF5 C++ libraries, Eigen3, and
Boost headers. Set `CONDA_PREFIX` when those dependencies live in a specific
conda environment; otherwise the helper uses standard system/CMake search paths.

```bash
cd final_experiments/07_darth_ada-ef
JOBS=24 ./scripts/build_adaef_simd_avx512.sh
```

The expected binary after build is:

```text
experiments_scripts/ada-ef/build-simd-avx512/backend_runner
```
