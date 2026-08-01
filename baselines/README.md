# Baselines

This directory keeps source-only baseline code for DARTH and Ada-EF. Build outputs,
run outputs, caches, indexes, datasets, and generated model artifacts are not
committed here.

## Provenance

The DARTH code was copied from the live target-0.99 session running on
2026-07-28:

```text
PID: 2184416
cwd: /home/kyungmin/vectordb/hnsw-playground/trials_on_fixing_search_process/adaptive_efsearch/papers/ours/final_experiments/08_darth_ada-ef
cmd: /home/kyungmin/anaconda3/envs/hnsw/bin/python3 -u scripts/run_darth_target099_main4.py --run-root /home/kyungmin/vectordb/hnsw-playground/index/darth_m32_efc500_target099_main4_20260728_run02 --datasets agnews,landmark-nomic,cohere,msmarco --threads 24 --train-threads 24
```

Copied DARTH source paths:

```text
baselines/darth/scripts/run_darth_target099_main4.py
baselines/darth/imported/final_implementation_reference/darth/scripts/
baselines/darth/benchmarking-darth/
```

The patched DARTH C++ tree in `baselines/darth/benchmarking-darth/` is the
metric-aware tree used by that session. Generated build directories, model
outputs, and run artifacts are intentionally omitted.

Ada-EF is available in two source copies: the final experiment uses
`experiments_scripts/ada-ef/`, while `baselines/ada-ef/` is kept as a provenance
snapshot. Generated Ada-EF run directories are intentionally omitted.

## Build DARTH

Install the DARTH Python dependencies, then build the HNSW test binary and
local FAISS shared library with AVX-512 flags:

```bash
cd $SAGE_ROOT
python3 -m pip install -r requirements-darth.txt

cd $SAGE_ROOT/final_experiments/07_darth_ada-ef
./scripts/build_darth_simd_avx512.sh
```

The expected binary after build is:

```text
baselines/darth/benchmarking-darth/build-simd-avx512/hnsw-test/hnsw_test
```

DARTH requires the Python `lightgbm` package and LightGBM headers. The build
helper downloads the LightGBM source distribution for headers when needed; in an
offline environment, set `LIGHTGBM_SRC_DIR`, `LIGHTGBM_LIB`, and related
overrides explicitly.

## Build Ada-EF

Ada-EF requires CMake, a C++17 compiler, OpenMP, HDF5 C++ libraries, Eigen3, and
Boost headers. Install the Python wrapper dependencies, then build the standalone
runner used by the target-0.99 comparison:

```bash
cd $SAGE_ROOT
python3 -m pip install -r requirements-adaef.txt

cd $SAGE_ROOT/final_experiments/07_darth_ada-ef
./scripts/build_adaef_simd_avx512.sh
```

The expected binary after build is:

```text
experiments_scripts/ada-ef/build-simd-avx512/backend_runner
```

Set `CONDA_PREFIX` if Eigen3, Boost, or HDF5 live in a conda environment.
