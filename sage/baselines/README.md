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
cd baselines/darth/benchmarking-darth
BUILD_DIR=build-local JOBS=24 ./build_hnsw_local.sh
```

The expected binary after build is:

```text
baselines/darth/benchmarking-darth/build-local/hnsw-test/hnsw_test
```

`build_hnsw_local.sh` locates the installed Python `lightgbm` package and uses a
LightGBM source distribution for headers. Override `LIGHTGBM_VERSION`,
`LIGHTGBM_SRC_DIR`, `LIGHTGBM_LIB`, or `LIGHTGBM_INCLUDE_DIR` if the local
environment differs.

## Build Ada-EF

Ada-EF requires CMake, a C++17 compiler, OpenMP, HDF5 C++ libraries, Eigen3, and
Boost headers. The helper script auto-detects common conda env locations when
`CONDA_PREFIX` is unset.

```bash
cd baselines/ada-ef
CONDA_PREFIX=/path/to/adaef ./scripts/build.sh
```

The expected binary after build is:

```text
baselines/ada-ef/build/backend_runner
```
