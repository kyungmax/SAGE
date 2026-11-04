# DARTH on FAISS 1.14.1

`src/search.cpp` and `include/darth1141/search.h` provide DARTH search for CPU
`IndexHNSWFlat`, L2/IP, and bounded candidate queues. `src/benchmark.cpp` is the
standalone batch runner. The retained search creates worker workspaces and
model clones for each call; it does not implement a persistent model pool.

Build from the `sage_icde` root:

```bash
DARTH_PYTHON=/path/to/python bash baselines/darth_faiss1141/scripts/build.sh
```

Dependencies: CMake 3.24+, C++20, OpenMP, BLAS/LAPACK, AVX512-capable x86 CPU,
and the LightGBM shared library. Install `lightgbm` for the selected Python,
or set `LIGHTGBM_LIB=/path/lib_lightgbm.so`. Vendored LightGBM headers and its
license are under `third_party/lightgbm/`. An existing local
`reference/lib/lib_lightgbm.so` is only a compatibility fallback; a fresh
checkout uses the installed library. `DARTH_BUILD_JOBS` controls parallelism.
The default build target and executable are `build/benchmark_new`.

Supply an existing index, model and query/ground-truth files:

```bash
baselines/darth_faiss1141/build/benchmark_new \
  --index /path/index.faiss --model /path/model.txt \
  --query-file /path/query.fvecs --gt-file /path/truth.ivecs \
  --gt-dist-file /path/truth-distances.fvecs \
  --queries 1000 --k 10 --threads 24 --ef 640 \
  --initial-interval 100 --min-interval 20 --target 0.99 \
  --mode darth --visited auto --normalize 1 --warmups 1 --repeats 4 \
  --result-prefix /path/new-run/result
```

Create the output parent directory first. The files must contain exactly the
specified number of rows, and ground truth must have exactly k columns.
Intervals and model must match the chosen training configuration. IP queries
are normalized when `--normalize 1`; use `0` only when that is intentional.
Results are written as `.ids.i64` / `.distances.f32`, and timing/recall records
are printed as JSON lines. Inputs and output-prefix paths are caller supplied.
An optional `--audit-prefix` adds a separate untimed counter pass.

The original observation-data collection and training tools remain in
[../darth](../darth/README.md). This search library does not train a model.
