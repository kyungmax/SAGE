# DARTH training and FAISS 1.14.1 search

This directory builds the corrected DARTH observation collector and the
FAISS 1.14.1 online runner. Online search supports CPU `IndexHNSWFlat`, L2/IP
and bounded candidate queues. Its persistent pool reuses worker models while
each call creates its own query workspaces. Training and inference share the
corrected eleven-feature contract from the completed October 30 experiments.

## Build

Run from the `sage_icde` root, using the Python environment for the experiment:

```bash
export SAGE_PYTHON=/path/to/python
"$SAGE_PYTHON" -m pip install lightgbm scikit-learn numpy pandas h5py
bash baselines/darth_faiss1141/scripts/build.sh
```

Native dependencies are Linux, CMake 3.24+, C++20, OpenMP, BLAS/LAPACK and an
AVX512-capable x86 CPU. The build locates LightGBM in `SAGE_PYTHON` (or the
build-specific `DARTH_PYTHON` override); alternatively set
`LIGHTGBM_LIB=/path/to/lib_lightgbm.so`. Vendored LightGBM headers and their
license are in `third_party/lightgbm/`. Integrated Python training additionally
uses scikit-learn, NumPy, pandas and h5py, plus the repository's FAISS extension.
Follow the [root build instructions](../../README.md) for that extension.

The default build produces both:

- `build/benchmark_new`: online DARTH search on FAISS 1.14.1.
- `build/train_darth_corrected`: original DARTH FAISS 1.8 observation collection,
  linked with the corrected distance-feature implementation.

The collector compiles the retained [../darth](../darth/README.md) source into
`build/training_faiss18/`; it needs no downloaded or frozen FAISS binary. Keep
that source directory when using this build. Corrected executable definitions
replace the original shared library's feature helpers, including the maximum
of negative IP distances. Original collector checkpoints, CSV columns and
formatting remain intact.

`DARTH_BUILD_JOBS` controls the main build's parallelism. The nested training
FAISS build defaults to four jobs, configurable through CMake's
`DARTH_TRAIN_BUILD_JOBS`. To build one executable, pass its target name:

```bash
bash baselines/darth_faiss1141/scripts/build.sh train_darth_corrected
```

## Complete one dataset

[`run_darth_pipeline.py`](../../experiments_scripts/01_main_results/run_darth_pipeline.py)
prepares training/validation inputs, collects observations, fits a model, reports
validation and measures the original test queries. Start after the FAISS main
run has generated its exact SAGE hit target:

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset cohere-768-angular \
  --targets runs/01_main_results/k10/faiss/targets.json \
  --catalog experiments_scripts/01_main_results/config/main6_online24/datasets.json \
  --output runs/01_main_results/darth/cohere-768-angular
```

Use a new output directory and a catalog pointing to your dataset and graph.
The pipeline uses k=10 and defaults to **one separate warmup plus five measured
calls**. It records preparation, training, validation and final measurement
artifacts under the chosen output. Source datasets and production graphs remain
inputs; the preparation stage creates its auxiliary training graph separately.

Training consumes all selected observations from a pipe rather than retaining
a large raw CSV. The eleven named float64 features use rows whose logical
distance count is divisible by the logging interval (default 2). Prediction
intervals come from the **training trace**: take the mean first distance count
reaching the target, then round half and one tenth of that mean, clamping both
to at least one. Final trace observations also participate in finding the first
attainment, even if they are excluded from model fitting. Validation evaluates
the frozen model and intervals; it does not tune them.

## Native collector and online runner

The collector can also write a CSV directly. Supply training queries and exact
neighbors for the same indexed corpus:

```bash
baselines/darth_faiss1141/build/train_darth_corrected \
  --index /path/training.index \
  --query-file /path/learn.fvecs --gt-file /path/learn.groundtruth.ivecs \
  --gt-dist-file /path/learn.groundtruth.fvecs \
  --queries 10000 --k 10 --ef 640 --threads 24 --logging-interval 2 \
  --output /path/new-run/observations.csv
```

Create the output parent first. The collector reads the requested number of
rows and requires at least k ground-truth columns. IP queries are normalized;
L2 queries are unchanged. `--output` also accepts an inherited pipe path. CSV
contains named features and recall labels; stdout ends with JSON describing
query count, work, recall and collection time. This executable collects
observations; the Python pipeline fits LightGBM and derives the intervals.

For an existing model, the lower-level
[`run_darth.py`](../../experiments_scripts/01_main_results/run_darth.py) records
inputs and summarizes online measurements. Its native executable is also usable
directly:

```bash
baselines/darth_faiss1141/build/benchmark_new \
  --index /path/index.faiss --model /path/model.txt \
  --query-file /path/query.fvecs --gt-file /path/truth.ivecs \
  --gt-dist-file /path/truth-distances.fvecs \
  --queries 1000 --k 10 --threads 24 --ef 640 \
  --initial-interval 100 --min-interval 20 --target 0.99 \
  --mode darth --visited auto --normalize 1 --warmups 1 --repeats 5 \
  --model-lifetime persistent --result-prefix /path/new-run/result
```

The online vecs files must contain exactly the specified number of rows, with
exactly k ground-truth columns. Replace the example intervals and target with
those recorded by training. `--normalize 1` normalizes IP queries; L2 is
unchanged. Results are `.ids.i64` and `.distances.f32`, while JSON lines report
timing and recall. `--audit-prefix` adds a separate untimed completed-expansion
counter pass. `--model-lifetime per-call` provides the model-recreation comparison.
Model initialization is reported separately from steady search and combined
with the first warmup for cold-model cost. Audit counters do not alter features
or timed search calls.

## Source

The persistent pool and feature correction come from
`darth_persistent_models_main6_k10_20261030/code`; the restored collector is its
`src/train_darth_corrected.cpp`. Training preparation and trace-to-model rules
follow `darth_holdout_transfer_main6_k10_20261029/scripts`. The observational
expansion counter follows `steps_methods_10pct_darth_holdout_20261030/collector/code`.
These historical directories are provenance, not runtime dependencies.
