# Ada-EF Standalone Runner

This directory vendors the Ada-EF/HNSW headers from `hnsw-ada-ef` and adds a small phase-based runner. The upstream algorithm code under `third_party/hnswlib/` is copied from the authors' public implementation; the runner, artifact layout, JSON output, and default early-statistics length are tuned for our benchmark workflow. It avoids the original `experiments_driver/run.cpp` behavior where all experiments are hardcoded into one executable.

## Build

External dependencies: CMake, a C++17 compiler, OpenMP, HDF5 C++ library, Eigen3, and Boost headers. On this machine, `scripts/build.sh` auto-detects `~/anaconda3/envs/adaef` when `CONDA_PREFIX` is not already set.

```bash
cd baselines/ada-ef
./scripts/build.sh
```

Explicit env:

```bash
CONDA_PREFIX=/home/kyungmin/anaconda3/envs/adaef ./scripts/build.sh
```

The binary is `build/backend_runner`.

## Data Layout

Default input path:

```text
experiments/data/<dataset>.hdf5
```

The HDF5 file should follow the ANN-Benchmarks layout: `train`, `test`, and `neighbors`. For data elsewhere, pass `--data-path /path/file.hdf5` or `--dataset-root /path/to/data-root`.

## Run

Example for `glove-200-angular.hdf5` under `experiments/data/`:

```bash
./build/backend_runner \
  --phase build \
  --dataset glove-200-angular \
  --experiments-root ./experiments \
  --m 32 \
  --ef-construction 500 \
  --k 10 \
  --num-threads 24 \
  --output-json /tmp/adaef_build.json
```

```bash
./build/backend_runner \
  --phase offline \
  --dataset glove-200-angular \
  --experiments-root ./experiments \
  --m 32 \
  --ef-construction 500 \
  --k 10 \
  --expected-recall 0.95 \
  --num-threads 24 \
  --output-json /tmp/adaef_offline.json
```

```bash
./build/backend_runner \
  --phase online \
  --dataset glove-200-angular \
  --experiments-root ./experiments \
  --m 32 \
  --ef-construction 500 \
  --k 10 \
  --expected-recall 0.95 \
  --warmup-runs 1 \
  --measured-runs 5 \
  --num-threads 24 \
  --output-json /tmp/adaef_online.json
```

Add `--per-query-csv <path>` to online to write `dataset,qid,initial_ef,chosen_ef,score,recall,latency_ms`. That extra pass is not included in JSON timing.

## Upstream Compatibility and Tuning

The authors' original experiment driver usually fixes `statics_length` to `1 + 32 + 31 * 32 = 1025`, which is the 2-hop base-layer neighbor count for `M=16`. Our benchmark uses `M=32, efConstruction=500`, so this runner defaults to an M-aware 2-hop value when `--statics-length` is omitted:

```text
statics_length = 1 + 2M + (2M - 1) * 2M
```

For `M=32`, that default is `4097`. This is the main tuned default relative to the upstream driver. To reproduce the upstream fixed setting exactly, pass `--statics-length 1025` explicitly. The chosen value is recorded in JSON as `statics_length`; `statics_length_auto=1` means the M-aware default was used.

## Artifact Naming

Default artifacts:

```text
experiments/index/<dataset>-M<M>-efc-<efc>-parallel.hnsw
experiments/statistics/<dataset>-estimator--k-<k>.bin
experiments/sampling/<dataset>-samplings--k<k>-ef.bin
experiments/estimation_table/<dataset>-ef_adaptor--k<k>-ef.bin
```

For `M=16` with the upstream-compatible default, the adaptor path omits `-sl...`. When `--statics-length` is explicit, or when `M != 16`, the adaptor path records the effective statistics length:

```text
experiments/estimation_table/<dataset>-ef_adaptor--k<k>-sl<statics_length>-ef.bin
```

Use `--force` to rebuild existing artifacts.

## Metric Notes

`--metric auto` maps names containing `ip`, `inner`, or `msmarco` to `ipd`; all other names use `cd`. The upstream Ada-EF estimator supports offline/online for `cd` and `ipd`.
