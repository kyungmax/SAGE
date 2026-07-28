# TODO: 10K Learn Split DARTH `ipi` / `mpi`

## Confirmed Setup

- Use `benchmarking-darth` as the runnable DARTH comparison pipeline.
- Keep shared storage:
  - processed datasets: `/home/kyungmin/vectordb/hnsw-playground/datasets/processed/DARTH`
  - indexes: `/home/kyungmin/vectordb/hnsw-playground/index/DARTH`
- `benchmarking-darth/datasets/processed` already points to the shared dataset root.
- `benchmarking-darth/hnsw-index` already points to the shared index root.
- `benchmarking-darth` HNSW runner now supports both `IP` and `L2`, so `gist` and `agnews` can be included in the same pipeline.

## Thread Count Note

- Historical note:
  - the previous paper-side `ours` vs `ada-ef` harness default was `12` threads
  - source: `papers/backend_comparison_harness/run_hnsw_python_backend.py`
  - `--offline-num-threads` default: `12`
  - `--query-num-threads` default: `12`
- Current decision for the next comparison run:
  - use `24` threads for `ours` and `ada-ef` offline/build work
  - this host has `96` logical threads, and `ada-ef` already defaults to `hardware_concurrency() / 4 = 24`
  - keep actual DARTH search/testing single-threaded, because `benchmarking-darth/hnsw_test.cpp` explicitly sets search to `1` thread and only uses multi-threading for index building and offline data generation

## Target Datasets

- `gist-960-euclidean`
- `agnews-mxbai-1024-euclidean`
- `glove-200-angular`
- `nytimes-256-angular`
- `dbpedia-openai-1000k-angular`
- `deep-image-96-angular`
- `msmarco-v1-openai-ada2-1M-ip`

## Fixed DARTH Settings

- `M = 16`
- `efConstruction = 200`
- `efSearch = 2000`
- `k = 10`
- `target recall Rt = 0.95`
- training log mode: `early-stop-training`
- training log `logging-interval = 2`
- thread policy:
  - index build / training-data generation: `24`
  - actual search / evaluation: `1`

## Required Dataset Preparation Rule

- Follow the `benchmarking-darth/notebooks_scripts/utils/organize_datasets.py` logic.
- Build `learn` from the raw HDF5 `train` split, not from the HDF5 `test` split.
- Use:
  - `learn = 10000`
  - `validation = 1000`
- Preserve `query` as the original test set.
- Use the same random sampling convention as the benchmarking script:
  - seed `987`

## Index Reuse Plan

- Reuse existing shared `M16_efC200` indexes where already available:
  - `gist-960-euclidean`
  - `agnews-mxbai-1024-euclidean`
  - `nytimes-256-angular`
  - `dbpedia-openai-1000k-angular`
  - `deep-image-96-angular`
  - `msmarco-v1-openai-ada2-1M-ip`
- Build missing shared index:
  - `glove-200-angular`

## Execution TODO

- [x] Create benchmarking-style `10K learn / 1K validation` processed datasets for all target datasets.
- [x] Make sure each dataset is accessible from the shared processed root with the file layout expected by `benchmarking-darth`.
- [x] Build or reuse `M16_efC200` indexes from the shared index root.
- [ ] Run `early-stop-training` with `query-type=training` and `query-num=10000` for each dataset.
- [ ] Save training logs under the shared root, not under local repo-only directories.
- [ ] Compute `avg_dists_rt` at `Rt=0.95` for each dataset.
- [ ] Derive:
  - `ipi = round(avg_dists_rt / 2)`
  - `mpi = round(avg_dists_rt / 10)`
- [ ] Save final interval JSON outputs in a shared, stable location.

## Output Convention To Use

- training logs:
  - `/home/kyungmin/vectordb/hnsw-playground/index/DARTH/et_training_data/...`
- interval summaries:
  - `/home/kyungmin/vectordb/hnsw-playground/index/DARTH/intervals/...`

## Important Notes

- Do not use the earlier `800/1000/6980` temporary runs for paper comparison.
- For this task, `10K learn` is the authoritative setting.
- `gist` and `agnews` are now allowed in `benchmarking-darth` because the runner supports `L2`.
