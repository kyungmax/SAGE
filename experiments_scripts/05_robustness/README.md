# Robustness

This directory runs the paper's embedding-quality and RaBitQ experiments with
the current local SAGE implementation: raw CFR over pops 1–16, shadow ef128,
and Top1 next-bucket recovery. Each setting has **five measured runs plus one
separate warmup**. Calibration uses 500 training-only probes from a fixed
10,000-node LID pool; evaluation queries are loaded after the policy is frozen.

| Experiment | Inputs | Implementation |
| --- | --- | --- |
| Embedding quality | The same ordered 10M Cohere/Wikipedia passages and 1,000 held-out queries, embedded with GloVe, Cohere and BGE-M3 | Local FAISS; the three models in the paper's figure |
| RaBitQ | MSMARCO-Qwen, YouTube15M and Cohere | Local 8-bit RaBitQ HNSW, M32/efConstruction500, no exact-vector reranking |

The three RaBitQ datasets are not the embedding-quality model variants.
YouTube uses normalized vectors with L2; MSMARCO and Cohere use normalized
inner product. Each embedding model needs its own exact ground truth, with
identical passage/query row ordering across models.

## Inputs and build

Run from the repository root. The catalogs contain portable relative paths
under `SAGE_DATA_DIR` and `SAGE_INDEX_DIR`. A custom catalog can instead use
`dataset_path` and `faiss_index.path` / `rabitq_index.path`. No historical
results, calibration policies, sibling checkout, or native snapshot is needed.

```bash
export SAGE_DATA_DIR=/path/to/datasets
export SAGE_INDEX_DIR=/path/to/index
SAGE_OPT_LEVEL=avx512 bash scripts/build.sh faiss
bash scripts/build.sh rabitq
```

Provide HDF5 `train`, `test` and `neighbors` arrays. Embedding HDF5 files must
record `ground_truth_method` containing `exact` or `exhaustive`. The supplied
embedding catalog selects the paper's three models from the existing aligned input layout;
embedding generation and exact-ground-truth generation are external input
preparation, not restored experiments. If an HDF5 dataset uses virtual or
external storage, all referenced files must also be present. Preflight checks
their existence and records their identities without loading vector matrices.

If a FAISS index is absent, build each model's graph with the local producer:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py \
  --dataset /path/to/model.hdf5 --index /path/to/model.M32.efC500.index \
  --state-dir runs/05_robustness/build/model \
  --space cosine --m 32 --ef-construction 500 --threads 24
```

For missing RaBitQ graphs, this prepares a build plan and then runs the local
builder. Inspect the plan's CPU/NUMA placement before launching:

```bash
bash scripts/python.sh experiments_scripts/rabitq/build_main_indexes.py prepare \
  --datasets-config experiments_scripts/05_robustness/config/rabitq_datasets.json \
  --datasets msmarco-qwen-1024-normalized,youtube-15M-angular,cohere-768-angular \
  --plan runs/05_robustness/build/rabitq/plan.json \
  --destination "$SAGE_INDEX_DIR/rabitq_b8"
bash scripts/python.sh experiments_scripts/rabitq/build_main_indexes.py launch \
  --plan runs/05_robustness/build/rabitq/plan.json
```

Graph construction is separate from measurement. These full datasets require
substantial RAM and build time; the benchmark itself only loads existing graphs.

## Preflight and execution

Review `config/experiment.json` for CPU placement. Its default uses 24 physical
cores with IDs 1,3,…,47. `--cpus` overrides both the CPU list and offline/online
thread counts. `--numa-node` additionally binds memory through `numactl`.
Preflight checks configuration, HDF5 metadata, dependent files, index existence
and the local native module; it does not build, load graphs, or search.

```bash
bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding --preflight
bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq --preflight

bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding \
  --output runs/05_robustness/embedding_run1
bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq \
  --output runs/05_robustness/rabitq_run1
```

Use `--catalog /path/to/catalog.json`, `--config /path/to/experiment.json`, or
`--datasets name1,name2` for explicit inputs/scope. Output directories must be
new. Workers run sequentially in separate processes and hold `runs/timing.lock`.
RaBitQ preserves the full CPU mask for its native `std::thread` workers.

Embedding execution first freezes calibration, then runs the online curve with
`01_main_results/run_full_search.py`. RaBitQ uses `rabitq/benchmark.py`'s shared
calibration and paired-curve worker. Both retain all five raw timings, per-query
results, measured curves, offline timings, and input/source identities under the
chosen output directory. Vector preprocessing is included in online timing;
index loading, calibration and diagnostics are excluded.

```bash
bash scripts/python.sh experiments_scripts/05_robustness/report.py \
  --run runs/05_robustness/embedding_run1 \
  --output runs/05_robustness/embedding_report1
bash scripts/python.sh experiments_scripts/05_robustness/report.py \
  --run runs/05_robustness/rabitq_run1 \
  --output runs/05_robustness/rabitq_report1
```

The exporter verifies all five timings against each reported mean/deviation and
preserves measured points in `curves.csv`, marking each method's Pareto points.
`matched_recall.csv` compares recommended SAGE recall with the vanilla frontier.
Exact matches are marked `measured`; interpolation between two measured points
is marked `linear_interpolation`. Unbracketed targets are reported without a
speedup; no extrapolation is used. The output can be plotted without retaining
historical CSVs or figures in this repository.

| Location | Output |
| --- | --- |
| Run root | `manifest.json`, `config.json`, `catalog.json`, input/source identities and `complete.json` |
| Run root | `summary.csv`, `offline_times.csv`, `raw_timings.csv` across selected datasets |
| `results/<dataset>/` | Per-query NPZ files, per-setting measurements and diagnostics |
| `calibration/<dataset>/` (embedding) or `results/<dataset>/` (RaBitQ) | Frozen `policy.json` and calibration records |
| Report directory | `curves.csv`, `matched_recall.csv`, `validation.json` |

`batch_mean_seconds` is the mean whole-batch wall time; `amortized_query_ms` is
that time divided by query count, in milliseconds. A report contains fresh
numeric results, not a rendered PDF. The standalone `rabitq/runner.py sweep`
also defaults to five measurements plus one warmup, but retains its median
and preprocessing-excluded timing; use this directory's paired worker for the
paper figure.

## Source provenance and remaining scope

The runnable protocol is restored from
`sage_analysis/embedding_rabitq_refresh_20261028/embedding/{run,prepare_plan}.py`
and `rabitq/{run,prepare_plan}.py`, using the current repository's shared workers.
The curve export follows the measured-point/Pareto conventions in
`sage_analysis/rabitq_top4_plot_csv_20261029/export.py`, narrowed to the three
requested datasets. Machine-specific supervisors, binary hashes, old receipts,
results and native snapshots were not copied. The default embedding catalog
contains only GloVe, Cohere and BGE-M3; additional aligned models require an
explicit custom catalog.

The attached manuscript's adaptive-index-build study is an optional exploratory
extension enclosed in `\comment{...}`, outside the active paper experiments. It describes
200/300/500 construction tiers with a 30/50/20 split. The inspected
`hnsw-playground/topic_search/ada_efc` sources contain later cutoff, reference-CFR
oracle, and graph-refinement experiments; they do not establish that exact
paper configuration as a portable current runner. In particular,
`cfr_cutoff_native.cpp` and related instrumentation require `collect_build_cfr`
from a separate FAISS fork. The current FAISS exposes per-insertion construction
budgets but lacks that build-CFR instrumentation. This experiment is therefore
**not restored as runnable** here; reproducing its reported build-time result
requires identifying the exact original routing source and porting its native
instrumentation. This does not block the active paper's combined embedding and
RaBitQ figure; the source-backed experiments above do not depend on it.
