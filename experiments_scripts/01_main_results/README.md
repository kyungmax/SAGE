# 01 — Main results

Run from the repository root after the [root build instructions](../../README.md).
The paper catalog contains GooAQ, GloVe100, Cohere, YouTube15M, MSMARCO-Qwen and
SpaceV. [experiment.json](config/main6_online24/experiment.json) selects CPUs/NUMA,
k and timing;
the default is k10, 24 threads, **one warmup plus five measured calls** per setting.
The paper's additional k100 comparison uses FAISS; see the separate command below.

## Data and indexes

Set `SAGE_DATA_DIR` and `SAGE_INDEX_DIR` to the roots containing the catalog's
relative paths, or supply a custom `--datasets-config`. The catalog needs HDF5
`train/test/neighbors`, graph metric and dimensions, M/efConstruction, and each
backend index path. [datasets.json](config/main6_online24/datasets.json) records
the six datasets' shapes and original graph locations. Optional size/mtime
fingerprints are enforced; otherwise
the current input identities are captured when launching. Graph shape, stored
vectors and metric are checked by the measurement workers.

To build missing graphs, use the local backend builders; for example, GloVe:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py \
  --dataset /path/glove-100-angular.hdf5 --space cosine \
  --index /path/index/glove.faiss --state-dir /path/build-state/glove-faiss \
  --m 32 --ef-construction 500 --threads 24
bash scripts/python.sh experiments_scripts/hnswlib/build_benchmark_index.py \
  --dataset /path/glove-100-angular.hdf5 --space cosine \
  --index /path/index/glove.hnsw --state-dir /path/build-state/glove-hnsw \
  --m 32 --ef-construction 500 --threads 24
```

Repeat for each needed dataset, using `--space l2` for SpaceV. These commands
do not update the catalog. Copy it and replace `dataset_file` with an absolute
`dataset_path`, and each index's `{"file": "..."}` with `{"path": "/absolute/index"}`.
Preserve the selected datasets' order and correct shape/metric/M/efConstruction.
Pass the resulting file as `--datasets-config` to main commands, or `--catalog`
to Ada-EF, the DARTH pipeline and ground-truth preparation. Reuse each backend's
graph across methods.

## Main execution

The default CPU list is `1,3,...,47`, on NUMA node 1. Adjust the config to this
machine's allowed physical cores and NUMA node before running. Main workers
require `numactl`; avoid concurrent timing jobs on the same cores.

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

`--dry-run` prints paths and the execution plan. `--preflight` checks HDF5/graph
metadata, GT width, CPU availability and local binary files, then writes
`runs/01_main_results/preparation/main6_online24/preflight.json`. It does not
load/test native extensions or run ANN, and actual execution does not invoke
preflight automatically.

`run.sh` executes FAISS, hnswlib, PiP and ABS sequentially. It creates
`runs/01_main_results/k10/{faiss,hnswlib,pip,abs}/`; an existing output cell is
never overwritten. Use `--output /path/new-run`, `--config /path/experiment.json`
and `--datasets-config /path/datasets.json` to change those locations.
Select one method with the same entry point and options:

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss --dry-run
bash experiments_scripts/01_main_results/run.sh run-cell --cell hnswlib
bash experiments_scripts/01_main_results/run.sh run-cell --cell pip
bash experiments_scripts/01_main_results/run.sh run-cell --cell abs
```

`run-all` is the default when no command is specified. Run FAISS before PiP/ABS
so their matched-recall target file exists.

FAISS/hnswlib calibration is frozen before loading evaluation queries. PiP/ABS
use the new FAISS SAGE recommendation's exact integer neighbor-hit count as the
target, tune in separate processes, and report achieved recall. Policies,
per-query labels/diagnostics, all repeat timings and summaries are generated
anew. `run_search_main6.py` and `run_baselines_main6.py` implement these workers.
The separate `run_full_search.py` worker is used by the embedding-quality
experiment in `05_robustness`: it separates calibration and online execution,
measures the recommended point first, then prepares the remaining curve.

| Output below `runs/01_main_results/k10/` | Contents |
| --- | --- |
| `faiss/summary.csv`, `hnswlib/summary.csv` | All measured SAGE/vanilla curve points and recommended-point flags |
| `{faiss,hnswlib}/datasets/DATASET/` | Frozen policy, offline timings, raw online timings, labels and diagnostics |
| `faiss/targets.json` | Exact SAGE target hits, recommended ef and input identities for prior methods |
| `{pip,abs}/tuning/`, `{pip,abs}/online/` | Tuning decisions and separate final measurements; cell-level `summary.csv` aggregates them |

## Ground truth for k100

The original YouTube HDF5 contains only top10 neighbors. Generate exact top100
from the existing FAISS index's flat vector storage before selecting that cell:

```bash
bash scripts/python.sh experiments_scripts/01_main_results/prepare_ground_truth.py \
  --dataset youtube-15M-angular --k 100 \
  --output runs/01_main_results/inputs/youtube_top100 --preflight
bash scripts/python.sh experiments_scripts/01_main_results/prepare_ground_truth.py \
  --dataset youtube-15M-angular --k 100 \
  --output runs/01_main_results/inputs/youtube_top100
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss --k 100 \
  --datasets-config runs/01_main_results/inputs/youtube_top100/datasets.json --preflight
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss --k 100 \
  --datasets-config runs/01_main_results/inputs/youtube_top100/datasets.json
```

The preparation preflight checks metadata and the local FAISS import without
loading the graph. Actual preparation loads the full FAISS graph and performs
an exhaustive search: allow memory for that graph.
It creates a small HDF5 overlay and catalog, reusing the
original train/test arrays through external links. It does not duplicate the
vector corpus. The generated catalog retains all six datasets and graph identities;
keep the original source file in place. Preflight reports
insufficient GT rather than silently excluding a dataset or using top10 at k100.
Main `--k 100` selects k100 only; `--k all` selects both k values. Without
`run-cell --cell faiss`, either option runs all four main cells.

## Ada-EF and DARTH

These prior methods need separate offline artifacts, so they have explicit
entry points. Build their binaries using the [Ada-EF](../../baselines/ada-ef/README.md)
and [DARTH runtime](../../baselines/darth_faiss1141/README.md) instructions first.
Both runners inherit CPU/memory affinity: wrap their commands with the same
`numactl --physcpubind=CPU_LIST --membind=NODE` placement used by main.

Ada-EF uses the completed FAISS target and the catalog's existing hnswlib graph:

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_adaef.py \
  --dataset glove-100-angular \
  --targets runs/01_main_results/k10/faiss/targets.json \
  --output-root runs/01_main_results/adaef --preflight
```

Remove `--preflight` to calibrate and measure. For a custom main catalog, supply
the identical file with `--catalog`. Outputs, including `summary.csv`, are under
`OUTPUT_ROOT/DATASET/k10/target_hits_N/`; offline artifacts default to that run's
`artifacts/`. `--phase offline` / `--phase online` split the stages. Repeat for
the five cosine datasets; SpaceV/L2 is unsupported, as in the paper. Achieved
recall can fall below the requested target and is retained in the output.

DARTH's integrated runner prepares its holdout data, trains the model, derives
prediction intervals, validates them, then measures the full official test set.
Build both `benchmark_new` and `train_darth_corrected` with
`baselines/darth_faiss1141/scripts/build.sh`. The Python environment also needs
`lightgbm` and `scikit-learn`, alongside the root runtime requirements.

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular \
  --targets runs/01_main_results/k10/faiss/targets.json \
  --preflight
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular \
  --targets runs/01_main_results/k10/faiss/targets.json
```

Use the same `--catalog` file as the FAISS run when overriding the default catalog.
The pipeline supports k10 and takes both ef and target recall from that completed
FAISS target manifest. It inherits CPU/NUMA affinity, as described above;
`--threads` defaults to 24. Preflight checks prerequisites without preparing
data, building a graph, training or running searches.

With seed 12345, preparation selects 10,000 learning vectors and 1,000 validation
vectors from the training corpus. It builds an auxiliary graph excluding both
splits and all exactly equivalent vectors, and computes exact GT in that graph's
ID space. The final test uses the unchanged full FAISS graph, raw official test
queries and original top10 neighbor IDs.

Training fits all selected observations with the corrected 11-feature order and
LightGBM's 100-tree regression recipe. Prediction intervals come from the
learning trace's mean first target-reaching distance count, divided by 2 and 10
with rounding and a minimum of 1. **Validation does not tune intervals or models.**
Both validation and final test use one warmup plus five measured calls and a
separate untimed counter audit. A recall miss is reported without retuning.

Default output is `runs/01_main_results/darth/DATASET`; override it with `--output`.
Use a new directory for a new run. To split the work into stages:

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --output /path/new-darth-run --stop-after prepare
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --output /path/new-darth-run --resume
```

`--stop-after` accepts `prepare`, `train`, `validation` or `test` (default).
`--resume` reuses verified completed stages; it does not treat partial artifacts
as complete. Partial stage outputs require a new output directory. Keep inputs,
options, source files, Python environment and CPU affinity unchanged when resuming.

| Pipeline output | Contents |
| --- | --- |
| `data/prepared.json`, `data/auxiliary.index`, `data/auxiliary_to_original.npy` | Split/graph/GT provenance and auxiliary-to-original ID mapping |
| `data/learn.*`, `data/validation.*`, `data/evaluation_inputs/` | Prepared learning, validation and official evaluation vectors/GT |
| `training/` | `model.txt`, `training.json`, `intervals.json`, trace metadata/logs; raw trace CSV is not retained |
| `evaluation/validation/`, `evaluation/test/` | Input plan, native timings, `summary.csv`, returned labels/distances and audit counters |
| `summary.csv`, `complete.json` | Final test summary and whole-pipeline completion record |

Preparation builds a nearly full-size auxiliary graph and runs exact searches
for both holdout splits. Report preparation overhead with the auxiliary-index
preparation stage excluded: `data/prepared.json`'s `timings.total_seconds` minus
`timings.auxiliary_build_seconds`, plus `training/training.json`'s `total_seconds`.
This retains exact GT generation, trace collection and model training. Keep the
original stage timings separately; the excluded stage also includes streaming
input processing, serialization and reload verification.

Training streams the CSV but keeps **all selected
feature rows in RAM** for model fitting. Allow memory and disk for those steps.
`--learn-queries`, `--validation-queries` and `--seed` can change the defaults
for a bounded validation run; record such changes as departures from the paper
protocol, and use an output directory outside the checkout for validation.

Use the pipeline's root `summary.csv` with `summarize_main.py --summary`.
For [02's DARTH counter import](../02_detailed_analysis/README.md#performance-drill-down),
set `DARTH_RUN=PIPELINE_OUTPUT/evaluation/test`, not the pipeline root.

`run_darth.py` remains the measurement worker for already prepared models and
vecs inputs (`--help` lists the explicit arguments). It accepts a FAISS manifest
with `--targets`, an exact `--target-hits`, or `--target-recall` for a validation
set. The vecs files require exactly the requested number of rows and k GT columns;
use raw queries and the matching graph/ID space. The integrated pipeline supplies
these inputs automatically.

Final DARTH measurement uses the persistent worker-model pool from October30,
includes query preprocessing/output allocation in steady timing, and reports one-time
model initialization separately.

## Matched-recall summary

```bash
bash scripts/python.sh experiments_scripts/01_main_results/summarize_main.py \
  --input runs/01_main_results/k10 \
  --output runs/01_main_results/k10/matched_recall.csv
```

Add repeated `--summary /path/adaef/summary.csv` and
`--summary runs/01_main_results/darth/DATASET/summary.csv` options for separately
measured datasets.
The output file must be new. Prior methods with a FAISS target use that FAISS
SAGE reference; hnswlib SAGE/vanilla is summarized separately.

Vanilla time is read at SAGE's recommended recall on its measured Pareto curve:
an exact point is used when available, otherwise the two enclosing points are
linearly interpolated. Equal-recall points use the fastest time. The CSV records
`comparison` and both bracket ef/recall values; there is no extrapolation.
`unbracketed` rows have no comparison time. Interpolated rows leave measured
`hits` and `target_met` blank, and report the interpolated recall/time explicitly.

PiP/ABS/Ada-EF/DARTH retain measured hits, recall and time, including below-target
results (`target_met=False`, `matched=False`). If multiple points are supplied,
the fastest point meeting the target is selected; otherwise the highest-recall
point is retained. `method_seconds_over_sage` divides the reported method time
by SAGE time; for a below-target method this is a timing ratio, not a
matched-recall speedup. Raw curve CSVs are unchanged. The paper's figure-rendering
script is not included; these CSVs are the inputs for plotting.

## Source

The retained main-six workers are relocated from the repository's former
`main/` directory. Scope and current policy follow
`current_sage_full_sweep_20261028` and `current_sage_k100_20261029`; dataset
bindings are recovered from the completed current-default FAISS/hnswlib runs.
Historical result anchors and fixed binary hashes are replaced by fresh
calibration and current input fingerprints. The main scripts no longer require
removed `experiment_results/reproduction` directories.
DARTH preparation/training follows `darth_holdout_transfer_main6_k10_20261029`;
its final measurement follows `darth_persistent_models_main6_k10_20261030`.
