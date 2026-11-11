# Start Wide, Cut Early: Self-Calibrating Adaptive HNSW Search

## Overview

This repository contains **SAGE**, a self-calibrating adaptive HNSW search
method built around the *start wide, cut early* strategy. Each query starts
with a wide baseline `efSearch` to protect hard queries, then reduces its
search budget when the runtime Candidate-to-Furthest Ratio (CFR) signal
indicates that a smaller budget is sufficient.

SAGE calibrates from the index itself using LID-stratified probes. It requires
no external query workload, learned model, or rebuild of the existing HNSW
graph. The runtime combines main and shadow CFR observations with a one-time
budget increase when Top1 improves after a reduction.

The repository includes patched **FAISS**, **hnswlib**, and **RaBitQ** search
implementations, shared calibration code, and runners for the paper's main
results, detailed analysis, calibration, parameter sensitivity, and robustness
experiments. FAISS and hnswlib provide the main HNSW comparisons; RaBitQ tests
the policy on quantized HNSW search.

**Repository layout**

- [FAISS](faiss/) -- patched FAISS search engine and Python bindings.
- [hnswlib](hnswlib/) -- patched hnswlib search engine and Python bindings.
- [RaBitQ](rabitq/) -- quantized HNSW implementation and native bindings.
- [Shared policy](experiments_scripts/common/) -- index-only calibration,
  threshold fitting, and runtime policy.
- Backend directories under [experiments_scripts](experiments_scripts/README.md)
  -- adapters, index builders, and standalone RaBitQ calibration/sweep tools.
- [01_main_results](experiments_scripts/01_main_results/README.md)
  -- six-dataset curves and matched-recall baseline comparisons.
- [02_detailed_analysis](experiments_scripts/02_detailed_analysis/README.md)
  -- difficulty drill-down, misclassification, and shadow contribution.
- [03_calibration](experiments_scripts/03_calibration/README.md)
  -- recommendation quality, offline cost, sampling stability, and pseudo-GT validation.
- [04_parameter_sensitivity](experiments_scripts/04_parameter_sensitivity/README.md)
  -- CFR window, tier count, and safety-margin sensitivity.
- [05_robustness](experiments_scripts/05_robustness/README.md)
  -- embedding-quality and RaBitQ experiments.
- [baselines](baselines/README.md) -- PiP, ABS, Ada-EF, and DARTH runtimes.
- [scripts](scripts/README.md) -- native builds, Python launcher, and checks.
- [datasets](datasets/README.md) and the local index directory -- dataset and
  graph locations, initialized by the root setup/environment scripts.

Large inputs and generated results are excluded from Git. Supply datasets and
indexes separately; experiment runners create fresh outputs under `runs/`.

## Quick Start

After installing the [software dependencies](#software-dependencies), run the
following from a fresh checkout:

```bash
cd /path/to/sage_icde
python3 -m venv .venv
source .venv/bin/activate
export SAGE_PYTHON="$PWD/.venv/bin/python"
export SAGE_OPT_LEVEL=avx512

"$SAGE_PYTHON" -m pip install -r requirements.txt
bash setup.sh
source ./sage_env.sh
bash scripts/build.sh all
bash scripts/check_env.sh
```

If you already use a virtual environment or Conda environment, activate it
and set `SAGE_PYTHON` to its Python executable instead of creating `.venv`.

`setup.sh` creates the local data/index directories. `build.sh all` builds
FAISS, hnswlib, PiP, and ABS. Use the separate build targets below for RaBitQ,
Ada-EF, and DARTH.

By default, data and indexes are read from `datasets/` and `index/`. To use
another volume, export `SAGE_DATA_DIR` and `SAGE_INDEX_DIR` before setup. Place
the inputs and configure their paths as described in [Data and Indexes](#data-and-indexes),
then inspect and run the main experiment:

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

The main entry point runs FAISS, hnswlib, PiP, and ABS sequentially. Ada-EF and
DARTH use separate entry points because they require additional preparation.

## Hardware Configuration

The experiment host uses:

- Dell PowerEdge R760.
- Two Intel Xeon Gold 6442Y CPUs, with 48 physical cores in total.
- 1 TiB DRAM.
- Ubuntu 24.04.1 LTS.

The supplied [main configuration](experiments_scripts/01_main_results/config/main6_online24/experiment.json)
uses **24 physical cores on NUMA node 1**, with CPU IDs `1,3,...,47`. Adjust the
CPU list and NUMA node for your machine before running measurements. Keep
timing jobs sequential on the assigned cores.

The paper runners expect AVX-512 builds. Core FAISS, PiP, and ABS builds also
support `generic` and `avx2`; selecting one does not change the supplied paper
runner requirements. Full datasets and graphs, especially SpaceV, require
substantial memory and storage.

## Software Dependencies

Use a Python environment with development headers and a C++20 compiler.
The native builds require CMake 3.24 or later, SWIG, OpenMP, and BLAS/LAPACK.
The paper runners also require `numactl`.

On Ubuntu, install the build dependencies with:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake swig pkg-config \
  python3-dev python3-pip python3-venv \
  libopenblas-dev libomp-dev libhdf5-dev \
  libeigen3-dev libboost-all-dev numactl
```

The root requirements cover NumPy, pandas, h5py, pybind11, setuptools, and
wheel. Install them in the Python environment created in [Quick Start](#quick-start).
Baseline-specific dependencies are documented in the
[baseline guide](baselines/README.md).

Run repository Python commands through `bash scripts/python.sh`. It selects
`SAGE_PYTHON` and adds this checkout's local bindings to the import path. The
patched FAISS tree is required for SAGE's APIs. Rebuild native bindings after
changing machines or Python ABI, and retain the Python/NumPy versions in
experiment records when comparing calibrated thresholds.

## Build

All commands below run from the repository root. `SAGE_BUILD_JOBS` defaults to
8, and `SAGE_OPT_LEVEL` defaults to `avx512`. The optional, Git-ignored
`sage.local.env` file supplies local environment overrides. Keep `SAGE_PYTHON`
set to the environment used to install the requirements.

### FAISS

```bash
export SAGE_OPT_LEVEL=avx512
bash scripts/build.sh faiss
```

The helper builds the patched CPU library and Python bindings under
`faiss/build_sage_avx512/`. `scripts/python.sh` selects that package without
requiring a global installation.

### hnswlib

```bash
bash scripts/build.sh hnswlib
```

This builds the extension in `hnswlib/` for the selected Python interpreter.
The default native build uses the host's instruction set.

### RaBitQ

```bash
bash scripts/build.sh rabitq
```

RaBitQ is a separate target and is not included in `build.sh all`. Its paper
experiments use the paired-curve worker in
[05_robustness](experiments_scripts/05_robustness/README.md).

### Baselines

```bash
bash scripts/build.sh pip
bash scripts/build.sh abs
bash baselines/ada-ef/scripts/build.sh
bash baselines/darth_faiss1141/scripts/build.sh
```

PiP and ABS are already included in `build.sh all`. Ada-EF and DARTH have
additional dependencies and build instructions in their
[baseline READMEs](baselines/README.md). Load the FAISS-based baseline modules
in separate Python processes, as the experiment runners do.

## Data and Indexes

The main experiments use six HDF5 datasets containing `train`, `test`, and
`neighbors` arrays. The table lists the inputs expected by the
[main dataset catalog](experiments_scripts/01_main_results/config/main6_online24/datasets.json).

| Dataset | Vectors | Dim | Metric | Queries | HDF5 filename |
| --- | ---: | ---: | --- | ---: | --- |
| GooAQ | 1,475,024 | 768 | cosine | 1,000 | `gooaq-distilroberta-768-normalized.hdf5` |
| GloVe100 | 1,183,514 | 100 | cosine | 10,000 | `glove-100-angular.hdf5` |
| CohereWiki | 10,000,000 | 768 | cosine | 1,000 | `cohere-768-angular.hdf5` |
| YouTube | 15,686,309 | 1024 | cosine | 1,000 | `youtube-15M-angular.hdf5` |
| MSMARCO-Qwen | 8,840,823 | 1024 | cosine | 1,000 | `msmarco-qwen-1024-normalized.hdf5` |
| SpaceV | 100,000,000 | 100 | L2 | 29,316 | `msspacev-100M-i8-euclidean.hdf5` |

Set external input roots before loading the environment:

```bash
export SAGE_DATA_DIR=/path/to/datasets
export SAGE_INDEX_DIR=/path/to/index
bash setup.sh
source ./sage_env.sh
```

The main catalog resolves dataset filenames under `SAGE_DATA_DIR` and both
backends' index filenames under `SAGE_INDEX_DIR`. It records each graph's
metric, dimension, vector count, `M=32`, and `efConstruction=500`.

Supply compatible prebuilt graphs or build them with the local producers:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py \
  --dataset "$SAGE_DATA_DIR/glove-100-angular.hdf5" --space cosine \
  --index "$SAGE_INDEX_DIR/glove.faiss" --state-dir runs/build/glove-faiss \
  --m 32 --ef-construction 500 --threads 24
bash scripts/python.sh experiments_scripts/hnswlib/build_benchmark_index.py \
  --dataset "$SAGE_DATA_DIR/glove-100-angular.hdf5" --space cosine \
  --index "$SAGE_INDEX_DIR/glove.hnsw" --state-dir runs/build/glove-hnsw \
  --m 32 --ef-construction 500 --threads 24
```

Use `--space l2` for SpaceV. Builders do not update the catalog: copy it and
set `dataset_path` and each backend index's `path` to the actual absolute
paths, replacing the corresponding relative `dataset_file`/`file` entries.
Pass that catalog with `--datasets-config` to the main runner. Other studies
use `--catalog`; see their READMEs for the schema and options.

For Recall@100, all datasets need at least 100 ground-truth neighbors. The
original YouTube input contains only top10. The
[top100 preparation guide](experiments_scripts/01_main_results/README.md#ground-truth-for-k100)
computes exact neighbors from the existing FAISS index and creates an overlay
catalog for the full six-dataset run.

RaBitQ accepts `RABITQ_DATA_DIR`, `RABITQ_INDEX_DIR`, and explicit input-directory
options. Embedding-quality and RaBitQ studies have their own
[input catalogs](experiments_scripts/05_robustness/README.md#inputs-and-build).

## SAGE Configuration

The current default policy is implemented in
[final_policy.py](experiments_scripts/common/final_policy.py):

| Setting | Default |
| --- | --- |
| Calibration source | Index/training vectors only, with each probe's own ID hidden |
| LID sampling | 10,000 nodes, seed 42, `k_lid=15`, `ef=512` |
| Probe selection | 500 LID-quantile probes after 1st--99th percentile trimming |
| Pseudo ground truth | Vanilla search at `ef=4096` on the same graph |
| Recommended ef | Smallest ef within 0.001 recall of the best measured calibration point |
| Offline ef range | Minimum 64; initial cap 1024, extended by 1024 up to 4096 when needed |
| CFR signal | Raw arithmetic mean over a 16-pop pre-expansion window |
| Main routing | Four tiers, with safety margin `g=2` |
| Shadow routing | Rank 128, routes 32/64/96, with safety margin `g=4` |
| Shadow activation | Starting ef at least 256; disabled for `k=100` |
| Reassessment | One promotion to the next tier when Top1 improves after a cut |
| Main timing | 24 threads; one warmup and five measured batch calls |

The calibration grid divides each doubling interval into four steps, with the
step capped at 128. Measurements are cached while the cap expands. The
recommendation is selected using calibration probes before evaluation queries
are loaded.

Main and shadow CFR are collected in one uncut traversal at the selected ef.
Main uses valid CFR values within its first 16 pops after its result set is
full; shadow uses its first 16 valid observations after 128 candidates are
available. Each has its own usable-probe mask and threshold fit.

If shadow does not select a smaller budget, main classification continues.
After a shadow cut, Top1 reassessment can promote `32 -> 64`, `64 -> 96`, or
`96 -> 128` once. Actual budgets are bounded below by `k`; `k=100` uses the
main path only.

## Reproducing Experiments

The numbered directories follow the paper's experiment order. All commands
run from the repository root through the local Python launcher.

| Experiment | Entry points | Backend |
| --- | --- | --- |
| [Main results](experiments_scripts/01_main_results/README.md) | `01_main_results/run.sh` | FAISS, hnswlib, PiP, ABS |
| [Detailed analysis](experiments_scripts/02_detailed_analysis/README.md) | `run_drilldown.py`, `run_misclassification.py`, `run_shadow.py` | FAISS |
| [Calibration](experiments_scripts/03_calibration/README.md) | `03_calibration/run.py` | FAISS |
| [Parameter sensitivity](experiments_scripts/04_parameter_sensitivity/README.md) | `04_parameter_sensitivity/run.py` | FAISS |
| [Robustness](experiments_scripts/05_robustness/README.md) | `05_robustness/run.py`, `report.py` | FAISS, RaBitQ |
| [Ada-EF and DARTH](experiments_scripts/01_main_results/README.md#ada-ef-and-darth) | `run_adaef.py`, `run_darth_pipeline.py` | Baseline-specific runtimes |

### Main Six-Dataset Results

Inspect the plan and validate prepared inputs before executing:

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

`--dry-run` prints paths and the plan without running searches. `--preflight`
checks input metadata, ground-truth width, CPU availability, and binary files,
and writes a preparation receipt. It does not test native imports or run ANN;
use the [validation checks](#validation) for the local engines. Preflight is
an explicit step rather than an automatic part of execution.

Select a single cell with:

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss
bash experiments_scripts/01_main_results/run.sh run-cell --cell hnswlib
```

Use a fresh `--output /path/to/new-run` for another run. Supply `--config` and
`--datasets-config` for host-specific settings and input paths. The default
output layout is:

```text
runs/01_main_results/k10/
  faiss/summary.csv
  faiss/targets.json
  faiss/datasets/<dataset>/policy.json
  hnswlib/summary.csv
  pip/summary.csv
  abs/summary.csv
```

Each cell also retains raw timings and per-query results. Run FAISS before
PiP/ABS so their target file exists. The
[main guide](experiments_scripts/01_main_results/README.md) documents baseline
tuning, k100 execution, and `summarize_main.py` for matched-recall comparisons.

### Individual Studies

Calibration and sensitivity experiments calibrate independently from the
supplied graphs. For example:

```bash
bash scripts/python.sh experiments_scripts/03_calibration/run.py \
  --dataset glove-100-angular --experiment quality
bash scripts/python.sh experiments_scripts/04_parameter_sensitivity/run.py \
  --dataset glove-100-angular
```

The calibration `quality` run records offline phase times and evaluates the
recommended ef. Use `--experiment stability` or `--experiment pseudo-gt` for
the other calibration studies. Detailed analysis instead consumes the frozen
policy and receipts from a completed FAISS main run.

Robustness experiments validate their separate inputs before execution:

```bash
bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding --preflight
bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq --preflight
```

Follow each study's README for execution, CPU/NUMA placement, and output
fields. The [experiment guide](experiments_scripts/README.md) describes the
dependency order and numeric outputs for all five sections.

### Timing and Outputs

Paper runners default to one warmup plus five measured calls and report
arithmetic means. Online timing covers the full query batch, including query
preparation and output allocation; index loading, calibration, recall
evaluation, diagnostics, and file writes are excluded. Amortized milliseconds
are batch time divided by query count, not individual-query latency percentiles.

Main speedups compare methods at matched recall. Vanilla time is taken from
an exact measured recall point or interpolated between measured points,
without extrapolation. The shadow-contribution study compares the same
starting ef. The standalone RaBitQ `sweep` retains a median summary and
excludes preprocessing; use `05_robustness` for the paper's timing scope.

Generated policies, CSV/JSON summaries, raw timings, and per-query arrays are
written under `runs/`. Historical result bundles, final paper figures, and
LaTeX table exports are not included in the source artifact.

## Baselines

The main comparison includes vanilla HNSW, SAGE, PiP, ABS, Ada-EF, and DARTH.
PiP and ABS are launched by the main runner after the FAISS SAGE target is
available. Ada-EF and DARTH have separate preparation and measurement stages:

- **Ada-EF** uses the catalog's hnswlib graph and the completed FAISS target.
  Its paper evaluation covers the five cosine datasets; SpaceV/L2 is unsupported.
- **DARTH** prepares disjoint learning/validation inputs, builds an auxiliary
  graph, computes exact holdout ground truth, and trains its model. Final
  measurement uses the original full FAISS graph and the main cell's target.

The [baseline build guide](baselines/README.md) lists dependencies and native
targets. The [main baseline instructions](experiments_scripts/01_main_results/README.md#ada-ef-and-darth)
cover CPU affinity, preflight, training, staged execution, and result summaries.

## Validation

After building all four main components, check local imports, synthetic
searches, and PiP/ABS behavior:

```bash
bash scripts/check_env.sh
```

Inspect available options without starting an experiment:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py --help
bash scripts/python.sh experiments_scripts/hnswlib/build_benchmark_index.py --help
bash scripts/python.sh experiments_scripts/01_main_results/run_search_main6.py --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner calibrate --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner sweep --help
```

These checks complement the experiment-specific input preflights; they do not
require the full benchmark datasets. See the [scripts guide](scripts/README.md)
for build selection and environment details.
