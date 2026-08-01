# SAGE: Query-Adaptive Early Termination via HNSW-Inherent Signals

## Overview

This repository contains the artifact for **Query-Adaptive Early Termination via
HNSW-Inherent Signals** and its **SAGE** (**S**ignal-driven **A**daptive
**G**reedy **E**arly-stop) implementation. SAGE is an adaptive HNSW search method
built around the *start wide, cut early* strategy: it begins each query with a
wide baseline `efSearch` to protect hard queries, then reduces the budget
mid-traversal for easy queries using the runtime Candidate-to-Furthest Ratio
(CFR) signal.

SAGE requires no external query workload, no learned model, and no index rebuild.
It performs a lightweight one-time calibration after index construction using
LID-stratified probes sampled from the index itself. The core artifact
implementation is the patched FAISS runtime path; a secondary hnswlib
implementation is included to validate that the policy ports across independent
HNSW implementations. In both backends, index construction, distance
computation, and the HNSW graph layout are unchanged; SAGE adds only scalar
bookkeeping in the base-layer search loop and exposes the adaptive query path
through `knn_query_adaptive_light`.

**Repository layout**
- `faiss/` -- core SAGE implementation in FAISS. The main C++ changes are in
  `faiss/impl/HNSW.{h,cpp}` with Python exposure through
  `faiss/python/class_wrappers.py`.
- `hnswlib/` -- reference/secondary SAGE implementation used to validate that
  the policy ports across independent HNSW implementations.
- `datasets/` -- local drop-in directory for HDF5 datasets. The directory is
  tracked, but dataset files are ignored by git.
- `index/` -- local drop-in directory for downloaded prebuilt FAISS and hnswlib
  indexes. The directory is tracked, but index binaries are ignored by git.
- `sage_env.sh`, `setup.sh`, `run_main8.sh` -- root-level artifact entrypoints
  for environment defaults, directory setup, and the main eight-dataset run.
- `experiments_scripts/` -- runnable experiment scripts, backend adapters, shared
  calibration logic, and lower-level recall-latency drivers.
- `experiments_scripts/faiss/` -- FAISS-backed SAGE adapter and backend-specific
  recall/QPS/latency sweep.
- `experiments_scripts/hnswlib/` -- hnswlib-backed SAGE adapter and
  backend-specific recall/QPS/latency sweep.
- `experiments_scripts/ada-ef/` -- Ada-EF baseline runner, wrapper scripts, and
  vendored HNSW headers needed by the runner.
- `experiments_scripts/darth/` -- DARTH baseline rerun wrappers.
- `final_experiments/01_main_results/` -- main eight-dataset experiment
  wrappers and manifests.
- `final_experiments/02_drill_down/` -- easy/medium/hard query-group analysis
  and false-easy replay scripts.
- `final_experiments/03_offline_cost/` -- offline calibration-cost study.
- `final_experiments/04_ablation_study/` -- SAGE component ablations.
- `final_experiments/05_embedding_model_effects/` -- embedding-model comparison.
- `final_experiments/06_better_index_quality/` -- HNSW graph-quality study.
- `final_experiments/07_darth_ada-ef/` -- DARTH and Ada-EF target-0.99 comparison scripts.
- `baselines/` -- baseline source snapshots, wrappers, and provenance notes.

## Quick Start

From a fresh clone, use the root entrypoints first:

```bash
cd /path/to/sage
./setup.sh
source ./sage_env.sh
```

By default, datasets are read from `$SAGE_ROOT/datasets`, hnswlib indexes from
`$SAGE_ROOT/index`, and FAISS indexes from
`$SAGE_ROOT/index/faiss_m32_efc500_main8_20260707/index`. To keep large
files on another volume, export `SAGE_DATA_DIR` and/or `SAGE_INDEX_DIR` before
running `setup.sh` or sourcing `sage_env.sh`.

After building FAISS and hnswlib and placing the data/index files, the main
artifact entrypoint is:

```bash
./run_main8.sh
```

Use the underlying runner subcommands through the same wrapper when needed:

```bash
./run_main8.sh run-cell --cell faiss
./run_main8.sh run-cell --cell hnswlib
```

## Hardware Configuration

**Testing configuration** (from the paper experiments):
- Dell PowerEdge R760
- 2x Intel Xeon Gold 6442Y CPUs
- 1 TiB DDR5 DRAM
- NVMe storage
- Ubuntu 24.04.1 LTS (Noble Numbat)

## Software Dependencies

SAGE uses patched CPU FAISS and hnswlib builds with Python bindings, plus
Python utilities for calibration and benchmarking.

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake g++ make swig pkg-config \
    python3 python3-dev python3-pip \
    libopenblas-dev libomp-dev libhdf5-dev \
    libeigen3-dev libboost-all-dev

python3 -m pip install -r requirements.txt
```

FAISS requires a C++20 compiler and BLAS. The final paper runs used the
optimized CPU path; AVX-512 is recommended on matching hardware. Do not install
`faiss-cpu` into the main SAGE environment; build the patched FAISS tree below
and use `FAISS_PYTHON_PATH`.

## Build

### FAISS

```bash
source /path/to/sage/sage_env.sh
cd $SAGE_ROOT/faiss

cmake -S . -B build_sage_avx512 \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_OPT_LEVEL=avx512 \
    -DBUILD_TESTING=ON

make -C build_sage_avx512 -j faiss_avx512
make -C build_sage_avx512 -j swigfaiss

cd build_sage_avx512/faiss/python
python3 setup.py install
```

If you do not install the package into the active environment, point the
experiment scripts to the built Python package:

```bash
export FAISS_PYTHON_PATH=$SAGE_ROOT/faiss/build_sage_avx512/faiss/python
```

Sanity check:

```bash
python3 - <<'PY'
import faiss
print(faiss.__file__)
print(hasattr(faiss, "SearchParametersHNSWAdaptiveLight"))
PY
```

The final line must print `True`.

### hnswlib

The hnswlib implementation is used for cross-backend validation and for
comparisons against the FAISS implementation. Build and install it from the
vendored patched source:

```bash
source /path/to/sage/sage_env.sh
cd $SAGE_ROOT/hnswlib
python3 -m pip install --no-build-isolation -e .
```

By default, `hnswlib/setup.py` builds the extension with `-O3`, `-march=native`,
and OpenMP (`-fopenmp`) on Unix-like systems. To build a more portable binary
without `-march=native`, set `HNSWLIB_NO_NATIVE=1` before installing.

Sanity check:

```bash
python3 - <<'PY'
import hnswlib
p = hnswlib.Index(space="l2", dim=4)
print(hnswlib.__file__)
print(hasattr(p, "knn_query_adaptive_light"))
PY
```

The final line must print `True`.

## Data and Indexes

Initialize the standard artifact directories before running experiments:

```bash
./setup.sh
source ./sage_env.sh
```

To keep large files outside the repository, export `SAGE_DATA_DIR` or
`SAGE_INDEX_DIR` first; `sage_env.sh` preserves those overrides.

The main paper experiments use eight HDF5 datasets with `train`, `test`, and
`neighbors` arrays. Download or prepare the HDF5 files under `SAGE_DATA_DIR`.
The public sources are listed below using the dataset names from the paper. The
scripts still expect the HDF5 filenames listed in the main sweep command.

The author-provided Google Drive folder is for converted inputs that are not
direct public HDF5 downloads: the YouTube HDF5 used in the main benchmark and
the non-ada MSMARCO embedding variants used in the embedding-model effects study
(GloVe, FastText, BGE-M3, and EmbeddingGemma):
[SAGE data artifacts](https://drive.google.com/drive/folders/12tu88Hx0D4BYGeIzqqbG60bzov57fyKb?usp=sharing).

| Datasets | Size | Dim | Metric | Source |
|----------|------|-----|--------|--------|
| `NYTimes` | 290K | 256 | angular | [ANN-Benchmarks HDF5](https://ann-benchmarks.com/nytimes-256-angular.hdf5) |
| `GloVe100` | 1.18M | 100 | angular | [ANN-Benchmarks HDF5](https://ann-benchmarks.com/glove-100-angular.hdf5) |
| `AGNews` | 769K | 1024 | L2 | [VIBE dataset bundle](https://huggingface.co/datasets/vector-index-bench/vibe) |
| `Landmark` | 761K | 768 | angular | [VIBE dataset bundle](https://huggingface.co/datasets/vector-index-bench/vibe); use `landmark-nomic-768-normalized.hdf5` as `landmark-nomic-768-angular.hdf5` |
| `CohereWiki` | 10M | 768 | angular | [Cohere Wikipedia embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings); HDF5 mirror: [Hugging Face `hhy3/ann-datasets`](https://huggingface.co/datasets/hhy3/ann-datasets/tree/main) |
| `YouTube` | 15M | 1024 | angular | [SeahorseDB YouTube source](https://huggingface.co/datasets/dnotitia/SeahorseDB-dataset/tree/main); converted HDF5 in the SAGE data artifacts |
| `MSMARCOV1` | 8.84M | 1536 | IP | [MS MARCO Passage Ranking Dataset](https://microsoft.github.io/msmarco/) with [Pyserini OpenAI ada2 corpus index](https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.msmarco-v1-passage.openai-ada2.20230530.e3a58f.tar.gz) and cached dev queries; convert to the expected HDF5 layout |
| `SpaceV` | 100M | 100 | L2 | [SPACEV1B source](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B); convert to the expected HDF5 layout |

Prebuilt indexes from the SAGE artifact bundle should be copied or extracted
into `index/`. To keep them elsewhere, set `SAGE_INDEX_DIR`,
`SAGE_FAISS_INDEX_ROOT`, or `SAGE_HNSWLIB_INDEX_ROOT` before running the
experiments.

Expected default FAISS layout:

```text
index/faiss_m32_efc500_main8_20260707/index/<dataset-stem>/<dataset-stem>.M32.efC500.index
```

hnswlib indexes are read directly from `SAGE_INDEX_DIR` using filenames produced
by `experiments_scripts/hnswlib/hnsw_index_utils.py`, for example:

```text
index/glove-100-angular_M32_M32_efC500_n1183514_dim100
```

Large HDF5 datasets, index files, generated CSV files, raw logs, and cache
directories should not be committed to git.

## SAGE Configuration

Default paper configuration:
- EF sweep: `64,80,96,128,160,192,256,320,384,512,640,768,896,1024`
- calibration probes: `ncal=100`
- calibration sampling: random 10,000-node LID pool, trimmed to the 1st--99th percentiles, then 100 LID-quantile calibration probes
- classification window: `[4,16]`
- CFR EMA decay: `alpha=0.8`
- routing buckets: `B=4`
- conservative threshold pair gap: `g=2`
- offline/calibration threads: `24`
- online search threads: `24`

## Reproducing Experiments

The commands below generate per-dataset run outputs in the `run/` directory for
each experiment and consolidated CSV/Markdown summaries under `final/`; these are
expected generated artifacts, not source files to maintain in git.

| Experiment | Driver | Backend |
|------------|--------|---------|
| Main eight-dataset sweep | `final_experiments/01_main_results/main8_online24/run_all.sh` | FAISS + hnswlib |
| Backend-specific sweep | `experiments_scripts/{faiss,hnswlib}/run_main_qps_latency_sweep.py` | FAISS / hnswlib |
| Difficulty drill-down | `final_experiments/02_drill_down/run_faiss_simd_24t.sh` | FAISS |
| Offline calibration cost | `final_experiments/03_offline_cost/run_all_simd_24t.sh` | FAISS + hnswlib |
| Ablation study | `final_experiments/04_ablation_study/run_all_faiss_glove_cohere_24t.sh` | FAISS |
| Embedding-model effects | `final_experiments/05_embedding_model_effects/run_msmarco_embedding_models_faiss_24t.sh` | FAISS |
| Index-quality study | `final_experiments/06_better_index_quality/run_faiss_simd_ndis_ef1024.py` | FAISS |
| DARTH / Ada-EF target-0.99 | `final_experiments/07_darth_ada-ef/run_darth_adaef_cohere_msmarco_simd_target099.sh` | FAISS + hnswlib |

### Run the Main Eight-Dataset Sweep

```bash
cd /path/to/sage
./setup.sh
source ./sage_env.sh
./run_main8.sh run-all --out-root final_experiments/01_main_results/main8_online24
```

By default this runs all eight datasets:

```text
glove-100-angular.hdf5
nytimes-256-angular.hdf5
msmarco-v1-openai-ada2-full-ip.hdf5
msspacev-100M-i8-euclidean.hdf5
cohere-768-angular.hdf5
youtube-15M-angular.hdf5
agnews-mxbai-1024-euclidean.hdf5
landmark-nomic-768-angular.hdf5
```

Run one backend cell only:

```bash
./run_main8.sh run-cell --cell faiss
./run_main8.sh run-cell --cell hnswlib
```

Expected generated outputs:

```text
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/main_qps_latency_sweep.csv
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/offline_recommended_efsearch.csv
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/offline_predicted_recall_curve.csv
```


## Running Individual Studies

### Difficulty Drill-Down

```bash
cd $SAGE_ROOT/final_experiments/02_drill_down
./run_faiss_simd_24t.sh
```

Expected generated outputs include:
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/query_groups.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_ef_sweep.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_pair_metrics.csv`
- `drilldown_faiss_SIMD_on_main8_24t/hard_loss_querywise_exactgt_24t/hard_loss_querywise.csv`
- `drilldown_faiss_SIMD_on_main8_24t/large_false_easy_drop_analysis/large_false_easy_summary_by_dataset_ef.csv`

### Offline Calibration Cost

```bash
cd $SAGE_ROOT/final_experiments/03_offline_cost
./run_all_simd_24t.sh
```

Run one backend only:

```bash
./run_faiss_simd_24t.sh
./run_hnswlib_simd_24t.sh
```

Expected generated outputs:
- `offline_cost_main8_faiss_SIMD_on_24t/final/faiss_offline_cost_median.csv`
- `offline_cost_main8_hnswlib_SIMD_on_24t/final/hnswlib_offline_cost_median.csv`

### Ablation Study

```bash
cd $SAGE_ROOT/final_experiments/04_ablation_study
python3 scripts/preflight_faiss_glove_cohere_ablation.py
export OUT_ROOT=$PWD/sage_ablation_faiss_glove_cohere_24t_m32_efc500_ef1024
./run_faiss_glove_cohere_ablation_24t.sh
export PSEUDOGT_OUT_DIR=$PWD/probe_pseudo_gt_vs_exact_glove_cohere_faiss_p100_ef4096
./run_faiss_glove_cohere_pseudogt_24t.sh
```

This reproduces the FAISS paper ablation on `glove-100-angular.hdf5` and `cohere-768-angular.hdf5`: `ncal`, CFR window, tier count `B`, EMA `alpha`, safety margin `g`, and the pseudo-GT vs exact-GT calibration check.

### Embedding-Model Effects

This study evaluates the same MSMARCO passage/query corpus under five embedding spaces: mean-pooled GloVe, mean-pooled FastText, OpenAI ada-002, BGE-M3, and EmbeddingGemma-300M. Each HDF5 must contain exact ground truth for its own embedding space.

```bash
cd $SAGE_ROOT/final_experiments/05_embedding_model_effects
export OUT_ROOT=$PWD/msmarco_embedding_models_faiss_SIMD_on_24t
python3 scripts/preflight_msmarco_embedding_models.py
./run_msmarco_embedding_models_faiss_24t.sh
python3 scripts/summarize_msmarco_embedding_model_effects.py --final-dir "$OUT_ROOT/sage_results/final"
```

The runner uses FAISS, SIMD on, 24 offline/online threads, `M=32`, `efConstruction=500`, and the paper EF ladder. Missing FAISS indexes are built on first use; missing HDF5 embedding datasets must be prepared first.

### Index Quality

```bash
cd $SAGE_ROOT/final_experiments/06_better_index_quality
python3 run_faiss_simd_ndis_ef1024.py \
    --policy-csv /path/to/combined_faiss_main_qps_latency_sweep.csv \
    --build-missing-indexes
```

This study records FAISS SIMD-on distance-computation reduction and recall loss, not QPS.

### DARTH / Ada-EF Target-0.99

```bash
cd $SAGE_ROOT/final_experiments/07_darth_ada-ef
python3 scripts/preflight_darth_adaef_cohere_msmarco.py
./run_darth_adaef_cohere_msmarco_simd_target099.sh
```

This reproduces the SOTA comparison scope from the paper: CohereWiki and MSMARCO only, target recall `0.99`, SIMD-on local builds, offline preparation on 24 threads, and full-query single-thread online latency. Use `scripts/build_darth_simd_avx512.sh` and `scripts/build_adaef_simd_avx512.sh` first if preflight reports missing binaries.

## Baselines

The main combined result uses:
- Vanilla HNSW in FAISS
- Vanilla HNSW in hnswlib
- SAGE in FAISS
- SAGE in hnswlib
- Ada-EF
- DARTH

Ada-EF and DARTH code is included in this repository. The final target-0.99
wrappers live under `final_experiments/07_darth_ada-ef/`; Ada-EF's standalone
runner is under `experiments_scripts/ada-ef/`, and the DARTH source tree is under
`baselines/darth/benchmarking-darth/`.

Install baseline Python dependencies and build the binaries used by the final
wrappers:

```bash
cd $SAGE_ROOT
python3 -m pip install -r requirements-darth.txt
python3 -m pip install -r requirements-adaef.txt

cd $SAGE_ROOT/final_experiments/07_darth_ada-ef
./scripts/build_darth_simd_avx512.sh
./scripts/build_adaef_simd_avx512.sh
python3 scripts/preflight_darth_adaef_cohere_msmarco.py
```

DARTH requires the Python `lightgbm` package and LightGBM headers. The build
helper downloads the LightGBM source distribution for headers when needed; in an
offline environment, pre-populate `LIGHTGBM_SRC_DIR` and `LIGHTGBM_LIB`. Ada-EF
requires CMake, a C++17 compiler, OpenMP, HDF5 C++ libraries, Eigen3, and Boost
headers; set `CONDA_PREFIX` if those headers live in a conda environment.


## Validation

Run the FAISS API sanity check:

```bash
python3 - <<'PY'
import faiss
assert hasattr(faiss, "SearchParametersHNSWAdaptiveLight")
print("SAGE FAISS API available")
PY
```

Run a smoke sweep on one dataset:

```bash
python3 experiments_scripts/faiss/run_main_qps_latency_sweep.py \
    --base-path "$SAGE_DATA_DIR" \
    --index-dir "$SAGE_FAISS_INDEX_ROOT" \
    --faiss-python-path "$FAISS_PYTHON_PATH" \
    --datasets glove-100-angular.hdf5 \
    --ef-sweep 512,1024 \
    --k-values 10 \
    --num-calibration-queries 20 \
    --measured-runs 1 \
    --offline-num-threads 24 \
    --online-num-threads 24 \
    --no-skip-existing
```

The smoke sweep writes a small FAISS result set under the runner default `run/`
and `final/` directories.

## Citation

```bibtex
@inproceedings{sage2026,
  title     = {Query-Adaptive Early Termination via HNSW-Inherent Signals},
  author    = {Kyungmin Kim and Dongseob Kim and Jihyo Jang and Joobo Shim and Jaeyoung Do and Sang-Won Lee},
  booktitle = {Proceedings of the VLDB Endowment},
  year      = {2026}
}
```
