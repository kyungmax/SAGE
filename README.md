# SAGE: Signal-driven Adaptive Greedy Early-stop

## Overview

This repository contains the artifact for **SAGE** (**S**ignal-driven
**A**daptive **G**reedy **E**arly-stop), an adaptive HNSW search method that
keeps the index fixed, starts from a deliberately wide `efSearch`, and stops the
greedy layer-0 search early for easy queries using a runtime
Candidate-to-Furthest Ratio (CFR) signal.

SAGE requires no offline training workload, no historical queries, and no index
rebuild. The core artifact implementation is the patched FAISS runtime path; a
secondary hnswlib implementation is included to validate that the policy ports
across independent HNSW implementations. In both backends, index construction,
distance computation, and the HNSW graph layout are unchanged; SAGE adds scalar
bookkeeping in the layer-0 search loop and exposes the adaptive query path
through `knn_query_adaptive_light`.

**Repository layout**
- `faiss/` -- core SAGE implementation in FAISS. The main C++ changes are in
  `faiss/impl/HNSW.{h,cpp}` with Python exposure through
  `faiss/python/class_wrappers.py`.
- `hnswlib/` -- reference/secondary SAGE implementation used to validate that
  the policy ports across independent HNSW implementations.
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
  wrappers, manifests, final CSVs, and combined recall-latency plot assets.
- `final_experiments/02_drill_down/` -- easy/medium/hard query-group analysis
  and false-easy replay scripts.
- `final_experiments/03_offline_cost/` -- offline calibration-cost study.
- `final_experiments/04_ablation_study/` -- SAGE component ablations.
- `final_experiments/05_embedding_model_effects/` -- embedding-model comparison.
- `final_experiments/06_better_index_quality/` -- HNSW graph-quality study.
- `final_experiments/07_darth_ada-ef/` -- DARTH and Ada-EF target-0.99 comparison scripts.
- `final_analysis/` -- analysis scripts and submission-figure generation for
  probe representativeness, difficulty drill-down, CFR behavior, and backend
  distance-count checks.
- `baselines/` -- wrappers and copied outputs for Ada-EF and DARTH.
- `plots/` -- plot-only scripts and gnuplot templates.
- `docs/` -- artifact notes, troubleshooting, and dataset/index instructions.

> **Artifact packaging model:** modified research code is included in this
> repository (`faiss/`, `hnswlib/`, `rabitq/`, `experiments_scripts/ada-ef`,
> `experiments_scripts/darth`, and `baselines/`). Standard system/Python
> dependencies such as LightGBM, OpenBLAS, HDF5, Eigen, Boost, CMake, and
> plotting packages are installed or downloaded by the documented setup steps.

## Hardware Configuration

**Testing configuration** (from the paper experiments):
- Dell PowerEdge R760
- 2x Intel Xeon Gold 6442Y CPUs
- 1 TiB DDR5 DRAM
- NVMe storage
- Ubuntu 24.04.1 LTS (Noble Numbat)

## Software Dependencies

SAGE uses a patched CPU FAISS build with Python bindings, plus Python utilities
for calibration, benchmarking, and plotting.

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake g++ make swig \
    python3 python3-dev python3-pip \
    libopenblas-dev libomp-dev \
    gnuplot

pip3 install \
    numpy pandas scipy h5py matplotlib seaborn \
    scikit-learn pytest
```

FAISS requires a C++20 compiler and BLAS. The final paper runs used the
optimized CPU path; AVX-512 is recommended on matching hardware.

## Build

### FAISS

```bash
export SAGE_PROJECT_ROOT=$(pwd)
cd $SAGE_PROJECT_ROOT/faiss

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
export FAISS_PYTHON_PATH=$SAGE_PROJECT_ROOT/faiss/build_sage_avx512/faiss/python
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
comparisons against the FAISS implementation.

```bash
cd $SAGE_PROJECT_ROOT/hnswlib
python3 -m pip install -e .
```

## Data and Indexes

By default the scripts detect the repository root from their location, then look for
datasets under `<repo>/datasets` and indexes under `<repo>/index`. Set
`SAGE_PROJECT_ROOT` only when you want to override the detected repository root;
set explicit data/index roots for large artifact storage outside the git checkout:

```bash
export SAGE_PROJECT_ROOT=/path/to/SAGE
export SAGE_DATA_DIR=$SAGE_PROJECT_ROOT/datasets
export SAGE_INDEX_DIR=$SAGE_PROJECT_ROOT/index
export SAGE_FAISS_INDEX_ROOT=$SAGE_INDEX_DIR/faiss_m32_efc500_main8_20260707/darth/index
export SAGE_HNSWLIB_INDEX_ROOT=$SAGE_INDEX_DIR
```

The main paper experiments use eight HDF5 datasets with `train`, `test`, and
`neighbors` arrays:

| Dataset file | Size | Dim | Metric | Modality |
|--------------|------|-----|--------|----------|
| `glove-100-angular.hdf5` | 1.18M | 100 | angular | text |
| `nytimes-256-angular.hdf5` | 290K | 256 | angular | text |
| `msmarco-v1-openai-ada2-full-ip.hdf5` | 8.84M | 1536 | inner product | text |
| `msspacev-100M-i8-euclidean.hdf5` | 100M | 100 | Euclidean | vector benchmark |
| `cohere-768-angular.hdf5` | 10M | 768 | angular | text |
| `youtube-15M-angular.hdf5` | 15M | 1024 | angular | image/video |
| `agnews-mxbai-1024-euclidean.hdf5` | 769K | 1024 | Euclidean | text |
| `landmark-nomic-768-angular.hdf5` | 761K | 768 | angular | image |

All main indexes use:
- `M=32`
- `efConstruction=500`
- `k=10`
- target recall `0.95`

TODO: add public download links, expected directory layout, file sizes, and
checksums for datasets and prebuilt indexes. Large HDF5 datasets, index files,
raw logs, and cache directories should not be committed to git.

## SAGE Configuration

Default paper configuration:
- EF sweep: `64,80,96,128,160,192,256,320,384,512,640,768,896,1024`
- calibration probes: `ncal=100`
- calibration sampling: random 10,000-node LID pool, trimmed to the 1st--99th percentiles, then 100 LID-quantile calibration probes
- classification window: `[4,16]`
- CFR EMA decay: `alpha=0.8`
- routing buckets: `B=4`
- conservative threshold pair gap: `g=2`
- `tmin_pops=25`
- offline/calibration threads: `24`
- online search threads: `24`

## Reproducing Experiments

The scripts write per-dataset run outputs under each experiment's `run/`
directory and consolidated CSV/Markdown summaries under `final/`.

| Paper result | Driver | Description | Backend |
|--------------|--------|-------------|---------|
| Main eight-dataset sweep | `final_experiments/01_main_results/run_main8_online24_20260707.py` | Vanilla HNSW vs SAGE over the EF ladder for both FAISS and hnswlib | FAISS + hnswlib |
| Main plotting | `final_experiments/01_main_results/main8_online24/combined_faiss_SIMD/` | Builds the eight-dataset recall-latency plot from final sweep CSVs | FAISS + hnswlib |
| Backend-specific sweep | `experiments_scripts/{faiss,hnswlib}/run_main_qps_latency_sweep.py` | Lower-level per-backend runners used by the main8 wrapper | FAISS / hnswlib |
| Difficulty drill-down | `final_experiments/02_drill_down/scripts/` | Easy/medium/hard group analysis and false-easy replay | FAISS + hnswlib |
| Offline calibration cost | `final_experiments/03_offline_cost/run_all_simd_24t.sh` | SAGE offline calibration cost with 24-thread SIMD-on FAISS and hnswlib | FAISS + hnswlib |
| Ablation study | `final_experiments/04_ablation_study/` | FAISS GloVe/Cohere parameter sensitivity and pseudo-GT check | FAISS |
| Embedding-model effects | `final_experiments/05_embedding_model_effects/` | MSMARCO five-embedding comparison with FAISS SIMD-on 24-thread runner | FAISS |
| Index-quality study | `final_experiments/06_better_index_quality/run_faiss_simd_ndis_ef1024.py` | Varying HNSW graph quality (`M`, `efConstruction`) | FAISS |
| DARTH / Ada-EF target-0.99 | `final_experiments/07_darth_ada-ef/` | SOTA comparison on Cohere/MSMARCO with SIMD-on full-query runs | FAISS + hnswlib |

TODO: replace `Paper result` labels with final VLDB figure/table numbers.

### Run the Main Eight-Dataset Sweep

```bash
export SAGE_PROJECT_ROOT=/path/to/SAGE
export SAGE_DATA_DIR=$SAGE_PROJECT_ROOT/datasets
export SAGE_INDEX_DIR=$SAGE_PROJECT_ROOT/index
export SAGE_FAISS_INDEX_ROOT=$SAGE_INDEX_DIR/faiss_m32_efc500_main8_20260707/darth/index
export SAGE_HNSWLIB_INDEX_ROOT=$SAGE_INDEX_DIR
export FAISS_PYTHON_PATH=$SAGE_PROJECT_ROOT/faiss/build_sage_avx512/faiss/python

cd $SAGE_PROJECT_ROOT/final_experiments/01_main_results
python3 run_main8_online24_20260707.py run-all \
    --out-root main8_online24
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
python3 run_main8_online24_20260707.py run-cell --cell faiss
python3 run_main8_online24_20260707.py run-cell --cell hnswlib
```

Main outputs:

```text
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/main_qps_latency_sweep.csv
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/offline_recommended_efsearch.csv
final_experiments/01_main_results/main8_online24/{faiss,hnswlib}/final/offline_predicted_recall_curve.csv
```

### Build the Combined Main Plot

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/01_main_results/main8_online24/combined_faiss_SIMD
gnuplot main8_recall_total_time_faiss_SIMD_smoothed_plot_ready.gp
```

TODO: wire the final all8 plot-only wrapper into `plots/`.

## Running Individual Studies

### Difficulty Drill-Down

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/02_drill_down
./run_faiss_simd_24t.sh
```

Primary outputs include:
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/query_groups.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_ef_sweep.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_pair_metrics.csv`
- `drilldown_faiss_SIMD_on_main8_24t/hard_loss_querywise_exactgt_24t/hard_loss_querywise.csv`
- `drilldown_faiss_SIMD_on_main8_24t/large_false_easy_drop_analysis/large_false_easy_summary_by_dataset_ef.csv`

### Offline Calibration Cost

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/03_offline_cost
./run_all_simd_24t.sh
```

Run one backend only:

```bash
./run_faiss_simd_24t.sh
./run_hnswlib_simd_24t.sh
```

Primary outputs:
- `offline_cost_main8_faiss_SIMD_on_24t/final/faiss_offline_cost_median.csv`
- `offline_cost_main8_hnswlib_SIMD_on_24t/final/hnswlib_offline_cost_median.csv`

### Ablation Study

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/04_ablation_study
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
cd $SAGE_PROJECT_ROOT/final_experiments/05_embedding_model_effects
export OUT_ROOT=$PWD/msmarco_embedding_models_faiss_SIMD_on_24t
python3 scripts/preflight_msmarco_embedding_models.py
./run_msmarco_embedding_models_faiss_24t.sh
python3 scripts/summarize_msmarco_embedding_model_effects.py --final-dir "$OUT_ROOT/sage_results/final"
```

The runner uses FAISS, SIMD on, 24 offline/online threads, `M=32`, `efConstruction=500`, and the paper EF ladder. Missing FAISS indexes are built on first use; missing HDF5 embedding datasets must be prepared first.

### Index Quality

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/06_better_index_quality
python3 run_faiss_simd_ndis_ef1024.py \
    --policy-csv /path/to/combined_faiss_main_qps_latency_sweep.csv \
    --build-missing-indexes
```

This study records FAISS SIMD-on distance-computation reduction and recall loss, not QPS.

### DARTH / Ada-EF Target-0.99

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
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

Ada-EF and DARTH outputs used by the combined plot are stored under
`final_experiments/Ada-EF/` and `final_experiments/DARTH/`. The main-result
tree under `final_experiments/01_main_results/` keeps the copied baseline JSONs
used by the plot; the main8 wrapper reruns the SAGE and vanilla HNSW cells.

From-scratch target-0.99 baseline rerun scripts are in `final_experiments/07_darth_ada-ef/`.
Lower-level wrappers are preserved under `experiments_scripts/ada-ef/` and
`experiments_scripts/darth/`. Source-only DARTH and Ada-EF baseline code lives under
`baselines/`; see `baselines/README.md` for provenance and build commands. The
baseline wrappers default to the local source trees and build outputs in this
repository; use `SAGE_DARTH_*` or `SAGE_ADAEF_*` only to override those paths.

## Plotting Only

Regenerate the main combined plot from existing CSV/JSON outputs:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/01_main_results/main8_online24/combined_faiss_SIMD
gnuplot main8_recall_total_time_faiss_SIMD_smoothed_plot_ready.gp
```

Regenerate FAISS analysis figures:

```bash
cd $SAGE_PROJECT_ROOT
python3 final_analysis/regenerate_faiss_01_02.py
```

Additional submission-figure scripts live under:
- `final_analysis/02_glove_cohere_target_recall_easy_hard_analysis/`
- `final_analysis/04_lid_stratified_probe100_online_recall_cfr_reproducibility/`
- `final_analysis/05_cfr_bin_difficulty_correlation_online_and_probe/`

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
    --index-dir "$SAGE_INDEX_DIR" \
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

TODO: add expected smoke-test output ranges for recall and QPS after the
artifact bundle is frozen.

## Citation

```bibtex
@inproceedings{sage2026,
  title     = {SAGE: Signal-driven Adaptive Greedy Early-stop},
  author    = {TODO},
  booktitle = {Proceedings of the VLDB Endowment},
  year      = {2026}
}
```
