# RaBitQ + SAGE

This directory contains the scripts for reproducing the RaBitQ + SAGE experiments. In the submitted repository, the expected layout is:

```text
SAGE/
  rabitq/                         # RaBitQ-Library source tree
  experiments_scripts/rabitq/     # these scripts
```


## Base Library Attribution

This implementation is built on top of the open-source RaBitQ Library. We keep the original RaBitQ source tree under `SAGE/rabitq` and add the SAGE adaptive HNSW search path, Python bindings, and reproduction scripts used in our experiments.

- Original project: The RaBitQ Library
- Upstream GitHub: https://github.com/VectorDB-NTU/RaBitQ-Library
- Original developers listed by the upstream project: Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan Shi, and Zhonghao Yang
- Research group: VectorDB group, Nanyang Technological University, Singapore
- License: Apache License 2.0
- RaBitQ reference: Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, Raymond Chi-Wing Wong, "Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search", SIGMOD 2025, https://arxiv.org/abs/2409.09913


## Modified Components

Compared with the upstream RaBitQ Library, the SAGE-specific changes are concentrated in the RaBitQ HNSW path and Python bindings:

- `include/rabitqlib/index/hnsw/hnsw.hpp`
  - Adds adaptive-light HNSW search entry points.
  - Computes the online CHR/CFR difficulty signal during base-layer traversal.
  - Applies EMA smoothing with `alpha = 0.8` over the early classification window.
  - Supports hide-node traversal for train-probe calibration.
  - Implements paper bucket routing with `paper_bucket_count=4` and calibrated `bucket_gamma_ratios`.

- `python_bindings/hnsw_bindings.cpp`
  - Exposes `search_adaptive_light` and `search_adaptive_light_with_stats` to Python.
  - Exposes adaptive controls: `paper_bucket_mode`, `paper_bucket_count`, `bucket_gamma_ratios`, and `hide_labels`.
  - Returns per-query adaptive statistics used by the calibration scripts.

- `experiments_scripts/rabitq/`
  - Contains the paper reproduction scripts for building/calibrating policies, measuring recall/QPS, and plotting recall/search-time results.

## RaBitQ + SAGE Logic

RaBitQ performs HNSW search using binary-quantized distance estimates. SAGE adds adaptive `efSearch` routing to the RaBitQ HNSW base-layer search.

During traversal, the C++ search loop computes a CHR/CFR-style difficulty signal from the current popped candidate distance and the frontier distance. The signal is smoothed with EMA decay `alpha = 0.8`, and the early-window mean is exposed as `classify_chr_mean`.

The calibration script builds an offline policy as follows:

1. Sample train nodes and estimate local intrinsic dimensionality (LID).
2. Select LID-representative train probes.
3. Search each probe while hiding its own node with `hide_labels`, so calibration does not rely on trivially rediscovering the probe itself.
4. For each `efSearch`, estimate which lower routed `efSearch` values preserve recall.
5. Convert those decisions into monotone thresholds for four buckets.

Online `RaBitQ+SAGE` search uses the calibrated thresholds with:

```text
bucket 0 -> about ef/4
bucket 1 -> about ef/2
bucket 2 -> about 3ef/4
bucket 3 -> ef
```

The Python scripts compute and save the policy. The actual online bucket decision and `efSearch` shrink happen inside `SAGE/rabitq/include/rabitqlib/index/hnsw/hnsw.hpp`. The Python binding entry points are in `SAGE/rabitq/python_bindings/hnsw_bindings.cpp` and expose `search_adaptive_light`, `search_adaptive_light_with_stats`, `paper_bucket_mode`, `paper_bucket_count`, `bucket_gamma_ratios`, and `hide_labels`.

## Scripts

- `rabitq_paper_build_calibrate.py`: build/load RaBitQ indexes and write SAGE calibration policies.
- `rabitq_paper_recall_qps_sweep.py`: run test-query sweeps for `RaBitQ` and `RaBitQ+SAGE`.
- `rabitq_paper_recall_latency.py`: plot recall-QPS, recall-latency, iso-recall curves, and matched-recall search-time speedup.

## Setup

Install the RaBitQ Python package and Python dependencies:

```bash
export SAGE_PROJECT_ROOT=/path/to/SAGE
cd $SAGE_PROJECT_ROOT/rabitq
python -m pip install -v .
python -m pip install h5py matplotlib numpy pandas
```

Set paths from the SAGE repository root. These are also the script defaults, except `RABITQ_MSSPACEV_RAW` is shown explicitly because MSSPACEV needs its raw i8bin file:

```bash
export SAGE_PROJECT_ROOT=/path/to/SAGE
cd $SAGE_PROJECT_ROOT
export RABITQ_REPO_DIR="$SAGE_PROJECT_ROOT/rabitq"
export RABITQ_DATA_DIR="$SAGE_PROJECT_ROOT/datasets"
export RABITQ_MSSPACEV_RAW="$RABITQ_DATA_DIR/spacev100m_raw/spacev100m_base.i8bin"
export RABITQ_INDEX_DIR="$SAGE_PROJECT_ROOT/artifacts/rabitq/rabitq_m32_efc500"
export RABITQ_OUT_DIR="$SAGE_PROJECT_ROOT/artifacts/rabitq"
export RABITQ_PLOT_DIR="$SAGE_PROJECT_ROOT/artifacts/rabitq/plots"
```

Supported dataset aliases are `agnews`, `cohere`, `msspacev`, and `youtube`. The data directory should contain the corresponding `.hdf5` files with `train`, `test`, and `neighbors`; MSSPACEV also needs the raw i8bin path above.

## Run

Build indexes and calibrate policies:

```bash
python experiments_scripts/rabitq/rabitq_paper_build_calibrate.py \
  --datasets agnews cohere msspacev youtube \
  --data-dir "$RABITQ_DATA_DIR" \
  --msspacev-raw "$RABITQ_MSSPACEV_RAW" \
  --index-dir "$RABITQ_INDEX_DIR" \
  --degree 32 \
  --ef-construction 500 \
  --total-bits 8 \
  --build-threads 24 \
  --calibration-threads 24 \
  --lid-pool-size 10000 \
  --num-calibration-queries 100
```

Run the recall/QPS sweep:

```bash
python experiments_scripts/rabitq/rabitq_paper_recall_qps_sweep.py \
  --datasets agnews cohere msspacev youtube \
  --data-dir "$RABITQ_DATA_DIR" \
  --msspacev-raw "$RABITQ_MSSPACEV_RAW" \
  --index-dir "$RABITQ_INDEX_DIR" \
  --calibration-threads 24 \
  --query-threads 24 \
  --rounds 3 \
  --out "$RABITQ_OUT_DIR/rabitq_paper_lid_hide_node_recall_qps.csv"
```

Plot recall and search-time results:

```bash
python experiments_scripts/rabitq/rabitq_paper_recall_latency.py \
  --csv "$RABITQ_OUT_DIR/rabitq_paper_lid_hide_node_recall_qps.csv" \
  --datasets agnews cohere msspacev youtube \
  --out-dir "$RABITQ_PLOT_DIR" \
  --suffix m32_efc500_24thread_rounds3 \
  --plot-kind all
```

The sweep CSV contains recall, QPS, and latency per query for each dataset/method/`efSearch`. The plotting script also writes `iso_recall_search_time_summary_{suffix}.csv` with matched-recall QPS, latency, and speedup.
