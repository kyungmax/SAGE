# Main Results

This directory contains script-only launchers for the main eight-dataset SAGE
experiment. Generated CSVs, plots, logs, caches, and prebuilt indexes are not
committed here.

Configuration:
- Backends: hnswlib and FAISS
- SIMD: hnswlib uses its `-march=native` build; FAISS uses the AVX-512 Python build and sets `FAISS_OPT_LEVEL=AVX512`
- Thread settings: `main8_online24` uses 24 online/offline threads; `main8_online1` uses 1 online thread and 24 offline threads
- Index: `M=32`, `efConstruction=500`
- Search: `k=10`, `ncal=100`
- Calibration: sampled 10,000-node LID pool, 1st--99th percentile trim, 100 LID-quantile probes
- EF sweep: `64,80,96,128,160,192,256,320,384,512,640,768,896,1024`

Datasets:
- `glove-100-angular.hdf5`
- `nytimes-256-angular.hdf5`
- `msmarco-v1-openai-ada2-full-ip.hdf5`
- `msspacev-100M-i8-euclidean.hdf5`
- `cohere-768-angular.hdf5`
- `youtube-15M-angular.hdf5`
- `agnews-mxbai-1024-euclidean.hdf5`
- `landmark-nomic-768-angular.hdf5`

Environment variables:
- `SAGE_DATA_DIR`: dataset HDF5 root. Defaults to `$SAGE_ROOT/datasets`.
- `SAGE_INDEX_DIR`: hnswlib index root. Defaults to `<repo>/index`.
- `SAGE_FAISS_INDEX_ROOT` or `FAISS_INDEX_ROOT`: FAISS HNSW index root. Defaults to `$SAGE_INDEX_DIR/faiss_m32_efc500_main8_20260707/index`.
- `FAISS_PYTHON_PATH`: patched FAISS Python package. Defaults to `<repo>/faiss/build_sage_avx512/faiss/python`.
- `SAGE_PYTHON`: Python executable for child cells. Defaults to the Python executable used to launch the runner.

Run 24-thread cells from the repository root:

```bash
./setup.sh
source ./sage_env.sh
./run_main8.sh
```

The underlying runner remains available when working inside this directory:

```bash
python3 run_main8_online24_20260707.py run-all
```

Run single-thread cells:

```bash
cd $SAGE_ROOT/final_experiments/01_main_results
python3 run_main8_online1.py run-all
```

Run one backend cell from the repository root:

```bash
./run_main8.sh run-cell --cell hnswlib
./run_main8.sh run-cell --cell faiss
```

Single-thread cells still use the local runner directly:

```bash
python3 run_main8_online1.py run-cell --cell hnswlib
python3 run_main8_online1.py run-cell --cell faiss
```
