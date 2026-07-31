# 07 DARTH / Ada-EF Scripts

This directory contains script-only artifacts for the paper SOTA comparison against DARTH and Ada-EF. It intentionally excludes generated JSONs, logs, model files, indexes, build directories, plots, and old broad-scope reruns.

Imported scope:

- datasets: `cohere-768-angular.hdf5` and `msmarco-v1-openai-ada2-full-ip.hdf5`
- target recall: `0.99`
- DARTH backend: FAISS, `efSearch=1000`, `M=32`, `efConstruction=500`
- Ada-EF backend: hnswlib, `M=32`, `efConstruction=500`
- SIMD: on via AVX512 build helpers and default binary paths
- offline threads: 24
- online latency threads: 1, matching the paper text
- query scope: full HDF5 `test` set (`cohere=1000`, `msmarco=6980` when the paper datasets are present)

Build the local SIMD binaries if they are missing:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
./scripts/build_darth_simd_avx512.sh
./scripts/build_adaef_simd_avx512.sh
```

Run preflight:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
python3 scripts/preflight_darth_adaef_cohere_msmarco.py
```

Run DARTH only:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
./run_darth_cohere_msmarco_simd_target099.sh
```

Run Ada-EF only:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
./run_adaef_cohere_msmarco_simd_target099.sh
```

Run both baselines:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/07_darth_ada-ef
./run_darth_adaef_cohere_msmarco_simd_target099.sh
```

Important environment overrides:

- `SAGE_PROJECT_ROOT`: optional repository-root override. Defaults to the detected checkout root.
- `SAGE_DATA_DIR`: HDF5 dataset root.
- `SAGE_FAISS_INDEX_ROOT`: reusable FAISS HNSW index root for DARTH.
- `SAGE_HNSWLIB_INDEX_ROOT`: reusable hnswlib index root for Ada-EF symlinks.
- `SAGE_DARTH_ROOT`, `SAGE_DARTH_SIMD_BUILD_ROOT`, `SAGE_DARTH_BIN`, `SAGE_DARTH_FAISS_LIB_DIR`: DARTH source/build/binary overrides.
- `SAGE_ADAEF_ROOT`, `SAGE_ADAEF_BACKEND_BIN`: Ada-EF source/binary overrides.

Primary outputs:

- `darth_target099_simd_cohere_msmarco_fullquery/summary/darth_target099_simd_fullquery_summary.csv`
- `adaef_target099_simd_cohere_msmarco_fullquery/adaef/results/offline_cost_summary.csv`
- per-dataset wrapper JSONs under each result tree.

`PAPER_TEX_CHECK.md` records the consistency check against the paper experiment section.
