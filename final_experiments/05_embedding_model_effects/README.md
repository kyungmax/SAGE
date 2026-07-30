# 05 Embedding Model Effects Scripts

This directory contains script-only artifacts for the MSMARCO embedding-model experiment. It intentionally excludes generated CSVs, logs, indexes, caches, plots, and unrelated exploratory runs from the old experiment tree.

Imported scope:

- backend: FAISS only
- SIMD: on (`FAISS_OPT_LEVEL=AVX512` and `FAISS_PYTHON_PATH`)
- threads: 24 offline and 24 online threads
- corpus/query source: MSMARCO passages and plain dev queries
- models: mean-pooled GloVe, mean-pooled FastText, OpenAI ada-002, BGE-M3, EmbeddingGemma-300M
- ground truth: each HDF5 is evaluated against its own exact nearest-neighbor ground truth
- default index: `M=32, efConstruction=500`
- default sweep: `efSearch=64,80,96,128,160,192,256,320,384,512,640,768,896,1024`
- default calibration: `ncal=100`, `B=4`, `g=2`, `[4,16]`, `alpha=0.8`

Default dataset files under `SAGE_DATA_DIR`:

```text
msmarco-v1-glove6b300d-full-ip.hdf5
msmarco-v1-fasttext-cc300d-full-ip.hdf5
msmarco-v1-openai-ada2-full-ip.hdf5
marco_embeddings/msmarco-v1-bge-m3-fp32-dev6980-ip.hdf5
marco_embeddings/msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip.hdf5
```

Run a preflight check before the full sweep:

```bash
cd $SAGE_ROOT/final_experiments/05_embedding_model_effects
python3 scripts/preflight_msmarco_embedding_models.py
```

Missing HDF5 files must be prepared before running. Missing FAISS indexes are acceptable by default: the runner builds them on first use under `SAGE_MSMARCO_EMBEDDING_FAISS_INDEX_ROOT`.

Run the five-model FAISS sweep:

```bash
cd $SAGE_ROOT/final_experiments/05_embedding_model_effects
export OUT_ROOT=$PWD/msmarco_embedding_models_faiss_SIMD_on_24t
./run_msmarco_embedding_models_faiss_24t.sh
```

Useful environment overrides:

```bash
export SAGE_DATA_DIR=/path/to/datasets
export SAGE_MSMARCO_EMBEDDING_FAISS_INDEX_ROOT=/path/to/index/msmarco_embedding_models_faiss_m32_efc500_20260715/darth/index
export FAISS_PYTHON_PATH=/path/to/faiss/python
export OUT_ROOT=$PWD/msmarco_embedding_models_faiss_SIMD_on_24t
```

Primary outputs after running:

- `${OUT_ROOT}/sage_results/final/main_qps_latency_sweep.csv`
- `${OUT_ROOT}/sage_results/final/offline_recommended_efsearch.csv`
- `${OUT_ROOT}/sage_results/final/offline_predicted_recall_curve.csv`
- `${OUT_ROOT}/logs/run_msmarco_embedding_models_faiss_24t.log`

Summarize iso-recall speedups from the final sweep CSV:

```bash
python3 scripts/summarize_msmarco_embedding_model_effects.py --final-dir "$OUT_ROOT/sage_results/final"
```

Optional input-preparation helpers:

- `scripts/prepare_msmarco_glove_static_hdf5.py`: build the mean-pooled GloVe HDF5 from MSMARCO TSVs and GloVe vectors.
- `scripts/download_msmarco_fasttext_model.sh`: download and unpack `cc.en.300.bin`.
- `scripts/build_msmarco_fasttext_static_hdf5.sh`: build the mean-pooled FastText HDF5.
- `scripts/make_msmarco_embedding_dev6980_subsets.py`: build dev-query subsets for precomputed BGE/Gemma MSMARCO embeddings.

`PAPER_TEX_CHECK.md` records the paper-section consistency check for this imported scope.
