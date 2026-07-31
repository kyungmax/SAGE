# Six-Dataset Recall-Latency Benchmark

This directory contains the experiment-script run/build scripts for the six-dataset combined recall-latency result.

Scripts:
- `run_combined_recall_latency_six.py`: runs missing hnswlib/Faiss final sweeps, then rebuilds the combined outputs.
- `build_combined_recall_latency_six.py`: rebuilds only the combined CSV/PNG/PDF from existing sweep CSVs.

By default the scripts detect the SAGE checkout root and write outputs under:

```text
$SAGE_PROJECT_ROOT/final_experiments/combined_recall_latency_six_m32_efc500
```

`SAGE_PROJECT_ROOT` is optional when running from inside the checkout. Set it to
override the detected repository root. Data and index defaults are code-aligned:

```bash
export SAGE_PROJECT_ROOT=/path/to/SAGE
export SAGE_DATA_DIR=$SAGE_PROJECT_ROOT/datasets
export SAGE_HNSWLIB_INDEX_ROOT=$SAGE_PROJECT_ROOT/index
export SAGE_FAISS_INDEX_ROOT=$SAGE_PROJECT_ROOT/index/m32_efc500_target095_adaef_darth_efs1000_20260603/darth/index
export FAISS_PYTHON_PATH=$SAGE_PROJECT_ROOT/faiss/build_sage_avx512/faiss/python
```

Ada-EF and DARTH are not rerun. Their copied result JSONs are read from the corresponding `final_experiments` directory.

Run:

```bash
cd $SAGE_PROJECT_ROOT
python3 experiments_scripts/six_dataset_recall_latency_benchmark/run_combined_recall_latency_six.py
```

Only rebuild the combined plot:

```bash
cd $SAGE_PROJECT_ROOT
python3 experiments_scripts/six_dataset_recall_latency_benchmark/run_combined_recall_latency_six.py --mode combine-only
```

