# Six-Dataset Recall-Latency Benchmark

This directory contains the experiment-script run/build scripts for the six-dataset combined recall-latency result.

Scripts:
- `run_combined_recall_latency_six.py`: runs missing hnswlib/Faiss final sweeps, then rebuilds the combined outputs.
- `build_combined_recall_latency_six.py`: rebuilds only the combined CSV/PNG/PDF from existing sweep CSVs.

Outputs are written under:

```text
../../final_experiments/combined_recall_latency_six_m32_efc500
```

Ada-EF and DARTH are not rerun. Their copied result JSONs are read from the corresponding `final_experiments` directory.

Run:

```bash
python3 experiments_scripts/six_dataset_recall_latency_benchmark/run_combined_recall_latency_six.py
```

Only rebuild the combined plot:

```bash
python3 experiments_scripts/six_dataset_recall_latency_benchmark/run_combined_recall_latency_six.py --mode combine-only
```

