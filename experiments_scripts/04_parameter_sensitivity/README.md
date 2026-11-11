# Parameter sensitivity

Run all commands from the repository root.

## 1. Prepare inputs

Follow the [main README](../../README.md) to build FAISS and prepare the HDF5
dataset, FAISS index, and dataset catalog. This experiment performs its own
calibration; neither the main-results nor calibration experiment must run first.

## 2. Run one dataset

Set the CPU list, NUMA node, and `--threads` to the same placement used for the
main experiments. The example uses 24 physical cores on NUMA node 1:

```bash
numactl --physcpubind=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47 --membind=1 \
  bash scripts/python.sh experiments_scripts/04_parameter_sensitivity/run.py \
  --dataset glove-100-angular --threads 24 \
  --catalog experiments_scripts/01_main_results/config/main6_online24/datasets.json \
  --output-root runs/04_parameter_sensitivity
```

One invocation runs all W, B, and g settings and the vanilla comparison.
Repeat with each dataset identifier in your catalog, using a fresh output root
for reruns. This runner has no `--preflight` option.

## 3. Read results

Results are under `runs/04_parameter_sensitivity/DATASET/`:

- `summary.csv` / `summary.json`: recall, latency, and speedup for each setting.
- `timings.csv`: individual warmup and measured calls.
- `default_policy.json`, `policy_*.json`, `protocol.json`: calibration and settings.
