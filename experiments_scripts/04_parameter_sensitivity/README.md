# Parameter sensitivity

Run all commands from the repository root.

## 1. Prepare inputs

Follow the [main README](../../README.md) to build FAISS and prepare the HDF5
dataset, FAISS index, and dataset catalog. This experiment performs its own
calibration; neither the main-results nor calibration experiment must run first.

## 2. Run one dataset

Set `--threads` to match the main experiments. The example uses 24 threads:

```bash
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
