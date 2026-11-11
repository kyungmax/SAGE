# Calibration

Run all commands from the repository root.

## 1. Prepare inputs

Follow the [main README](../../README.md) to build FAISS and prepare the HDF5
dataset, FAISS index, and dataset catalog. This experiment calibrates its own
policy; a completed main-results run is not required.

## 2. Run one dataset

Set the CPU list, NUMA node, and `--threads` to the same placement used for the
main experiments. The example uses 24 physical cores on NUMA node 1:

```bash
numactl --physcpubind=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47 --membind=1 \
  bash scripts/python.sh experiments_scripts/03_calibration/run.py \
  --dataset glove-100-angular --experiment all --threads 24 \
  --catalog experiments_scripts/01_main_results/config/main6_online24/datasets.json \
  --output-root runs/03_calibration
```

Repeat with each dataset identifier in your catalog, using a fresh output root
for reruns. `all` runs recommendation quality/offline timing, sampling stability,
and pseudo-ground-truth validation. To run only one study, replace `all` with
`quality`, `stability`, or `pseudo-gt`. This runner has no `--preflight` option.

## 3. Read results

Results are under `runs/03_calibration/DATASET/EXPERIMENT/`:

- `summary.json`: recommendation quality, calibration cost, and selected studies.
- `offline_timings.csv`, `timings.csv`, `pseudo_exact_timings.csv`: calibration,
  online search, and pseudo/exact ground-truth timing for the selected studies.
- `stability_per_seed.csv`: per-seed sampling results; `policy.json`: calibrated policy.
