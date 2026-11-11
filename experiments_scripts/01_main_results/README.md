# Main results

Run from the repository root after the [build and data setup](../../README.md).

## 1. Configure inputs and timing

Set the paths in [datasets.json](config/main6_online24/datasets.json) and the
thread counts in [experiment.json](config/main6_online24/experiment.json).
The runner applies the CPU/NUMA settings from that configuration.
For copies, pass `--config` and `--datasets-config` to each `run.sh` command.
Use a fresh `--output` directory for another run; the examples use default paths.
If you change it, update the later `--targets` and `--input` paths to match.

## 2. Run FAISS, hnswlib, PiP, and ABS

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

This runs all six datasets in that method order, including calibration/tuning
and final measurements. Results: `runs/01_main_results/k10/METHOD/summary.csv`.
To select one method, use `run-cell --cell faiss|hnswlib|pip|abs` (choose one).
FAISS must finish before PiP/ABS because it produces `faiss/targets.json`.

## 3. Run Ada-EF and DARTH separately

Build their [optional runtimes](../../README.md#build) first. After step 2,
run these commands per dataset. Pass the same catalog with `--catalog` if using
a copy, and set `--threads` to match step 1.
Add `--preflight` to check inputs before execution.

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_adaef.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --threads 24 --output-root runs/adaef
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --threads 24 --output runs/darth/glove
```

- Ada-EF runs offline calibration then online measurement for the five cosine
  datasets. Result: `runs/adaef/DATASET/k10/target_hits_N/summary.csv`.
- DARTH prepares holdouts, trains, validates, and tests for each of the six
  datasets at k=10. Choose a different `--output` per dataset;
  the example produces `runs/darth/glove/summary.csv`.

## 4. Summarize matched-recall comparisons

```bash
bash scripts/python.sh experiments_scripts/01_main_results/summarize_main.py \
  --input runs/01_main_results/k10 \
  --output runs/01_main_results/k10/matched_recall.csv
```

Include Ada-EF/DARTH by adding `--summary /path/to/summary.csv` once per completed
dataset/method to that command. The output CSV must not already exist.

For Recall@100, follow [the top100 preparation and FAISS commands](../../README.md#recall100)
first, then summarize `runs/01_main_results/k100` in the same way.
