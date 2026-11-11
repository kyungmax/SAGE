# Robustness

Run all commands from the repository root. Common setup is in the
[main README](../../README.md).

## 1. Prepare inputs

- **Embedding:** edit [embedding_datasets.json](config/embedding_datasets.json)
  for the aligned embeddings of the same ordered corpus/query split, their FAISS
  indexes, and exact ground truth for each embedding. Each HDF5 must record an
  exact `ground_truth_method` attribute.
- **RaBitQ:** edit [rabitq_datasets.json](config/rabitq_datasets.json) for the
  datasets and prepared 8-bit RaBitQ indexes, then build the optional runtime:

```bash
bash scripts/build.sh rabitq
```

Paths use `SAGE_DATA_DIR` / `SAGE_INDEX_DIR`, or absolute `dataset_path` / index
`path` entries. Set offline/online thread counts in
[experiment.json](config/experiment.json). CPU/NUMA binding is disabled by default.
Both studies calibrate independently.

## 2. Check and run each study

Each subcommand defaults to all entries in its own catalog. Run timing jobs
sequentially, and use fresh output directories for another run:

```bash
bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding --preflight
bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding \
  --output runs/05_robustness/embedding

bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq --preflight
bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq \
  --output runs/05_robustness/rabitq
```

For copied inputs, pass `--config` and `--catalog` to both checks and runs.
Use `--datasets NAME1,NAME2` for a subset. Optional `--cpus` enables CPU binding
and sets both thread counts; `--numa-node` adds NUMA binding and requires `numactl`.

## 3. Generate reports

```bash
bash scripts/python.sh experiments_scripts/05_robustness/report.py \
  --run runs/05_robustness/embedding --output runs/05_robustness/embedding-report
bash scripts/python.sh experiments_scripts/05_robustness/report.py \
  --run runs/05_robustness/rabitq --output runs/05_robustness/rabitq-report
```

Runs contain `summary.csv`, `offline_times.csv`, and `raw_timings.csv`.
Each report directory must be new; it contains `curves.csv`,
`matched_recall.csv`, and `validation.json`.
