# Benchmark execution

Run commands from the repository root with `scripts/python.sh`. These controllers
require NumPy, h5py, local native builds, external HDF5 datasets and built indexes.
The main controllers use the AVX512 FAISS/PiP/ABS builds and local hnswlib extension;
PiP and ABS run in separate Python processes. They use Linux CPU affinity and
`numactl` when NUMA placement is configured.

Review `config/main6_online24/experiment.json` and `datasets.json` before execution:
they specify dataset selection, thread/CPU placement, warmups and timed calls.
`SAGE_DATA_DIR` and `SAGE_INDEX_DIR` resolve catalog filenames. Custom catalogs may
instead supply absolute dataset and index paths. Native files must match the
catalog's recorded sizes, timestamps and graph parameters.

Inspect the bundled plan without running ANN:

```bash
bash experiments_scripts/main/main6_online24/run_all.sh --dry-run --k 10
bash scripts/python.sh experiments_scripts/main/preflight_main6_online24.py --help
```

The shell wrappers run the bundled configuration and write under repository-root
`runs/main/k10/` or `runs/main/k100/`. Preparation reports and launch state go under
`runs/main/preparation/main6_online24/`. Relative timing-lock paths are resolved
under `runs/main/`; absolute lock paths remain unchanged. For explicit configuration
and output locations, call the worker directly:

```bash
bash scripts/python.sh experiments_scripts/main/run_search_main6.py \
  --backend faiss --k 10 --config /path/to/experiment.json \
  --datasets-config /path/to/datasets.json --output runs/faiss

bash scripts/python.sh experiments_scripts/main/run_baselines_main6.py \
  --method pip --k 10 --config /path/to/experiment.json \
  --datasets-config /path/to/datasets.json --targets runs/faiss/targets.json \
  --output runs/pip
```

Use `--backend hnswlib` or `--method abs` for the other workers. Configuration must
enable execution, and output directories must be unused. PiP/ABS require the
matching FAISS target file produced by the same configuration and catalog.

`run_full_search.py` separates calibration and online execution for one dataset;
`run_topk_full_search.py` additionally supports k=1. Their `--dataset-config` takes
one dataset object, with resolved paths, rather than a catalog. Run each script
with `--help` for its required arguments.

RaBitQ entry points remain under `../rabitq/`. Its batch index preparer requires
explicit `--datasets` names and accepts a `--datasets-config` catalog.

```bash
bash scripts/python.sh experiments_scripts/rabitq/build_main_indexes.py prepare \
  --datasets-config /path/to/datasets.json \
  --datasets msmarco-qwen-1024-normalized,youtube-15M-angular,cohere-768-angular \
  --plan runs/rabitq_indexes/plan.json --destination /path/to/rabitq_indexes
```

The supplied catalog must contain all three requested datasets. `prepare` checks
the inputs and writes a build plan; `launch --plan runs/rabitq_indexes/plan.json`
starts the builds.
