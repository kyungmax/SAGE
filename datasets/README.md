# Datasets

Place HDF5 datasets here or set `SAGE_DATA_DIR` before loading `sage_env.sh`.
Backend runners generally expect `train`, `test` and `neighbors` arrays; check the
selected runner's `--help` and dataset configuration for its metric and paths.

RaBitQ accepts `RABITQ_DATA_DIR` or `--data-dir`.
The main controller's dataset catalog is
[`experiments_scripts/main/config/main6_online24/datasets.json`](../experiments_scripts/main/config/main6_online24/datasets.json).
Edit the catalog or supply another one for your host.

Dataset payloads and local links are excluded from Git.
