# Start Wide, Cut Early: Self-Calibrating Adaptive HNSW Search

SAGE implements adaptive HNSW search in FAISS, hnswlib, and RaBitQ.
Run the commands below from the repository root.

## Experiments

Each directory guide below lists its prerequisites and execution order.

| Directory | Experiments | Entry points |
| --- | --- | --- |
| [01_main_results](experiments_scripts/01_main_results/README.md) | Recall/latency curves, matched-recall comparisons, Recall@100 | [run.sh](experiments_scripts/01_main_results/run.sh), [run_adaef.py](experiments_scripts/01_main_results/run_adaef.py), [run_darth_pipeline.py](experiments_scripts/01_main_results/run_darth_pipeline.py) |
| [02_detailed_analysis](experiments_scripts/02_detailed_analysis/README.md) | Performance by query difficulty, misclassification, shadow contribution | [run_drilldown.py](experiments_scripts/02_detailed_analysis/run_drilldown.py), [run_misclassification.py](experiments_scripts/02_detailed_analysis/run_misclassification.py), [run_shadow.py](experiments_scripts/02_detailed_analysis/run_shadow.py) |
| [03_calibration](experiments_scripts/03_calibration/README.md) | Recommended ef quality, offline calibration time, sampling stability, pseudo-ground-truth validation | [run.py](experiments_scripts/03_calibration/run.py) |
| [04_parameter_sensitivity](experiments_scripts/04_parameter_sensitivity/README.md) | CFR window W, number of tiers B, safety margin g | [run.py](experiments_scripts/04_parameter_sensitivity/run.py) |
| [05_robustness](experiments_scripts/05_robustness/README.md) | Embedding quality, quantized HNSW (RaBitQ) | [run.py](experiments_scripts/05_robustness/run.py), [report.py](experiments_scripts/05_robustness/report.py) |

## Build

Use Linux, a C++20 compiler, CMake 3.24+, and an AVX-512 CPU for the supplied
paper runners. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake swig pkg-config \
  python3-dev python3-venv libopenblas-dev libomp-dev \
  libhdf5-dev libeigen3-dev libboost-all-dev

python3 -m venv .venv
source .venv/bin/activate
export SAGE_PYTHON="$PWD/.venv/bin/python"
export SAGE_OPT_LEVEL=avx512
export SAGE_DATA_DIR="$PWD/datasets"  # change to your data volume
export SAGE_INDEX_DIR="$PWD/index"   # change to your index volume

"$SAGE_PYTHON" -m pip install -r requirements.txt
bash setup.sh
source ./sage_env.sh
bash scripts/build.sh all
bash scripts/check_env.sh
```

`all` builds FAISS, hnswlib, PiP, and ABS. Use `bash scripts/python.sh` for
Python commands so they load this checkout's bindings. Build optional engines
only when needed:

```bash
bash scripts/build.sh rabitq
bash baselines/ada-ef/scripts/build.sh
bash scripts/python.sh -m pip install lightgbm scikit-learn
bash baselines/darth_faiss1141/scripts/build.sh
```

The DARTH build includes its training collector. Set `SAGE_BUILD_JOBS` to
change the main build's parallelism (default: 8).

## Datasets

Download the inputs below into `SAGE_DATA_DIR`. The runners expect HDF5 arrays
named `train`, `test`, and `neighbors`. Metrics are cosine except for SpaceV (L2).

| Dataset | Vectors / dimensions | Download | Expected HDF5 filename |
| --- | --- | --- | --- |
| GooAQ | 1,475,024 / 768 | [HDF5](https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/gooaq-distilroberta-768-normalized.hdf5?download=true) | `gooaq-distilroberta-768-normalized.hdf5` |
| GloVe100 | 1,183,514 / 100 | [HDF5](https://ann-benchmarks.com/glove-100-angular.hdf5) | `glove-100-angular.hdf5` |
| CohereWiki | 10,000,000 / 768 | [Train shard 00](https://assets.zilliz.com/benchmark/cohere_large_10m/train-00-of-10.parquet), [queries](https://assets.zilliz.com/benchmark/cohere_large_10m/test.parquet), [ground truth](https://assets.zilliz.com/benchmark/cohere_large_10m/neighbors.parquet) | `cohere-768-angular.hdf5` |
| YouTube | 15,686,309 / 1024 | [15M dataset](https://huggingface.co/datasets/dnotitia/SeahorseDB-dataset/tree/main/youtube-15M-dataset) | `youtube-15M-angular.hdf5` |
| MSMARCO-Qwen | 8,840,823 / 1024 | [HDF5](https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/msmarco-qwen-1024-normalized.hdf5?download=true) | `msmarco-qwen-1024-normalized.hdf5` |
| SpaceV | 100,000,000 / 100 | [SPACEV1B source](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) | `msspacev-100M-i8-euclidean.hdf5` |

CohereWiki has 10 training shards: download `train-00-of-10.parquet` through
`train-09-of-10.parquet` from the same directory, plus the queries and ground truth.

CohereWiki, YouTube, and SpaceV links provide source data requiring conversion
to the HDF5 layout above; they are not the prepared experiment files. Match the
corpus/query split in the [dataset catalog](experiments_scripts/01_main_results/config/main6_online24/datasets.json)
and compute ground truth for that corpus. SpaceV uses a 100M subset of the
larger source. YouTube requires Hugging Face login and acceptance of the dataset
access conditions. Conversion scripts for these three inputs are not included.

## Indexes

Supply prebuilt graphs or build both backends for each dataset. For example:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py \
  --dataset "$SAGE_DATA_DIR/glove-100-angular.hdf5" --space cosine \
  --index "$SAGE_INDEX_DIR/glove.faiss" --state-dir runs/build/glove-faiss \
  --m 32 --ef-construction 500 --threads 24
bash scripts/python.sh experiments_scripts/hnswlib/build_benchmark_index.py \
  --dataset "$SAGE_DATA_DIR/glove-100-angular.hdf5" --space cosine \
  --index "$SAGE_INDEX_DIR/glove.hnsw" --state-dir runs/build/glove-hnsw \
  --m 32 --ef-construction 500 --threads 24
```

Use `--space l2` for SpaceV. Copy
[datasets.json](experiments_scripts/01_main_results/config/main6_online24/datasets.json)
and replace `dataset_file` with an absolute `dataset_path`, and each backend's
`file` with an absolute `path`. Remove the old keys; they take precedence over
absolute paths. Index builders do not update this catalog.

## Run

### Common settings

Set `offline_threads` and `online_threads` in
[experiment.json](experiments_scripts/01_main_results/config/main6_online24/experiment.json)
for your host (default: 24 each). CPU and NUMA binding are disabled by default:
`timing.cpus` and `timing.numa_node` are `null`. Explicit binding remains optional;
setting a NUMA node requires `numactl`.

- **Main runner:** pass `--config /path/experiment.json --datasets-config /path/datasets.json`
  when using copies, and a fresh `--output /path/new-run` for another run.
- **Ada-EF / DARTH:** build their optional runtimes first. Pass the same catalog
  with `--catalog` and set `--threads`; these runners do not read `experiment.json`.
  The examples below use 24 threads.
- **Timing:** one warmup and five measured calls. Keep timing jobs sequential
  on the same machine.

### Full main run

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

`--dry-run` prints the plan; `--preflight` checks prepared inputs. The full run
executes FAISS, hnswlib, PiP, and ABS sequentially at k=10. Ada-EF and DARTH use
the separate commands below.

Results are written to
`runs/01_main_results/k10/{faiss,hnswlib,pip,abs}/summary.csv`; policies and raw
measurements are kept alongside them. To run methods individually:

### FAISS and hnswlib

Calibrate SAGE and measure the SAGE/vanilla recall-latency curves on each backend.
Run FAISS first: its `runs/01_main_results/k10/faiss/targets.json` supplies the
SAGE recall target at the recommended ef for all four baselines below.

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss
bash experiments_scripts/01_main_results/run.sh run-cell --cell hnswlib
```

### PiP

- **Input:** the same FAISS graph and the completed FAISS target from this run.
- **Tuning:** fix the overlap threshold at `0.95`; tune `efSearch` and patience
  against the target using evaluation ground truth. Select the fastest passing
  configuration, then measure it in a separate process.

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell pip
```

### ABS

- **Input:** the same FAISS graph and the completed FAISS target from this run.
- **Tuning:** tune the distance-slack parameter `gamma` using evaluation ground
  truth. Select the smallest tested value meeting the target, then measure it
  in a separate process. ABS does not use `efSearch`.

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell abs
```

### Ada-EF

- **Input:** the catalog's existing hnswlib graph and the FAISS target manifest.
- **Stages:** offline calibration of the score-to-ef table, followed by online
  measurement. Run for the five cosine datasets; SpaceV/L2 is unsupported.

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_adaef.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --threads 24 --output-root runs/adaef
```

Repeat per dataset. Add `--preflight` to check inputs, or use `--phase offline`
and `--phase online` to split the stages. Results:
`runs/adaef/DATASET/k10/target_hits_N/summary.csv`.

### DARTH

- **Preparation:** select 10,000 learning and 1,000 validation queries, build an
  auxiliary graph excluding those holdouts, compute exact ground truth, and
  train the LightGBM model and derive prediction intervals.
- **Evaluation:** validate the frozen model, then measure the official test
  queries on the original full FAISS graph using the manifest's recommended ef
  and target recall. This pipeline supports k=10 on all six datasets.

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --threads 24 --output runs/darth/glove
```

Repeat per dataset with a new output directory. Add `--preflight` to check inputs;
`--stop-after prepare` and `--resume` split preparation from the remaining stages.
Results: `runs/darth/glove/summary.csv`.

### Recall@100

Prepare YouTube's top100 ground truth, then run FAISS using the new catalog
(pass your input catalog to preparation with `--catalog`):

```bash
bash scripts/python.sh experiments_scripts/01_main_results/prepare_ground_truth.py \
  --dataset youtube-15M-angular --k 100 --output runs/inputs/youtube_top100
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss --k 100 \
  --datasets-config runs/inputs/youtube_top100/datasets.json
```

### Other experiments

For the other [experiments](#experiments), run `bash scripts/python.sh PATH --help`
for required inputs and output options. Detailed analysis consumes a completed
FAISS main policy and its adjacent receipts. Calibration and parameter sensitivity
use `--dataset`, `--catalog`, and `--output-root`; calibration also selects
`--experiment`. Robustness uses `embedding` or `rabitq` with its own
[input catalogs](experiments_scripts/05_robustness/config/).
