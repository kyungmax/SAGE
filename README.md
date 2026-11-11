# Start Wide, Cut Early: Self-Calibrating Adaptive HNSW Search

SAGE implements adaptive HNSW search in FAISS, hnswlib, and RaBitQ.
Run the commands below from the repository root.

## Build

Use Linux, a C++20 compiler, CMake 3.24+, and an AVX-512 CPU for the supplied
paper runners. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake swig pkg-config \
  python3-dev python3-venv libopenblas-dev libomp-dev numactl \
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

Edit [experiment.json](experiments_scripts/01_main_results/config/main6_online24/experiment.json)
for your host: set `timing.cpus`, `timing.numa_node`, `offline_threads`, and
`online_threads`, with thread counts equal to the CPU count. The supplied
configuration uses 24 physical cores (`1,3,...,47`) on NUMA node 1.

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash experiments_scripts/01_main_results/run.sh
```

Pass `--config /path/experiment.json --datasets-config /path/datasets.json`
to each command when using copies. `--dry-run` prints the plan; `--preflight`
checks prepared inputs. Use a fresh `--output /path/new-run` for another run.

The main runner executes FAISS, hnswlib, PiP, and ABS sequentially at k=10,
with one warmup and five measured calls. Results are written to
`runs/01_main_results/k10/{faiss,hnswlib,pip,abs}/summary.csv`; policies and raw
measurements are kept alongside them. Run only one cell with:

```bash
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss
```

Other cell names are `hnswlib`, `pip`, and `abs`. Run FAISS first because
PiP/ABS consume its target file. Keep timing jobs sequential on the same cores.

For Recall@100, prepare YouTube's top100 ground truth, then run FAISS using the
new catalog (pass your input catalog to preparation with `--catalog`):

```bash
bash scripts/python.sh experiments_scripts/01_main_results/prepare_ground_truth.py \
  --dataset youtube-15M-angular --k 100 --output runs/inputs/youtube_top100
bash experiments_scripts/01_main_results/run.sh run-cell --cell faiss --k 100 \
  --datasets-config runs/inputs/youtube_top100/datasets.json
```

Ada-EF and DARTH run separately after the FAISS main cell completes. Use matching
`numactl` placement and `--threads N`; supply your catalog with `--catalog`.
These runners do not read the main `experiment.json`:

```bash
bash scripts/python.sh experiments_scripts/01_main_results/run_adaef.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --output-root runs/adaef
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --output runs/darth/glove
```

Repeat per dataset; Ada-EF supports the five cosine datasets only. Both commands
accept `--preflight` to check inputs before execution.

Additional experiment entry points are listed below. Run each with
`bash scripts/python.sh PATH --help` for its required inputs and output options.

| Experiment | Entry point | Inputs |
| --- | --- | --- |
| Detailed analysis | [run_drilldown.py](experiments_scripts/02_detailed_analysis/run_drilldown.py), [run_misclassification.py](experiments_scripts/02_detailed_analysis/run_misclassification.py), [run_shadow.py](experiments_scripts/02_detailed_analysis/run_shadow.py) | Dataset, FAISS graph, completed main policy and adjacent receipts |
| Calibration | [03_calibration/run.py](experiments_scripts/03_calibration/run.py) | `--dataset`, `--catalog`, `--experiment all`, `--output-root` |
| Parameter sensitivity | [04_parameter_sensitivity/run.py](experiments_scripts/04_parameter_sensitivity/run.py) | `--dataset`, `--catalog`, `--output-root` |
| Robustness | [05_robustness/run.py](experiments_scripts/05_robustness/run.py) | `embedding` or `rabitq`, with its own [input catalogs](experiments_scripts/05_robustness/config/) |
