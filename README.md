# SAGE

FAISS, hnswlib and RaBitQ implementations of SAGE adaptive HNSW search.
The policy uses raw CFR observations 1–16, shadow128 and Top1 second chance.

## Layout

| Directory | Purpose |
| --- | --- |
| `faiss/`, `hnswlib/`, `rabitq/` | Native search engines and Python bindings |
| `experiments_scripts/common/` | Shared calibration and search policy |
| `experiments_scripts/{faiss,hnswlib,rabitq}/` | Index builders, backend adapters and RaBitQ sweep |
| [`experiments_scripts/01_main_results/` … `05_robustness/`](experiments_scripts/README.md) | Paper-ordered experiment runners, configuration and analysis |
| [`baselines/`](baselines/README.md) | PiP, ABS, Ada-EF and DARTH build/runtime code |
| [`scripts/`](scripts/README.md) | Build, Python environment and execution checks |
| [`datasets/`](datasets/README.md), `index/` | Local data and index locations, excluded from Git |

Saved results, figures, paper exports and historical archives are not included.
Supply datasets and indexes separately; benchmark commands create new outputs.

## Build

Install a C++20 compiler, CMake, SWIG, OpenMP, OpenBLAS and Python development headers.
From this repository root:

```bash
python3 -m pip install -r requirements.txt
export SAGE_PYTHON=python3
export SAGE_OPT_LEVEL=avx512  # generic / avx2 are also supported by core builds
bash setup.sh
bash scripts/build.sh all
bash scripts/build.sh rabitq
bash scripts/check_env.sh
```

`all` builds FAISS, hnswlib, PiP and ABS. RaBitQ is a separate target.
Ada-EF and DARTH have their own [baseline build instructions](baselines/README.md).
`SAGE_BUILD_JOBS` defaults to 8. Rebuild native binaries for another machine or Python ABI.

`SAGE_DATA_DIR` and `SAGE_INDEX_DIR` configure shared data/index locations.
RaBitQ also accepts `RABITQ_DATA_DIR`, `RABITQ_INDEX_DIR`, `--data-dir` and `--index-dir`.
`sage.local.env` is an optional local override and is excluded from Git.

## Run

Display backend options without starting an experiment:

```bash
bash scripts/python.sh experiments_scripts/faiss/build_benchmark_index.py --help
bash scripts/python.sh experiments_scripts/hnswlib/build_benchmark_index.py --help
bash scripts/python.sh experiments_scripts/01_main_results/run_search_main6.py --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner calibrate --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner sweep --help
```

The [experiment guide](experiments_scripts/README.md) maps the paper sections to
their runnable entry points. Main results, detailed analysis, calibration,
parameter sensitivity and robustness write new outputs under `runs/`.
Timing defaults to five measured calls plus a separate warmup; main comparisons
use matched recall and the shadow study uses the same ef.

The [DARTH pipeline](experiments_scripts/01_main_results/README.md#ada-ef-and-darth)
prepares disjoint learning/validation inputs, trains the model and measures it
on the original full graph using the FAISS main cell's target.

The standalone RaBitQ `sweep` also defaults to five measured calls after one
warmup. It retains its median summary and excludes query preprocessing from
timing; use `05_robustness/run.py rabitq` for the paper's mean and timing scope.

Inspect the main plan before execution:

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
```

The main catalog covers GooAQ, GloVe100, Cohere, YouTube15M, MSMARCO-Qwen and
SpaceV. RaBitQ covers MSMARCO, YouTube and Cohere. Input catalogs, baseline
models and CPU/NUMA settings must match your machine; the supplied main
configuration targets the original AVX512 host.
