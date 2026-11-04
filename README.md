# SAGE

FAISS, hnswlib and RaBitQ implementations of SAGE adaptive HNSW search.
The policy uses raw CFR observations 1–16, shadow128 and Top1 second chance.

## Layout

| Directory | Purpose |
| --- | --- |
| `faiss/`, `hnswlib/`, `rabitq/` | Native search engines and Python bindings |
| `experiments_scripts/common/` | Shared calibration and search policy |
| `experiments_scripts/{faiss,hnswlib,rabitq}/` | Index builders and backend runners |
| [`experiments_scripts/main/`](experiments_scripts/main/README.md) | Experiment controllers, configuration and launch scripts |
| [`baselines/`](baselines/README.md) | PiP, ABS, Ada-EF and DARTH build/runtime code |
| `scripts/` | Build, Python environment and execution checks |
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
bash scripts/python.sh experiments_scripts/hnswlib/run_final_policy_benchmark.py --help
bash scripts/python.sh experiments_scripts/main/run_search_main6.py --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner calibrate --help
bash scripts/python.sh -m experiments_scripts.rabitq.runner sweep --help
```

The [main controllers](experiments_scripts/main/README.md) require explicit input
configuration and output locations. Their retained baseline build paths and NUMA
settings target the original AVX512 host; review them before using another host.
Warmup and measured-run counts are explicit settings; inspect them before each run.
