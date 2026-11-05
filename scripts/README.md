# Build and runtime helpers

Run these commands from the repository root. Paper experiments live in
[experiments_scripts](../experiments_scripts/README.md).

| Script | Purpose |
| --- | --- |
| `build.sh` | Compile local native engines and Python bindings |
| `python.sh` | Run the selected Python with this checkout's FAISS, hnswlib and RaBitQ on its import path |
| `check_env.sh` | Check local FAISS/hnswlib imports and small synthetic searches, then PiP/ABS tests |

## Build

Install the [root build dependencies](../README.md#build), then:

```bash
export SAGE_PYTHON=python3
export SAGE_OPT_LEVEL=avx512
bash scripts/build.sh all
bash scripts/build.sh rabitq
```

`all` includes FAISS, hnswlib, PiP and ABS. Each is also an individual target.
RaBitQ is separate; Ada-EF and DARTH use the
[baseline build commands](../baselines/README.md). `SAGE_BUILD_JOBS` defaults to 8.
Core FAISS/PiP/ABS builds accept `generic`, `avx2` and `avx512`; the supplied
paper runner preflights expect the AVX512 build. Build support for another
instruction set alone does not change those experiment requirements.

## Run and check

```bash
bash scripts/python.sh experiments_scripts/03_calibration/run.py --help
bash scripts/check_env.sh
bash scripts/python.sh -c 'from experiments_scripts.rabitq.final_policy import load_library; print(load_library().__file__)'
```

`check_env.sh` needs all four components from `build.sh all`. It does not check
RaBitQ, Ada-EF, DARTH, external datasets or full experiment reproducibility.
The last command checks the local RaBitQ binding and required SAGE API only.

`python.sh` sources [sage_env.sh](../sage_env.sh), including optional
`sage.local.env`. Set `SAGE_PYTHON`, `SAGE_DATA_DIR` and `SAGE_INDEX_DIR` before
execution to select the interpreter and external inputs. Rebuild bindings
after changing Python ABI. [setup.sh](../setup.sh) creates the data/index
directories; it does not download datasets, build indexes or compile engines.
