# Baseline runtimes

Commands below run from the `sage_icde` root. Each FAISS-based baseline has its
own source and build; load PiP and ABS in separate Python processes.

| Directory | Runtime |
| --- | --- |
| [pip](pip/README.md) | PiP top-k set-overlap stopping, default threshold 0.95 |
| [abs](abs/README.md) | ABS search with a caller-supplied gamma |
| [ada-ef](ada-ef/README.md) | Ada-EF index build, offline calibration, online search |
| [darth_faiss1141](darth_faiss1141/README.md) | DARTH search on FAISS 1.14.1 |
| [darth](darth/README.md) | Original DARTH observation-data collection and model-training tools |

```bash
bash scripts/build.sh pip
bash scripts/build.sh abs
bash baselines/ada-ef/scripts/build.sh
bash baselines/darth_faiss1141/scripts/build.sh
bash baselines/darth/scripts/build_parallel.sh
```

The PiP/ABS batch runners and parameter-tuning entry points are under
[experiments_scripts/main](../experiments_scripts/main/README.md).
Ada-EF and DARTH accept explicit datasets, indexes and models through their
standalone interfaces. Those inputs and generated measurements are not source
code and must be supplied separately.

Quick local runtime checks:

```bash
bash scripts/python.sh baselines/pip/tests/test_pip.py -q
bash scripts/python.sh baselines/abs/tests/test_abs.py -q
```
