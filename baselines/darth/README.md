# Original DARTH training tools

This source tree retains the original DARTH observation-data collector and
training tools. The separate [FAISS 1.14.1 runtime](../darth_faiss1141/README.md)
can search with a trained model but does not replace this training path.
The nested `.git` directory and upstream license are preserved.

From this folder:

```bash
bash scripts/build_parallel.sh
build-parallel/hnsw-test/hnsw_test --help
```

Building needs CMake 3.24+, C++17, OpenMP, BLAS/LAPACK, and LightGBM. The build
finds LightGBM in the selected conda environment or accepts explicit
`LIGHTGBM_INCLUDE_DIR` and `LIGHTGBM_LIB` paths. `DARTH_BUILD_DIR` and
`DARTH_BUILD_JOBS` override its output directory and parallelism.

`hnsw_test --mode early-stop-training` collects training observations;
`--mode early-stop-testing` searches with `--predictor-model-path` and explicit
prediction intervals. Dataset names and prepared fvecs/ivecs layouts are
mapped in `hnsw-test/VectorDataLoader.cpp`. Supply `--dataset-dir-prefix` and
`--index-filepath` for the input files.

The original model-training entry point is
`notebooks_scripts/predictor_training.py`. Its dataset locations and choices
are configured inside that file; it is not a generic data-path CLI. It needs
NumPy, pandas, LightGBM, scikit-learn and Dask; install them with
`python -m pip install -r requirements.txt`. No models, datasets or previous
measurement outputs are bundled.
