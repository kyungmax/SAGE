# Ada-EF

The standalone `backend_runner` builds HNSW indexes, calibrates the score-to-ef
table, and searches with that table. Offline and online Ada-EF support cosine
and inner product; L2 is unsupported for those phases.

Dependencies: CMake, C++17, OpenMP, HDF5 C++, Eigen3 and Boost headers. From this
folder, build and inspect the CLI:

```bash
bash scripts/build.sh
build/backend_runner --help
```

`CONDA_PREFIX` supplies library/header search paths. `BUILD_DIR` and `JOBS`
control the build directory and parallelism; the default executable is
`build/backend_runner`.

Use an ANN-Benchmarks HDF5 file with `train`, `test` and `neighbors`. Run build,
offline, then online with the same dataset, index settings and artifact root:

```bash
build/backend_runner --phase build --dataset example \
  --data-path /path/example.hdf5 --experiments-root /path/adaef-run \
  --m 32 --ef-construction 500 --k 10 --num-threads 24
build/backend_runner --phase offline --dataset example \
  --data-path /path/example.hdf5 --experiments-root /path/adaef-run \
  --m 32 --ef-construction 500 --k 10 --num-threads 24 --expected-recall 0.99
build/backend_runner --phase online --dataset example \
  --data-path /path/example.hdf5 --experiments-root /path/adaef-run \
  --m 32 --ef-construction 500 --k 10 --num-threads 24 --expected-recall 0.99 \
  --parallel-queries --e2e-timing --warmup-runs 1 --measured-runs 5 \
  --output-json /path/adaef-run/online.json
```

Use a different artifact root for each target/configuration: table filenames
omit the target recall. The default statistics length is 1025. Offline tables
use `author-v1-zero-recall-v1`, which retains zero-recall samples; the original
secant update and no-gain exit are preserved. Offline calibration can therefore
stop below the requested recall. Online reports the achieved recall and does
not load tables from earlier revisions implicitly.
