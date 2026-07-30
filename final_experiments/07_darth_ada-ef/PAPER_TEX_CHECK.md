# Paper Tex Consistency Check

Checked against `paper_tex/6.Experimental Evaluation.tex`, subsection `Comparison with DARTH and Ada-EF` (`sec:exp:sota`).

The paper states:

- target recall is `0.99`;
- only `CohereWiki` and `MSMARCOV1` are retained for this comparison;
- DARTH uses FAISS and Ada-EF uses hnswlib;
- DARTH receives `efSearch=1000` for training and query time;
- offline preparation uses 24 threads;
- query latency is measured on the full query set on a single thread.

Script check:

- `run_darth_cohere_msmarco_simd_target099.py` restricts to `cohere,msmarco`, uses `target_recall=0.99`, `ef_search=1000`, offline/train threads `24`, and calls the DARTH lower-level runner once per dataset.
- DARTH full-query handling is fixed here: the old source wrapper had one global `--online-query-num 1000`, which is full for Cohere but not for MSMARCO. The new wrapper reads each HDF5 `test.shape[0]` and passes that value per dataset.
- `run_paper_offline_fromscratch.py` sets `OMP_NUM_THREADS=1` in the DARTH online phase, matching the paper's single-thread latency setting.
- `run_adaef_cohere_msmarco_simd_target099.py` restricts to `cohere,msmarco`, uses `target_recall=0.99`, offline threads `24`, and online threads `1` by default.
- Ada-EF full-query handling is already in the current backend: `run_online()` calls `load_full_dataset()` and summarizes `neighbors.rows()` as `metrics.query_count`. The wrapper verifies that reported count equals HDF5 `test.shape[0]`.
- SIMD-on is represented by the AVX512 build helpers and default binary paths: DARTH `baselines/darth/benchmarking-darth/build-simd-avx512/hnsw-test/hnsw_test`, Ada-EF `experiments_scripts/ada-ef/build-simd-avx512/backend_runner`.

Conclusion: the imported 08 scripts now match the paper scope and fix the MSMARCO full-query gap in the old DARTH wrapper. Preflight still must pass on the target machine because large datasets, reusable indexes, and SIMD build artifacts are local state.
