# Paper Experiment-Section Check

Checked against `paper_tex/6.Experimental Evaluation.tex`, subsection `Index Quality`.

Matched settings:

- backend scope for this import: FAISS only, as requested.
- SIMD is forced via `FAISS_OPT_LEVEL=AVX512` and the configured SAGE FAISS Python path.
- paper index-quality section uses six datasets and excludes AGNews/Landmark because vanilla recall is already above 0.999; the runner defaults to the same six datasets.
- paper compares `M=16, efConstruction=200` against `M=32, efConstruction=500`; the runner defaults to these two endpoints.
- paper uses baseline `efSearch=1024`; the runner defaults to `1024`.
- paper's x/y quantities are distance-computation reduction and recall loss, not QPS; the runner writes `saved_ndis_pct`, `ndis_speedup`, and recall-loss columns.

Intentional differences from old source bundles:

- The old source index-quality folder contains hnswlib outputs, four-setting FAISS QPS/latency sweeps, repeated QPS runs, and generated result files. Those are not imported.
- This artifact keeps one FAISS SIMD-on ndis runner and no generated results.
