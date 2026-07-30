# Paper Experiment-Section Check

Checked against `paper_tex/6.Experimental Evaluation.tex`.

Matched settings:

- datasets: the imported runner defaults to the paper's eight benchmark files: NYTimes, GloVe, AGNews, Landmark, Cohere, YouTube, MS MARCO, and SpaceV.
- backend scope for this import: FAISS only, as requested. The paper setup states both FAISS and hnswlib are SIMD-enabled, but this `02_drill_down` import intentionally includes only the FAISS SIMD-on path.
- threads: 24 offline calibration threads and 24 online search threads.
- index/search setup: `M=32`, `efConstruction=500`, Recall@10, `ncal=100`, `B=4`, `g=2`, `alpha=0.8`, CFR window `[4,16]`, and the `64..1024` calibration efSearch grid.
- drill-down tables: the paper reports four representative datasets (`glove`, `cohere`, `agnews`, `youtube`) at `efSearch=1024`; the wrapper default also uses `1024`.

Intentional differences from the old source bundle:

- The old `02_drill_down` source bundle also contains hnswlib, non-SIMD/legacy FAISS, six-dataset no-SIFT outputs, and generated result files. Those are not imported.
- The old source wrapper ran `efSearch=512,1024`; the paper drill-down and false-easy tables use `1024`, so the new wrapper defaults to `1024`. Set `SAGE_DRILLDOWN_EFS=512,1024` to reproduce the broader diagnostic bundle.
- The old source bundle contains generated CSV/TSV/log outputs. This artifact imports scripts only.
