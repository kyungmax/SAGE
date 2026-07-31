# 02 Drill-Down Scripts

This directory contains the script-only artifact for the drill-down experiment. It intentionally excludes generated CSV, TSV, log, PDF, and image outputs.

Imported scope:

- backend: FAISS only
- SIMD: on, via `FAISS_OPT_LEVEL=AVX512` and the SAGE FAISS Python path
- threads: 24 offline calibration threads and 24 online search threads
- index: HNSW `M=32`, `efConstruction=500`
- Recall@10, `ncal=100`, `B=4`, `g=2`, `alpha=0.8`, CFR window `[4,16]`
- default `efSearch`: `1024`, matching the paper drill-down and false-easy tables

Run the full main8 FAISS drill-down:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/02_drill_down
./run_faiss_simd_24t.sh
```

The paper tables in `paper_tex/6.Experimental Evaluation.tex` report the representative four-dataset subset (`glove`, `cohere`, `agnews`, `youtube`) at `efSearch=1024`. To regenerate that subset only:

```bash
SAGE_RUN_ID=drilldown_faiss_SIMD_on_paper4_24t \
SAGE_DRILLDOWN_DATASETS=glove-100-angular.hdf5,cohere-768-angular.hdf5,agnews-mxbai-1024-euclidean.hdf5,youtube-15M-angular.hdf5 \
SAGE_ANALYSIS_DATASETS=glove-100-angular,cohere-768-angular,agnews-mxbai-1024-euclidean,youtube-15M-angular \
./run_faiss_simd_24t.sh
```

Primary outputs when the script is run:

- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/query_groups.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_ef_sweep.csv`
- `drilldown_faiss_SIMD_on_main8_24t/difficulty_exactgt_24t/group_pair_metrics.csv`
- `drilldown_faiss_SIMD_on_main8_24t/hard_loss_querywise_exactgt_24t/hard_loss_querywise.csv`
- `drilldown_faiss_SIMD_on_main8_24t/large_false_easy_drop_analysis/large_false_easy_summary_by_dataset_ef.csv`
