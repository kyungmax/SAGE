# 06 Better Index Quality Scripts

This directory contains the script-only artifact for the index-quality experiment.
It intentionally excludes generated CSV, log, PDF, and image outputs.

Imported scope:

- backend: FAISS only
- SIMD: on, via `FAISS_OPT_LEVEL=AVX512` and the SAGE FAISS Python path
- metric: actual distance computations (`ndis`), not QPS
- default datasets: the six datasets used in the paper index-quality section, excluding AGNews and Landmark
- default settings: weak `M=16, efConstruction=200` and strong `M=32, efConstruction=500`
- default efSearch: `1024`
- recall loss is measured and written with the ndis summary

Run:

```bash
cd $SAGE_ROOT/final_experiments/06_better_index_quality
python3 run_faiss_simd_ndis_ef1024.py   --policy-csv /path/to/combined_faiss_main_qps_latency_sweep.csv   --m32-index-root /path/to/faiss_m32_efc500_main8_20260707/index   --index-root-base /path/to/faiss_graph_quality_ndis/index   --build-missing-indexes
```

The policy CSV is used only to read SAGE's calibrated `early_stop_ratio`, route signature, and bucket gammas for each dataset/build/efSearch. This runner's outputs do not report QPS.

Primary outputs after running:

- `faiss_simd_ndis_ef1024/combined_summary_actual_ndis.csv`
- `faiss_simd_ndis_ef1024/m16_m32_ndis_reduction_comparison.csv`
- `faiss_simd_ndis_ef1024/<dataset>/M*_efC*/ef1024/per_query_actual_ndis.csv`
