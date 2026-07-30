# 03 Offline Cost Scripts

This directory contains script-only artifacts for the offline calibration cost experiment. It intentionally excludes generated CSVs, logs, caches, and result bundles.

Imported scope:

- backends: FAISS and hnswlib
- SIMD: on
  - FAISS uses `FAISS_OPT_LEVEL=AVX512` and the configured AVX-512 FAISS Python path
  - hnswlib uses the compiled extension from `SAGE_HNSWLIB_EXTENSION_ROOT`
- threads: 24 offline calibration threads
- default datasets: the eight datasets in the paper offline-cost table
- default index: `M=32, efConstruction=500`
- default calibration: 10,000-node LID pool, 100 LID-stratified calibration queries, pseudo-GT at `efSearch=4096`, EF sweep `64..1024`, `B=4`, `g=2`, `[4,16]`, `alpha=0.8`

Run FAISS:

```bash
cd $SAGE_ROOT/final_experiments/03_offline_cost
./run_faiss_simd_24t.sh
```

Run hnswlib:

```bash
cd $SAGE_ROOT/final_experiments/03_offline_cost
./run_hnswlib_simd_24t.sh
```

Run both:

```bash
cd $SAGE_ROOT/final_experiments/03_offline_cost
./run_all_simd_24t.sh
```

Primary outputs after running:

- `offline_cost_main8_faiss_SIMD_on_24t/final/faiss_offline_cost_raw.csv`
- `offline_cost_main8_faiss_SIMD_on_24t/final/faiss_offline_cost_median.csv`
- `offline_cost_main8_hnswlib_SIMD_on_24t/final/hnswlib_offline_cost_raw.csv`
- `offline_cost_main8_hnswlib_SIMD_on_24t/final/hnswlib_offline_cost_median.csv`

After both backends finish, optionally assemble one comparison CSV:

```bash
python3 scripts/assemble_offline_cost_medians.py
```

The paper table columns map to the median CSV as:

- `Samp.`: `paper_samp_s` / `step1_lid_sampling_wall_s`
- `Select`: `paper_select_s` / `step2_pre_evaluation_wall_s`
- `Eval.`: `paper_eval_s` / `step3_eval_wall_s`
- `Total`: `paper_total_s` / `offline_calibration_wall_s`
