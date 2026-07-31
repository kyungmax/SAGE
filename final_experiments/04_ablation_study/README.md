# 04 Ablation Study Scripts

This directory contains script-only artifacts for the paper parameter-sensitivity ablation. It intentionally excludes generated CSVs, logs, calibration caches, indexes, and old broad-scope reruns.

Imported scope:

- backend: FAISS only
- SIMD: on (`FAISS_OPT_LEVEL=AVX512` and `FAISS_PYTHON_PATH`)
- threads: 24 offline, 24 online, 24 FAISS index-build threads by default
- datasets: `glove-100-angular.hdf5` and `cohere-768-angular.hdf5` (paper names: GloVe100 and CohereWiki)
- `k`: 10
- ablation search budget: `efSearch=1024`
- index: `M=32`, `efConstruction=500`
- threshold mode: `paper_floor_half`
- default SAGE settings: `ncal=100`, `B=4`, `g=2`, `[4,16]`, `alpha=0.8`, `tmin_pops=25`

Paper ablation cells:

| Study | Values |
|-------|--------|
| `01_ncal` | `100`, `500`, `1000` |
| `02_classification_window` | `[0,12]`, `[4,16]` |
| `03_tiers` | `B=2`, `B=4`, `B=6` |
| `04_ema_alpha` | `alpha=0.0`, `alpha=0.4`, `alpha=0.8` |
| `05_pair_gap` | `g=1`, `g=2`, `g=3`, `g=4` |

Run a preflight check:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/04_ablation_study
python3 scripts/preflight_faiss_glove_cohere_ablation.py
```

Missing HDF5 files must be prepared before running. Missing FAISS indexes are acceptable by default: the ablation and pseudo-GT runners build `M=32, efConstruction=500` indexes on first use under `SAGE_FAISS_INDEX_ROOT` / `FAISS_INDEX_ROOT`.

Run the parameter-sensitivity ablation:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/04_ablation_study
export OUT_ROOT=$PWD/sage_ablation_faiss_glove_cohere_24t_m32_efc500_ef1024
./run_faiss_glove_cohere_ablation_24t.sh
```

The runner executes one dataset per subprocess by default to release large FAISS indexes between jobs. Use `--studies ncal,window` or `--max-cells N --dry-run` for partial execution/planning.

Primary outputs after running:

- `${OUT_ROOT}/<study>/<variant>/final/main_qps_latency_sweep.csv`
- `${OUT_ROOT}/<study>/<variant>/final/offline_predicted_recall_curve.csv`
- `${OUT_ROOT}/<study>/<variant>/final/offline_recommended_efsearch.csv`
- `${OUT_ROOT}/summary/paper_ablation_glove_cohere_faiss.csv`
- `${OUT_ROOT}/summary/paper_ablation_glove_cohere_faiss_paper_table.csv`

Run the pseudo-GT vs exact-GT calibration check:

```bash
cd $SAGE_PROJECT_ROOT/final_experiments/04_ablation_study
export PSEUDOGT_OUT_DIR=$PWD/probe_pseudo_gt_vs_exact_glove_cohere_faiss_p100_ef4096
./run_faiss_glove_cohere_pseudogt_24t.sh
```

Pseudo-GT defaults match the paper check: 100 LID-stratified calibration probes, FAISS hide-node pseudo-GT at `efSearch=4096`, brute-force exact train-set neighbors, and the full baseline EF ladder `64..1024` for recommended-ef agreement.

Primary pseudo-GT outputs:

- `${PSEUDOGT_OUT_DIR}/summary.csv`
- `${PSEUDOGT_OUT_DIR}/querywise.csv`
- `${PSEUDOGT_OUT_DIR}/baseline_recommended_efsearch.csv`
- `${PSEUDOGT_OUT_DIR}/baseline_recommendation_curve.csv`

Run both parts:

```bash
./run_all_faiss_glove_cohere_24t.sh
```

`PAPER_TEX_CHECK.md` records the consistency check against the paper experiment section.
