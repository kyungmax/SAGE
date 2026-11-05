# Paper experiments

Run commands from the repository root with `bash scripts/python.sh`. Numbered
directories follow the active sections of the experiment manuscript.
`common/` and the backend directories supply reusable policies, adapters and
index builders; numbered directories contain the experiment entry points.

| Directory | Paper experiment | Entry points |
| --- | --- | --- |
| [01_main_results](01_main_results/README.md) | Main recall/latency curves and matched-recall comparisons | `run.sh`, `run_adaef.py`, `run_darth_pipeline.py`, `summarize_main.py` |
| [02_detailed_analysis](02_detailed_analysis/README.md) | Performance drill-down, misclassification, shadow contribution | `run_drilldown.py`, `run_misclassification.py`, `run_shadow.py` |
| [03_calibration](03_calibration/README.md) | Recommended ef quality, offline cost, sampling stability, pseudo-GT validation | `run.py` |
| [04_parameter_sensitivity](04_parameter_sensitivity/README.md) | CFR window, tier count and safety margin | `run.py` |
| [05_robustness](05_robustness/README.md) | Embedding quality and RaBitQ | `run.py`, `report.py` |

Timed settings default to **five measured calls plus one separate warmup**.
Sampling seeds are an independent experimental dimension. Main comparisons
use matched recall; the shadow contribution uses the same starting ef. Every
runner preserves measured outputs rather than embedding historical results.
The standalone `rabitq/runner.py sweep` keeps its median summary and timing
scope, with five measured calls; `05_robustness` supplies the paper's paired
curves and mean summaries.

The main six datasets are GooAQ, GloVe100, Cohere, YouTube15M, MSMARCO-Qwen and
SpaceV. RaBitQ covers MSMARCO, YouTube and Cohere. Embedding quality compares
GloVe, Cohere and BGE-M3 on the same Cohere/Wikipedia corpus.

## Preparation and execution order

1. Follow the [root build instructions](../README.md#build). The
   [scripts guide](../scripts/README.md) explains native build and environment checks;
   [baselines](../baselines/README.md) lists the additional Ada-EF/DARTH builds.
2. Supply HDF5 `train`, `test` and `neighbors` arrays and matching indexes.
   Set `SAGE_DATA_DIR` / `SAGE_INDEX_DIR`, or use an explicit catalog. Main index
   builders and exact top100 ground-truth preparation are documented in `01`.
   Dataset payloads, indexes, trained models and measured results are not bundled.
3. Run `01` FAISS first to generate the frozen SAGE policy and exact recall targets.
   Its `run.sh` handles FAISS, hnswlib, PiP and ABS. Ada-EF and DARTH have separate
   entry points. `run_darth_pipeline.py` prepares the auxiliary graph and exact
   holdout GT, trains the model, derives intervals from its learning trace, then
   validates and measures the full test set. It takes ef/recall from the FAISS
   targets and supports k10; validation does not tune the model or intervals.
   Build both DARTH native executables and install `lightgbm`/`scikit-learn` as
   described in `01`. Ada-EF does not support SpaceV's L2 metric.
4. Run `02` from the FAISS policy and receipts produced by `01`. For all-method
   drill-down, also collect/import PiP, ABS, Ada-EF and DARTH counters as described
   in its README. DARTH imports use the pipeline's `evaluation/test/` directory.
5. `03` and `04` independently calibrate from the main catalog's FAISS indexes;
   they do not consume a previous main run. `04` uses GloVe and Cohere in the paper.
   `05` uses its own embedding/RaBitQ catalogs and also calibrates independently.

```bash
bash experiments_scripts/01_main_results/run.sh --dry-run
bash experiments_scripts/01_main_results/run.sh --preflight
bash scripts/python.sh experiments_scripts/01_main_results/run_darth_pipeline.py \
  --dataset glove-100-angular --targets runs/01_main_results/k10/faiss/targets.json \
  --preflight
bash scripts/python.sh experiments_scripts/02_detailed_analysis/run_shadow.py --help
bash scripts/python.sh experiments_scripts/03_calibration/run.py --help
bash scripts/python.sh experiments_scripts/04_parameter_sensitivity/run.py --help
bash scripts/python.sh experiments_scripts/05_robustness/run.py embedding --preflight
bash scripts/python.sh experiments_scripts/05_robustness/run.py rabitq --preflight
```

Preflight requires prepared inputs. `03` and `04` have no preflight switch;
`--help` only displays options, and execution validates inputs before calibration.
Review CPU/NUMA placement for your machine and run timing experiments sequentially
on the same assigned cores. Use new output directories for each run; directory
overwrite protection differs between entry points.

## Outputs and reproduction boundary

Outputs default to `runs/<numbered-experiment>/` and are ignored by Git.
Each experiment README identifies the files and fields for its paper results:

| Directory | Main numeric outputs |
| --- | --- |
| `01` | Per-method `summary.csv`, raw timing samples, frozen policies/targets, DARTH preparation/model artifacts, and `summarize_main.py` comparisons |
| `02` | Cohort work/recall summaries, false-easy/false-hard diagnostics and the same-ef shadow comparison |
| `03` | Per-dataset `summary.json`, offline phase timings, seed statistics and pseudo/exact ground truth |
| `04` | Per-dataset `summary.csv` for W/B/g settings and raw timings |
| `05` | Validated `curves.csv` and `matched_recall.csv` from `report.py` |

These entry points generate measurements and numeric summaries. Final paper PDF
figures, LaTeX table formatting and a single cross-dataset paper exporter are not
included. DARTH's integrated output defaults to
`runs/01_main_results/darth/DATASET/`: `data/`, `training/`,
`evaluation/{validation,test}/`, and a final root `summary.csv`/`complete.json`.
Pass the root summary to `summarize_main.py --summary`; `--stop-after` and
`--resume` split execution across verified completed stages. Its default split
is 10,000 learning plus 1,000 validation vectors with seed 12345. Smaller split
options are for bounded validation and must be identified as protocol deviations.
The auxiliary graph, exact GT and all selected training feature rows require
substantial disk/RAM; raw trace CSV is streamed, not retained. Keep validation
outputs outside the checkout.

`05` requires aligned embeddings and their exact ground
truth as external inputs; it does not generate embeddings.

The manuscript's commented-out adaptive-index-build extension is outside this
execution scope; see [05's remaining scope](05_robustness/README.md#source-provenance-and-remaining-scope).
