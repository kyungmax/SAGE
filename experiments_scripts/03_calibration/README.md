# 03 — Index-only calibration

This folder produces the paper's recommended-ef quality/offline-cost table,
calibration-stability table, and pseudo-ground-truth table, all at **k=10**.
It runs independently of `01_main_results`: no saved main result or policy is
required. Each run calibrates the supplied FAISS graph afresh.

Run from the repository root. Build with `bash scripts/build.sh faiss`, then
prepare the HDF5 data and matching indexes described in
[01_main_results](../01_main_results/README.md). The default input catalog is
[datasets.json](../01_main_results/config/main6_online24/datasets.json).
`SAGE_DATA_DIR` and `SAGE_INDEX_DIR` prefix its relative data/index filenames;
use `--catalog /path/to/datasets.json` for a different layout. Inputs must have
at least 10,000 indexed training vectors and the official `test`/`neighbors`
arrays. The runner checks graph metadata and sampled stored vectors against
the dataset.

```bash
bash scripts/python.sh experiments_scripts/03_calibration/run.py \
  --dataset glove-100-angular --experiment quality
bash scripts/python.sh experiments_scripts/03_calibration/run.py \
  --dataset glove-100-angular --experiment stability
bash scripts/python.sh experiments_scripts/03_calibration/run.py \
  --dataset glove-100-angular --experiment pseudo-gt
```

`--experiment all` is the default and combines the three into one output
directory. To run the six paper datasets sequentially:

```bash
for dataset in glove-100-angular gooaq-distilroberta-768-normalized \
  msmarco-qwen-1024-normalized cohere-768-angular youtube-15M-angular \
  msspacev-100M-i8-euclidean; do
  bash scripts/python.sh experiments_scripts/03_calibration/run.py \
    --dataset "$dataset" --experiment all --output-root runs/03_calibration/run1
done
```

Without `--output-root`, outputs go to
`runs/03_calibration/<dataset>/<experiment>/`; the loop above writes under
`runs/03_calibration/run1/<dataset>/all/`. Use a new output root for each attempt.

- `quality`: fresh production calibration, phase costs, and vanilla online
  recall at recommended ef and the final calibration cap. The recommendation
  uses 500 self-hidden, LID-stratified probes from a fixed 10,000-node pool
  sampled with seed 42. Recommendation and cap are recomputed, not hardcoded
  to the paper's printed values.
- `stability`: seeds 0–9 compare 500 uniform random graph nodes with 500
  LID-stratified probes from a fresh 10,000-node pool. All recommendations are
  frozen before loading real queries. Vanilla is timed once per distinct ef
  and reused for every seed selecting that ef. Reported variability is the
  **sample standard deviation across the 10 seeds**, not timing-repeat noise.
- `pseudo-gt`: freshly compute self-hidden ef4096 pseudo neighbors and exact
  brute-force top-10 neighbors for the same 500 calibration probes. Exact
  search scans all stored index vectors on every repeat, including block
  reconstruction and top-k merging in its cost. `--exact-chunk-size` defaults
  to 50,000 rows and bounds block memory; it does not sample the dataset.

Timing uses **5 measured runs plus 1 separate warmup**, configurable with
`--measured-runs` / `--warmup-runs`; `--seed-count` independently controls the
10 stability seeds. Time summaries are arithmetic means of measured runs.
Online latency is the full-batch mean divided by query count, in milliseconds;
it includes query preparation and output allocation, excludes validation and
file writes, and is not an individual-query latency percentile. `--threads`
defaults to 24. Index loading is excluded from the reported costs. Apply the
same external CPU/NUMA placement as other experiments.
`--help` lists options; this runner has no separate preflight command.

Each experiment writes `summary.json`. Paper columns map to these outputs:

| Paper quantity | Output field | Unit |
| --- | --- | --- |
| Recommended ef | `summary.json: recommended_ef` | Search budget |
| Samp., Sel., Eval., Total | `calibration_seconds: Samp., Select., Eval., total_seconds` | Seconds |
| Recommended/cap recall and gap | `recall_recommended`, `recall_cap`, `recall_gap` | Recall fractions; gap = cap minus recommended |
| Random/LID ef variability | `stability.<method>.ef_sample_sd` | Search budget |
| Random/LID recall variability | `stability.<method>.recall_pp_sample_sd` | Percentage points |
| Random/LID latency variability | `stability.<method>.latency_ms_sample_sd` | Milliseconds |
| Pseudo-GT overlap and cost | `pseudo_exact.json: recall, pseudo_seconds, exact_seconds` | Recall fraction; seconds |

`quality`/`all` write `offline_timings.csv` and online `timings.csv` with
warmup/measured phase tags. Stability writes `stability_per_seed.csv` with
all seed choices and online metrics. Pseudo-GT validation writes
`pseudo_exact_timings.csv` and `groundtruth.npz` with construction samples
and exact/pseudo labels. `policy.json` records default calibration in every
mode except stability-only.
These are per-dataset outputs; no combined six-dataset paper table is generated.

Source: the current production flow in
`faiss_followup_experiments_20261028/calibration/run.py`, the random sampling
stream in `e6_lid_random_500/design.py`, and the self-hidden, chunked exact
comparison in the earlier VLDB `compare_probe_pseudo_gt_to_exact_faiss.py`.
The portable runner uses this checkout's APIs; no historical result bundle,
native snapshot, or sibling checkout is a runtime dependency.
