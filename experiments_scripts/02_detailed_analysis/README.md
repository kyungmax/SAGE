# Detailed analysis

Run from the repository root after the [setup](../../README.md) and the
[FAISS main experiment](../01_main_results/). Repeat these steps per dataset.

## 1. Select the completed main-run inputs

Use the same HDF5 file, FAISS graph, and frozen `policy.json` as the main run.
Keep `target.json` and `online_plan.json` beside the policy. Set the metric and
thread count to match your run (`l2` for SpaceV; `cosine` for the other datasets).

```bash
detail=experiments_scripts/02_detailed_analysis
out=runs/02_detailed_analysis/glove
common=(
  --dataset /path/to/glove-100-angular.hdf5
  --index /path/to/glove.faiss
  --policy /path/to/main-run/policy.json
  --space cosine --threads 24
)
bash scripts/python.sh "$detail/run_drilldown.py" "${common[@]}" --preflight
```

Use fresh output directories and keep timing jobs sequential.

## 2. Prepare baseline counters (optional)

The default drilldown compares Vanilla and SAGE. To include all methods, first
finish their main runs and prepare one `counters.npz` per baseline:

| Methods | Collector | Required inputs in addition to its output directory |
| --- | --- | --- |
| PiP / ABS | `collect_baseline.py` | `"${common[@]}"`, `--method pip` or `abs`, `--settings /path/to/tuning/DATASET/selection.json`, `--reference /path/to/online/DATASET/result.npz` |
| Ada-EF | `import_baseline_counters.py` | `--method adaef --dataset /path/to/data.hdf5 --k 10 --run /path/to/adaef/DATASET/k10/target_hits_N` |
| DARTH | `import_baseline_counters.py` | `--method darth --dataset /path/to/data.hdf5 --k 10 --run /path/to/darth/pipeline/evaluation/test` |

Invoke either collector with `bash scripts/python.sh "$detail/SCRIPT.py"` and
`--output /path/to/new-counter-directory`; use `--help` for all options. The
DARTH `--run` must point to `evaluation/test` inside the pipeline output, not
the pipeline root. Match `--k` to the policy.

## 3. Run the analyses

These three analyses are independent; run timing jobs sequentially.

```bash
bash scripts/python.sh "$detail/run_drilldown.py" "${common[@]}" \
  --output "$out/drilldown"
bash scripts/python.sh "$detail/run_misclassification.py" "${common[@]}" \
  --output "$out/misclassification"
bash scripts/python.sh "$detail/run_shadow.py" "${common[@]}" \
  --output "$out/shadow"
```

For a full drilldown, append `--baseline NAME=/path/to/counters.npz` once per
baseline to the first command; use names `PiP`, `ABS`, `Ada-EF`, and `DARTH`.

| Analysis | Main outputs under its output directory |
| --- | --- |
| Query difficulty | `mean_steps_recall.csv`, `cohort_query_ids.csv`, `per_query.csv` |
| Misclassification | `summary.json`, `per_query.csv` |
| Shadow contribution | `summary.csv`, `timings.csv`, `protocol.json` |

Drilldown and misclassification collect work counters, not latency. Shadow
compares Vanilla and SAGE with/without shadow at the same recommended ef.
