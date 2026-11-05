# 02 — Detailed performance analysis

These runners collect the three analyses in the paper's **Detailed Performance
Analysis** section using the current FAISS policy. They reuse the policy frozen
by [01_main_results](../01_main_results/README.md), without recalibrating on test
queries. Run each dataset separately at **k=10** for the paper.

| Paper result | Runner | Result to use |
| --- | --- | --- |
| Easy/Hard 10% steps and Recall@10 figure | `run_drilldown.py` | `mean_steps_recall.csv`: `mean_steps` for bars, `mean_recall` for dots |
| False-easy recall-drop table and false-hard work analysis | `run_misclassification.py` | `summary.json`, with definitions below |
| Shadow result-set table | `run_shadow.py` | `summary.csv`: `recall_loss` and `speedup` for `without_shadow` / `with_shadow` |

The runners write per-dataset CSV/JSON and query-level evidence. They do not
combine the six datasets into the final paper PDF or LaTeX tables.

## Inputs and preflight

Build FAISS using the [root instructions](../../README.md), then complete the
FAISS main run. The example below uses the default main output and Cohere;
change `DATASET`, `DATA` and `MAIN` to match your run. The graph path is read from
the main receipt so the analysis uses exactly the calibrated graph.

```bash
DATASET=cohere-768-angular
DATA="${SAGE_DATA_DIR:-$PWD/datasets}/$DATASET.hdf5"
MAIN=runs/01_main_results/k10
POLICY="$MAIN/faiss/datasets/$DATASET/policy.json"
INDEX="$(bash scripts/python.sh -c 'import json,sys; print(json.load(open(sys.argv[1]))["faiss_index"]["path"])' "${POLICY%/*}/target.json")"
OUT="runs/02_detailed_analysis/$DATASET"
DETAIL_INPUTS=(--dataset "$DATA" --index "$INDEX" --policy "$POLICY" --space cosine)

bash scripts/python.sh experiments_scripts/02_detailed_analysis/run_drilldown.py \
  "${DETAIL_INPUTS[@]}" --preflight
```

Run these Bash commands from the repository root. The paper catalog contains
GooAQ, GloVe100, Cohere, YouTube15M, MSMARCO-Qwen and SpaceV. SpaceV uses `--space
l2`; the other five use `--space cosine`. Defaults are `--M 32
--ef-construction 500 --threads 24`. Set `--cpus` to a comma-separated list and
use the main run's CPU/NUMA placement for shadow timing.

Keep `target.json` and `online_plan.json` beside `policy.json`. They bind the
policy hash, graph path/size/mtime, recommended ef, k and query/ground-truth order.
`--policy-receipt` can specify a target receipt elsewhere; a receipt containing
`policy_sha256` does not need the adjacent `online_plan.json`. Preflight checks
these identities, array shapes, the current raw CFR window 1–16 and native APIs,
without loading the graph or running searches. The three analysis runners
additionally check the graph metric, count, M/efConstruction, 33 stored training
vectors and complete query/ground-truth hashes when executing. Baseline NPZ contents are checked when executing the
drill-down, not by its common preflight.

Each execution needs a **new output directory**; an existing directory is never
overwritten. The three analysis runners write `inputs.json` with the graph,
policy, query hashes and execution placement. The PiP/ABS collector instead
validates the frozen graph identity, query/GT hashes and selected result, and
records its receipt in `complete.json`. No historical `experiment_results`
directory is required.

## Performance drill-down

First collect the baseline work counters using the settings selected by the
main experiment. PiP and ABS load their own FAISS builds in separate processes:

```bash
for METHOD in pip abs; do
  bash scripts/python.sh experiments_scripts/02_detailed_analysis/collect_baseline.py \
    "${DETAIL_INPUTS[@]}" --method "$METHOD" \
    --settings "$MAIN/$METHOD/tuning/$DATASET/selection.json" \
    --reference "$MAIN/$METHOD/online/$DATASET/result.npz" \
    --output "$OUT/$METHOD"
done
```

`selection.json` supplies `selected.config`; the collector checks its method,
matched target and output hashes, then requires diagnostic labels and distances
to equal the saved online result. It writes `counters.npz` and `complete.json`.
PiP and ABS use their native completed-expansion counter, `nexpan`. These are
untimed replays; they do not retune the selected settings or measure latency.
Add `--preflight` to check the selection/reference inputs before searching.

DARTH and Ada-EF counters come from their separate
[main runners](../01_main_results/README.md#ada-ef-and-darth). Rebuild the native
baselines after updating their source. Set `DARTH_RUN` to the integrated
pipeline's `OUTPUT/evaluation/test` directory (or the standalone
`run_darth.py --output` directory). Ada-EF's directory includes its target hit count:

```bash
DARTH_RUN=/path/to/completed/darth/pipeline/evaluation/test
TARGET_HITS="$(bash scripts/python.sh -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_hits"])' "${POLICY%/*}/target.json")"
ADAEF_RUN="runs/01_main_results/adaef/$DATASET/k10/target_hits_$TARGET_HITS"

bash scripts/python.sh experiments_scripts/02_detailed_analysis/import_baseline_counters.py \
  --method darth --dataset "$DATA" --run "$DARTH_RUN" --k 10 \
  --output "$OUT/darth"
bash scripts/python.sh experiments_scripts/02_detailed_analysis/import_baseline_counters.py \
  --method adaef --dataset "$DATA" --run "$ADAEF_RUN" --k 10 \
  --output "$OUT/adaef"

bash scripts/python.sh experiments_scripts/02_detailed_analysis/run_drilldown.py \
  "${DETAIL_INPUTS[@]}" \
  --baseline "PiP=$OUT/pip/counters.npz" \
  --baseline "ABS=$OUT/abs/counters.npz" \
  --baseline "DARTH=$OUT/darth/counters.npz" \
  --baseline "Ada-EF=$OUT/adaef/counters.npz" \
  --output "$OUT/drilldown"
```

DARTH conversion needs `plan.json`, `timings.json`, `result.ids.i64`,
`audit.counters.csv` and the original query/GT vecs paths recorded in the plan.
It checks those arrays against the HDF5. Ada-EF conversion needs `run_plan.json`,
`online.json`, `labels.u64` and `per_query.csv`, and verifies the recorded HDF5
identity. Both check native diagnostic parity against measured outputs before
writing `counters.npz`. DARTH excludes a neighbor expansion interrupted by model
termination; Ada-EF includes its statistics-collection prefix. Their counters
use an untimed observer and leave search decisions unchanged.

Ada-EF supports cosine/IP and cannot produce a SpaceV L2 row. Omit its conversion
and `--baseline` there; do not substitute fabricated values. Without any
`--baseline` options, the drill-down runs only Vanilla and SAGE. `complete.json`
means the requested collection finished; inspect `methods` and
`missing_paper_methods` before treating it as the full comparison.

Every method uses the same cohort IDs: sort by Vanilla's final hits descending,
then the first step attaining those hits ascending, then query ID ascending.
Easy uses positions before `ceil(0.1*n)`; Hard starts at `ceil(0.9*n)`. Steps mean
**completed level-0 expansions**. Vanilla/SAGE upper-layer hops are measured by
the native greedy descent and subtracted, with exact ordinary/diagnostic output
parity checks. Baselines retain their selected main settings; `reference_ef` in
the summary identifies the Vanilla/SAGE reference, not a forced baseline ef.

`mean_steps_recall.csv` gives one row per method/cohort. `cohort_query_ids.csv`
preserves membership; `per_query.csv` contains steps, hits and Vanilla first
attainment. `vanilla.npz` and `sage.npz` retain the native outputs. A checked
external collector can also supply NPZ fields `labels`, `hits`, integer `steps`,
contiguous `qids`, scalar `query_sha256`, `truth_sha256` and
`step_definition='completed_L0_expansions'`.

## Misclassification drill-down

```bash
bash scripts/python.sh experiments_scripts/02_detailed_analysis/run_misclassification.py \
  "${DETAIL_INPUTS[@]}" --output "$OUT/misclassification"
```

The following `summary.json` fields fill the false-easy table directly:

| Paper column | Field |
| --- | --- |
| All queries | `all_queries` |
| Reduced queries | `reduced_queries` |
| False-easy queries (percentage of reduced queries) | `false_easy_queries`, `false_easy_percent_reduced` |
| Recall@10 drop = 0.1 | `recall_drop_01_queries` |
| Recall@10 drop = 0.2 | `recall_drop_02_queries` |
| Drop ≤ 0.2, percentage of false-easy queries | `recall_drop_le_02_percent` |

“Reduced” in the paper table means the **final budget** remains below the
recommended ef. False-easy means these queries actually lose hits relative to
Vanilla at that ef. Counts, percentages and drop distributions above all use
this final-reduced population, not merely assignment below an oracle tier.
`ever_reduced_queries` also counts queries later promoted back to full budget;
`ever_reduced_false_easy_queries` and `promoted_full_budget_false_easy_queries`
retain any residual losses after those promotions. These distinctions are also
explicit per-query flags. A percentage is `null` if its denominator is zero;
the two exact drop-count fields require k=10.

The oracle is the smallest **measured assignable Vanilla tier** reaching the
recommended-budget Vanilla hit count for that query. All main/shadow tiers are
replayed, including shadow's promotion-only tier 128; no recall monotonicity is
assumed. Initial budgets are reconstructed from native float32 CFR/routing
observations and validated against the final budget and one-tier promotion.

The paper's false-hard prose uses queries whose final budget remains full even
though a smaller tier preserves recall: `full_budget_false_hard_queries` and
`full_budget_false_hard_percent_all`. Its excess-work percentage is
`full_budget_false_hard_extra_share_sage_steps_pct`: positive SAGE-minus-oracle
steps on these queries, divided by **all SAGE steps**. The broader
`tier_false_hard_queries` also includes over-allocation below full budget and
must not replace that paper count.

`per_query.csv` exposes initial/final/oracle budgets, hits, classification and
excess work. `vanilla_ef*.npz` and `sage.npz` preserve the underlying measured
steps and outputs. This analysis is untimed; it does not estimate latency from
tier sizes or claim an oracle speedup.

## Effect of the shadow result set

```bash
bash scripts/python.sh experiments_scripts/02_detailed_analysis/run_shadow.py \
  "${DETAIL_INPUTS[@]}" --warmups 1 --measured-runs 5 \
  --output "$OUT/shadow"
```

Vanilla, SAGE without shadow and full SAGE use the **same recommended ef**.
The main thresholds and Top1 second chance remain identical. `recall_loss` is
Vanilla recall minus the method's recall; `speedup` is Vanilla mean seconds
divided by the method's mean seconds. Use the two SAGE rows for the paper table.
This comparison is independent of the main matched-recall speedup.

Each method gets **one separate warmup plus five measured calls**. Timing covers
whole query calls, with index loading, recall calculation and repeated-output
parity checks outside timing. `timings.csv` retains all six calls per method;
`summary.csv` uses only the five measured calls. `protocol.json` records the
counts, same-ef contract, Top1 and CFR window. Shadow is inactive below ef=256
(e.g. the paper's GooAQ ef=160); this is recorded as `shadow_active: false`.

## Source

Traversal/cohort logic comes from
`faiss_followup_experiments_20261028/drilldown/run.py`; tier classification from
`false_easy_tier_oracle_20261029/aggregate.py`; full-budget false-hard separation
from `false_hard_full_budget_20261030` and `false_hard_oracle_fullbatch_20261031`;
shadow treatment from `faiss_followup_experiments_20261028/ablation/run.py`.
The DARTH observer follows `steps_methods_10pct_darth_holdout_20261030/collector/prepare.py`;
the Ada-EF completion boundary follows `steps_methods_10pct_20261028/adaef`.
Historical directories, outputs and private native snapshots are not runtime
dependencies. The retained code uses the current four-tier/shadow128 policy,
raw CFR window 1–16 and Top1 second chance.
