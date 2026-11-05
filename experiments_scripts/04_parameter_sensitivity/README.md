# 04 — Parameter sensitivity

This folder produces the current paper's **k=10 W/B/g sensitivity table** on
GloVe100 and Cohere. It needs a built FAISS backend, HDF5 data, and matching
indexes from the [main data/index preparation](../01_main_results/README.md).
It can run independently of the main and calibration experiments: it computes
its own default calibration and does not consume saved outputs from `01` or `03`.

Run from the repository root:

```bash
for dataset in glove-100-angular cohere-768-angular; do
  bash scripts/python.sh experiments_scripts/04_parameter_sensitivity/run.py \
    --dataset "$dataset" --output-root runs/04_parameter_sensitivity/run1
done
```

The runner first obtains the recommended ef with the current production
calibration, then holds it fixed across all settings. The paper's ef values
(1664 for GloVe100 and 640 for Cohere) are historical observations, not forced
CLI settings. The runner varies one parameter
at a time: CFR window W = 8/16/24, main tier count B = 2/4/6, and main safety
margin g = 1/2/3/4. Defaults are W16/B4/g2. W applies to both main and shadow
observations; shadow stays at rank128, B4/g4, with pre-expansion observation
and the native Top1 second chance. Calibration always uses the same 500
self-hidden probes, selected with seed 42 from the fixed 10,000-node LID pool;
thresholds are refitted for each variant. Shadow is active only when the fixed
recommended ef is at least 256.

Without `--output-root`, outputs go to
`runs/04_parameter_sensitivity/<dataset>/`; the example writes under
`runs/04_parameter_sensitivity/run1/<dataset>/`. Use a new output root for each
attempt. The default variant must match production thresholds and public API
labels/distances before measurements begin.

`summary.csv` contains eight unique settings: `default`, `W_8`, `W_24`,
`B_2`, `B_6`, `g_1`, `g_3`, and `g_4`. To lay out the paper's three parameter
groups, reuse the same `default` row for W16, B4, and g2.

| Paper quantity | `summary.csv` column | Definition |
| --- | --- | --- |
| Fixed recommended budget | `ef` | Same budget for every variant and Vanilla |
| Parameter values | `W`, `B`, `g` | Only one changes from W16/B4/g2 |
| Recall loss | `recall_loss` | Vanilla Recall@10 minus SAGE Recall@10, as a fraction rather than percentage points |
| Speedup | `same_ef_speedup` | Mean Vanilla batch time divided by mean variant batch time at the same ef |

`summary.json` also includes the Vanilla row. `timings.csv` retains each
warmup/measured sample; `vanilla.npz` and `<setting>.npz` retain result
labels/distances/hits.
`default_policy.json`, `policy_<setting>.json`, `calibration.npz`, and
`protocol.json` record the fitted policies and probe inputs. No combined
two-dataset publication table is generated.

Timing uses **5 measured runs plus 1 separate warmup**. `--threads` defaults
to 24; `--measured-runs`, `--warmup-runs`, `--catalog`, and `--output-root`
are configurable. `--catalog` defaults to
[the main dataset catalog](../01_main_results/config/main6_online24/datasets.json);
its relative filenames use `SAGE_DATA_DIR` / `SAGE_INDEX_DIR`. Each repeat
randomizes method order. The mean is over complete online batch calls including
query preparation and output allocation; calibration, parity checks, recall
evaluation and file writes are excluded. `latency_ms` is batch mean divided
by query count, not a percentile. Use the same CPU/NUMA placement for both
datasets. `--help` lists options; there is no separate preflight command.

Source: `parameter_sensitivity_recommended_20261102/run.py` and its W/B/g
configuration. The removed experiment policy helper is reconstructed from
the current production fitting and query APIs; the default-parity checks
guard that port. No historical result files or sibling source imports are
required. The earlier fixed-ef1024 variant and optional calibration-size
rows are omitted because the current paper uses recommended ef and W/B/g.
