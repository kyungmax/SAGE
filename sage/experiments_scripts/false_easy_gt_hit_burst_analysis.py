#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import false_easy_first_pass_gt_spread_local_minima as first_pass  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    first_pass.DEFAULT_FALSE_EASY_DIR / "first_pass_gt_spread_local_minima_20260622" / "gt_hit_burst_20260622"
)
WINDOWS = (8, 16, 32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether false-easy GT hits arrive as late bursts or gradual one-by-one hits."
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=first_pass.DEFAULT_FALSE_EASY_DIR / "hard_false_easy_chr_ratio_margins.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=first_pass.DEFAULT_DATASET_ROOT)
    parser.add_argument("--index-dir", type=Path, default=first_pass.DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset stems or .hdf5 names.")
    parser.add_argument("--cohorts", default="hard_false_easy_loss")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--trace-batch-size", type=int, default=64)
    parser.add_argument("--max-queries-per-dataset", type=int, default=0)
    return parser.parse_args()


def finite_quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def finite_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def finite_pct(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return 100.0 * float(mask.mean())


def max_hits_in_window(steps: list[int], window: int) -> int:
    if not steps:
        return 0
    best = 1
    left = 0
    for right, step in enumerate(steps):
        while step - steps[left] > window:
            left += 1
        best = max(best, right - left + 1)
    return int(best)


def last_n_span(steps: list[int], n: int) -> float:
    if len(steps) < n:
        return float("nan")
    return float(steps[-1] - steps[-n])


def gap_stats(steps: list[int], prefix: str) -> dict[str, float]:
    if len(steps) < 2:
        return {
            f"{prefix}_gap_mean": float("nan"),
            f"{prefix}_gap_p50": float("nan"),
            f"{prefix}_gap_p90": float("nan"),
            f"{prefix}_gap_max": float("nan"),
            f"{prefix}_last_gap": float("nan"),
        }
    gaps = np.diff(np.asarray(steps, dtype=np.float64))
    return {
        f"{prefix}_gap_mean": finite_mean(gaps),
        f"{prefix}_gap_p50": finite_quantile(gaps, 0.50),
        f"{prefix}_gap_p90": finite_quantile(gaps, 0.90),
        f"{prefix}_gap_max": float(np.max(gaps)),
        f"{prefix}_last_gap": float(gaps[-1]),
    }


def trace_hit_metrics(path: list[dict[str, Any]], gt_labels: np.ndarray, first_final_step: float) -> dict[str, Any]:
    gt_set = {int(x) for x in np.asarray(gt_labels, dtype=np.int64).tolist()}
    seen: set[int] = set()
    path_steps: list[int] = []
    fullpop_steps: list[int] = []
    labels: list[int] = []
    total_events = 0

    for step_idx, step in enumerate(path):
        label = int(step.get("node_label", -1))
        if label not in gt_set:
            continue
        total_events += 1
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
        path_steps.append(step_idx + 1)
        try:
            fullpop = int(step.get("full_pop_count_after", 0) or 0)
        except Exception:
            fullpop = 0
        fullpop_steps.append(fullpop)

    hit_count = len(path_steps)
    first_hit = float(path_steps[0]) if path_steps else float("nan")
    last_hit = float(path_steps[-1]) if path_steps else float("nan")
    first_fullpop = float(fullpop_steps[0]) if fullpop_steps else float("nan")
    last_fullpop = float(fullpop_steps[-1]) if fullpop_steps else float("nan")
    first_final = float(first_final_step) if np.isfinite(float(first_final_step)) else float("nan")

    row: dict[str, Any] = {
        "gt_hit_unique_count": int(hit_count),
        "gt_hit_total_pop_events": int(total_events),
        "gt_hit_label_order": "|".join(str(x) for x in labels),
        "gt_hit_path_steps": "|".join(str(x) for x in path_steps),
        "gt_hit_fullpop_steps": "|".join(str(x) for x in fullpop_steps),
        "first_gt_hit_path_step": first_hit,
        "last_gt_hit_path_step": last_hit,
        "gt_hit_path_span": last_hit - first_hit if hit_count else float("nan"),
        "first_gt_hit_fullpop_step": first_fullpop,
        "last_gt_hit_fullpop_step": last_fullpop,
        "gt_hit_fullpop_span": last_fullpop - first_fullpop if hit_count else float("nan"),
        "first_final_minus_first_gt_hit_path": first_final - first_hit
        if np.isfinite(first_final) and np.isfinite(first_hit)
        else float("nan"),
        "first_final_minus_last_gt_hit_path": first_final - last_hit
        if np.isfinite(first_final) and np.isfinite(last_hit)
        else float("nan"),
        "final2_hit_path_span": last_n_span(path_steps, 2),
        "final3_hit_path_span": last_n_span(path_steps, 3),
        "final5_hit_path_span": last_n_span(path_steps, 5),
        "final2_hit_fullpop_span": last_n_span(fullpop_steps, 2),
        "final3_hit_fullpop_span": last_n_span(fullpop_steps, 3),
        "final5_hit_fullpop_span": last_n_span(fullpop_steps, 5),
    }
    row.update(gap_stats(path_steps, "path_inter_hit"))
    row.update(gap_stats(fullpop_steps, "fullpop_inter_hit"))

    for window in WINDOWS:
        tail_count = int(sum((last_hit - float(step)) <= float(window) for step in path_steps)) if hit_count else 0
        row[f"tail_hits_within_{window}_path_steps"] = tail_count
        row[f"tail_hit_fraction_within_{window}_path_steps"] = float(tail_count) / float(hit_count) if hit_count else float("nan")
        max_count = max_hits_in_window(path_steps, window)
        row[f"max_hits_in_{window}_path_step_window"] = max_count
        row[f"max_hit_fraction_in_{window}_path_step_window"] = (
            float(max_count) / float(hit_count) if hit_count else float("nan")
        )
    return row


def run_dataset(dataset: str, cohort_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    dataset_file = first_pass.dataset_file_from_stem(dataset)
    dataset_path = Path(args.dataset_root) / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    print(f"[DATASET] {dataset} rows={len(cohort_df)}", flush=True)
    with h5py.File(dataset_path, "r") as handle:
        train_ds = handle["train"]
        test_ds = handle["test"]
        neighbors_ds = handle["neighbors"]
        n_train = int(train_ds.shape[0])
        dim = int(train_ds.shape[1])

        qids = cohort_df["qid"].to_numpy(dtype=np.int64)
        query_vectors = first_pass.read_rows(test_ds, qids)
        gt_labels_all = np.asarray(neighbors_ds[qids, : int(args.k)], dtype=np.int64)

        index, _index_path = first_pass.load_index(
            Path(args.index_dir),
            dataset_file,
            n_train,
            dim,
            int(args.m),
            int(args.ef_construction),
            int(args.num_threads),
        )
        trace_fn = getattr(index, "search_layer0_path_with_dist_metrics_batch", None)
        if trace_fn is None:
            raise RuntimeError("Loaded hnswlib does not expose search_layer0_path_with_dist_metrics_batch.")

        rows: list[dict[str, Any]] = []
        start_time = time.time()
        for start in range(0, len(qids), int(args.trace_batch_size)):
            end = min(start + int(args.trace_batch_size), len(qids))
            paths, _, _ = trace_fn(
                query_vectors[start:end],
                k=int(args.k),
                ef=int(args.ef),
                num_threads=int(args.num_threads),
            )
            for offset, path in enumerate(paths):
                source = cohort_df.iloc[start + offset]
                row = {
                    "dataset": dataset,
                    "qid": int(source["qid"]),
                    "cohort": str(source["cohort"]),
                    "route": int(source["route"]),
                    "drop": float(source["drop"]),
                    "classify_chr_mean": float(source["chr"]),
                    "classify_chr_ratio": float(source["ratio"]),
                    "feature_first_final_step": float(source["first_step"]),
                    "gt_k": int(args.k),
                    "trace_path_len": int(len(path)),
                }
                row.update(trace_hit_metrics(path, gt_labels_all[start + offset], float(source["first_step"])))
                rows.append(row)
            elapsed = time.time() - start_time
            print(f"[TRACE] {dataset} {end}/{len(qids)} elapsed={elapsed:.1f}s", flush=True)

        del index
        return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    metrics = [
        "feature_first_final_step",
        "gt_hit_unique_count",
        "first_gt_hit_path_step",
        "last_gt_hit_path_step",
        "gt_hit_path_span",
        "path_inter_hit_gap_p50",
        "path_inter_hit_gap_p90",
        "path_inter_hit_gap_max",
        "final3_hit_path_span",
        "final5_hit_path_span",
        "tail_hits_within_32_path_steps",
        "tail_hits_within_64_path_steps",
        "tail_hits_within_128_path_steps",
        "max_hits_in_32_path_step_window",
        "max_hits_in_64_path_step_window",
    ]
    for (dataset, cohort), group in rows.groupby(["dataset", "cohort"], sort=True):
        row: dict[str, Any] = {"dataset": dataset, "cohort": cohort, "n": int(len(group))}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
            row[f"{metric}_p25"] = finite_quantile(values, 0.25)
            row[f"{metric}_p50"] = finite_quantile(values, 0.50)
            row[f"{metric}_p75"] = finite_quantile(values, 0.75)
            row[f"{metric}_mean"] = finite_mean(values)
        row["pct_gt10_observed"] = finite_pct(group["gt_hit_unique_count"].ge(10))
        row["pct_gt9_observed"] = finite_pct(group["gt_hit_unique_count"].ge(9))
        row["pct_tail3_within_16_steps"] = finite_pct(group["final3_hit_path_span"].le(16))
        row["pct_tail3_within_32_steps"] = finite_pct(group["final3_hit_path_span"].le(32))
        row["pct_tail3_within_64_steps"] = finite_pct(group["final3_hit_path_span"].le(64))
        row["pct_tail5_within_64_steps"] = finite_pct(group["final5_hit_path_span"].le(64))
        row["pct_tail5_within_128_steps"] = finite_pct(group["final5_hit_path_span"].le(128))
        row["pct_max32_window_has_3plus_hits"] = finite_pct(group["max_hits_in_32_path_step_window"].ge(3))
        row["pct_last64_window_has_3plus_hits"] = finite_pct(group["tail_hits_within_64_path_steps"].ge(3))
        row["pct_last128_window_has_5plus_hits"] = finite_pct(group["tail_hits_within_128_path_steps"].ge(5))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def write_markdown(output_dir: Path, summary: pd.DataFrame, rows: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# False-Easy GT Hit Burst Diagnostics",
        "",
        f"- cohort CSV: `{args.cohort_csv}`",
        f"- ef: `{int(args.ef)}`",
        f"- k: `{int(args.k)}`",
        f"- cohorts: `{args.cohorts}`",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | hit10 observed | first final p50 | first GT p50 | last GT p50 | span p50 | gap p50 | tail3<=32 | tail5<=128 | last64 has 3+ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.n} | {row.pct_gt10_observed:.1f}% | "
            f"{row.feature_first_final_step_p50:.0f} | {row.first_gt_hit_path_step_p50:.0f} | "
            f"{row.last_gt_hit_path_step_p50:.0f} | {row.gt_hit_path_span_p50:.0f} | "
            f"{row.path_inter_hit_gap_p50_p50:.0f} | {row.pct_tail3_within_32_steps:.1f}% | "
            f"{row.pct_tail5_within_128_steps:.1f}% | {row.pct_last64_window_has_3plus_hits:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `first GT p50` is the first unique ground-truth label popped in the layer-0 trace.",
            "- `last GT p50` and `span p50` summarize how long it takes to see the observed GT labels.",
            "- `tail3<=32` / `tail5<=128` test a late-burst pattern: several final GT hits packed near the last hit.",
            "- This is a pop-trace diagnostic, not an exact intermediate top-k snapshot. In these false-easy cohorts, the trace observes at least 9 of 10 GT labels for nearly all queries, so it is a good proxy for hit-arrival shape.",
            "",
            "## Files",
            "",
            "- `per_query_gt_hit_burst_metrics.csv`: per-query hit step lists and burst metrics.",
            "- `gt_hit_burst_summary.csv`: per-dataset aggregate quantiles and burst rates.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = pd.read_csv(args.cohort_csv)
    wanted_cohorts = {x.strip() for x in str(args.cohorts).split(",") if x.strip()}
    cohorts = cohorts[cohorts["cohort"].isin(wanted_cohorts)].copy()
    if str(args.datasets).strip():
        wanted = {first_pass.dataset_stem(token.strip()) for token in str(args.datasets).split(",") if token.strip()}
        cohorts = cohorts[cohorts["dataset"].map(first_pass.dataset_stem).isin(wanted)].copy()
    if int(args.max_queries_per_dataset) > 0:
        cohorts = (
            cohorts.groupby("dataset", group_keys=False, sort=False)
            .head(int(args.max_queries_per_dataset))
            .reset_index(drop=True)
        )

    frames = []
    for dataset, group in cohorts.groupby("dataset", sort=True):
        frame = run_dataset(str(dataset), group.reset_index(drop=True), args)
        frame.to_csv(output_dir / f"{dataset}_gt_hit_burst_metrics.csv", index=False)
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows.to_csv(output_dir / "per_query_gt_hit_burst_metrics.csv", index=False)
    summary = summarize(rows) if not rows.empty else pd.DataFrame()
    summary.to_csv(output_dir / "gt_hit_burst_summary.csv", index=False)
    if not summary.empty:
        write_markdown(output_dir, summary, rows, args)
    print(f"[DONE] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
