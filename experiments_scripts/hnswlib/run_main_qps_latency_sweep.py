#!/usr/bin/env python3
"""Main SAGE recall/QPS/latency sweep with separate offline and online threads."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
EXPERIMENTS_SCRIPT_ROOT = EXP_ROOT.parent


def _find_default_project_root() -> Path:
    env_root = os.environ.get("HNSW_PLAYGROUND_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (EXP_ROOT, *EXP_ROOT.parents):
        if (candidate / "datasets").exists():
            return candidate
    return EXPERIMENTS_SCRIPT_ROOT.parent


PROJECT_ROOT = _find_default_project_root()
# Preload the compiled extension before adding experiments_scripts to sys.path;
# otherwise the sibling hnswlib/ directory can be resolved as a namespace package.
import hnswlib as _hnswlib_extension  # noqa: F401,E402
if str(EXPERIMENTS_SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_SCRIPT_ROOT))


from common.adaptive_runtime import (  # noqa: E402
    DEFAULT_CALIBRATION_SAMPLE_SEED,
    DEFAULT_EF_SWEEP_VALUES,
    DEFAULT_MIXED_GT_EF,
    DEFAULT_MIXED_GT_SOURCE,
    dedupe_preserve_order,
    evaluate_recall_per_query,
    flush_cache,
    format_float_signature,
    format_int_signature,
    load_dataset_with_special_cases,
    parse_ef_sweep,
    resolve_runtime_threshold_scales,
    run_adaptive_query,
    validate_query_method,
)
from common.benchmark_utils import benchmark_query_batch  # noqa: E402
from common.offline_calibration import (  # noqa: E402
    compute_fixed_calibration_lid_pool,
    resolve_mixed_policy_with_status,
)
from common.projected_local_acceptable_runtime import (  # noqa: E402
    _compute_gt_neighbors,
    _compute_recall_by_ef,
    _extract_chr_mean_by_query,
    _select_dummy_queries,
)
from final_index_utils import build_original_index  # noqa: E402


DATASETS = (
    "glove-100-angular.hdf5",
    "nytimes-256-angular.hdf5",
    "deep-100M.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "sift-100M-euclidean.hdf5",
    "cohere-768-angular.hdf5",
)
RUN_ROOT = EXP_ROOT / "main_qps_latency_total6_m32_efc500_ncal100_offline24_online1" / "run"
FINAL_DIR = EXP_ROOT / "main_qps_latency_total6_m32_efc500_ncal100_offline24_online1" / "final"
RESULT_CSV = FINAL_DIR / "main_qps_latency_sweep.csv"
RESULT_MD = FINAL_DIR / "main_qps_latency_sweep.md"
OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS = 0.001
OFFLINE_CALIBRATION_PROBE_ROUTING = "hide_node"


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty.")
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive.")
    return dedupe_preserve_order(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--base-path", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--index-dir", default=str(PROJECT_ROOT / "index"))
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("10"))
    parser.add_argument("--ef-sweep", type=parse_ef_sweep, default=DEFAULT_EF_SWEEP_VALUES)
    parser.add_argument("--offline-num-threads", type=int, default=24)
    parser.add_argument("--online-num-threads", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--query-method", choices=("adaptive-light", "adaptive"), default="adaptive-light")
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--internal-lid-k", type=int, default=15)
    parser.add_argument("--calibration-sample-seed", type=int, default=DEFAULT_CALIBRATION_SAMPLE_SEED)
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument(
        "--mixed-threshold-mode",
        choices=("paper_floor_half",),
        default="paper_floor_half",
    )
    parser.add_argument("--mixed-bucket-count", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--skip-unavailable-k", action="store_true", default=True)
    parser.add_argument("--fail-unavailable-k", action="store_false", dest="skip_unavailable_k")
    parser.add_argument(
        "--enable-drilldown",
        action="store_true",
        help="Also run the hnswlib-only easy/medium/hard group efSearch sweep.",
    )
    parser.add_argument(
        "--drilldown-output-dir",
        default="",
        help="Directory for drilldown CSV/MD outputs. Defaults under --final-dir.",
    )
    parser.add_argument("--drilldown-pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--drilldown-group-def-ef", type=int, default=1024)
    parser.add_argument(
        "--drilldown-ef-sweep",
        default="",
        help="Optional efSearch list/range for drilldown. Defaults to the main ef sweep.",
    )
    args = parser.parse_args()
    (
        args.easy_threshold_scale,
        args.mid_threshold_scale,
        args.super_threshold_scale,
    ) = resolve_runtime_threshold_scales(
        easy_threshold_scale=None,
        mid_threshold_scale=None,
        super_threshold_scale=None,
    )
    args.datasets = tuple(part.strip() for part in str(args.datasets).split(",") if part.strip())
    if not args.datasets:
        raise ValueError("--datasets must not be empty.")
    if int(args.offline_num_threads) < 1:
        raise ValueError("--offline-num-threads must be positive.")
    if int(args.online_num_threads) < 1:
        raise ValueError("--online-num-threads must be positive.")
    if int(args.measured_runs) < 1:
        raise ValueError("--measured-runs must be positive.")
    if int(args.warmup_runs) < 0:
        raise ValueError("--warmup-runs must be >= 0.")
    if int(args.num_calibration_queries) != 100:
        raise ValueError("This main run is fixed to calibration n=100.")
    if args.drilldown_ef_sweep:
        args.drilldown_ef_sweep = parse_ef_sweep(str(args.drilldown_ef_sweep))
    else:
        args.drilldown_ef_sweep = None
    if int(args.drilldown_pseudo_gt_ef) < 1:
        raise ValueError("--drilldown-pseudo-gt-ef must be positive.")
    if int(args.drilldown_group_def_ef) < 1:
        raise ValueError("--drilldown-group-def-ef must be positive.")
    if int(args.drilldown_pseudo_gt_ef) < int(args.drilldown_group_def_ef):
        raise ValueError("--drilldown-pseudo-gt-ef must be >= --drilldown-group-def-ef.")
    if int(args.mixed_bucket_count) < 2 and str(args.mixed_threshold_mode) == "paper_floor_half":
        raise ValueError("--mixed-bucket-count must be at least 2 for paper_floor_half.")
    return args


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = "" if pd.isna(value) else f"{float(value):.3f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def latency_metrics(benchmark: dict[str, Any], query_count: int) -> dict[str, float]:
    durations = np.asarray(benchmark["durations_s"], dtype=np.float64)
    query_count = int(query_count)
    mean_ms = float(benchmark["avg_duration_s"] * 1000.0)
    return {
        "query_count": query_count,
        "batch_latency_mean_ms": mean_ms,
        "batch_latency_p50_ms": float(np.percentile(durations, 50) * 1000.0),
        "batch_latency_p95_ms": float(np.percentile(durations, 95) * 1000.0),
        "batch_latency_min_ms": float(benchmark["min_duration_s"] * 1000.0),
        "batch_latency_max_ms": float(benchmark["max_duration_s"] * 1000.0),
        "latency_per_query_mean_ms": mean_ms / float(query_count),
    }


def benchmark_baseline_k(
    *,
    index,
    test: np.ndarray,
    neighbors: np.ndarray,
    ef: int,
    k: int,
    warmup_runs: int,
    measured_runs: int,
    num_threads: int,
) -> dict[str, float]:
    def run_once():
        index.set_ef(int(ef))
        return index.knn_query(test, k=int(k), num_threads=int(num_threads))

    flush_cache()
    benchmark = benchmark_query_batch(
        run_once=run_once,
        query_count=len(test),
        warmup_runs=int(warmup_runs),
        measured_runs=int(measured_runs),
    )
    labels, _ = benchmark["last_output"]
    recalls = evaluate_recall_per_query(labels, neighbors, int(k))
    metrics = {
        "recall": float(np.mean(recalls)),
        "qps": float(benchmark["qps"]),
    }
    metrics.update(latency_metrics(benchmark, query_count=len(test)))
    return metrics


def benchmark_ours_k(
    *,
    index,
    test: np.ndarray,
    neighbors: np.ndarray,
    ef: int,
    k: int,
    query_method: str,
    tau: float,
    super_gamma: float,
    mid_gamma: float,
    tmin_pops: int,
    mixed_threshold_mode: str,
    paper_bucket_count: int,
    paper_bucket_gamma_ratios: tuple[float, ...],
    warmup_runs: int,
    measured_runs: int,
    num_threads: int,
) -> dict[str, float]:
    def run_once():
        return run_adaptive_query(
            index=index,
            test=test,
            k_search=int(k),
            ef=int(ef),
            query_method=str(query_method),
            enable_stop=True,
            early_stop_ratio=float(tau),
            super_easy_gamma_ratio=float(super_gamma),
            mid_easy_upper_gamma_ratio=float(mid_gamma),
            tmin_pops=int(tmin_pops),
            mixed_threshold_mode=str(mixed_threshold_mode),
            paper_bucket_count=int(paper_bucket_count),
            paper_bucket_gamma_ratios=paper_bucket_gamma_ratios,
            num_threads=int(num_threads),
        )

    flush_cache()
    benchmark = benchmark_query_batch(
        run_once=run_once,
        query_count=len(test),
        warmup_runs=int(warmup_runs),
        measured_runs=int(measured_runs),
    )
    adaptive_output = benchmark["last_output"]
    if str(query_method) == "adaptive-light":
        labels, dists = adaptive_output[:2]
        reduced_steps = None
        stop_count = np.nan
    else:
        labels, dists, reduced_steps, stop_count = adaptive_output
    recalls = evaluate_recall_per_query(labels, neighbors, int(k))
    metrics = {
        "recall": float(np.mean(recalls)),
        "qps": float(benchmark["qps"]),
        "adaptive_max_dist_mean": float(np.mean(np.max(dists, axis=1))),
        "stop_count": stop_count,
        "reduced_steps_mean": np.nan if reduced_steps is None else float(np.mean(reduced_steps)),
        "reduced_steps_max": np.nan if reduced_steps is None else int(np.max(reduced_steps)),
    }
    metrics.update(latency_metrics(benchmark, query_count=len(test)))
    return metrics



def write_summary(
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    *,
    final_dir: Path,
    result_csv: Path,
    result_md: Path,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["dataset", "k", "ef", "method"]).reset_index(drop=True)
    result.to_csv(result_csv, index=False)

    with result_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Main SAGE Recall/QPS/Latency Sweep\n\n")
        handle.write(
            "Setup: vanilla HNSW vs ours, M=32, efConstruction=500, "
            "calibration n=100, offline threads=24, online search threads=1.\n\n"
        )
        if result.empty:
            handle.write("No completed result rows yet.\n")
        else:
            ours = result[result["method"] == "Ours"].copy()
            handle.write(f"- Result rows: {len(result)}.\n")
            handle.write(f"- Completed ours cells: {len(ours)}.\n")
            handle.write(f"- Datasets: {', '.join(sorted(result['dataset'].unique()))}.\n")
            handle.write(f"- efSearch values completed: {', '.join(str(int(v)) for v in sorted(result['ef'].unique()))}.\n\n")
            if not ours.empty:
                latest = (
                    ours.sort_values(["dataset", "k", "ef"])
                    .groupby(["dataset", "k"], as_index=False)
                    .tail(1)
                    .sort_values(["dataset", "k"])
                )
                handle.write("## Highest ef per dataset/k\n\n")
                handle.write(
                    markdown_table(
                        latest[
                            [
                                "dataset",
                                "k",
                                "ef",
                                "recall",
                                "recall_loss_vs_vanilla_pp",
                                "qps",
                                "qps_gain_vs_vanilla_pct",
                                "batch_latency_mean_ms",
                                "offline_calibration_wall_s",
                            ]
                        ]
                    )
                )
                handle.write("\n")
        if skips:
            skip_df = pd.DataFrame(skips)
            handle.write("\n## Skips\n\n")
            handle.write(markdown_table(skip_df))
            handle.write("\n")


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def result_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    return run_root / dataset_stem(dataset) / f"{dataset_stem(dataset)}__k{int(k)}__main_qps_latency.csv"


def offline_curve_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    stem = dataset_stem(dataset)
    return run_root / stem / f"{stem}__k{int(k)}__offline_predicted_recall_curve.csv"


def offline_recommended_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    stem = dataset_stem(dataset)
    return run_root / stem / f"{stem}__k{int(k)}__offline_recommended_efsearch.csv"


def route_count_signature(routed_efs: np.ndarray) -> str:
    values, counts = np.unique(np.asarray(routed_efs, dtype=np.int64), return_counts=True)
    return ";".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def route_ef_for_chr_ratio(
    *,
    selection_ef: int,
    k: int,
    route_efs: tuple[int, ...],
    bucket_gamma_ratios: tuple[float, ...],
    chr_ratio: float,
) -> int:
    if not np.isfinite(chr_ratio):
        return int(selection_ef)
    for route_ef, gamma in zip(route_efs, bucket_gamma_ratios):
        if float(chr_ratio) <= float(gamma) + 1e-12:
            return max(int(k), int(route_ef))
    return int(selection_ef)


def compute_offline_recommended_efsearch(
    *,
    index,
    train: np.ndarray,
    dataset: str,
    stem: str,
    k: int,
    gt_k: int,
    ef_values: list[int],
    tau_by_ef: dict[int, float],
    policy,
    calibration_lid_df: pd.DataFrame,
    num_calibration_queries: int,
    trim_low_percentile: float,
    trim_high_percentile: float,
    num_threads: int,
    mixed_threshold_mode: str,
    mixed_bucket_count: int,
    cache_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_df = _select_dummy_queries(
        index=index,
        num_nodes=len(train),
        lid_df=calibration_lid_df,
        num_calibration_queries=int(num_calibration_queries),
        selection_mode="quantile",
        trim_low_percentile=float(trim_low_percentile),
        trim_high_percentile=float(trim_high_percentile),
    )
    query_ids = selected_df["query_id"].to_numpy(dtype=np.int64)
    query_vectors = np.asarray(train[query_ids], dtype=np.float32)
    gt_neighbors = _compute_gt_neighbors(
        index=index,
        train=np.asarray(train, dtype=np.float32),
        dataset_name=dataset,
        query_ids=query_ids,
        query_vectors=query_vectors,
        gt_source=DEFAULT_MIXED_GT_SOURCE,
        gt_ef=int(DEFAULT_MIXED_GT_EF),
        num_threads=int(num_threads),
        k=int(k),
    )

    recall_cache: dict[int, np.ndarray] = {}

    def recall_for(ef_value: int) -> np.ndarray:
        resolved_ef = int(max(int(k), int(ef_value)))
        if resolved_ef not in recall_cache:
            recall_cache[resolved_ef] = _compute_recall_by_ef(
                index=index,
                query_ids=query_ids,
                query_vectors=query_vectors,
                gt_neighbors=gt_neighbors,
                ef_value=resolved_ef,
                num_threads=int(num_threads),
                k=int(k),
            )
        return recall_cache[resolved_ef]

    curve_rows: list[dict[str, Any]] = []
    for ef in sorted({int(value) for value in ef_values}):
        full_recalls = recall_for(ef)
        routed_recalls = np.array(full_recalls, dtype=np.float64, copy=True)
        routed_efs = np.full(len(selected_df), int(ef), dtype=np.int64)
        route_efs = tuple(int(value) for value in getattr(policy, "route_efs_by_ef", {}).get(int(ef), ()))
        bucket_gammas = tuple(
            float(value) for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(int(ef), ())
        )
        usable_count = 0
        cfr_mean = float("nan")
        cfr_std = float("nan")
        cfr_p10 = float("nan")
        cfr_p50 = float("nan")
        cfr_p90 = float("nan")
        cfr_p95 = float("nan")
        tau = float(tau_by_ef[int(ef)])

        anchor_df = _extract_chr_mean_by_query(
            index=index,
            selected_df=selected_df,
            query_vectors=query_vectors,
            query_ids=query_ids,
            selection_ef=int(ef),
            num_threads=int(num_threads),
            k=int(k),
        )
        usable_mask = anchor_df["usable_for_mean_window_calibration"].astype(bool).to_numpy(dtype=bool)
        chr_values = pd.to_numeric(
            anchor_df["mean_smoothed_chr_classify_window"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_cfr_values = chr_values[usable_mask & np.isfinite(chr_values)]
        usable_count = int(finite_cfr_values.size)
        if finite_cfr_values.size:
            cfr_mean = float(np.mean(finite_cfr_values))
            cfr_std = float(np.std(finite_cfr_values))
            cfr_p10 = float(np.quantile(finite_cfr_values, 0.10))
            cfr_p50 = float(np.quantile(finite_cfr_values, 0.50))
            cfr_p90 = float(np.quantile(finite_cfr_values, 0.90))
            cfr_p95 = float(np.quantile(finite_cfr_values, 0.95))

        if route_efs and bucket_gammas and len(route_efs) == len(bucket_gammas):
            for idx, chr_value in enumerate(chr_values):
                if not usable_mask[idx] or not np.isfinite(chr_value):
                    continue
                chr_ratio = float(chr_value) / max(float(tau), 1e-6)
                routed_ef = route_ef_for_chr_ratio(
                    selection_ef=int(ef),
                    k=int(k),
                    route_efs=route_efs,
                    bucket_gamma_ratios=bucket_gammas,
                    chr_ratio=chr_ratio,
                )
                routed_efs[idx] = int(routed_ef)
                if int(routed_ef) != int(ef):
                    routed_recalls[idx] = float(recall_for(int(routed_ef))[idx])

        curve_rows.append(
            {
                "dataset": stem,
                "dataset_file": dataset,
                "k": int(k),
                "groundtruth_k": int(gt_k),
                "ef": int(ef),
                "offline_predicted_recall": float(np.mean(routed_recalls)),
                "offline_vanilla_recall": float(np.mean(full_recalls)),
                "calibration_query_count": int(len(selected_df)),
                "calibration_lid_pool_count": int(len(calibration_lid_df)),
                "usable_chr_query_count": int(usable_count),
                "cfr_metric": "mean_smoothed_chr_classify_window",
                "cfr_mean": cfr_mean,
                "cfr_std": cfr_std,
                "cfr_p10": cfr_p10,
                "cfr_p50": cfr_p50,
                "cfr_p90": cfr_p90,
                "cfr_p95": cfr_p95,
                "offline_num_threads": int(num_threads),
                "mixed_threshold_mode": str(mixed_threshold_mode),
                "mixed_bucket_count": int(mixed_bucket_count) if str(mixed_threshold_mode) == "paper_floor_half" else np.nan,
                "route_signature": format_int_signature(route_efs + (int(ef),)) if route_efs else "",
                "bucket_gamma_signature": format_float_signature(bucket_gammas) if bucket_gammas else "",
                "routed_ef_count_signature": route_count_signature(routed_efs),
                "early_stop_ratio": float(tau),
                "stop_config_source": str(policy.source_label),
                "cache_path": str(cache_path),
                "recommendation_source": "offline_calibration_proxy",
                "calibration_probe_routing": OFFLINE_CALIBRATION_PROBE_ROUTING,
            }
        )

    curve_df = pd.DataFrame(curve_rows).sort_values("ef", kind="stable").reset_index(drop=True)
    # Start-wide reads the baseline efSearch off the PRE-ROUTE (vanilla) probe recall
    # curve, matching the paper's saturation definition (Sec. 4.2). Using the routed
    # (post-cut) recall here would entangle baseline selection with the cut policy.
    cumulative = np.maximum.accumulate(curve_df["offline_vanilla_recall"].to_numpy(dtype=np.float64))
    max_cumulative = float(cumulative[-1]) if cumulative.size else float("nan")
    remaining = max_cumulative - cumulative
    previous_gain = np.empty_like(cumulative)
    previous_gain[:] = np.nan
    if cumulative.size > 1:
        previous_gain[1:] = cumulative[1:] - cumulative[:-1]
    curve_df["offline_cumulative_recall"] = cumulative
    curve_df["max_cumulative_recall"] = max_cumulative
    curve_df["remaining_cumulative_recall_gain"] = remaining
    curve_df["previous_step_cumulative_gain"] = previous_gain
    curve_df["recommendation_eps"] = OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS

    eligible = np.flatnonzero(remaining <= OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS)
    selected_idx = int(eligible[0]) if eligible.size else int(len(curve_df) - 1)
    selected = curve_df.iloc[selected_idx]
    recommended_df = pd.DataFrame(
        [
            {
                "dataset": stem,
                "dataset_file": dataset,
                "k": int(k),
                "recommended_ef": int(selected["ef"]),
                "offline_predicted_recall": float(selected["offline_predicted_recall"]),
                "offline_cumulative_recall": float(selected["offline_cumulative_recall"]),
                "max_cumulative_recall": float(selected["max_cumulative_recall"]),
                "remaining_cumulative_recall_gain": float(selected["remaining_cumulative_recall_gain"]),
                "previous_step_cumulative_gain": float(selected["previous_step_cumulative_gain"]),
                "recommendation_eps": OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS,
                "selection_rule": (
                    "first ef where max cumulative pre-route (vanilla) probe Recall@10 minus "
                    f"current cumulative Recall@10 <= {OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS:g}"
                    if eligible.size
                    else "fallback last ef; no offline saturation point found"
                ),
                "calibration_query_count": int(len(selected_df)),
                "calibration_lid_pool_count": int(len(calibration_lid_df)),
                "recommendation_source": "offline_calibration_proxy",
                "calibration_probe_routing": OFFLINE_CALIBRATION_PROBE_ROUTING,
                "curve_csv": str(offline_curve_csv_for(Path(""), dataset, int(k))).lstrip("/"),
                "cache_path": str(cache_path),
            }
        ]
    )
    return curve_df, recommended_df


def write_offline_recommendation_summary(
    *,
    final_dir: Path,
    curve_rows: list[dict[str, Any]],
    recommended_rows: list[dict[str, Any]],
) -> None:
    if curve_rows:
        pd.DataFrame(curve_rows).sort_values(["dataset", "k", "ef"]).to_csv(
            final_dir / "offline_predicted_recall_curve.csv",
            index=False,
        )
    if recommended_rows:
        pd.DataFrame(recommended_rows).sort_values(["dataset", "k"]).to_csv(
            final_dir / "offline_recommended_efsearch.csv",
            index=False,
        )




def drilldown_output_dir(args: argparse.Namespace, final_dir: Path) -> Path:
    if str(args.drilldown_output_dir).strip():
        return Path(args.drilldown_output_dir).expanduser().resolve()
    return (
        final_dir
        / f"easy_medium_hard_drilldown_pseudogt{int(args.drilldown_pseudo_gt_ef)}"
        / f"groupdef{int(args.drilldown_group_def_ef)}"
    )


def drilldown_dataset_dir(args: argparse.Namespace, final_dir: Path, dataset: str) -> Path:
    return drilldown_output_dir(args, final_dir) / dataset_stem(dataset)


def drilldown_sweep_csv_for(args: argparse.Namespace, final_dir: Path, dataset: str, k: int) -> Path:
    stem = dataset_stem(dataset)
    return drilldown_dataset_dir(args, final_dir, dataset) / f"{stem}__k{int(k)}__group_ef_sweep.csv"


def replace_csv_rows(
    path: Path,
    new_df: pd.DataFrame,
    *,
    key_values: dict[str, Any],
    sort_columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        drop_mask = pd.Series(True, index=existing.index)
        for col, value in key_values.items():
            if col not in existing.columns:
                drop_mask &= False
            else:
                drop_mask &= existing[col].astype(str) == str(value)
        combined = pd.concat([existing.loc[~drop_mask], new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    if sort_columns:
        present = [col for col in sort_columns if col in combined.columns]
        if present:
            combined = combined.sort_values(present).reset_index(drop=True)
    combined.to_csv(path, index=False)


def compute_vanilla_labels(
    *,
    index,
    test: np.ndarray,
    ef: int,
    k: int,
    num_threads: int,
) -> np.ndarray:
    index.set_ef(int(ef))
    labels, _ = index.knn_query(test, k=int(k), num_threads=int(num_threads))
    return labels


def compute_hit_counts(labels: np.ndarray, gt_labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels)
    gt_labels = np.asarray(gt_labels)
    k = int(k)
    if labels.shape[0] != gt_labels.shape[0]:
        raise ValueError("labels and gt_labels must have the same query count.")

    hit_counts = np.zeros(labels.shape[0], dtype=np.int64)
    for row in range(labels.shape[0]):
        hit_counts[row] = int(np.intersect1d(labels[row][:k], gt_labels[row][:k]).size)
    return hit_counts


def compute_first_final_recall_steps(
    *,
    index,
    test: np.ndarray,
    gt_labels: np.ndarray,
    target_hits: np.ndarray,
    ef: int,
    k: int,
    num_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(index, "knn_query_beam_width_first_target_hit_step"):
        raise RuntimeError(
            "Drilldown tie-break requires knn_query_beam_width_first_target_hit_step, "
            "but the loaded hnswlib index does not expose it."
        )

    first_steps, reached_flags, achieved_hits, _reached_count = index.knn_query_beam_width_first_target_hit_step(
        test,
        np.asarray(gt_labels, dtype=np.uint64),
        np.asarray(target_hits, dtype=np.uint64),
        k=int(k),
        ef_before=int(ef),
        switch_pop=0,
        switch_full_pop=0,
        ef_after=int(ef),
        num_threads=int(num_threads),
    )
    return (
        np.asarray(first_steps, dtype=np.int64),
        np.asarray(reached_flags, dtype=np.int64),
        np.asarray(achieved_hits, dtype=np.int64),
    )


def assign_easiness_groups(
    recalls: np.ndarray,
    first_final_recall_steps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recalls = np.nan_to_num(np.asarray(recalls, dtype=np.float64), nan=-1.0)
    query_count = int(len(recalls))
    if first_final_recall_steps is None:
        first_steps = np.full(query_count, np.inf, dtype=np.float64)
    else:
        first_steps = np.nan_to_num(
            np.asarray(first_final_recall_steps, dtype=np.float64),
            nan=np.inf,
            posinf=np.inf,
            neginf=np.inf,
        )
        if len(first_steps) != query_count:
            raise ValueError("first_final_recall_steps must match recalls length.")

    order = np.lexsort((np.arange(query_count, dtype=np.int64), first_steps, -recalls))
    easy_count = int(np.floor(query_count * 0.30))
    hard_count = int(np.floor(query_count * 0.30))
    medium_count = query_count - easy_count - hard_count
    groups = np.empty(query_count, dtype=object)
    groups[order[:easy_count]] = "easy"
    groups[order[easy_count : easy_count + medium_count]] = "medium"
    groups[order[easy_count + medium_count :]] = "hard"
    ranks = np.empty(query_count, dtype=np.int64)
    ranks[order] = np.arange(1, query_count + 1, dtype=np.int64)
    if query_count > 1:
        percentiles = (ranks.astype(np.float64) - 1.0) / float(query_count - 1)
    else:
        percentiles = np.zeros(query_count, dtype=np.float64)
    return groups, ranks, percentiles


def write_drilldown_readme(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# HNSW Easy/Medium/Hard Drilldown\n\n"
        "Groups are defined once per dataset/k using Vanilla recall against a pseudo ground truth.\n\n"
        f"- pseudo ground truth: Vanilla HNSW at efSearch={int(args.drilldown_pseudo_gt_ef)}\n"
        f"- group definition: Vanilla HNSW at efSearch={int(args.drilldown_group_def_ef)}\n"
        "- split: top 30% easy, middle 40% medium, bottom 30% hard by group-definition recall\n"
        "- tie-break: recall ties use the earliest first step reaching that final recall when "
        "the loaded hnswlib exposes that instrumentation; otherwise qid is used\n"
        "- comparison: Vanilla and Ours are evaluated on the same fixed query groups\n"
        "- scope: hnswlib only\n\n"
        "Primary files:\n\n"
        "- `query_groups.csv`: fixed query group assignment\n"
        "- `group_ef_sweep.csv`: per-group recall/QPS/latency rows for Vanilla and Ours\n"
        "- `group_pair_metrics.csv`: Ours-vs-Vanilla deltas on the fixed groups\n",
        encoding="utf-8",
        newline="\n",
    )


def metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "query_count",
        "batch_latency_mean_ms",
        "batch_latency_p50_ms",
        "batch_latency_p95_ms",
        "batch_latency_min_ms",
        "batch_latency_max_ms",
        "latency_per_query_mean_ms",
    ]
    return {key: metrics[key] for key in keys}


def run_drilldown_for_k(
    *,
    args: argparse.Namespace,
    final_dir: Path,
    index,
    test: np.ndarray,
    dataset: str,
    k: int,
    gt_k: int,
    k_ef_values: list[int],
    tau_by_ef: dict[int, float],
    policy,
    calibration_lid_df: pd.DataFrame,
    calibration_lid_pool_wall_s: float,
    threshold_calibration_wall_s: float,
    offline_calibration_wall_s: float,
    dataset_load_wall_s: float,
    index_load_wall_s: float,
    cache_path: Path,
    emit,
) -> None:
    if int(args.drilldown_pseudo_gt_ef) < int(k):
        raise ValueError("--drilldown-pseudo-gt-ef must be >= k.")
    if int(args.drilldown_group_def_ef) < int(k):
        raise ValueError("--drilldown-group-def-ef must be >= k.")

    output_dir = drilldown_output_dir(args, final_dir)
    dataset_dir = drilldown_dataset_dir(args, final_dir, dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_drilldown_readme(output_dir, args)

    if args.drilldown_ef_sweep is None:
        drilldown_efs = [int(ef) for ef in k_ef_values if int(ef) >= int(k)]
    else:
        drilldown_efs = [int(ef) for ef in args.drilldown_ef_sweep if int(ef) >= int(k)]
    drilldown_efs = dedupe_preserve_order(drilldown_efs)
    if not drilldown_efs:
        emit(f"[DRILLDOWN] {dataset} k={int(k)} no valid efSearch values; skipping")
        return
    missing_policy_efs = [ef for ef in drilldown_efs if int(ef) not in tau_by_ef]
    if missing_policy_efs:
        raise ValueError(
            "Drilldown ef values must also be present in the main --ef-sweep so Ours policy is calibrated: "
            f"missing={missing_policy_efs}"
        )

    stem = dataset_stem(dataset)
    query_count = int(len(test))
    emit(
        f"[DRILLDOWN] {dataset} k={int(k)} pseudo_gt_ef={int(args.drilldown_pseudo_gt_ef)} "
        f"group_def_ef={int(args.drilldown_group_def_ef)} ef_sweep={drilldown_efs}"
    )

    pseudo_start = time.perf_counter()
    pseudo_gt_labels = compute_vanilla_labels(
        index=index,
        test=test,
        ef=int(args.drilldown_pseudo_gt_ef),
        k=int(k),
        num_threads=int(args.online_num_threads),
    )
    group_def_labels = compute_vanilla_labels(
        index=index,
        test=test,
        ef=int(args.drilldown_group_def_ef),
        k=int(k),
        num_threads=int(args.online_num_threads),
    )
    group_def_recalls = evaluate_recall_per_query(group_def_labels, pseudo_gt_labels, int(k))
    group_def_target_hits = compute_hit_counts(group_def_labels, pseudo_gt_labels, int(k))
    first_step_tie_break_available = hasattr(index, "knn_query_beam_width_first_target_hit_step")
    (
        group_def_first_steps,
        group_def_first_step_reached,
        group_def_first_step_achieved_hits,
    ) = compute_first_final_recall_steps(
        index=index,
        test=test,
        gt_labels=pseudo_gt_labels,
        target_hits=group_def_target_hits,
        ef=int(args.drilldown_group_def_ef),
        k=int(k),
        num_threads=int(args.online_num_threads),
    )
    group_tie_breaker = "first_final_recall_step"
    groups, easiness_ranks, easiness_percentiles = assign_easiness_groups(
        group_def_recalls,
        group_def_first_steps,
    )
    pseudo_wall_s = time.perf_counter() - pseudo_start
    if group_def_first_steps is None:
        group_def_first_steps_for_csv = np.full(query_count, -1, dtype=np.int64)
    else:
        group_def_first_steps_for_csv = group_def_first_steps.astype(np.int64)

    query_group_df = pd.DataFrame(
        {
            "dataset": stem,
            "dataset_file": dataset,
            "k": int(k),
            "qid": np.arange(query_count, dtype=np.int64),
            "pseudo_gt_ef": int(args.drilldown_pseudo_gt_ef),
            "group_def_ef": int(args.drilldown_group_def_ef),
            "group_tie_breaker": group_tie_breaker,
            "first_step_tie_break_available": bool(first_step_tie_break_available),
            "group_def_vanilla_recall": group_def_recalls.astype(np.float64),
            "group_def_target_hits": group_def_target_hits.astype(np.int64),
            "group_def_first_final_recall_step": group_def_first_steps_for_csv,
            "group_def_first_step_reached_target": group_def_first_step_reached.astype(np.int64),
            "group_def_first_step_achieved_hits": group_def_first_step_achieved_hits.astype(np.int64),
            "easiness_group": groups,
            "easiness_rank": easiness_ranks,
            "easiness_percentile": easiness_percentiles,
        }
    )
    replace_csv_rows(
        output_dir / "query_groups.csv",
        query_group_df,
        key_values={"dataset_file": dataset, "k": int(k)},
        sort_columns=["dataset", "k", "qid"],
    )
    query_group_df.to_csv(dataset_dir / f"{stem}__k{int(k)}__query_groups.csv", index=False)

    group_counts = query_group_df.groupby("easiness_group").size().to_dict()
    easy_group_count = int(group_counts.get("easy", 0))
    medium_group_count = int(group_counts.get("medium", 0))
    hard_group_count = int(group_counts.get("hard", 0))
    emit(
        f"[DRILLDOWN] groups easy={easy_group_count} medium={medium_group_count} "
        f"hard={hard_group_count} tie_break={group_tie_breaker} pseudo_wall_s={pseudo_wall_s:.3f}"
    )

    sweep_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for ef in drilldown_efs:
        ef = int(ef)
        route_efs = tuple(int(value) for value in getattr(policy, "route_efs_by_ef", {}).get(ef, ()))
        bucket_gammas = tuple(
            float(value) for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(ef, ())
        )
        paper_bucket_count = (
            int(args.mixed_bucket_count)
            if str(args.mixed_threshold_mode) == "paper_floor_half"
            else 0
        )
        for group_name in ("easy", "medium", "hard"):
            group_mask = groups == group_name
            group_query_count = int(np.sum(group_mask))
            if group_query_count < 1:
                continue
            group_test = np.ascontiguousarray(test[group_mask])
            group_neighbors = np.ascontiguousarray(pseudo_gt_labels[group_mask])
            vanilla_metrics = benchmark_baseline_k(
                index=index,
                test=group_test,
                neighbors=group_neighbors,
                ef=ef,
                k=int(k),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                num_threads=int(args.online_num_threads),
            )
            ours_metrics = benchmark_ours_k(
                index=index,
                test=group_test,
                neighbors=group_neighbors,
                ef=ef,
                k=int(k),
                query_method=str(args.query_method),
                tau=float(tau_by_ef[ef]),
                super_gamma=float(policy.gamma_ratio_by_ef.get(ef, float("nan"))),
                mid_gamma=float(policy.mid_easy_upper_gamma_ratio_by_ef.get(ef, float("nan"))),
                tmin_pops=int(args.tmin_pops),
                mixed_threshold_mode=str(args.mixed_threshold_mode),
                paper_bucket_count=paper_bucket_count,
                paper_bucket_gamma_ratios=bucket_gammas,
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                num_threads=int(args.online_num_threads),
            )
            vanilla_recall = float(vanilla_metrics["recall"])
            ours_recall = float(ours_metrics["recall"])
            vanilla_qps = float(vanilla_metrics["qps"])
            ours_qps = float(ours_metrics["qps"])
            vanilla_latency = float(vanilla_metrics["latency_per_query_mean_ms"])
            ours_latency = float(ours_metrics["latency_per_query_mean_ms"])
            recall_delta_pp = float(ours_recall - vanilla_recall) * 100.0
            recall_loss_pp = -recall_delta_pp
            qps_gain_pct = (
                float((ours_qps / vanilla_qps - 1.0) * 100.0)
                if vanilla_qps > 0.0
                else float("nan")
            )
            latency_speedup = vanilla_latency / ours_latency if ours_latency > 0.0 else float("nan")
            base_common = {
                "dataset": stem,
                "dataset_file": dataset,
                "k": int(k),
                "groundtruth_k": int(gt_k),
                "pseudo_gt_ef": int(args.drilldown_pseudo_gt_ef),
                "group_def_ef": int(args.drilldown_group_def_ef),
                "group_tie_breaker": group_tie_breaker,
                "first_step_tie_break_available": bool(first_step_tie_break_available),
                "easiness_group": group_name,
                "group_query_count": group_query_count,
                "ef": ef,
                "offline_num_threads": int(args.offline_num_threads),
                "online_num_threads": int(args.online_num_threads),
                "num_threads": int(args.online_num_threads),
                "warmup_runs": int(args.warmup_runs),
                "measured_runs": int(args.measured_runs),
                "M": int(args.param_m),
                "efConstruction": int(args.ef_construction),
                "query_method": str(args.query_method),
                "num_calibration_queries": int(args.num_calibration_queries),
                "calibration_lid_pool_count": int(len(calibration_lid_df)),
                "calibration_lid_pool_wall_s": float(calibration_lid_pool_wall_s),
                "threshold_calibration_wall_s": float(threshold_calibration_wall_s),
                "offline_calibration_wall_s": float(offline_calibration_wall_s),
                "dataset_load_wall_s": float(dataset_load_wall_s),
                "index_load_wall_s": float(index_load_wall_s),
                "drilldown_pseudo_wall_s": float(pseudo_wall_s),
                "mixed_threshold_mode": str(args.mixed_threshold_mode),
                "mixed_bucket_count": (
                    int(args.mixed_bucket_count)
                    if str(args.mixed_threshold_mode) == "paper_floor_half"
                    else np.nan
                ),
                "route_signature": format_int_signature(route_efs + (ef,)) if route_efs else "",
                "bucket_gamma_signature": format_float_signature(bucket_gammas) if bucket_gammas else "",
                "early_stop_ratio": float(tau_by_ef[ef]),
                "stop_config_source": str(policy.source_label),
                "cache_path": str(cache_path),
            }
            vanilla_row = {
                **base_common,
                "method": "Vanilla",
                "enable_stop": False,
                "recall": vanilla_recall,
                "qps": vanilla_qps,
                "adaptive_max_dist_mean": np.nan,
                "stop_count": np.nan,
                "reduced_steps_mean": np.nan,
                "reduced_steps_max": np.nan,
            }
            vanilla_row.update(metric_subset(vanilla_metrics))
            ours_row = {
                **base_common,
                "method": "Ours",
                "enable_stop": True,
                "recall": ours_recall,
                "qps": ours_qps,
                "adaptive_max_dist_mean": float(ours_metrics["adaptive_max_dist_mean"]),
                "stop_count": ours_metrics["stop_count"],
                "reduced_steps_mean": ours_metrics["reduced_steps_mean"],
                "reduced_steps_max": ours_metrics["reduced_steps_max"],
            }
            ours_row.update(metric_subset(ours_metrics))
            sweep_rows.extend([vanilla_row, ours_row])
            pair_rows.append(
                {
                    **base_common,
                    "vanilla_recall": vanilla_recall,
                    "ours_recall": ours_recall,
                    "recall_delta_ours_minus_vanilla_pp": recall_delta_pp,
                    "recall_loss_vs_vanilla_pp": recall_loss_pp,
                    "recall_loss_clamped_pp": max(0.0, recall_loss_pp),
                    "vanilla_qps": vanilla_qps,
                    "ours_qps": ours_qps,
                    "qps_gain_vs_vanilla_pct": qps_gain_pct,
                    "latency_speedup_vs_vanilla": latency_speedup,
                    "vanilla_latency_per_query_mean_ms": vanilla_latency,
                    "ours_latency_per_query_mean_ms": ours_latency,
                }
            )
            emit(
                f"[DRILLDOWN k={int(k)} ef={ef} {group_name}] "
                f"vanilla={vanilla_recall:.5f}/{vanilla_qps:.1f}qps "
                f"ours={ours_recall:.5f}/{ours_qps:.1f}qps "
                f"delta={recall_delta_pp:+.3f}pp speedup={latency_speedup:.3f}x"
            )

    sweep_df = pd.DataFrame(sweep_rows)
    pair_df = pd.DataFrame(pair_rows)
    replace_csv_rows(
        output_dir / "group_ef_sweep.csv",
        sweep_df,
        key_values={"dataset_file": dataset, "k": int(k)},
        sort_columns=["dataset", "k", "ef", "easiness_group", "method"],
    )
    replace_csv_rows(
        output_dir / "group_pair_metrics.csv",
        pair_df,
        key_values={"dataset_file": dataset, "k": int(k)},
        sort_columns=["dataset", "k", "ef", "easiness_group"],
    )
    sweep_df.to_csv(dataset_dir / f"{stem}__k{int(k)}__group_ef_sweep.csv", index=False)
    pair_df.to_csv(dataset_dir / f"{stem}__k{int(k)}__group_pair_metrics.csv", index=False)
    emit(f"[DRILLDOWN] Wrote {dataset_dir}")



def append_status(status_path: Path, fields: list[Any]) -> None:
    with status_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(str(field) for field in fields) + "\n")


def offline_recommendation_outputs_current(curve_csv: Path, recommended_csv: Path) -> bool:
    if not curve_csv.exists() or not recommended_csv.exists():
        return False
    try:
        curve_df = pd.read_csv(curve_csv)
        recommended_df = pd.read_csv(recommended_csv)
    except Exception:
        return False
    for frame in (curve_df, recommended_df):
        if "calibration_probe_routing" not in frame.columns:
            return False
        values = frame["calibration_probe_routing"].fillna("").astype(str)
        if not values.eq(OFFLINE_CALIBRATION_PROBE_ROUTING).all():
            return False
    return True


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path(args.final_dir).expanduser().resolve()
    summary_csv = final_dir / RESULT_CSV.name
    summary_md = final_dir / RESULT_MD.name
    final_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    offline_curve_rows: list[dict[str, Any]] = []
    offline_recommended_rows: list[dict[str, Any]] = []
    offline_curve_summary_csv = final_dir / "offline_predicted_recall_curve.csv"
    offline_recommended_summary_csv = final_dir / "offline_recommended_efsearch.csv"
    if summary_csv.exists():
        rows.extend(pd.read_csv(summary_csv).to_dict("records"))
    if offline_curve_summary_csv.exists():
        offline_curve_rows.extend(pd.read_csv(offline_curve_summary_csv).to_dict("records"))
    if offline_recommended_summary_csv.exists():
        offline_recommended_rows.extend(pd.read_csv(offline_recommended_summary_csv).to_dict("records"))

    status_tsv = run_root / "status.tsv"
    if not status_tsv.exists():
        append_status(
            status_tsv,
            ["timestamp", "dataset", "k", "status", "result_csv", "log_path"],
        )
    main_log = run_root / "main_qps_latency_sweep.log"
    with main_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Main sweep started_at={now_label()}\n")
        handle.write(f"# run_root={run_root}\n")
        handle.write(f"# datasets={args.datasets}\n")
        handle.write(f"# k_values={args.k_values}\n")
        handle.write(f"# ef_sweep={args.ef_sweep}\n")
        handle.write(
            f"# offline_num_threads={int(args.offline_num_threads)} "
            f"online_num_threads={int(args.online_num_threads)} "
            f"measured_runs={int(args.measured_runs)}\n\n"
        )

    completed = {
        (str(row.get("dataset_file")), int(row.get("k")), str(row.get("method")), int(row.get("ef")))
        for row in rows
        if "dataset_file" in row and "k" in row and "method" in row and "ef" in row
    }

    for dataset in args.datasets:
        stem = dataset_stem(dataset)
        dataset_dir = run_root / stem
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_log = log_dir / f"{stem}__main_qps_latency.log"
        append_status(
            status_tsv,
            [now_label(), dataset, "*", "dataset_start", "", dataset_log],
        )

        with dataset_log.open("a", encoding="utf-8", newline="\n") as log:
            def emit(message: str) -> None:
                print(message)
                log.write(message + "\n")
                log.flush()

            dataset_load_start = time.perf_counter()
            emit(f"[1/5] Loading dataset: {dataset}")
            train, test, neighbors = load_dataset_with_special_cases(
                str(Path(args.base_path).expanduser().resolve()),
                dataset,
            )
            train = train.astype("float32")
            test = test.astype("float32")
            gt_k = int(neighbors.shape[1])
            dataset_load_wall_s = time.perf_counter() - dataset_load_start
            emit(
                f"[1/5] shapes train={train.shape} test={test.shape} "
                f"neighbors={neighbors.shape} load_wall_s={dataset_load_wall_s:.3f}"
            )

            index_load_start = time.perf_counter()
            emit(
                "[2/5] Loading original index "
                f"M={int(args.param_m)} efC={int(args.ef_construction)} "
                f"offline_threads={int(args.offline_num_threads)}"
            )
            index, _space, _index_dataset_name = build_original_index(
                train=train,
                dataset_name=dataset,
                index_dir=str(Path(args.index_dir).expanduser().resolve()),
                param_m=int(args.param_m),
                ef_construction=int(args.ef_construction),
                num_threads=int(args.offline_num_threads),
            )
            index_load_wall_s = time.perf_counter() - index_load_start
            emit(f"[2/5] Index ready index_load_wall_s={index_load_wall_s:.3f}")

            lid_pool_start = time.perf_counter()
            emit("[2/5] Building sampled calibration LID pool")
            calibration_lid_df = compute_fixed_calibration_lid_pool(
                index,
                internal_lid_k=int(args.internal_lid_k),
                num_nodes=len(train),
                lid_sample_seed=int(args.calibration_sample_seed),
                num_threads=int(args.offline_num_threads),
                dataset_name=dataset,
            )
            calibration_lid_pool_wall_s = time.perf_counter() - lid_pool_start
            emit(
                f"[2/5] Calibration LID pool count={len(calibration_lid_df)} "
                f"wall_s={calibration_lid_pool_wall_s:.3f}"
            )

            for k in args.k_values:
                result_csv = result_csv_for(run_root, dataset, int(k))
                if int(k) > gt_k:
                    message = f"groundtruth_k={gt_k} < requested_k={int(k)}"
                    if not args.skip_unavailable_k:
                        raise ValueError(f"{dataset}: {message}")
                    emit(f"[SKIP] {dataset} k={int(k)} {message}")
                    skip_row = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "reason": message,
                    }
                    skips.append(skip_row)
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_unavailable_k", result_csv, dataset_log],
                    )
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    continue

                k_ef_values = [int(ef) for ef in args.ef_sweep if int(ef) >= int(k)]
                skipped_low_efs = [int(ef) for ef in args.ef_sweep if int(ef) < int(k)]
                if skipped_low_efs:
                    emit(
                        f"[SKIP-EF] {dataset} k={int(k)} skipping efSearch<k: "
                        f"{skipped_low_efs}"
                    )
                if not k_ef_values:
                    message = f"no efSearch values >= requested_k={int(k)}"
                    emit(f"[SKIP] {dataset} k={int(k)} {message}")
                    skip_row = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "reason": message,
                    }
                    skips.append(skip_row)
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_no_valid_ef", result_csv, dataset_log],
                    )
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    continue

                needed = {
                    (dataset, int(k), "Vanilla", int(ef)) for ef in k_ef_values
                } | {
                    (dataset, int(k), "Ours", int(ef)) for ef in k_ef_values
                }
                skip_main_sweep = bool(
                    args.skip_existing and result_csv.exists() and needed.issubset(completed)
                )
                offline_curve_csv = offline_curve_csv_for(run_root, dataset, int(k))
                offline_recommended_csv = offline_recommended_csv_for(run_root, dataset, int(k))
                offline_recommendation_complete = offline_recommendation_outputs_current(
                    offline_curve_csv,
                    offline_recommended_csv,
                )
                drilldown_csv = drilldown_sweep_csv_for(args, final_dir, dataset, int(k))
                drilldown_complete = bool(args.enable_drilldown and drilldown_csv.exists())
                if (
                    skip_main_sweep
                    and offline_recommendation_complete
                    and (not args.enable_drilldown or drilldown_complete)
                ):
                    emit(f"[SKIP] {dataset} k={int(k)} existing result={result_csv}")
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_existing", result_csv, dataset_log],
                    )
                    continue
                if skip_main_sweep:
                    pending_parts = []
                    if not offline_recommendation_complete:
                        pending_parts.append(f"offline_recommended={offline_recommended_csv}")
                    if args.enable_drilldown and not drilldown_complete:
                        pending_parts.append(f"drilldown={drilldown_csv}")
                    emit(
                        f"[SKIP-MAIN] {dataset} k={int(k)} existing main result; "
                        f"continuing for {'; '.join(pending_parts)}"
                    )

                emit(f"[3/5] Calibrating ours thresholds for k={int(k)}")
                cache_path = (
                    dataset_dir
                    / f"{stem}__k{int(k)}__{args.mixed_threshold_mode}_b{int(args.mixed_bucket_count)}"
                    / f"{stem}__k{int(k)}__mixed_original_M{int(args.param_m)}_efC{int(args.ef_construction)}.json"
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_exists_before = cache_path.exists()
                threshold_start = time.perf_counter()
                tau_by_ef, policy, calibration_cache_status = resolve_mixed_policy_with_status(
                    index=index,
                    train=train,
                    dataset_name=dataset,
                    ef_values=k_ef_values,
                    k=int(k),
                    acceptable_recall_threshold=1.0,
                    tmin_pops=int(args.tmin_pops),
                    num_calibration_queries=int(args.num_calibration_queries),
                    selection_mode="quantile",
                    trim_low_percentile=float(args.trim_low_percentile),
                    trim_high_percentile=float(args.trim_high_percentile),
                    internal_lid_k=int(args.internal_lid_k),
                    num_threads=int(args.offline_num_threads),
                    easy_threshold_scale=float(args.easy_threshold_scale),
                    mid_threshold_scale=float(args.mid_threshold_scale),
                    super_threshold_scale=float(args.super_threshold_scale),
                    lid_df=calibration_lid_df,
                    lid_sample_seed=int(args.calibration_sample_seed),
                    cache_path=cache_path,
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    mixed_bucket_count=int(args.mixed_bucket_count),
                )
                threshold_calibration_wall_s = time.perf_counter() - threshold_start
                offline_calibration_wall_s = (
                    float(calibration_lid_pool_wall_s) + float(threshold_calibration_wall_s)
                )
                emit(
                    f"[3/5] Calibration done cache={calibration_cache_status or 'disabled'} "
                    f"threshold_wall_s={threshold_calibration_wall_s:.3f} "
                    f"offline_calibration_wall_s={offline_calibration_wall_s:.3f}"
                )

                offline_curve_df, offline_recommended_df = compute_offline_recommended_efsearch(
                    index=index,
                    train=train,
                    dataset=dataset,
                    stem=stem,
                    k=int(k),
                    gt_k=int(gt_k),
                    ef_values=k_ef_values,
                    tau_by_ef=tau_by_ef,
                    policy=policy,
                    calibration_lid_df=calibration_lid_df,
                    num_calibration_queries=int(args.num_calibration_queries),
                    trim_low_percentile=float(args.trim_low_percentile),
                    trim_high_percentile=float(args.trim_high_percentile),
                    num_threads=int(args.offline_num_threads),
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    mixed_bucket_count=int(args.mixed_bucket_count),
                    cache_path=cache_path,
                )
                offline_curve_df.to_csv(offline_curve_csv, index=False)
                offline_recommended_df.to_csv(offline_recommended_csv, index=False)
                offline_curve_rows = [
                    old
                    for old in offline_curve_rows
                    if not (str(old.get("dataset_file")) == dataset and int(old.get("k")) == int(k))
                ]
                offline_curve_rows.extend(offline_curve_df.to_dict("records"))
                offline_recommended_rows = [
                    old
                    for old in offline_recommended_rows
                    if not (str(old.get("dataset_file")) == dataset and int(old.get("k")) == int(k))
                ]
                offline_recommended_rows.extend(offline_recommended_df.to_dict("records"))
                write_offline_recommendation_summary(
                    final_dir=final_dir,
                    curve_rows=offline_curve_rows,
                    recommended_rows=offline_recommended_rows,
                )
                offline_rec = offline_recommended_df.iloc[0]
                emit(
                    f"[3/5] Offline recommended efSearch={int(offline_rec['recommended_ef'])} "
                    f"pred_recall={float(offline_rec['offline_predicted_recall']):.5f} "
                    f"remaining_gain={float(offline_rec['remaining_cumulative_recall_gain']):.6f} "
                    f"curve={offline_curve_csv}"
                )
                append_status(
                    status_tsv,
                    [now_label(), dataset, int(k), "offline_recommended", offline_recommended_csv, dataset_log],
                )

                first_ef = int(k_ef_values[0])
                first_bucket_gammas = tuple(
                    float(value)
                    for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(first_ef, ())
                )
                first_paper_bucket_count = (
                    int(args.mixed_bucket_count)
                    if str(args.mixed_threshold_mode) == "paper_floor_half"
                    else 0
                )
                validate_query_method(
                    index,
                    query_method=str(args.query_method),
                    sample_query=test[:1],
                    enable_stop=True,
                    early_stop_ratio=float(tau_by_ef[first_ef]),
                    super_easy_gamma_ratio=float(policy.gamma_ratio_by_ef.get(first_ef, float("nan"))),
                    mid_easy_upper_gamma_ratio=float(
                        policy.mid_easy_upper_gamma_ratio_by_ef.get(first_ef, float("nan"))
                    ),
                    tmin_pops=int(args.tmin_pops),
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    paper_bucket_count=first_paper_bucket_count,
                    paper_bucket_gamma_ratios=first_bucket_gammas,
                    num_threads=int(args.online_num_threads),
                )

                main_ef_values = [] if skip_main_sweep else k_ef_values
                if skip_main_sweep:
                    emit(
                        f"[4/5] Main efSearch sweep already complete for k={int(k)}; "
                        "skipping main timing rows"
                    )
                else:
                    emit(
                        f"[4/5] Running efSearch sweep for k={int(k)} "
                        f"online_threads={int(args.online_num_threads)}"
                    )
                k_rows: list[dict[str, Any]] = []
                for ef in main_ef_values:
                    ef = int(ef)
                    vanilla_metrics = benchmark_baseline_k(
                        index=index,
                        test=test,
                        neighbors=neighbors,
                        ef=ef,
                        k=int(k),
                        warmup_runs=int(args.warmup_runs),
                        measured_runs=int(args.measured_runs),
                        num_threads=int(args.online_num_threads),
                    )
                    route_efs = tuple(
                        int(value)
                        for value in getattr(policy, "route_efs_by_ef", {}).get(ef, ())
                    )
                    bucket_gammas = tuple(
                        float(value)
                        for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(ef, ())
                    )
                    paper_bucket_count = (
                        int(args.mixed_bucket_count)
                        if str(args.mixed_threshold_mode) == "paper_floor_half"
                        else 0
                    )
                    ours_metrics = benchmark_ours_k(
                        index=index,
                        test=test,
                        neighbors=neighbors,
                        ef=ef,
                        k=int(k),
                        query_method=str(args.query_method),
                        tau=float(tau_by_ef[ef]),
                        super_gamma=float(policy.gamma_ratio_by_ef.get(ef, float("nan"))),
                        mid_gamma=float(policy.mid_easy_upper_gamma_ratio_by_ef.get(ef, float("nan"))),
                        tmin_pops=int(args.tmin_pops),
                        mixed_threshold_mode=str(args.mixed_threshold_mode),
                        paper_bucket_count=paper_bucket_count,
                        paper_bucket_gamma_ratios=bucket_gammas,
                        warmup_runs=int(args.warmup_runs),
                        measured_runs=int(args.measured_runs),
                        num_threads=int(args.online_num_threads),
                    )
                    recall_loss = float(vanilla_metrics["recall"] - ours_metrics["recall"])
                    qps_gain_pct = (
                        float((ours_metrics["qps"] / vanilla_metrics["qps"] - 1.0) * 100.0)
                        if float(vanilla_metrics["qps"]) > 0.0
                        else float("nan")
                    )
                    base_common = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "groundtruth_k": gt_k,
                        "ef": ef,
                        "offline_num_threads": int(args.offline_num_threads),
                        "online_num_threads": int(args.online_num_threads),
                        "num_threads": int(args.online_num_threads),
                        "warmup_runs": int(args.warmup_runs),
                        "measured_runs": int(args.measured_runs),
                        "M": int(args.param_m),
                        "efConstruction": int(args.ef_construction),
                        "query_method": str(args.query_method),
                        "num_calibration_queries": int(args.num_calibration_queries),
                        "calibration_lid_pool_count": int(len(calibration_lid_df)),
                        "calibration_lid_pool_wall_s": float(calibration_lid_pool_wall_s),
                        "threshold_calibration_wall_s": float(threshold_calibration_wall_s),
                        "offline_calibration_wall_s": float(offline_calibration_wall_s),
                        "calibration_cache_status": str(calibration_cache_status or ""),
                        "calibration_cache_exists_before": bool(cache_exists_before),
                        "dataset_load_wall_s": float(dataset_load_wall_s),
                        "index_load_wall_s": float(index_load_wall_s),
                        "mixed_threshold_mode": str(args.mixed_threshold_mode),
                        "mixed_bucket_count": (
                            int(args.mixed_bucket_count)
                            if str(args.mixed_threshold_mode) == "paper_floor_half"
                            else np.nan
                        ),
                        "route_signature": (
                            format_int_signature(route_efs + (ef,)) if route_efs else ""
                        ),
                        "bucket_gamma_signature": (
                            format_float_signature(bucket_gammas) if bucket_gammas else ""
                        ),
                        "early_stop_ratio": float(tau_by_ef[ef]),
                        "stop_config_source": str(policy.source_label),
                        "cache_path": str(cache_path),
                    }
                    vanilla_row = {
                        **base_common,
                        "method": "Vanilla",
                        "enable_stop": False,
                        "recall": float(vanilla_metrics["recall"]),
                        "qps": float(vanilla_metrics["qps"]),
                        "recall_loss_vs_vanilla_pp": 0.0,
                        "qps_gain_vs_vanilla_pct": 0.0,
                        "adaptive_max_dist_mean": np.nan,
                        "stop_count": np.nan,
                        "reduced_steps_mean": np.nan,
                        "reduced_steps_max": np.nan,
                    }
                    vanilla_row.update(
                        {
                            key: vanilla_metrics[key]
                            for key in [
                                "query_count",
                                "batch_latency_mean_ms",
                                "batch_latency_p50_ms",
                                "batch_latency_p95_ms",
                                "batch_latency_min_ms",
                                "batch_latency_max_ms",
                                "latency_per_query_mean_ms",
                            ]
                        }
                    )
                    ours_row = {
                        **base_common,
                        "method": "Ours",
                        "enable_stop": True,
                        "recall": float(ours_metrics["recall"]),
                        "qps": float(ours_metrics["qps"]),
                        "recall_loss_vs_vanilla_pp": recall_loss * 100.0,
                        "qps_gain_vs_vanilla_pct": qps_gain_pct,
                        "adaptive_max_dist_mean": float(ours_metrics["adaptive_max_dist_mean"]),
                        "stop_count": ours_metrics["stop_count"],
                        "reduced_steps_mean": ours_metrics["reduced_steps_mean"],
                        "reduced_steps_max": ours_metrics["reduced_steps_max"],
                    }
                    ours_row.update(
                        {
                            key: ours_metrics[key]
                            for key in [
                                "query_count",
                                "batch_latency_mean_ms",
                                "batch_latency_p50_ms",
                                "batch_latency_p95_ms",
                                "batch_latency_min_ms",
                                "batch_latency_max_ms",
                                "latency_per_query_mean_ms",
                            ]
                        }
                    )
                    k_rows.extend([vanilla_row, ours_row])
                    rows = [
                        old
                        for old in rows
                        if not (
                            str(old.get("dataset_file")) == dataset
                            and int(old.get("k")) == int(k)
                            and int(old.get("ef")) == ef
                        )
                    ]
                    rows.extend([vanilla_row, ours_row])
                    completed.add((dataset, int(k), "Vanilla", ef))
                    completed.add((dataset, int(k), "Ours", ef))
                    emit(
                        f"[k={int(k)} ef={ef}] "
                        f"vanilla={vanilla_metrics['recall']:.5f}/{vanilla_metrics['qps']:.1f}qps "
                        f"ours={ours_metrics['recall']:.5f}/{ours_metrics['qps']:.1f}qps "
                        f"loss={recall_loss * 100.0:+.3f}pp gain={qps_gain_pct:+.1f}% "
                        f"lat_mean_ms(v/o)={vanilla_metrics['batch_latency_mean_ms']:.1f}/"
                        f"{ours_metrics['batch_latency_mean_ms']:.1f}"
                    )

                if k_rows:
                    pd.DataFrame(k_rows).to_csv(result_csv, index=False)
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    emit(f"[5/5] Wrote {result_csv}")
                else:
                    emit(f"[5/5] Main sweep already complete; kept {result_csv}")

                if args.enable_drilldown:
                    run_drilldown_for_k(
                        args=args,
                        final_dir=final_dir,
                        index=index,
                        test=test,
                        dataset=dataset,
                        k=int(k),
                        gt_k=int(gt_k),
                        k_ef_values=k_ef_values,
                        tau_by_ef=tau_by_ef,
                        policy=policy,
                        calibration_lid_df=calibration_lid_df,
                        calibration_lid_pool_wall_s=float(calibration_lid_pool_wall_s),
                        threshold_calibration_wall_s=float(threshold_calibration_wall_s),
                        offline_calibration_wall_s=float(offline_calibration_wall_s),
                        dataset_load_wall_s=float(dataset_load_wall_s),
                        index_load_wall_s=float(index_load_wall_s),
                        cache_path=cache_path,
                        emit=emit,
                    )
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "drilldown_done", drilldown_csv, dataset_log],
                    )

                append_status(
                    status_tsv,
                    [now_label(), dataset, int(k), "done", result_csv, dataset_log],
                )

        append_status(
            status_tsv,
            [now_label(), dataset, "*", "dataset_done", "", dataset_log],
        )

    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
    with main_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[COMPLETE] {now_label()} main sweep finished.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
