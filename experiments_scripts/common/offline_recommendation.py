"""Offline efSearch recommendation for the shared SAGE pipeline.

Extracted verbatim from the HNSWLib driver so the Faiss pipeline shares the
exact same offline recommendation logic. This module is backend-agnostic: it
only relies on index methods exposed by both backends (``set_ef``,
``knn_query_hide_node`` and ``search_layer0_cfr_trace_hide_node_batch``), all of
which are routed through ``common.projected_local_acceptable_runtime`` helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.adaptive_runtime import (
    DEFAULT_MIXED_GT_EF,
    DEFAULT_MIXED_GT_SOURCE,
    format_float_signature,
    format_int_signature,
)
from common.projected_local_acceptable_runtime import (
    _compute_gt_neighbors,
    _compute_recall_by_ef,
    _extract_chr_mean_by_query,
    _select_dummy_queries,
)

OFFLINE_RECOMMENDED_CUMULATIVE_GAIN_EPS = 0.001
OFFLINE_CALIBRATION_PROBE_ROUTING = "hide_node"


def _dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def offline_curve_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    stem = _dataset_stem(dataset)
    return run_root / stem / f"{stem}__k{int(k)}__offline_predicted_recall_curve.csv"


def offline_recommended_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    stem = _dataset_stem(dataset)
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
    classify_start: int = 4,
    classify_end: int = 16,
    chr_ema_decay: float = 0.8,
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
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
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
                "classify_start": int(classify_start),
                "classify_end": int(classify_end),
                "chr_ema_decay": float(chr_ema_decay),
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
