"""Shared offline calibration helpers for SAGE experiment scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .adaptive_runtime import (
    DEFAULT_MIXED_GT_EF,
    DEFAULT_MIXED_GT_SOURCE,
    compute_calibration_lid_df,
)
from .projected_local_acceptable_runtime import (
    load_or_build_dynamic_projected_local_acceptable_mixed_policy,
)


FIXED_CALIBRATION_LID_POOL_SIZE = 10000
# Positive sentinel required by current backend bindings; min_sample_size enforces 10,000.
FIXED_CALIBRATION_SAMPLE_FRACTION = 1e-9


def compute_fixed_calibration_lid_pool(
    index,
    *,
    internal_lid_k: int,
    num_nodes: int,
    lid_sample_seed: int,
    num_threads: int,
    dataset_name: str = "",
) -> pd.DataFrame:
    lid_df = compute_calibration_lid_df(
        index,
        internal_lid_k=int(internal_lid_k),
        num_nodes=int(num_nodes),
        lid_sampling_mode="sampled",
        lid_sample_fraction=FIXED_CALIBRATION_SAMPLE_FRACTION,
        lid_min_sample_size=FIXED_CALIBRATION_LID_POOL_SIZE,
        lid_sample_seed=int(lid_sample_seed),
        num_threads=int(num_threads),
    )
    if len(lid_df) != FIXED_CALIBRATION_LID_POOL_SIZE:
        suffix = f" for {dataset_name}" if dataset_name else ""
        raise RuntimeError(
            "Expected fixed calibration LID pool size "
            f"{FIXED_CALIBRATION_LID_POOL_SIZE}, got {len(lid_df)}{suffix}."
        )
    return lid_df


def resolve_mixed_policy_with_status(
    *,
    index,
    train,
    dataset_name: str,
    ef_values: list[int],
    k: int,
    acceptable_recall_threshold: float,
    tmin_pops: int,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
    internal_lid_k: int,
    num_threads: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
    lid_df: pd.DataFrame,
    lid_sample_seed: int,
    cache_path: Path,
    mixed_threshold_mode: str,
    mixed_bucket_count: int,
    ablation_name: str = "baseline",
    ablation_value: str = "default",
    classify_start: int = 4,
    classify_end: int = 16,
    cfr_ema_decay: float = 0.8,
    use_pre_frontier_cfr: bool = False,
    pair_gap: int = 2,
):
    return load_or_build_dynamic_projected_local_acceptable_mixed_policy(
        index=index,
        train=train,
        dataset_name=dataset_name,
        ef_values=ef_values,
        acceptable_recall_threshold=float(acceptable_recall_threshold),
        requested_tmin_pops=int(tmin_pops),
        num_calibration_queries=int(num_calibration_queries),
        selection_mode=selection_mode,
        trim_low_percentile=float(trim_low_percentile),
        trim_high_percentile=float(trim_high_percentile),
        gt_source=DEFAULT_MIXED_GT_SOURCE,
        gt_ef=int(DEFAULT_MIXED_GT_EF),
        internal_lid_k=int(internal_lid_k),
        num_threads=int(num_threads),
        k=int(k),
        easy_threshold_scale=float(easy_threshold_scale),
        mid_threshold_scale=float(mid_threshold_scale),
        super_threshold_scale=float(super_threshold_scale),
        lid_df=lid_df,
        lid_source_graph="original",
        lid_sampling_mode="sampled",
        lid_sample_fraction=FIXED_CALIBRATION_SAMPLE_FRACTION,
        lid_min_sample_size=FIXED_CALIBRATION_LID_POOL_SIZE,
        lid_sample_seed=int(lid_sample_seed),
        cache_path=cache_path,
        mixed_threshold_mode=str(mixed_threshold_mode),
        mixed_bucket_count=int(mixed_bucket_count),
        classify_start=int(classify_start),
        classify_end=int(classify_end),
        cfr_ema_decay=float(cfr_ema_decay),
        use_pre_frontier_cfr=bool(use_pre_frontier_cfr),
        paper_floor_pair_gap=int(pair_gap),
        cache_context={
            "calibration_graph_variant": "original",
            "calibration_lid_source_graph": "original",
            "calibration_probe_routing": "hide_node",
            "benchmark_family": "sage_main_qps_latency",
            "offline_num_threads": int(num_threads),
            "ablation_name": str(ablation_name),
            "ablation_value": str(ablation_value),
            "classify_start": int(classify_start),
            "classify_end": int(classify_end),
            "cfr_ema_decay": float(cfr_ema_decay),
            "use_pre_frontier_cfr": bool(use_pre_frontier_cfr),
            "cfr_observation_mode": "pre_frontier" if bool(use_pre_frontier_cfr) else "pre_expansion_full_pop",
            "pair_gap": int(pair_gap),
        },
    )
