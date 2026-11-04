"""Shared offline calibration helpers for SAGE experiment scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adaptive_runtime import (
    DEFAULT_LID_SEARCH_EF,
    compute_calibration_lid_df,
)


FIXED_CALIBRATION_LID_POOL_SIZE = 10000
FIXED_CALIBRATION_PROTOCOL = "fixed10000-lid-quantile-trim1-99-v1"
DEFAULT_CALIBRATION_PROBE_COUNT = 500
# Positive sentinel required by current backend bindings; min_sample_size enforces 10,000.
FIXED_CALIBRATION_SAMPLE_FRACTION = 1e-9


def validate_fixed_calibration_lid_pool(lid_df: pd.DataFrame) -> None:
    """Reject truncated, proportional or malformed pools before selecting probes."""
    if len(lid_df) != FIXED_CALIBRATION_LID_POOL_SIZE:
        raise ValueError(
            f"Expected fixed calibration LID pool size {FIXED_CALIBRATION_LID_POOL_SIZE}, "
            f"got {len(lid_df)}. Regenerate the pool with compute_fixed_calibration_lid_pool()."
        )
    if not {"query_id", "lid"} <= set(lid_df.columns):
        raise ValueError("Calibration pool requires query_id and lid columns.")
    ids = lid_df["query_id"].to_numpy()
    if (not np.isfinite(ids).all() or (ids < 0).any()
            or not np.equal(ids, np.floor(ids)).all()
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Calibration pool requires distinct nonnegative integer labels.")
    if not np.isfinite(lid_df["lid"].to_numpy()).all():
        raise ValueError("Calibration pool requires 10,000 finite LIDs; regenerate the pool.")


def calibration_sampling_metadata(lid_pool, *, probe_count=DEFAULT_CALIBRATION_PROBE_COUNT, source="supplied_pool",
                                  internal_lid_k=None, seed=None):
    """Record actual counts; external probes never imply a verified 10,000-node pool."""
    if lid_pool is None:
        return dict(protocol="external-frozen-probes-v1", source="supplied_probe_ids",
                    pool_size=None, valid_lid_count=None, probe_count=int(probe_count))
    validate_fixed_calibration_lid_pool(lid_pool)
    valid = int((lid_pool["lid"] > 0).sum())
    if valid < probe_count:
        raise ValueError(f"Only {valid} positive LIDs for {probe_count} calibration probes.")
    settings = {}
    for name in ("internal_lid_k", "lid_search_ef", "lid_exclude_self", "lid_sample_seed"):
        value = None
        if name in lid_pool:
            values = lid_pool[name].drop_duplicates()
            if len(values) != 1 or values.isna().any():
                raise ValueError(f"Calibration pool has inconsistent {name} metadata.")
            value = bool(values.iloc[0]) if name == "lid_exclude_self" else int(values.iloc[0])
        settings[name] = value
    if settings['internal_lid_k'] is not None:
        internal_lid_k = settings['internal_lid_k']
    if settings['lid_sample_seed'] is not None:
        seed = settings['lid_sample_seed']
    return dict(protocol=FIXED_CALIBRATION_PROTOCOL, source=source,
                pool_size=len(lid_pool), valid_lid_count=valid, probe_count=int(probe_count),
                internal_lid_k=internal_lid_k, seed=seed,
                lid_search_ef=settings['lid_search_ef'],
                lid_exclude_self=settings['lid_exclude_self'],
                selection="quantile", trim_low_percentile=1., trim_high_percentile=99.)


def compute_fixed_calibration_lid_pool(
    index,
    *,
    internal_lid_k: int,
    num_nodes: int,
    lid_sample_seed: int,
    num_threads: int,
    dataset_name: str = "",
    lid_search_ef: int = DEFAULT_LID_SEARCH_EF,
) -> pd.DataFrame:
    if int(num_nodes) < FIXED_CALIBRATION_LID_POOL_SIZE:
        raise ValueError(
            f"Fixed calibration requires at least {FIXED_CALIBRATION_LID_POOL_SIZE} nodes, "
            f"got {num_nodes}. For a small diagnostic graph, supply explicit frozen probe_ids."
        )
    lid_df = compute_calibration_lid_df(
        index,
        internal_lid_k=int(internal_lid_k),
        lid_sample_fraction=FIXED_CALIBRATION_SAMPLE_FRACTION,
        lid_min_sample_size=FIXED_CALIBRATION_LID_POOL_SIZE,
        lid_sample_seed=int(lid_sample_seed),
        num_threads=int(num_threads),
        lid_search_ef=lid_search_ef,
    )
    if len(lid_df) != FIXED_CALIBRATION_LID_POOL_SIZE:
        suffix = f" for {dataset_name}" if dataset_name else ""
        raise RuntimeError(
            "Expected fixed calibration LID pool size "
            f"{FIXED_CALIBRATION_LID_POOL_SIZE}, got {len(lid_df)}{suffix}."
        )
    validate_fixed_calibration_lid_pool(lid_df)
    return lid_df
