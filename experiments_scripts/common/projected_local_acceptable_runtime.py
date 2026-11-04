"""Probe selection and bucket routes shared by SAGE policy calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lid_selection import select_lid_representatives, trim_lid_outliers


@dataclass(frozen=True)
class MixedThresholdConfig:
    resolved_route_efs: tuple[int, ...]


def _coerce_lid_df(index, *, num_nodes: int, lid_df: pd.DataFrame | None) -> pd.DataFrame:
    if lid_df is not None:
        scoped = lid_df.copy()
        scoped["query_id"] = pd.to_numeric(scoped["query_id"], errors="raise").astype(np.int64)
        scoped["lid"] = pd.to_numeric(scoped["lid"], errors="raise").astype(np.float32)
        if "query_source" not in scoped.columns:
            scoped["query_source"] = "train"
        else:
            scoped["query_source"] = scoped["query_source"].fillna("train").astype(str)
        return scoped[["query_id", "query_source", "lid"]].sort_values("query_id").reset_index(drop=True)

    lids = np.asarray(index.get_lids(), dtype=np.float32)
    if lids.shape[0] != int(num_nodes):
        raise ValueError(f"Expected {int(num_nodes)} internal LIDs, but got {lids.shape[0]}.")
    return pd.DataFrame(
        {
            "query_id": np.arange(int(num_nodes), dtype=np.int64),
            "query_source": "train",
            "lid": lids,
        }
    )


def _select_dummy_queries(
    *,
    index,
    num_nodes: int,
    lid_df: pd.DataFrame | None,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
) -> pd.DataFrame:
    scoped_lid_df = _coerce_lid_df(index, num_nodes=num_nodes, lid_df=lid_df)
    trimmed_df, trimmed_lid_min, trimmed_lid_max = trim_lid_outliers(
        lid_df=scoped_lid_df,
        low_percentile=float(trim_low_percentile),
        high_percentile=float(trim_high_percentile),
    )
    selected_df = select_lid_representatives(
        trimmed_df=trimmed_df,
        num_samples=int(num_calibration_queries),
        selection_mode=str(selection_mode),
        lid_min=float(trimmed_lid_min),
        lid_max=float(trimmed_lid_max),
    ).copy()
    selected_df["query_id"] = pd.to_numeric(selected_df["query_id"], errors="raise").astype(np.int64)
    selected_df["selection_rank"] = pd.to_numeric(
        selected_df["selection_rank"],
        errors="raise",
    ).astype(np.int64)
    selected_df["lid"] = pd.to_numeric(selected_df["lid"], errors="raise").astype(np.float32)
    return selected_df.sort_values("selection_rank").reset_index(drop=True)



def _build_paper_floor_route_efs(
    *,
    selection_ef: int,
    mixed_bucket_count: int,
) -> tuple[int, ...]:
    selection_ef = int(selection_ef)
    mixed_bucket_count = int(mixed_bucket_count)
    if mixed_bucket_count < 2:
        raise ValueError(f"mixed_bucket_count must be >= 2, got {mixed_bucket_count}")
    route_efs: list[int] = []
    for bucket_idx in range(1, mixed_bucket_count):
        routed_ef = max(1, (selection_ef * int(bucket_idx)) // mixed_bucket_count)
        routed_ef = min(routed_ef, selection_ef - 1)
        if not route_efs or routed_ef != route_efs[-1]:
            route_efs.append(int(routed_ef))
    if not route_efs:
        raise ValueError(
            f"Could not build any routed efs for selection_ef={selection_ef}, mixed_bucket_count={mixed_bucket_count}"
        )
    return tuple(route_efs)


def _quantile_theta(cfr_values: np.ndarray, mass: float) -> float:
    return float(np.quantile(cfr_values, float(np.clip(mass, 0.0, 1.0))))



def _build_mixed_threshold_config(
    selection_ef: int,
    *,
    mixed_bucket_count: int = 4,
) -> MixedThresholdConfig:
    selection_ef = int(selection_ef)
    if selection_ef < 2:
        raise ValueError(f"selection_ef must be >= 2, got {selection_ef}")
    return MixedThresholdConfig(_build_paper_floor_route_efs(
        selection_ef=selection_ef, mixed_bucket_count=int(mixed_bucket_count)))
