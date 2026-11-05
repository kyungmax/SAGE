"""LID trimming and representative-query selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trim_lid_outliers(lid_df, low_percentile, high_percentile):
    low_value = float(np.percentile(lid_df["lid"].to_numpy(dtype=np.float64), low_percentile))
    high_value = float(np.percentile(lid_df["lid"].to_numpy(dtype=np.float64), high_percentile))
    trimmed = lid_df[(lid_df["lid"] >= low_value) & (lid_df["lid"] <= high_value)].copy()
    if trimmed.empty:
        raise ValueError("Outlier trimming removed all candidates.")
    return trimmed.sort_values(["lid", "query_id"]).reset_index(drop=True), low_value, high_value


def select_quantile_lid_representatives(trimmed_df, num_samples, lid_min, lid_max):
    selection_count = min(int(num_samples), len(trimmed_df))
    if selection_count == 0:
        raise ValueError("No trimmed LID candidates available.")
    ordered = trimmed_df.sort_values(["lid", "query_id"]).reset_index(drop=True)
    chunks = np.array_split(np.arange(len(ordered), dtype=np.int64), selection_count)
    selected_rows = []
    for rank, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        chosen = ordered.iloc[int(chunk[len(chunk) // 2])]
        chunk_lids = ordered.iloc[chunk]["lid"].to_numpy(dtype=np.float64)
        selected_rows.append({
            "selection_rank": rank,
            "selection_mode": "quantile",
            "query_id": int(chosen["query_id"]),
            "query_source": str(chosen["query_source"]),
            "lid": float(chosen["lid"]),
            "trimmed_lid_min": lid_min,
            "trimmed_lid_max": lid_max,
            "selection_bucket_id": rank,
            "selection_bucket_lower": float(np.min(chunk_lids)),
            "selection_bucket_upper": float(np.max(chunk_lids)),
            "selection_target_lid": float(np.median(chunk_lids)),
            "selection_used_fallback": False,
        })
    return pd.DataFrame(selected_rows).sort_values("selection_rank").reset_index(drop=True)
