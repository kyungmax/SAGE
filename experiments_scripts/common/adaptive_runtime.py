"""Internal graph-LID sampling for fixed-pool SAGE calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_CALIBRATION_SAMPLE_SEED = 42
DEFAULT_INTERNAL_LID_K = 15
DEFAULT_LID_SEARCH_EF = 512


def build_internal_lid_df_from_arrays(query_ids, lids) -> pd.DataFrame:
    query_ids = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    lids = np.asarray(lids, dtype=np.float32).reshape(-1)
    if query_ids.shape[0] != lids.shape[0]:
        raise ValueError(
            f"Expected matching query_ids/lids lengths, got {query_ids.shape[0]} and {lids.shape[0]}."
        )
    finite_mask = np.isfinite(lids)
    query_ids = query_ids[finite_mask]
    lids = lids[finite_mask]
    if query_ids.size == 0:
        raise ValueError("No finite internal LIDs available.")
    return pd.DataFrame(
        {
            "query_id": query_ids,
            "query_source": "train",
            "lid": lids,
        }
    ).sort_values("query_id").reset_index(drop=True)


def annotate_calibration_lid_df(
    lid_df: pd.DataFrame,
    *,
    lid_sample_fraction: float | None = None,
    lid_min_sample_size: int | None = None,
    lid_sample_seed: int | None = None,
    internal_lid_k: int,
    lid_search_ef: int,
) -> pd.DataFrame:
    annotated = lid_df.copy()
    annotated["internal_lid_k"] = int(internal_lid_k)
    annotated["lid_search_ef"] = max(int(lid_search_ef), int(internal_lid_k) + 1)
    annotated["lid_exclude_self"] = True
    annotated["lid_sampling_mode"] = "sampled"
    annotated["lid_sample_fraction"] = (
        float(lid_sample_fraction) if lid_sample_fraction is not None else np.nan
    )
    annotated["lid_min_sample_size"] = (
        int(lid_min_sample_size) if lid_min_sample_size is not None else np.nan
    )
    annotated["lid_sample_seed"] = (
        int(lid_sample_seed) if lid_sample_seed is not None else np.nan
    )
    return annotated


def compute_calibration_lid_df(
    index,
    *,
    internal_lid_k: int,
    lid_sample_fraction: float = 0.0,
    lid_min_sample_size: int = 10000,
    lid_sample_seed: int = DEFAULT_CALIBRATION_SAMPLE_SEED,
    num_threads: int = 24,
    lid_search_ef: int = DEFAULT_LID_SEARCH_EF,
) -> pd.DataFrame:
    if int(lid_search_ef) <= 0:
        raise ValueError("lid_search_ef must be positive.")
    lid_search_options = {"lid_search_ef": int(lid_search_ef)}

    calc_sampled = getattr(index, "sample_internal_lids", None)
    if calc_sampled is None:
        calc_sampled = getattr(index, "calc_lids_internal_sampled", None)
    if calc_sampled is None:
        raise RuntimeError(
            "Current index backend does not expose sample_internal_lids() or "
            "calc_lids_internal_sampled(). The selected backend must provide its own "
            "sampled internal-LID implementation before using sampled calibration."
        )
    sampled_query_ids, sampled_lids = calc_sampled(
        k_lid=int(internal_lid_k),
        sample_fraction=float(lid_sample_fraction),
        min_sample_size=int(lid_min_sample_size),
        random_seed=int(lid_sample_seed),
        num_threads=int(num_threads),
        **lid_search_options,
    )
    sampled_df = build_internal_lid_df_from_arrays(sampled_query_ids, sampled_lids)
    # The sampler returns external labels, which need not be dense row IDs.
    # Membership is checked by get_items when the selected probes are loaded.
    if (sampled_df["query_id"] < 0).any():
        raise ValueError("Sampled LID labels must be nonnegative.")
    return annotate_calibration_lid_df(
        sampled_df,
        lid_sample_fraction=float(lid_sample_fraction),
        lid_min_sample_size=int(lid_min_sample_size),
        lid_sample_seed=int(lid_sample_seed),
        internal_lid_k=internal_lid_k, lid_search_ef=lid_search_ef,
    )
