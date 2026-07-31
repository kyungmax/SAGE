#!/usr/bin/env python3
"""Shared adaptive-query and offline-calibration helpers for SAGE experiment scripts."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_utils import load_dataset
from .projected_local_acceptable_runtime import (
    DEFAULT_MIXED_THRESHOLD_MODE,
    resolve_runtime_bucket_routing,
)
NUM_THREADS = 24


DEFAULT_EF_SWEEP_VALUES = [64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024]
DEFAULT_THRESHOLD_SCALE = 1.0
DEFAULT_MIXED_GT_SOURCE = "hnsw"
DEFAULT_MIXED_GT_EF = 4096
DEFAULT_CALIBRATION_LID_MODE = "sampled"
DEFAULT_CALIBRATION_SAMPLE_SEED = 42


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


def build_internal_lid_df(index, num_nodes: int) -> pd.DataFrame:
    lids = np.asarray(index.get_lids(), dtype=np.float32)
    if lids.shape[0] != int(num_nodes):
        raise ValueError(f"Expected {int(num_nodes)} internal LIDs, but got {lids.shape[0]}.")
    return build_internal_lid_df_from_arrays(
        np.arange(int(num_nodes), dtype=np.int64),
        lids,
    )


def annotate_calibration_lid_df(
    lid_df: pd.DataFrame,
    *,
    lid_sampling_mode: str,
    lid_sample_fraction: float | None = None,
    lid_min_sample_size: int | None = None,
    lid_sample_seed: int | None = None,
) -> pd.DataFrame:
    annotated = lid_df.copy()
    annotated["lid_sampling_mode"] = str(lid_sampling_mode)
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
    num_nodes: int,
    lid_sampling_mode: str = DEFAULT_CALIBRATION_LID_MODE,
    lid_sample_fraction: float = 0.0,
    lid_min_sample_size: int = 10000,
    lid_sample_seed: int = DEFAULT_CALIBRATION_SAMPLE_SEED,
    num_threads: int = NUM_THREADS,
) -> pd.DataFrame:
    resolved_mode = str(lid_sampling_mode)
    if resolved_mode == "full":
        index.calc_lids_internal(
            k_lid=int(internal_lid_k),
            num_threads=int(num_threads),
        )
        return annotate_calibration_lid_df(
            build_internal_lid_df(index=index, num_nodes=int(num_nodes)),
            lid_sampling_mode="full",
        )
    if resolved_mode != "sampled":
        raise ValueError(f"Unsupported calibration lid mode: {resolved_mode!r}")

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
    )
    sampled_df = build_internal_lid_df_from_arrays(sampled_query_ids, sampled_lids)
    max_query_id = int(sampled_df["query_id"].max())
    if max_query_id >= int(num_nodes):
        raise ValueError(
            f"Sampled internal LID query_id out of range: max={max_query_id}, num_nodes={int(num_nodes)}"
        )
    return annotate_calibration_lid_df(
        sampled_df,
        lid_sampling_mode="sampled",
        lid_sample_fraction=float(lid_sample_fraction),
        lid_min_sample_size=int(lid_min_sample_size),
        lid_sample_seed=int(lid_sample_seed),
    )


def flush_cache() -> None:
    large_array = np.random.bytes(256 * 1024 * 1024)
    _ = np.frombuffer(large_array, dtype=np.uint8).sum()


def load_fbin(filename: str | Path, count: int = -1) -> np.ndarray:
    with open(filename, "rb") as handle:
        _, dim = struct.unpack("ii", handle.read(8))
        data = np.fromfile(handle, dtype=np.float32, count=count * dim if count > 0 else -1)
    return data.reshape(-1, dim)


def load_ibin(filename: str | Path) -> np.ndarray:
    with open(filename, "rb") as handle:
        header = np.fromfile(handle, dtype=np.uint32, count=2)
        if len(header) < 2:
            return np.array([], dtype=np.int32)
        count, k = int(header[0]), int(header[1])
        ids = np.fromfile(handle, dtype=np.uint32, count=count * k).reshape(count, k)
    return ids.astype(np.int32)


def load_dataset_with_special_cases(base_path: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_name = dataset_name.lower()
    if "text2image" in low_name or ("t2i" in low_name and "coco" not in low_name):
        dataset_root = Path(base_path)
        train = load_fbin(dataset_root / "t2i.10M.fbin")
        test = load_fbin(dataset_root / "t2i.query.public.100K.fbin")
        neighbors = load_ibin(dataset_root / "t2i_10M_top10_custom.ibin")
        return train, test, neighbors
    return load_dataset(base_path, file_name=dataset_name)


def evaluate_recall_per_query(labels, neighbors, k: int) -> np.ndarray:
    n_queries = len(labels)
    recalls = np.zeros(n_queries)
    for i in range(n_queries):
        intersect = np.intersect1d(labels[i][: int(k)], neighbors[i][: int(k)]).size
        recalls[i] = intersect / float(k)
    return recalls


def parse_ef_sweep(value):
    ef_values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not ef_values:
        raise argparse.ArgumentTypeError("ef sweep list cannot be empty.")
    if any(ef < 1 for ef in ef_values):
        raise argparse.ArgumentTypeError("all ef sweep values must be positive.")
    return ef_values


def dedupe_preserve_order(values):
    return list(dict.fromkeys(values))


def format_int_signature(values) -> str:
    resolved = tuple(int(value) for value in values)
    return "/".join(str(value) for value in resolved) if resolved else ""


def format_float_signature(values) -> str:
    resolved = tuple(float(value) for value in values)
    if not resolved:
        return ""
    parts: list[str] = []
    for value in resolved:
        parts.append(f"{value:.6f}" if np.isfinite(value) else "nan")
    return "/".join(parts)


def resolve_runtime_threshold_scales(
    *,
    easy_threshold_scale: float | None,
    mid_threshold_scale: float | None,
    super_threshold_scale: float | None,
) -> tuple[float, float, float]:
    resolved_easy = DEFAULT_THRESHOLD_SCALE if easy_threshold_scale is None else float(easy_threshold_scale)
    resolved_mid = DEFAULT_THRESHOLD_SCALE if mid_threshold_scale is None else float(mid_threshold_scale)
    resolved_super = DEFAULT_THRESHOLD_SCALE if super_threshold_scale is None else float(super_threshold_scale)
    return float(resolved_easy), float(resolved_mid), float(resolved_super)


def resolve_paper_bucket_runtime_config(
    *,
    policy,
    selection_ef: int,
    mixed_threshold_mode: str,
    require_present: bool = False,
) -> tuple[int, tuple[int, ...], tuple[float, ...], str, str]:
    if str(mixed_threshold_mode) != "paper_floor_half":
        return 0, tuple(), tuple(), "", ""
    route_efs, bucket_gamma_ratios = resolve_runtime_bucket_routing(
        policy=policy,
        selection_ef=int(selection_ef),
        mixed_threshold_mode=str(mixed_threshold_mode),
    )
    if not route_efs or len(route_efs) != len(bucket_gamma_ratios):
        if require_present:
            raise ValueError(
                f"Missing calibrated paper-bucket routing for selection_ef={int(selection_ef)}."
            )
        return 0, tuple(), tuple(), "", ""
    resolved_route_efs = tuple(int(value) for value in route_efs)
    resolved_bucket_gamma_ratios = tuple(float(value) for value in bucket_gamma_ratios)
    return (
        int(len(resolved_route_efs) + 1),
        resolved_route_efs,
        resolved_bucket_gamma_ratios,
        format_int_signature(resolved_route_efs + (int(selection_ef),)),
        format_float_signature(resolved_bucket_gamma_ratios),
    )


def run_adaptive_query(
    index,
    test,
    k_search,
    ef,
    query_method,
    enable_stop,
    early_stop_ratio,
    super_easy_gamma_ratio,
    mid_easy_upper_gamma_ratio,
    tmin_pops,
    mixed_threshold_mode=DEFAULT_MIXED_THRESHOLD_MODE,
    paper_bucket_count=0,
    paper_bucket_gamma_ratios=(),
    classify_start=4,
    classify_end=16,
    cfr_ema_decay=0.8,
    use_pre_frontier_cfr=False,
    num_threads=NUM_THREADS,
):
    paper_bucket_gamma_ratios = tuple(float(value) for value in paper_bucket_gamma_ratios)
    use_paper_bucket = str(mixed_threshold_mode) == "paper_floor_half"
    if use_paper_bucket:
        if int(paper_bucket_count) < 2:
            raise ValueError("paper_bucket_count must be at least 2 for paper_floor_half runtime routing.")
        if len(paper_bucket_gamma_ratios) != int(paper_bucket_count) - 1:
            raise ValueError("paper_bucket_gamma_ratios must contain exactly paper_bucket_count - 1 entries.")

    if query_method == "adaptive":
        if bool(use_pre_frontier_cfr):
            raise ValueError("use_pre_frontier_cfr is only supported for adaptive-light.")
        if use_paper_bucket:
            return index.knn_query_adaptive_analysis_paper_bucket(
                data=test,
                k=k_search,
                ef_init=ef,
                ef_max=ef,
                tmin_pops=tmin_pops,
                enable_stop=enable_stop,
                num_threads=num_threads,
                early_stop_ratio=early_stop_ratio,
                paper_bucket_count=int(paper_bucket_count),
                bucket_gamma_ratios=list(paper_bucket_gamma_ratios),
                classify_start=int(classify_start),
                classify_end=int(classify_end),
                cfr_ema_decay=float(cfr_ema_decay),
            )
        return index.knn_query_adaptive_analysis(
            data=test,
            k=k_search,
            ef_init=ef,
            ef_max=ef,
            tmin_pops=tmin_pops,
            early_stop_ratio=early_stop_ratio,
            super_easy_gamma_ratio=super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio=mid_easy_upper_gamma_ratio,
            enable_stop=enable_stop,
            num_threads=num_threads,
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            cfr_ema_decay=float(cfr_ema_decay),
        )

    if query_method == "adaptive-light":
        if use_paper_bucket:
            if bool(use_pre_frontier_cfr):
                query_sage = index.knn_query_adaptive_light_paper_bucket_pre_frontier
            else:
                query_sage = getattr(index, "knn_query_sage", None)
            if query_sage is None:
                query_sage = index.knn_query_adaptive_light_paper_bucket
            labels, dists = query_sage(
                test,
                k=k_search,
                ef_init=ef,
                enable_stop=enable_stop,
                early_stop_ratio=early_stop_ratio,
                tmin_pops=tmin_pops,
                paper_bucket_count=int(paper_bucket_count),
                bucket_gamma_ratios=list(paper_bucket_gamma_ratios),
                classify_start=int(classify_start),
                classify_end=int(classify_end),
                cfr_ema_decay=float(cfr_ema_decay),
                num_threads=num_threads,
            )
            return labels, dists
        adaptive_light = (
            index.knn_query_adaptive_light_pre_frontier
            if bool(use_pre_frontier_cfr)
            else index.knn_query_adaptive_light
        )
        labels, dists = adaptive_light(
            test,
            k=k_search,
            ef_init=ef,
            enable_stop=enable_stop,
            early_stop_ratio=early_stop_ratio,
            super_easy_gamma_ratio=super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio=mid_easy_upper_gamma_ratio,
            tmin_pops=tmin_pops,
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            cfr_ema_decay=float(cfr_ema_decay),
            num_threads=num_threads,
        )
        return labels, dists

    raise ValueError(f"Unsupported query method: {query_method}")


def validate_query_method(
    index,
    query_method,
    sample_query,
    enable_stop,
    early_stop_ratio,
    super_easy_gamma_ratio,
    mid_easy_upper_gamma_ratio,
    tmin_pops,
    mixed_threshold_mode=DEFAULT_MIXED_THRESHOLD_MODE,
    paper_bucket_count=0,
    paper_bucket_gamma_ratios=(),
    classify_start=4,
    classify_end=16,
    cfr_ema_decay=0.8,
    use_pre_frontier_cfr=False,
    num_threads=NUM_THREADS,
) -> None:
    try:
        run_adaptive_query(
            index=index,
            test=sample_query,
            k_search=1,
            ef=64,
            query_method=query_method,
            enable_stop=enable_stop,
            early_stop_ratio=early_stop_ratio,
            super_easy_gamma_ratio=super_easy_gamma_ratio,
            mid_easy_upper_gamma_ratio=mid_easy_upper_gamma_ratio,
            tmin_pops=tmin_pops,
            mixed_threshold_mode=mixed_threshold_mode,
            paper_bucket_count=paper_bucket_count,
            paper_bucket_gamma_ratios=paper_bucket_gamma_ratios,
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            cfr_ema_decay=float(cfr_ema_decay),
            use_pre_frontier_cfr=bool(use_pre_frontier_cfr),
            num_threads=num_threads,
        )
    except AttributeError as exc:
        available = ", ".join(name for name in dir(index) if "knn_query" in name)
        raise RuntimeError(
            f"Selected query method '{query_method}' is not supported by the current index backend. "
            f"Available query methods: {available}"
        ) from exc
    except TypeError as exc:
        use_paper_bucket = str(mixed_threshold_mode) == "paper_floor_half"
        if query_method == "adaptive":
            method_name = "knn_query_adaptive_analysis_paper_bucket" if use_paper_bucket else "knn_query_adaptive_analysis"
            fallback_method_name = method_name
        else:
            if use_paper_bucket:
                if bool(use_pre_frontier_cfr):
                    method_name = "knn_query_adaptive_light_paper_bucket_pre_frontier"
                    fallback_method_name = method_name
                else:
                    method_name = "knn_query_sage"
                    fallback_method_name = "knn_query_adaptive_light_paper_bucket"
            else:
                method_name = (
                    "knn_query_adaptive_light_pre_frontier"
                    if bool(use_pre_frontier_cfr)
                    else "knn_query_adaptive_light"
                )
                fallback_method_name = method_name
        method = getattr(index, method_name, None)
        if method is None and fallback_method_name != method_name:
            method = getattr(index, fallback_method_name, None)
        doc = getattr(method, "__doc__", None) if method is not None else None
        raise RuntimeError(
            f"Selected query method '{query_method}' has an incompatible runtime signature in the current index backend. "
            f"{method_name} doc: {doc}"
        ) from exc
