from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from .dataset_utils import exact_topk
from .dataset_utils import resolve_space_type
from .lid_selection import (
    select_lid_representatives,
    trim_lid_outliers,
)


CLASSIFY_START = 4
CLASSIFY_END = 16
CHR_EMA_DECAY = 0.8
PAPER_FLOOR_PAIR_GAP = 2
DEFAULT_MIXED_ACCEPTABLE_RECALL_THRESHOLD = 1.0
DEFAULT_MIXED_THRESHOLD_MODE = "paper_floor_half"
DEFAULT_MIXED_BUCKET_COUNT = 4
SUPPORTED_MIXED_THRESHOLD_MODES = ("paper_floor_half",)
MIXED_CALIBRATION_CACHE_FORMAT = "mixed-projected-local-acceptable-v1"


class MixedCalibrationCacheSettingsMismatchError(ValueError):
    """Raised when an on-disk mixed calibration cache was built with different settings."""


@dataclass(frozen=True)
class MixedProjectedLocalAcceptablePolicy:
    policy_name: str
    source_label: str
    enabled: bool
    tmin_pops: int
    accepted_threshold: float
    accepted_patience: int
    super_pct_by_ef: dict[int, float]
    gamma_ratio_by_ef: dict[int, float]
    mid_easy_upper_pct_by_ef: dict[int, float]
    mid_easy_upper_gamma_ratio_by_ef: dict[int, float]
    route_efs_by_ef: dict[int, tuple[int, ...]] = field(default_factory=dict)
    bucket_gamma_ratios_by_ef: dict[int, tuple[float, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MixedThresholdConfig:
    selection_ef: int
    route_super_ef: int
    route_mid_ef: int
    route_easy_ef: int
    pair_super_target_ef: int
    pair_mid_target_ef: int
    pair_easy_target_ef: int
    route_efs: tuple[int, ...] = ()
    pair_target_efs: tuple[int, ...] = ()

    @property
    def resolved_route_efs(self) -> tuple[int, ...]:
        if self.route_efs:
            return tuple(int(value) for value in self.route_efs)
        return (
            int(self.route_super_ef),
            int(self.route_mid_ef),
            int(self.route_easy_ef),
        )

    @property
    def resolved_pair_target_efs(self) -> tuple[int, ...]:
        if self.pair_target_efs:
            return tuple(int(value) for value in self.pair_target_efs)
        return (
            int(self.pair_super_target_ef),
            int(self.pair_mid_target_ef),
            int(self.pair_easy_target_ef),
        )

    @property
    def bucket_count(self) -> int:
        return len(self.resolved_route_efs) + 1

    @property
    def route_signature(self) -> str:
        route_parts = [str(int(value)) for value in self.resolved_route_efs]
        route_parts.append(str(int(self.selection_ef)))
        return "/".join(route_parts)

    @property
    def local_signature(self) -> str:
        return ",".join(
            f"{int(route_ef)}->{int(pair_target_ef)}"
            for route_ef, pair_target_ef in zip(
                self.resolved_route_efs,
                self.resolved_pair_target_efs,
            )
        )


def _resolve_cache_context(cache_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if cache_context is None:
        return {}
    resolved: dict[str, Any] = {}
    for key, value in dict(cache_context).items():
        if isinstance(value, Path):
            resolved[str(key)] = str(value)
        else:
            resolved[str(key)] = value
    return resolved


def _serialize_float(value: float) -> float | None:
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def _deserialize_float(value: float | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _serialize_float_map(values_by_ef: Mapping[int, float]) -> dict[str, float | None]:
    return {
        str(int(ef)): _serialize_float(float(value))
        for ef, value in sorted(values_by_ef.items(), key=lambda item: int(item[0]))
    }


def _deserialize_float_map(payload: Mapping[str, float | None]) -> dict[int, float]:
    return {
        int(ef): _deserialize_float(value)
        for ef, value in dict(payload).items()
    }


def _serialize_int_tuple_map(values_by_ef: Mapping[int, tuple[int, ...]]) -> dict[str, list[int]]:
    return {
        str(int(ef)): [int(value) for value in values]
        for ef, values in sorted(values_by_ef.items(), key=lambda item: int(item[0]))
    }


def _deserialize_int_tuple_map(payload: Mapping[str, list[int]] | None) -> dict[int, tuple[int, ...]]:
    if not payload:
        return {}
    return {
        int(ef): tuple(int(value) for value in values)
        for ef, values in dict(payload).items()
    }


def _serialize_float_tuple_map(values_by_ef: Mapping[int, tuple[float, ...]]) -> dict[str, list[float | None]]:
    return {
        str(int(ef)): [_serialize_float(float(value)) for value in values]
        for ef, values in sorted(values_by_ef.items(), key=lambda item: int(item[0]))
    }


def _deserialize_float_tuple_map(
    payload: Mapping[str, list[float | None]] | None,
) -> dict[int, tuple[float, ...]]:
    if not payload:
        return {}
    return {
        int(ef): tuple(_deserialize_float(value) for value in values)
        for ef, values in dict(payload).items()
    }


def _serialize_policy(policy: MixedProjectedLocalAcceptablePolicy) -> dict[str, Any]:
    return {
        "policy_name": str(policy.policy_name),
        "source_label": str(policy.source_label),
        "enabled": bool(policy.enabled),
        "tmin_pops": int(policy.tmin_pops),
        "accepted_threshold": _serialize_float(policy.accepted_threshold),
        "accepted_patience": int(policy.accepted_patience),
        "super_pct_by_ef": _serialize_float_map(policy.super_pct_by_ef),
        "gamma_ratio_by_ef": _serialize_float_map(policy.gamma_ratio_by_ef),
        "mid_easy_upper_pct_by_ef": _serialize_float_map(policy.mid_easy_upper_pct_by_ef),
        "mid_easy_upper_gamma_ratio_by_ef": _serialize_float_map(policy.mid_easy_upper_gamma_ratio_by_ef),
        "route_efs_by_ef": _serialize_int_tuple_map(policy.route_efs_by_ef),
        "bucket_gamma_ratios_by_ef": _serialize_float_tuple_map(policy.bucket_gamma_ratios_by_ef),
    }


def _deserialize_policy(payload: Mapping[str, Any]) -> MixedProjectedLocalAcceptablePolicy:
    return MixedProjectedLocalAcceptablePolicy(
        policy_name=str(payload["policy_name"]),
        source_label=str(payload["source_label"]),
        enabled=bool(payload["enabled"]),
        tmin_pops=int(payload["tmin_pops"]),
        accepted_threshold=_deserialize_float(payload.get("accepted_threshold")),
        accepted_patience=int(payload["accepted_patience"]),
        super_pct_by_ef=_deserialize_float_map(payload["super_pct_by_ef"]),
        gamma_ratio_by_ef=_deserialize_float_map(payload["gamma_ratio_by_ef"]),
        mid_easy_upper_pct_by_ef=_deserialize_float_map(payload["mid_easy_upper_pct_by_ef"]),
        mid_easy_upper_gamma_ratio_by_ef=_deserialize_float_map(payload["mid_easy_upper_gamma_ratio_by_ef"]),
        route_efs_by_ef=_deserialize_int_tuple_map(payload.get("route_efs_by_ef")),
        bucket_gamma_ratios_by_ef=_deserialize_float_tuple_map(payload.get("bucket_gamma_ratios_by_ef")),
    )


def _normalize_ef_values(ef_values: list[int] | tuple[int, ...]) -> list[int]:
    return sorted({int(ef) for ef in ef_values})


def _build_cache_settings(
    *,
    dataset_name: str,
    ef_values: list[int] | tuple[int, ...],
    acceptable_recall_threshold: float,
    requested_tmin_pops: int,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
    gt_source: str,
    gt_ef: int,
    internal_lid_k: int,
    k: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
    lid_source_graph: str,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
    mixed_bucket_count: int = DEFAULT_MIXED_BUCKET_COUNT,
    classify_start: int = CLASSIFY_START,
    classify_end: int = CLASSIFY_END,
    chr_ema_decay: float = CHR_EMA_DECAY,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
    lid_sampling_mode: str = "full",
    lid_sample_fraction: float | None = None,
    lid_min_sample_size: int | None = None,
    lid_sample_seed: int | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = {
        "dataset_name": str(dataset_name),
        "ef_values": _normalize_ef_values(ef_values),
        "acceptable_recall_threshold": float(acceptable_recall_threshold),
        "requested_tmin_pops": int(requested_tmin_pops),
        "num_calibration_queries": int(num_calibration_queries),
        "selection_mode": str(selection_mode),
        "trim_low_percentile": float(trim_low_percentile),
        "trim_high_percentile": float(trim_high_percentile),
        "gt_source": str(gt_source),
        "gt_ef": int(gt_ef),
        "internal_lid_k": int(internal_lid_k),
        "k": int(k),
        "easy_threshold_scale": float(easy_threshold_scale),
        "mid_threshold_scale": float(mid_threshold_scale),
        "super_threshold_scale": float(super_threshold_scale),
        "lid_source_graph": str(lid_source_graph),
        "cache_context": _resolve_cache_context(cache_context),
    }
    settings["mixed_threshold_mode"] = str(mixed_threshold_mode)
    settings["mixed_bucket_count"] = int(mixed_bucket_count)
    settings["classify_start"] = int(classify_start)
    settings["classify_end"] = int(classify_end)
    settings["chr_ema_decay"] = float(chr_ema_decay)
    settings["paper_floor_pair_gap"] = int(paper_floor_pair_gap)
    if (
        str(lid_sampling_mode) != "full"
        or lid_sample_fraction is not None
        or lid_min_sample_size is not None
        or lid_sample_seed is not None
    ):
        settings["lid_sampling_mode"] = str(lid_sampling_mode)
        settings["lid_sample_fraction"] = (
            float(lid_sample_fraction) if lid_sample_fraction is not None else None
        )
        settings["lid_min_sample_size"] = (
            int(lid_min_sample_size) if lid_min_sample_size is not None else None
        )
        settings["lid_sample_seed"] = (
            int(lid_sample_seed) if lid_sample_seed is not None else None
        )
    return settings


def _validate_cache_settings(
    *,
    cache_path: Path,
    loaded_settings: Mapping[str, Any],
    expected_settings: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    expected_ef_values = [int(ef) for ef in expected_settings.get("ef_values", [])]
    loaded_ef_values = [int(ef) for ef in loaded_settings.get("ef_values", [])]
    missing_ef_values = [ef for ef in expected_ef_values if ef not in set(loaded_ef_values)]
    if missing_ef_values:
        mismatches.append(
            f"ef_values: expected subset={expected_ef_values!r}, found={loaded_ef_values!r}, "
            f"missing={missing_ef_values!r}"
        )
    for key, expected_value in dict(expected_settings).items():
        if key == "ef_values":
            continue
        actual_value = loaded_settings.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, found={actual_value!r}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise MixedCalibrationCacheSettingsMismatchError(
            f"Mixed calibration cache settings mismatch for {cache_path}: {details}"
        )


def save_mixed_calibration_cache(
    *,
    cache_path: str | Path,
    tau_by_ef: Mapping[int, float],
    policy: MixedProjectedLocalAcceptablePolicy,
    cache_settings: Mapping[str, Any],
) -> Path:
    resolved_cache_path = Path(cache_path)
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MIXED_CALIBRATION_CACHE_FORMAT,
        "settings": dict(cache_settings),
        "tau_by_ef": _serialize_float_map(tau_by_ef),
        "policy": _serialize_policy(policy),
    }
    resolved_cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_cache_path


def load_mixed_calibration_cache(
    *,
    cache_path: str | Path,
    cache_settings: Mapping[str, Any],
) -> tuple[dict[int, float], MixedProjectedLocalAcceptablePolicy]:
    resolved_cache_path = Path(cache_path)
    payload = json.loads(resolved_cache_path.read_text(encoding="utf-8"))
    if payload.get("format") != MIXED_CALIBRATION_CACHE_FORMAT:
        raise ValueError(
            f"Unsupported mixed calibration cache format in {resolved_cache_path}: "
            f"{payload.get('format')!r}"
        )
    _validate_cache_settings(
        cache_path=resolved_cache_path,
        loaded_settings=payload.get("settings", {}),
        expected_settings=cache_settings,
    )
    tau_by_ef = _deserialize_float_map(payload["tau_by_ef"])
    policy = _deserialize_policy(payload["policy"])
    return tau_by_ef, policy


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


def _remove_self_from_neighbor_rows(
    neighbor_rows: np.ndarray,
    query_ids: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    filtered_rows: list[np.ndarray] = []
    for local_idx, qid in enumerate(np.asarray(query_ids, dtype=np.int64)):
        row = np.asarray(neighbor_rows[local_idx], dtype=np.int64)
        row_wo_self = row[row != int(qid)]
        if row_wo_self.size < int(k):
            row_wo_self = row[: int(k)]
        filtered_rows.append(row_wo_self[: int(k)])
    return np.vstack(filtered_rows).astype(np.int64, copy=False)


def _evaluate_recall_per_query(
    *,
    gt_neighbors: np.ndarray,
    predicted_neighbors: np.ndarray,
    k: int,
) -> np.ndarray:
    recalls = np.zeros(len(predicted_neighbors), dtype=np.float32)
    for idx in range(len(predicted_neighbors)):
        recalls[idx] = np.intersect1d(
            gt_neighbors[idx][: int(k)],
            predicted_neighbors[idx][: int(k)],
        ).size / float(k)
    return recalls


def _knn_query_hide_node(
    index,
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    *,
    k: int,
    num_threads: int,
) -> np.ndarray:
    query_hide = getattr(index, "knn_query_hide_node", None)
    if query_hide is None:
        raise RuntimeError(
            "Current index backend does not expose knn_query_hide_node(). "
            "The selected backend must provide its own hide-node probe method "
            "before using train-node probes for offline calibration."
        )
    labels, _ = query_hide(
        query_vectors,
        k=int(k),
        hide_labels=np.asarray(query_ids, dtype=np.int64),
        num_threads=int(num_threads),
    )
    return np.asarray(labels, dtype=np.int64)


def _map_exact_distance_type(dataset_name: str) -> str:
    space_type = str(resolve_space_type(dataset_name)).lower()
    if space_type in {"cosine", "ip", "angular"}:
        return "angular"
    if space_type == "l2":
        return "l2"
    raise ValueError(f"Unsupported space_type for exact GT: {space_type}")


def _compute_gt_neighbors(
    *,
    index,
    train: np.ndarray,
    dataset_name: str,
    query_ids: np.ndarray,
    query_vectors: np.ndarray,
    gt_source: str,
    gt_ef: int,
    num_threads: int,
    k: int,
) -> np.ndarray:
    if str(gt_source) == "hnsw":
        index.set_ef(int(gt_ef))
        return _knn_query_hide_node(
            index,
            query_vectors,
            query_ids,
            k=int(k),
            num_threads=int(num_threads),
        )

    if str(gt_source) == "exact":
        gt_neighbors_raw = exact_topk(
            train_subset=np.asarray(train, dtype=np.float32),
            queries=np.asarray(query_vectors, dtype=np.float32),
            K=int(k) + 1,
            distance_type=_map_exact_distance_type(dataset_name),
        )
        return _remove_self_from_neighbor_rows(gt_neighbors_raw, query_ids, k=int(k))

    raise ValueError(f"Unsupported gt_source={gt_source!r}. Expected 'hnsw' or 'exact'.")


def _compute_recall_by_ef(
    *,
    index,
    query_ids: np.ndarray,
    query_vectors: np.ndarray,
    gt_neighbors: np.ndarray,
    ef_value: int,
    num_threads: int,
    k: int,
) -> np.ndarray:
    index.set_ef(int(ef_value))
    predicted_neighbors = _knn_query_hide_node(
        index,
        query_vectors,
        query_ids,
        k=int(k),
        num_threads=int(num_threads),
    )
    return _evaluate_recall_per_query(
        gt_neighbors=gt_neighbors,
        predicted_neighbors=predicted_neighbors,
        k=int(k),
    )


def _extract_chr_mean_by_query(
    *,
    index,
    selected_df: pd.DataFrame,
    query_vectors: np.ndarray,
    selection_ef: int,
    num_threads: int,
    k: int,
    query_ids: np.ndarray | None = None,
    classify_start: int = CLASSIFY_START,
    classify_end: int = CLASSIFY_END,
    chr_ema_decay: float = CHR_EMA_DECAY,
) -> pd.DataFrame:
    hide_labels = None if query_ids is None else np.asarray(query_ids, dtype=np.int64)

    # Both backends (faiss + hnswlib) expose search_layer0_chr_summary, which
    # computes the classify-window smoothed-CHR mean entirely in the C++ search
    # and returns compact per-query arrays. This replaces the old per-step trace
    # path (search_layer0_path_with_dist_metrics_*), which marshalled every search
    # step into a Python dict and re-derived the CHR EMA in Python — the calibration
    # bottleneck. The summary is a faithful port of that aggregation, so calibrated
    # thresholds are unchanged (values identical up to float rounding).
    summary_fn = getattr(index, "search_layer0_chr_summary", None)
    if summary_fn is None:
        raise RuntimeError(
            "Index build does not expose search_layer0_chr_summary; rebuild the "
            "backend (faiss/hnswlib) with the layer-0 CHR summary API."
        )

    summary_kwargs = {
        "k": int(k),
        "ef": int(selection_ef),
        "num_threads": int(num_threads),
        "classify_start": int(classify_start),
        "classify_end": int(classify_end),
        "chr_ema_decay": float(chr_ema_decay),
    }
    if hide_labels is not None:
        summary_kwargs["hide_labels"] = hide_labels
    summary = summary_fn(query_vectors, **summary_kwargs)
    full_pop_counts = np.asarray(summary["full_pop_counts"], dtype=np.uint64)
    window_obs_counts = np.asarray(summary["window_obs_counts"], dtype=np.uint64)
    usable_flags = np.asarray(summary["usable_flags"], dtype=np.uint64)
    mean_smoothed_cfrs = np.asarray(summary["mean_smoothed_cfrs"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    for local_idx in range(len(selected_df)):
        selected_row = selected_df.iloc[local_idx]
        mean_window = float(mean_smoothed_cfrs[local_idx])
        usable = bool(int(usable_flags[local_idx]) != 0 and np.isfinite(mean_window))
        rows.append(
            {
                "selection_rank": int(selected_row["selection_rank"]),
                "query_id": int(selected_row["query_id"]),
                "lid": float(selected_row["lid"]),
                "selection_ef": int(selection_ef),
                "observed_full_pop_count": int(
                    min(int(full_pop_counts[local_idx]), int(classify_end))
                ),
                "window_obs_count": int(window_obs_counts[local_idx]),
                "usable_for_mean_window_calibration": usable,
                "mean_smoothed_chr_classify_window": mean_window,
            }
        )
    return pd.DataFrame(rows).sort_values("selection_rank").reset_index(drop=True)


def _resolve_pair_targets_for_mode(
    *,
    selection_ef: int,
    route_efs: tuple[int, ...],
    mixed_threshold_mode: str,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
) -> tuple[int, ...]:
    route_efs = tuple(int(route_ef) for route_ef in route_efs)
    if not route_efs:
        raise ValueError("route_efs must be non-empty.")
    resolved_mode = str(mixed_threshold_mode)
    if resolved_mode not in SUPPORTED_MIXED_THRESHOLD_MODES:
        raise ValueError(
            f"Unsupported mixed_threshold_mode={resolved_mode!r}. "
            f"Expected one of {SUPPORTED_MIXED_THRESHOLD_MODES!r}."
        )

    pair_gap = int(paper_floor_pair_gap)
    if pair_gap < 1:
        raise ValueError(f"paper_floor_pair_gap must be >= 1, got {pair_gap}")
    return tuple(max(int(route_ef) // pair_gap, 1) for route_ef in route_efs)


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


def _build_mixed_threshold_config(
    selection_ef: int,
    *,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
    mixed_bucket_count: int = DEFAULT_MIXED_BUCKET_COUNT,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
) -> MixedThresholdConfig:
    selection_ef = int(selection_ef)
    if selection_ef < 2:
        raise ValueError(f"selection_ef must be >= 2, got {selection_ef}")

    resolved_mode = str(mixed_threshold_mode)
    if resolved_mode not in SUPPORTED_MIXED_THRESHOLD_MODES:
        raise ValueError(
            f"Unsupported mixed_threshold_mode={resolved_mode!r}. "
            f"Expected one of {SUPPORTED_MIXED_THRESHOLD_MODES!r}."
        )
    route_efs = _build_paper_floor_route_efs(
        selection_ef=int(selection_ef),
        mixed_bucket_count=int(mixed_bucket_count),
    )

    pair_target_efs = _resolve_pair_targets_for_mode(
        selection_ef=int(selection_ef),
        route_efs=tuple(int(value) for value in route_efs),
        mixed_threshold_mode=resolved_mode,
        paper_floor_pair_gap=int(paper_floor_pair_gap),
    )
    route_super_ef = int(route_efs[0])
    route_mid_ef = int(route_efs[1]) if len(route_efs) > 1 else int(route_efs[0])
    route_easy_ef = int(route_efs[-1])
    pair_super_target_ef = int(pair_target_efs[0])
    pair_mid_target_ef = int(pair_target_efs[1]) if len(pair_target_efs) > 1 else int(pair_target_efs[0])
    pair_easy_target_ef = int(pair_target_efs[-1])

    return MixedThresholdConfig(
        selection_ef=int(selection_ef),
        route_super_ef=int(route_super_ef),
        route_mid_ef=int(route_mid_ef),
        route_easy_ef=int(route_easy_ef),
        pair_super_target_ef=int(pair_super_target_ef),
        pair_mid_target_ef=int(pair_mid_target_ef),
        pair_easy_target_ef=int(pair_easy_target_ef),
        route_efs=tuple(int(value) for value in route_efs),
        pair_target_efs=tuple(int(value) for value in pair_target_efs),
    )


def _quantile_theta(chr_values: np.ndarray, mass: float) -> float:
    return float(np.quantile(chr_values, float(np.clip(mass, 0.0, 1.0))))


def _build_source_label(
    configs: list[MixedThresholdConfig],
    *,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
    mixed_bucket_count: int = DEFAULT_MIXED_BUCKET_COUNT,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
) -> str:
    parts = []
    for config in configs:
        parts.append(
            f"sel{int(config.selection_ef)}:route={config.route_signature}:local={config.local_signature}"
        )
    prefix = (
        f"mixed-dynamic[{str(mixed_threshold_mode)}]"
        f"[B={int(mixed_bucket_count)}][g={int(paper_floor_pair_gap)}]"
    )
    return prefix + ":" + ";".join(parts)


def _threshold_scale_for_route_index(
    *,
    route_index: int,
    route_count: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
) -> float:
    if route_count <= 1:
        return float(easy_threshold_scale)
    if route_index <= 0:
        return float(super_threshold_scale)
    if route_index >= route_count - 1:
        return float(easy_threshold_scale)
    return float(mid_threshold_scale)


def build_dynamic_projected_local_acceptable_mixed_policy(
    *,
    index,
    train: np.ndarray,
    dataset_name: str,
    ef_values: list[int] | tuple[int, ...],
    acceptable_recall_threshold: float,
    requested_tmin_pops: int,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
    gt_source: str,
    gt_ef: int,
    internal_lid_k: int,
    num_threads: int,
    k: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
    mixed_bucket_count: int = DEFAULT_MIXED_BUCKET_COUNT,
    classify_start: int = CLASSIFY_START,
    classify_end: int = CLASSIFY_END,
    chr_ema_decay: float = CHR_EMA_DECAY,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
    lid_df: pd.DataFrame | None = None,
    lid_source_graph: str = "index",
) -> tuple[dict[int, float], MixedProjectedLocalAcceptablePolicy]:
    del internal_lid_k
    del lid_source_graph

    resolved_ef_values = sorted({int(ef) for ef in ef_values})
    if not resolved_ef_values:
        raise ValueError("ef_values must be non-empty.")

    selected_df = _select_dummy_queries(
        index=index,
        num_nodes=len(train),
        lid_df=lid_df,
        num_calibration_queries=int(num_calibration_queries),
        selection_mode=str(selection_mode),
        trim_low_percentile=float(trim_low_percentile),
        trim_high_percentile=float(trim_high_percentile),
    )
    query_ids = selected_df["query_id"].to_numpy(dtype=np.int64)
    query_vectors = np.asarray(train[query_ids], dtype=np.float32)
    gt_neighbors = _compute_gt_neighbors(
        index=index,
        train=np.asarray(train, dtype=np.float32),
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        query_vectors=query_vectors,
        gt_source=str(gt_source),
        gt_ef=int(gt_ef),
        num_threads=int(num_threads),
        k=int(k),
    )

    tau_by_ef: dict[int, float] = {}
    super_gamma_by_ef: dict[int, float] = {}
    mid_gamma_by_ef: dict[int, float] = {}
    route_efs_by_ef: dict[int, tuple[int, ...]] = {}
    bucket_gamma_ratios_by_ef: dict[int, tuple[float, ...]] = {}
    nan_by_ef: dict[int, float] = {}
    configs = [
        _build_mixed_threshold_config(
            ef,
            mixed_threshold_mode=str(mixed_threshold_mode),
            mixed_bucket_count=int(mixed_bucket_count),
            paper_floor_pair_gap=int(paper_floor_pair_gap),
        )
        for ef in resolved_ef_values
    ]

    recall_cache: dict[int, np.ndarray] = {}
    for config in configs:
        pair_target_efs = {int(pair_target_ef) for pair_target_ef in config.resolved_pair_target_efs}
        for pair_target_ef in pair_target_efs:
            if int(pair_target_ef) not in recall_cache:
                recall_cache[int(pair_target_ef)] = _compute_recall_by_ef(
                    index=index,
                    query_ids=query_ids,
                    query_vectors=query_vectors,
                    gt_neighbors=gt_neighbors,
                    ef_value=int(pair_target_ef),
                    num_threads=int(num_threads),
                    k=int(k),
                )

    for config in configs:
        anchor_df = _extract_chr_mean_by_query(
            index=index,
            selected_df=selected_df,
            query_vectors=query_vectors,
            query_ids=query_ids,
            selection_ef=int(config.selection_ef),
            num_threads=int(num_threads),
            k=int(k),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )
        usable_mask = anchor_df["usable_for_mean_window_calibration"].astype(bool).to_numpy(dtype=bool)
        anchor_chr_values = pd.to_numeric(
            anchor_df.loc[usable_mask, "mean_smoothed_chr_classify_window"],
            errors="coerce",
        ).to_numpy(dtype=float)
        anchor_chr_values = anchor_chr_values[np.isfinite(anchor_chr_values)]
        if anchor_chr_values.size == 0:
            raise RuntimeError(
                f"No usable mixed calibration CHR values for selection_ef={int(config.selection_ef)}."
            )

        route_thetas: list[float] = []
        route_efs = config.resolved_route_efs
        pair_target_efs = config.resolved_pair_target_efs
        for route_index, pair_target_ef in enumerate(pair_target_efs):
            acceptable_rate = float(
                np.mean(
                    recall_cache[int(pair_target_ef)][usable_mask] + 1e-12
                    >= float(acceptable_recall_threshold)
                )
            )
            route_theta = max(
                _quantile_theta(anchor_chr_values, acceptable_rate)
                * _threshold_scale_for_route_index(
                    route_index=route_index,
                    route_count=len(route_efs),
                    easy_threshold_scale=float(easy_threshold_scale),
                    mid_threshold_scale=float(mid_threshold_scale),
                    super_threshold_scale=float(super_threshold_scale),
                ),
                1e-6,
            )
            route_thetas.append(float(route_theta))

        monotone_route_thetas: list[float] = []
        running_theta = 0.0
        for route_theta in route_thetas:
            running_theta = max(float(route_theta), float(running_theta))
            monotone_route_thetas.append(float(running_theta))

        tau_value = float(monotone_route_thetas[-1])
        gamma_ratios = tuple(
            float(min(1.0, max(0.0, route_theta / max(tau_value, 1e-6))))
            for route_theta in monotone_route_thetas
        )

        tau_by_ef[int(config.selection_ef)] = float(tau_value)
        super_gamma_by_ef[int(config.selection_ef)] = (
            float(gamma_ratios[0]) if gamma_ratios else float("nan")
        )
        mid_gamma_by_ef[int(config.selection_ef)] = (
            float(gamma_ratios[1]) if len(gamma_ratios) > 1 else float("nan")
        )
        route_efs_by_ef[int(config.selection_ef)] = tuple(int(route_ef) for route_ef in route_efs)
        bucket_gamma_ratios_by_ef[int(config.selection_ef)] = tuple(float(value) for value in gamma_ratios)
        nan_by_ef[int(config.selection_ef)] = float("nan")

    policy = MixedProjectedLocalAcceptablePolicy(
        policy_name="mixed-dynamic",
        source_label=_build_source_label(
            configs,
            mixed_threshold_mode=str(mixed_threshold_mode),
            mixed_bucket_count=int(mixed_bucket_count),
            paper_floor_pair_gap=int(paper_floor_pair_gap),
        ),
        enabled=False,
        tmin_pops=int(requested_tmin_pops),
        accepted_threshold=float(acceptable_recall_threshold),
        accepted_patience=0,
        super_pct_by_ef=dict(nan_by_ef),
        gamma_ratio_by_ef=dict(super_gamma_by_ef),
        mid_easy_upper_pct_by_ef=dict(nan_by_ef),
        mid_easy_upper_gamma_ratio_by_ef=dict(mid_gamma_by_ef),
        route_efs_by_ef=dict(route_efs_by_ef),
        bucket_gamma_ratios_by_ef=dict(bucket_gamma_ratios_by_ef),
    )
    return tau_by_ef, policy


def load_or_build_dynamic_projected_local_acceptable_mixed_policy(
    *,
    index,
    train: np.ndarray,
    dataset_name: str,
    ef_values: list[int] | tuple[int, ...],
    acceptable_recall_threshold: float,
    requested_tmin_pops: int,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
    gt_source: str,
    gt_ef: int,
    internal_lid_k: int,
    num_threads: int,
    k: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
    mixed_bucket_count: int = DEFAULT_MIXED_BUCKET_COUNT,
    classify_start: int = CLASSIFY_START,
    classify_end: int = CLASSIFY_END,
    chr_ema_decay: float = CHR_EMA_DECAY,
    paper_floor_pair_gap: int = PAPER_FLOOR_PAIR_GAP,
    lid_df: pd.DataFrame | None = None,
    lid_source_graph: str = "index",
    lid_sampling_mode: str = "full",
    lid_sample_fraction: float | None = None,
    lid_min_sample_size: int | None = None,
    lid_sample_seed: int | None = None,
    cache_path: str | Path | None = None,
    cache_context: Mapping[str, Any] | None = None,
) -> tuple[dict[int, float], MixedProjectedLocalAcceptablePolicy, str | None]:
    cache_settings = _build_cache_settings(
        dataset_name=dataset_name,
        ef_values=ef_values,
        acceptable_recall_threshold=acceptable_recall_threshold,
        requested_tmin_pops=requested_tmin_pops,
        num_calibration_queries=num_calibration_queries,
        selection_mode=selection_mode,
        trim_low_percentile=trim_low_percentile,
        trim_high_percentile=trim_high_percentile,
        gt_source=gt_source,
        gt_ef=gt_ef,
        internal_lid_k=internal_lid_k,
        k=k,
        easy_threshold_scale=easy_threshold_scale,
        mid_threshold_scale=mid_threshold_scale,
        super_threshold_scale=super_threshold_scale,
        lid_source_graph=lid_source_graph,
        mixed_threshold_mode=mixed_threshold_mode,
        mixed_bucket_count=mixed_bucket_count,
        classify_start=int(classify_start),
        classify_end=int(classify_end),
        chr_ema_decay=float(chr_ema_decay),
        paper_floor_pair_gap=int(paper_floor_pair_gap),
        lid_sampling_mode=lid_sampling_mode,
        lid_sample_fraction=lid_sample_fraction,
        lid_min_sample_size=lid_min_sample_size,
        lid_sample_seed=lid_sample_seed,
        cache_context=cache_context,
    )
    resolved_cache_path = Path(cache_path) if cache_path is not None else None
    rebuilt_due_to_mismatch = False
    if resolved_cache_path is not None and resolved_cache_path.exists():
        try:
            tau_by_ef, policy = load_mixed_calibration_cache(
                cache_path=resolved_cache_path,
                cache_settings=cache_settings,
            )
        except MixedCalibrationCacheSettingsMismatchError as exc:
            rebuilt_due_to_mismatch = True
            print(f"[MIXED_CACHE] rebuild_due_to_settings_mismatch reason={exc}")
        else:
            return tau_by_ef, policy, "loaded"

    tau_by_ef, policy = build_dynamic_projected_local_acceptable_mixed_policy(
        index=index,
        train=train,
        dataset_name=dataset_name,
        ef_values=ef_values,
        acceptable_recall_threshold=acceptable_recall_threshold,
        requested_tmin_pops=requested_tmin_pops,
        num_calibration_queries=num_calibration_queries,
        selection_mode=selection_mode,
        trim_low_percentile=trim_low_percentile,
        trim_high_percentile=trim_high_percentile,
        gt_source=gt_source,
        gt_ef=gt_ef,
        internal_lid_k=internal_lid_k,
        num_threads=num_threads,
        k=k,
        easy_threshold_scale=easy_threshold_scale,
        mid_threshold_scale=mid_threshold_scale,
        super_threshold_scale=super_threshold_scale,
        mixed_threshold_mode=mixed_threshold_mode,
        mixed_bucket_count=mixed_bucket_count,
        classify_start=int(classify_start),
        classify_end=int(classify_end),
        chr_ema_decay=float(chr_ema_decay),
        paper_floor_pair_gap=int(paper_floor_pair_gap),
        lid_df=lid_df,
        lid_source_graph=lid_source_graph,
    )
    if resolved_cache_path is not None:
        save_mixed_calibration_cache(
            cache_path=resolved_cache_path,
            tau_by_ef=tau_by_ef,
            policy=policy,
            cache_settings=cache_settings,
        )
        return tau_by_ef, policy, "rebuilt" if rebuilt_due_to_mismatch else "saved"
    return tau_by_ef, policy, None


def resolve_runtime_bucket_routing(
    *,
    policy: MixedProjectedLocalAcceptablePolicy,
    selection_ef: int,
    mixed_threshold_mode: str = DEFAULT_MIXED_THRESHOLD_MODE,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if str(mixed_threshold_mode) != "paper_floor_half":
        return tuple(), tuple()
    route_efs = tuple(int(value) for value in policy.route_efs_by_ef.get(int(selection_ef), ()))
    gamma_ratios = tuple(
        float(value) for value in policy.bucket_gamma_ratios_by_ef.get(int(selection_ef), ())
    )
    if len(route_efs) != len(gamma_ratios):
        return tuple(), tuple()
    return route_efs, gamma_ratios
