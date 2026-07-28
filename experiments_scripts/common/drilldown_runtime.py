"""Backend-neutral easy/medium/hard drilldown helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .adaptive_runtime import (
    dedupe_preserve_order,
    evaluate_recall_per_query,
    format_float_signature,
    format_int_signature,
)


BenchmarkFn = Callable[..., dict[str, float]]


def _dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def drilldown_output_dir(args, final_dir: Path) -> Path:
    if str(args.drilldown_output_dir).strip():
        return Path(args.drilldown_output_dir).expanduser().resolve()
    return (
        final_dir
        / f"easy_medium_hard_drilldown_pseudogt{int(args.drilldown_pseudo_gt_ef)}"
        / f"groupdef{int(args.drilldown_group_def_ef)}"
    )


def drilldown_dataset_dir(args, final_dir: Path, dataset: str) -> Path:
    return drilldown_output_dir(args, final_dir) / _dataset_stem(dataset)


def drilldown_sweep_csv_for(args, final_dir: Path, dataset: str, k: int) -> Path:
    stem = _dataset_stem(dataset)
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
    return np.asarray(labels, dtype=np.int64)


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
    method = getattr(index, "knn_query_beam_width_first_target_hit_step", None)
    if method is None:
        raise RuntimeError(
            "Drilldown tie-break requires knn_query_beam_width_first_target_hit_step, "
            "but the selected index backend does not expose it."
        )

    first_steps, reached_flags, achieved_hits, _reached_count = method(
        test,
        np.asarray(gt_labels, dtype=np.int64),
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


def write_drilldown_readme(output_dir: Path, args, *, backend_label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {backend_label} Easy/Medium/Hard Drilldown",
        "",
        "Groups are defined once per dataset/k using Vanilla recall against a pseudo ground truth.",
        "",
        f"- pseudo ground truth: Vanilla {backend_label} at efSearch={int(args.drilldown_pseudo_gt_ef)}",
        f"- group definition: Vanilla {backend_label} at efSearch={int(args.drilldown_group_def_ef)}",
        "- split: top 30% easy, middle 40% medium, bottom 30% hard by group-definition recall",
        "- tie-break: recall ties use the earliest first step reaching that final recall",
        "- comparison: Vanilla and Ours are evaluated on the same fixed query groups",
        "",
        "Primary files:",
        "",
        "- `query_groups.csv`: fixed query group assignment",
        "- `group_ef_sweep.csv`: per-group recall/QPS/latency rows for Vanilla and Ours",
        "- `group_pair_metrics.csv`: Ours-vs-Vanilla deltas on the fixed groups",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n",
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
    args,
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
    benchmark_baseline_fn: BenchmarkFn,
    benchmark_ours_fn: BenchmarkFn,
    emit,
    backend_label: str = "Backend",
) -> None:
    if int(args.drilldown_pseudo_gt_ef) < int(k):
        raise ValueError("--drilldown-pseudo-gt-ef must be >= k.")
    if int(args.drilldown_group_def_ef) < int(k):
        raise ValueError("--drilldown-group-def-ef must be >= k.")

    output_dir = drilldown_output_dir(args, final_dir)
    dataset_dir = drilldown_dataset_dir(args, final_dir, dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_drilldown_readme(output_dir, args, backend_label=backend_label)

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

    stem = _dataset_stem(dataset)
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

    query_group_df = pd.DataFrame(
        {
            "dataset": stem,
            "dataset_file": dataset,
            "k": int(k),
            "qid": np.arange(query_count, dtype=np.int64),
            "pseudo_gt_ef": int(args.drilldown_pseudo_gt_ef),
            "group_def_ef": int(args.drilldown_group_def_ef),
            "group_tie_breaker": group_tie_breaker,
            "first_step_tie_break_available": True,
            "group_def_vanilla_recall": group_def_recalls.astype(np.float64),
            "group_def_target_hits": group_def_target_hits.astype(np.int64),
            "group_def_first_final_recall_step": group_def_first_steps.astype(np.int64),
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
    emit(
        f"[DRILLDOWN] groups easy={int(group_counts.get('easy', 0))} "
        f"medium={int(group_counts.get('medium', 0))} hard={int(group_counts.get('hard', 0))} "
        f"tie_break={group_tie_breaker} pseudo_wall_s={pseudo_wall_s:.3f}"
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
            vanilla_metrics = benchmark_baseline_fn(
                index=index,
                test=group_test,
                neighbors=group_neighbors,
                ef=ef,
                k=int(k),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                num_threads=int(args.online_num_threads),
            )
            ours_kwargs = {
                "index": index,
                "test": group_test,
                "neighbors": group_neighbors,
                "ef": ef,
                "k": int(k),
                "query_method": str(args.query_method),
                "tau": float(tau_by_ef[ef]),
                "super_gamma": float(policy.gamma_ratio_by_ef.get(ef, float("nan"))),
                "mid_gamma": float(policy.mid_easy_upper_gamma_ratio_by_ef.get(ef, float("nan"))),
                "tmin_pops": int(args.tmin_pops),
                "mixed_threshold_mode": str(args.mixed_threshold_mode),
                "paper_bucket_count": paper_bucket_count,
                "paper_bucket_gamma_ratios": bucket_gammas,
                "warmup_runs": int(args.warmup_runs),
                "measured_runs": int(args.measured_runs),
                "num_threads": int(args.online_num_threads),
            }
            optional_runtime_kwargs = {
                "classify_start": int(getattr(args, "classify_start", 4)),
                "classify_end": int(getattr(args, "classify_end", 16)),
                "chr_ema_decay": float(getattr(args, "chr_ema_decay", 0.8)),
            }
            try:
                ours_metrics = benchmark_ours_fn(**ours_kwargs, **optional_runtime_kwargs)
            except TypeError as exc:
                if not any(name in str(exc) for name in optional_runtime_kwargs):
                    raise
                ours_metrics = benchmark_ours_fn(**ours_kwargs)
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
                "first_step_tie_break_available": True,
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
