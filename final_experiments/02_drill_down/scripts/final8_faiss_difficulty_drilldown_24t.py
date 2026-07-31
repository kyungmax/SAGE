#!/usr/bin/env python3
"""FAISS SIMD-on 24-thread easy/medium/hard drill-down for the main8 setup."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
DRILLDOWN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = DRILLDOWN_ROOT.parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FINAL_FAISS_ROOT = EXPERIMENTS_ROOT / "faiss"

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_ROOT))


def _default_project_root() -> Path:
    for key in ("HNSW_PLAYGROUND_ROOT", "SAGE_PROJECT_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return REPO_ROOT


PROJECT_ROOT = _default_project_root()
DEFAULT_DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(PROJECT_ROOT / "index"))).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get("FAISS_PYTHON_PATH", str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"))
).expanduser()
DEFAULT_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get("FAISS_INDEX_ROOT", str(DEFAULT_INDEX_DIR / "faiss_m32_efc500_main8_20260707/darth/index")),
    )
).expanduser()

from common.adaptive_runtime import (  # noqa: E402
    DEFAULT_CALIBRATION_SAMPLE_SEED,
    evaluate_recall_per_query,
    format_float_signature,
    format_int_signature,
    load_dataset_with_special_cases,
    resolve_runtime_threshold_scales,
)
from common.offline_calibration import compute_fixed_calibration_lid_pool, resolve_mixed_policy_with_status  # noqa: E402
import common.projected_local_acceptable_runtime as projected_runtime  # noqa: E402


build_original_index = None
BACKEND_LABEL = "FAISS"


DEFAULT_DATASETS = (
    "glove-100-angular.hdf5",
    "nytimes-256-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
    "cohere-768-angular.hdf5",
    "youtube-15M-angular.hdf5",
    "agnews-mxbai-1024-euclidean.hdf5",
    "landmark-nomic-768-angular.hdf5",
)
GROUP_ORDER = ("easy", "medium", "hard")
DEFAULT_CALIBRATION_EFS = (64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("faiss",), default="faiss")
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--faiss-index-root", type=Path, default=DEFAULT_FAISS_INDEX_ROOT)
    parser.add_argument("--allow-system-faiss", action="store_true")
    parser.add_argument("--efs", default="1024")
    parser.add_argument("--calibration-efs", default=",".join(str(value) for value in DEFAULT_CALIBRATION_EFS))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-gt-source", choices=("exact", "pseudo_hnsw"), default="exact")
    parser.add_argument("--pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--group-def-ef", type=int, default=1024)
    parser.add_argument("--offline-num-threads", type=int, default=24)
    parser.add_argument("--online-num-threads", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--internal-lid-k", type=int, default=15)
    parser.add_argument("--calibration-sample-seed", type=int, default=DEFAULT_CALIBRATION_SAMPLE_SEED)
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument("--mixed-threshold-mode", default="paper_floor_half")
    parser.add_argument("--mixed-bucket-count", type=int, default=4)
    parser.add_argument("--classify-start", type=int, default=4)
    parser.add_argument("--classify-end", type=int, default=16)
    parser.add_argument("--cfr-ema-decay", type=float, default=0.8)
    parser.add_argument("--pair-gap", type=int, default=2)
    parser.add_argument("--groups", default=",".join(GROUP_ORDER), help="Comma list of groups to benchmark: easy,medium,hard")
    args = parser.parse_args()
    args.datasets = tuple(part.strip() for part in str(args.datasets).split(",") if part.strip())
    args.efs = tuple(int(part.strip()) for part in str(args.efs).split(",") if part.strip())
    args.calibration_efs = tuple(int(part.strip()) for part in str(args.calibration_efs).split(",") if part.strip())
    args.groups = tuple(part.strip().lower() for part in str(args.groups).split(",") if part.strip())
    invalid_groups = [group for group in args.groups if group not in GROUP_ORDER]
    if invalid_groups:
        raise ValueError(f"--groups contains unsupported values: {invalid_groups}")
    if not args.groups:
        raise ValueError("--groups must not be empty")
    (
        args.easy_threshold_scale,
        args.mid_threshold_scale,
        args.super_threshold_scale,
    ) = resolve_runtime_threshold_scales(
        easy_threshold_scale=None,
        mid_threshold_scale=None,
        super_threshold_scale=None,
    )
    if int(args.k) < 1:
        raise ValueError("--k must be positive")
    if not set(args.efs).issubset(set(args.calibration_efs)):
        raise ValueError("--calibration-efs must include every reported --efs value")
    if str(args.eval_gt_source) == "pseudo_hnsw" and int(args.pseudo_gt_ef) < int(args.group_def_ef):
        raise ValueError("--pseudo-gt-ef must be >= --group-def-ef")
    return args


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_backend(args: argparse.Namespace) -> None:
    global build_original_index, BACKEND_LABEL
    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(PROJECT_ROOT))
    os.environ["FAISS_PYTHON_PATH"] = str(args.faiss_python_path.expanduser().resolve())
    os.environ["FAISS_INDEX_ROOT"] = str(args.faiss_index_root.expanduser().resolve())
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    if str(FINAL_FAISS_ROOT) not in sys.path:
        sys.path.insert(0, str(FINAL_FAISS_ROOT))
    module = load_module_from_path("drilldown_faiss_final_index_utils", FINAL_FAISS_ROOT / "final_index_utils.py")
    module.configure_faiss_loader(
        python_path=args.faiss_python_path,
        index_root=args.faiss_index_root,
        allow_system_faiss=bool(args.allow_system_faiss),
    )
    build_original_index = module.build_original_index
    BACKEND_LABEL = "FAISS"


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def configure_runtime(args: argparse.Namespace) -> None:
    projected_runtime.CLASSIFY_START = int(args.classify_start)
    projected_runtime.CLASSIFY_END = int(args.classify_end)
    projected_runtime.CFR_EMA_DECAY = float(args.cfr_ema_decay)
    projected_runtime.CFR_EMA_UPDATE = 1.0 - float(args.cfr_ema_decay)
    projected_runtime.PAPER_FLOOR_PAIR_GAP = int(args.pair_gap)


def append_status(path: Path, fields: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(str(field) for field in fields) + "\n")


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def compute_vanilla_labels(index: Any, test: np.ndarray, *, ef: int, k: int, num_threads: int) -> np.ndarray:
    index.set_ef(int(ef))
    labels, _ = index.knn_query(test, k=int(k), num_threads=int(num_threads))
    return np.asarray(labels, dtype=np.int64)


def compute_hit_counts(labels: np.ndarray, gt_labels: np.ndarray, k: int) -> np.ndarray:
    hit_counts = np.zeros(labels.shape[0], dtype=np.int64)
    for row in range(labels.shape[0]):
        hit_counts[row] = int(np.intersect1d(labels[row][: int(k)], gt_labels[row][: int(k)]).size)
    return hit_counts


def compute_first_final_recall_steps(index: Any, test: np.ndarray, gt_labels: np.ndarray, target_hits: np.ndarray, *, ef: int, k: int, num_threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_steps, reached_flags, achieved_hits, _ = index.knn_query_beam_width_first_target_hit_step(
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


def assign_groups(recalls: np.ndarray, first_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_count = int(len(recalls))
    order = np.lexsort((np.arange(query_count, dtype=np.int64), np.asarray(first_steps, dtype=np.float64), -np.asarray(recalls, dtype=np.float64)))
    easy_count = int(np.floor(query_count * 0.30))
    hard_count = int(np.floor(query_count * 0.30))
    medium_count = query_count - easy_count - hard_count
    groups = np.empty(query_count, dtype=object)
    groups[order[:easy_count]] = "easy"
    groups[order[easy_count : easy_count + medium_count]] = "medium"
    groups[order[easy_count + medium_count :]] = "hard"
    ranks = np.empty(query_count, dtype=np.int64)
    ranks[order] = np.arange(1, query_count + 1, dtype=np.int64)
    percentiles = (ranks.astype(np.float64) - 1.0) / float(max(query_count - 1, 1))
    return groups, ranks, percentiles


def latency_fields(durations: list[float], query_count: int) -> dict[str, float]:
    arr = np.asarray(durations, dtype=np.float64)
    avg = float(arr.mean())
    return {
        "query_count": int(query_count),
        "batch_latency_mean_ms": avg * 1000.0,
        "batch_latency_p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "batch_latency_p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "batch_latency_min_ms": float(arr.min() * 1000.0),
        "batch_latency_max_ms": float(arr.max() * 1000.0),
        "latency_per_query_mean_ms": avg * 1000.0 / float(query_count),
        "qps": float(query_count) / avg if avg > 0.0 else float("nan"),
    }


def _faiss_hnsw_stats() -> Any | None:
    module = sys.modules.get("faiss")
    if module is None:
        return None
    cvar = getattr(module, "cvar", None)
    if cvar is None:
        return None
    stats = getattr(cvar, "hnsw_stats", None)
    if stats is None or not hasattr(stats, "ndis"):
        return None
    return stats


def benchmark_query(query_fn: Callable[[], tuple[np.ndarray, np.ndarray]], gt: np.ndarray, *, k: int, warmup_runs: int, measured_runs: int) -> dict[str, Any]:
    stats = _faiss_hnsw_stats()
    for _ in range(int(warmup_runs)):
        if stats is not None and hasattr(stats, "reset"):
            stats.reset()
        query_fn()
    durations: list[float] = []
    ndis_values: list[int] = []
    nhops_values: list[int] = []
    labels = None
    dists = None
    for _ in range(int(measured_runs)):
        if stats is not None and hasattr(stats, "reset"):
            stats.reset()
        start = time.perf_counter()
        labels, dists = query_fn()
        durations.append(time.perf_counter() - start)
        if stats is not None:
            ndis_values.append(int(getattr(stats, "ndis", 0)))
            nhops_values.append(int(getattr(stats, "nhops", 0)))
    assert labels is not None
    recall = float(np.mean(evaluate_recall_per_query(labels, gt, int(k))))
    out: dict[str, Any] = {"recall": recall}
    out.update(latency_fields(durations, len(gt)))
    out["adaptive_max_dist_mean"] = float(np.mean(np.max(dists, axis=1))) if dists is not None else np.nan
    if ndis_values:
        ndis_arr = np.asarray(ndis_values, dtype=np.float64)
        out["distance_computations_mean"] = float(ndis_arr.mean())
        out["distance_computations_per_query_mean"] = float(ndis_arr.mean() / float(len(gt)))
        out["distance_computations_min"] = float(ndis_arr.min())
        out["distance_computations_max"] = float(ndis_arr.max())
    else:
        out["distance_computations_mean"] = np.nan
        out["distance_computations_per_query_mean"] = np.nan
        out["distance_computations_min"] = np.nan
        out["distance_computations_max"] = np.nan
    if nhops_values:
        nhops_arr = np.asarray(nhops_values, dtype=np.float64)
        out["hnsw_hops_mean"] = float(nhops_arr.mean())
        out["hnsw_hops_per_query_mean"] = float(nhops_arr.mean() / float(len(gt)))
    else:
        out["hnsw_hops_mean"] = np.nan
        out["hnsw_hops_per_query_mean"] = np.nan
    return out


def benchmark_vanilla(index: Any, queries: np.ndarray, gt: np.ndarray, *, ef: int, k: int, args: argparse.Namespace) -> dict[str, Any]:
    def query() -> tuple[np.ndarray, np.ndarray]:
        index.set_ef(int(ef))
        labels, dists = index.knn_query(queries, k=int(k), num_threads=int(args.online_num_threads))
        return np.asarray(labels, dtype=np.int64), np.asarray(dists)

    return benchmark_query(query, gt, k=int(k), warmup_runs=int(args.warmup_runs), measured_runs=int(args.measured_runs))


def benchmark_sage(
    index: Any,
    queries: np.ndarray,
    gt: np.ndarray,
    *,
    ef: int,
    k: int,
    tau: float,
    gammas: tuple[float, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_sage = getattr(index, "knn_query_sage", None)
    if query_sage is None:
        query_sage = getattr(index, "knn_query_adaptive_light_paper_bucket")
    def query() -> tuple[np.ndarray, np.ndarray]:
        kwargs = {
            "k": int(k),
            "ef_init": int(ef),
            "enable_stop": True,
            "early_stop_ratio": float(tau),
            "tmin_pops": int(args.tmin_pops),
            "paper_bucket_count": int(len(gammas) + 1),
            "bucket_gamma_ratios": [float(value) for value in gammas],
            "num_threads": int(args.online_num_threads),
        }
        labels, dists = query_sage(queries, **kwargs)
        return np.asarray(labels, dtype=np.int64), np.asarray(dists)

    return benchmark_query(query, gt, k=int(k), warmup_runs=int(args.warmup_runs), measured_runs=int(args.measured_runs))


def write_readme(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.eval_gt_source) == "exact":
        gt_line = "- evaluation ground truth: exact dataset neighbors from the benchmark HDF5\n"
    else:
        gt_line = f"- evaluation ground truth: Vanilla {BACKEND_LABEL} efSearch={int(args.pseudo_gt_ef)}\n"
    (output_dir / "README.md").write_text(
        f"# Main8 {BACKEND_LABEL} Easy/Medium/Hard Drilldown\n\n"
        f"- backend: {str(args.backend)}\n"
        f"{gt_line}"
        f"- group definition: Vanilla {BACKEND_LABEL} efSearch={int(args.group_def_ef)}\n"
        "- split: top 30% easy, middle 40% medium, bottom 30% hard\n"
        f"- benchmarked groups: {','.join(args.groups)}\n"
        f"- online/offline threads: {int(args.online_num_threads)}/{int(args.offline_num_threads)}\n"
        f"- calibration policy remains the final implementation mixed policy calibrated against {BACKEND_LABEL} pseudo GT.\n"
        "- SAGE runtime call uses the current `knn_query_sage` signature exposed by the selected backend.\n",
        encoding="utf-8",
    )


def run_dataset(args: argparse.Namespace, dataset: str, all_query_groups: list[pd.DataFrame], all_sweep: list[pd.DataFrame], all_pair: list[pd.DataFrame]) -> None:
    stem = dataset_stem(dataset)
    dataset_dir = args.output_dir / stem
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DATASET] {dataset}", flush=True)

    t0 = time.perf_counter()
    train, test, neighbors = load_dataset_with_special_cases(str(args.base_path.resolve()), dataset)
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    gt_k = int(neighbors.shape[1])
    if gt_k < int(args.k):
        raise ValueError(f"{dataset}: groundtruth_k={gt_k} < k={int(args.k)}")
    dataset_load_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    index, _space, _ = build_original_index(
        train=train,
        dataset_name=dataset,
        index_dir=str(args.index_dir.resolve()),
        param_m=int(args.param_m),
        ef_construction=int(args.ef_construction),
        num_threads=int(args.offline_num_threads),
    )
    index_load_wall_s = time.perf_counter() - t0
    index.set_num_threads(int(args.online_num_threads))

    t0 = time.perf_counter()
    lid_df = compute_fixed_calibration_lid_pool(
        index,
        internal_lid_k=int(args.internal_lid_k),
        num_nodes=len(train),
        lid_sample_seed=int(args.calibration_sample_seed),
        num_threads=int(args.offline_num_threads),
        dataset_name=dataset,
    )
    lid_wall_s = time.perf_counter() - t0

    cache_path = (
        args.run_root
        / stem
        / "final6_24t_baseline__ncal100__cs4ce16__alpha0p8__g2__b4"
        / f"{stem}__k{int(args.k)}__mixed_original_M{int(args.param_m)}_efC{int(args.ef_construction)}.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tau_by_ef, policy, cache_status = resolve_mixed_policy_with_status(
        index=index,
        train=train,
        dataset_name=dataset,
        ef_values=list(int(ef) for ef in args.calibration_efs),
        k=int(args.k),
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
        lid_df=lid_df,
        lid_sample_seed=int(args.calibration_sample_seed),
        cache_path=cache_path,
        mixed_threshold_mode=str(args.mixed_threshold_mode),
        mixed_bucket_count=int(args.mixed_bucket_count),
    )
    threshold_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if str(args.eval_gt_source) == "exact":
        eval_gt = np.ascontiguousarray(np.asarray(neighbors[:, : int(args.k)], dtype=np.int64))
        eval_gt_label = "exact"
    else:
        eval_gt = compute_vanilla_labels(index, test, ef=int(args.pseudo_gt_ef), k=int(args.k), num_threads=int(args.online_num_threads))
        eval_gt_label = f"hnsw_ef{int(args.pseudo_gt_ef)}"
    group_def = compute_vanilla_labels(index, test, ef=int(args.group_def_ef), k=int(args.k), num_threads=int(args.online_num_threads))
    group_recalls = evaluate_recall_per_query(group_def, eval_gt, int(args.k))
    target_hits = compute_hit_counts(group_def, eval_gt, int(args.k))
    first_steps, reached, achieved_hits = compute_first_final_recall_steps(
        index,
        test,
        eval_gt,
        target_hits,
        ef=int(args.group_def_ef),
        k=int(args.k),
        num_threads=int(args.online_num_threads),
    )
    groups, ranks, percentiles = assign_groups(group_recalls, first_steps)
    pseudo_wall_s = time.perf_counter() - t0

    query_group_df = pd.DataFrame(
        {
            "dataset": stem,
            "dataset_file": dataset,
            "k": int(args.k),
            "qid": np.arange(len(test), dtype=np.int64),
            "evaluation_gt_source": eval_gt_label,
            "pseudo_gt_ef": int(args.pseudo_gt_ef),
            "group_def_ef": int(args.group_def_ef),
            "group_tie_breaker": "first_final_recall_step",
            "first_step_tie_break_available": True,
            "group_def_vanilla_recall": group_recalls.astype(np.float64),
            "group_def_target_hits": target_hits.astype(np.int64),
            "group_def_first_final_recall_step": first_steps.astype(np.int64),
            "group_def_first_step_reached_target": reached.astype(np.int64),
            "group_def_first_step_achieved_hits": achieved_hits.astype(np.int64),
            "easiness_group": groups,
            "easiness_rank": ranks,
            "easiness_percentile": percentiles,
        }
    )
    query_group_df.to_csv(dataset_dir / f"{stem}__k{int(args.k)}__query_groups.csv", index=False)
    all_query_groups.append(query_group_df)

    sweep_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for ef in args.efs:
        route_efs = tuple(int(value) for value in getattr(policy, "route_efs_by_ef", {}).get(int(ef), ()))
        gammas = tuple(float(value) for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(int(ef), ()))
        for group_name in args.groups:
            mask = groups == group_name
            group_test = np.ascontiguousarray(test[mask])
            group_gt = np.ascontiguousarray(eval_gt[mask])
            vanilla = benchmark_vanilla(index, group_test, group_gt, ef=int(ef), k=int(args.k), args=args)
            vanilla_recall = float(vanilla["recall"])
            vanilla_latency = float(vanilla["latency_per_query_mean_ms"])
            base_common = {
                "dataset": stem,
                "dataset_file": dataset,
                "k": int(args.k),
                "groundtruth_k": gt_k,
                "evaluation_gt_source": eval_gt_label,
                "pseudo_gt_ef": int(args.pseudo_gt_ef),
                "group_def_ef": int(args.group_def_ef),
                "group_tie_breaker": "first_final_recall_step",
                "first_step_tie_break_available": True,
                "easiness_group": group_name,
                "group_query_count": int(mask.sum()),
                "ef": int(ef),
                "offline_num_threads": int(args.offline_num_threads),
                "online_num_threads": int(args.online_num_threads),
                "num_threads": int(args.online_num_threads),
                "warmup_runs": int(args.warmup_runs),
                "measured_runs": int(args.measured_runs),
                "M": int(args.param_m),
                "efConstruction": int(args.ef_construction),
                "query_method": "adaptive-light",
                "num_calibration_queries": int(args.num_calibration_queries),
                "calibration_ef_values": format_int_signature(args.calibration_efs),
                "calibration_lid_pool_count": int(len(lid_df)),
                "calibration_lid_pool_wall_s": float(lid_wall_s),
                "threshold_calibration_wall_s": float(threshold_wall_s),
                "offline_calibration_wall_s": float(lid_wall_s + threshold_wall_s),
                "dataset_load_wall_s": float(dataset_load_wall_s),
                "index_load_wall_s": float(index_load_wall_s),
                "drilldown_pseudo_wall_s": float(pseudo_wall_s),
                "mixed_threshold_mode": str(args.mixed_threshold_mode),
                "mixed_bucket_count": int(args.mixed_bucket_count),
                "route_signature": format_int_signature(route_efs + (int(ef),)),
                "bucket_gamma_signature": format_float_signature(gammas),
                "early_stop_ratio": float(tau_by_ef[int(ef)]),
                "stop_config_source": str(policy.source_label),
                "cache_path": str(cache_path),
                "calibration_cache_status": str(cache_status or ""),
                "classify_start": int(args.classify_start),
                "classify_end": int(args.classify_end),
                "cfr_ema_decay": float(args.cfr_ema_decay),
                "pair_gap": int(args.pair_gap),
            }
            ours = benchmark_sage(
                index,
                group_test,
                group_gt,
                ef=int(ef),
                k=int(args.k),
                tau=float(tau_by_ef[int(ef)]),
                gammas=gammas,
                args=args,
            )
            ours_recall = float(ours["recall"])
            ours_latency = float(ours["latency_per_query_mean_ms"])
            for method, metrics in (("Vanilla", vanilla), ("Ours", ours)):
                row = {
                    **base_common,
                    "method": method,
                    "enable_stop": method == "Ours",
                    "recall": float(metrics["recall"]),
                    "qps": float(metrics["qps"]),
                    "adaptive_max_dist_mean": float(metrics.get("adaptive_max_dist_mean", np.nan)),
                    "stop_count": np.nan,
                    "reduced_steps_mean": np.nan,
                    "reduced_steps_max": np.nan,
                    "query_count": int(metrics["query_count"]),
                    "batch_latency_mean_ms": float(metrics["batch_latency_mean_ms"]),
                    "batch_latency_p50_ms": float(metrics["batch_latency_p50_ms"]),
                    "batch_latency_p95_ms": float(metrics["batch_latency_p95_ms"]),
                    "batch_latency_min_ms": float(metrics["batch_latency_min_ms"]),
                    "batch_latency_max_ms": float(metrics["batch_latency_max_ms"]),
                    "latency_per_query_mean_ms": float(metrics["latency_per_query_mean_ms"]),
                    "distance_computations_mean": float(metrics.get("distance_computations_mean", np.nan)),
                    "distance_computations_per_query_mean": float(metrics.get("distance_computations_per_query_mean", np.nan)),
                    "distance_computations_min": float(metrics.get("distance_computations_min", np.nan)),
                    "distance_computations_max": float(metrics.get("distance_computations_max", np.nan)),
                    "hnsw_hops_mean": float(metrics.get("hnsw_hops_mean", np.nan)),
                    "hnsw_hops_per_query_mean": float(metrics.get("hnsw_hops_per_query_mean", np.nan)),
                }
                sweep_rows.append(row)
            recall_delta_pp = (ours_recall - vanilla_recall) * 100.0
            recall_loss_pp = -recall_delta_pp
            pair_rows.append(
                {
                    **base_common,
                    "vanilla_recall": vanilla_recall,
                    "ours_recall": ours_recall,
                    "recall_delta_ours_minus_vanilla_pp": recall_delta_pp,
                    "recall_loss_vs_vanilla_pp": recall_loss_pp,
                    "recall_loss_clamped_pp": max(0.0, recall_loss_pp),
                    "vanilla_qps": float(vanilla["qps"]),
                    "ours_qps": float(ours["qps"]),
                    "qps_gain_vs_vanilla_pct": (float(ours["qps"]) / float(vanilla["qps"]) - 1.0) * 100.0 if float(vanilla["qps"]) > 0 else np.nan,
                    "latency_speedup_vs_vanilla": vanilla_latency / ours_latency if ours_latency > 0.0 else np.nan,
                    "vanilla_latency_per_query_mean_ms": vanilla_latency,
                    "ours_latency_per_query_mean_ms": ours_latency,
                    "vanilla_distance_computations_per_query_mean": float(vanilla.get("distance_computations_per_query_mean", np.nan)),
                    "ours_distance_computations_per_query_mean": float(ours.get("distance_computations_per_query_mean", np.nan)),
                    "distance_computation_reduction_vs_vanilla_pct": (
                        (1.0 - float(ours.get("distance_computations_per_query_mean", np.nan)) / float(vanilla.get("distance_computations_per_query_mean", np.nan))) * 100.0
                        if float(vanilla.get("distance_computations_per_query_mean", np.nan)) > 0.0
                        else np.nan
                    ),
                    "distance_computation_speedup_vs_vanilla": (
                        float(vanilla.get("distance_computations_per_query_mean", np.nan)) / float(ours.get("distance_computations_per_query_mean", np.nan))
                        if float(ours.get("distance_computations_per_query_mean", np.nan)) > 0.0
                        else np.nan
                    ),
                    "vanilla_distance_computations_mean": float(vanilla.get("distance_computations_mean", np.nan)),
                    "ours_distance_computations_mean": float(ours.get("distance_computations_mean", np.nan)),
                }
            )
            print(
                f"[GROUP] {stem} ef={int(ef)} {group_name} "
                f"vanilla={vanilla_recall:.5f} ours={ours_recall:.5f} "
                f"loss={recall_loss_pp:+.3f}pp speedup={pair_rows[-1]['latency_speedup_vs_vanilla']:.3f}x",
                flush=True,
            )
    sweep_df = pd.DataFrame(sweep_rows)
    pair_df = pd.DataFrame(pair_rows)
    sweep_df.to_csv(dataset_dir / f"{stem}__k{int(args.k)}__group_ef_sweep.csv", index=False)
    pair_df.to_csv(dataset_dir / f"{stem}__k{int(args.k)}__group_pair_metrics.csv", index=False)
    all_sweep.append(sweep_df)
    all_pair.append(pair_df)


def write_combined(args: argparse.Namespace, groups: list[pd.DataFrame], sweeps: list[pd.DataFrame], pairs: list[pd.DataFrame]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if groups:
        pd.concat(groups, ignore_index=True).sort_values(["dataset", "k", "qid"]).to_csv(args.output_dir / "query_groups.csv", index=False)
    if sweeps:
        pd.concat(sweeps, ignore_index=True).sort_values(["dataset", "k", "ef", "easiness_group", "method"]).to_csv(args.output_dir / "group_ef_sweep.csv", index=False)
    if pairs:
        pd.concat(pairs, ignore_index=True).sort_values(["dataset", "k", "ef", "easiness_group"]).to_csv(args.output_dir / "group_pair_metrics.csv", index=False)
    write_readme(args.output_dir, args)


def main() -> int:
    args = parse_args()
    configure_backend(args)
    configure_runtime(args)
    args.output_dir = args.output_dir.resolve()
    args.run_root = args.run_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    status = args.output_dir / "status.tsv"
    append_status(status, ["timestamp", "dataset", "status"])
    groups: list[pd.DataFrame] = []
    sweeps: list[pd.DataFrame] = []
    pairs: list[pd.DataFrame] = []
    for dataset in args.datasets:
        append_status(status, [now_label(), dataset, "start"])
        run_dataset(args, dataset, groups, sweeps, pairs)
        write_combined(args, groups, sweeps, pairs)
        append_status(status, [now_label(), dataset, "done"])
    print(f"[RESULT] {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
