#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HNSWLIB = Path(os.environ.get("SAGE_HNSWLIB_EXTENSION_ROOT", str(REPO_ROOT / "hnswlib"))).expanduser()
if str(LOCAL_HNSWLIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_HNSWLIB))

import hnswlib  # noqa: E402


DEFAULT_FALSE_EASY_DIR = REPO_ROOT / "final_analysis/false_easy_analysis"
DEFAULT_DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(REPO_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(REPO_ROOT / "index"))).expanduser()
DEFAULT_OUTPUT_DIR = DEFAULT_FALSE_EASY_DIR / "first_pass_gt_spread_local_minima_20260622"

CLASSIFY_START = 4
CLASSIFY_END = 16
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "First-pass quantitative check for two false-easy hypotheses: "
            "GT dispersion and early local-minimum / wrong-basin behavior."
        )
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=DEFAULT_FALSE_EASY_DIR / "hard_false_easy_chr_ratio_margins.csv",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset stems or .hdf5 names.")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--trace-batch-size", type=int, default=32)
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--max-queries-per-cohort", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def dataset_stem(name: str) -> str:
    return str(name).replace(".hdf5", "")


def dataset_file_from_stem(stem: str) -> str:
    return str(stem) if str(stem).endswith(".hdf5") else f"{stem}.hdf5"


def resolve_space(dataset_file: str) -> str:
    low = str(dataset_file).lower()
    if any(token in low for token in ("msmarco", "msmacro", "-ip", "_ip", "dot")):
        return "ip"
    if any(token in low for token in ("angular", "cosine", "nytimes", "glove", "cohere", "openai")):
        return "cosine"
    return "l2"


def index_candidates(index_dir: Path, dataset_file: str, n_train: int, dim: int, m: int, efc: int) -> list[Path]:
    stem = dataset_stem(dataset_file)
    names = [
        f"{stem}_M{int(m)}_M{int(m)}_efC{int(efc)}_n{int(n_train)}_dim{int(dim)}",
        f"{stem}_M{int(m)}_efC{int(efc)}_n{int(n_train)}_dim{int(dim)}",
        f"{stem}_M{int(m)}_efC{int(efc)}_n{int(n_train)}_dim{int(dim)}.bin",
    ]
    return [index_dir / name for name in names]


def load_index(index_dir: Path, dataset_file: str, n_train: int, dim: int, m: int, efc: int, num_threads: int):
    space = resolve_space(dataset_file)
    candidates = index_candidates(index_dir, dataset_file, n_train, dim, m, efc)
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        raise FileNotFoundError("Could not find an HNSW index. Tried: " + ", ".join(str(p) for p in candidates))
    index = hnswlib.Index(space=space, dim=int(dim))
    index.set_num_threads(int(num_threads))
    print(f"[INDEX] load {existing}", flush=True)
    index.load_index(str(existing), max_elements=int(n_train))
    index.set_num_threads(int(num_threads))
    return index, existing


def read_rows(dataset: h5py.Dataset, ids: np.ndarray) -> np.ndarray:
    ids_arr = np.asarray(ids, dtype=np.int64)
    if ids_arr.size == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=dataset.dtype)
    unique, inverse = np.unique(ids_arr, return_inverse=True)
    values = np.asarray(dataset[unique], dtype=np.float32)
    return values[inverse]


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix[tri], dtype=np.float64)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = normalize_rows(np.atleast_2d(a))
    bn = normalize_rows(np.atleast_2d(b))
    return np.maximum(0.0, 1.0 - np.matmul(an, bn.T)).astype(np.float64, copy=False)


def index_distance_matrix(space: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b2 = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if space == "cosine":
        return cosine_distance_matrix(a2, b2)
    if space == "ip":
        return (1.0 - np.matmul(a2, b2.T)).astype(np.float64, copy=False)
    diff = a2[:, None, :] - b2[None, :, :]
    return np.sum(diff * diff, axis=2).astype(np.float64, copy=False)


def safe_quantile(values: np.ndarray, q: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def safe_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def safe_min(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals))


def safe_max(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def safe_ratio(num: float, den: float) -> float:
    if not (np.isfinite(num) and np.isfinite(den)) or abs(float(den)) <= EPS:
        return float("nan")
    return float(num) / float(den)


def component_count_from_threshold(dist_matrix: np.ndarray, threshold: float) -> int:
    n = int(dist_matrix.shape[0])
    if n == 0 or not np.isfinite(threshold):
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if float(dist_matrix[i, j]) <= float(threshold):
                union(i, j)
    return len({find(i) for i in range(n)})


def compute_gt_spread_metrics(
    *,
    dataset: str,
    cohort: str,
    qid: int,
    route: int,
    drop: float,
    chr_value: float,
    chr_ratio: float,
    first_step: float,
    query_vector: np.ndarray,
    gt_labels: np.ndarray,
    gt_vectors: np.ndarray,
    space: str,
) -> dict[str, Any]:
    k = int(len(gt_labels))
    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    gt = np.asarray(gt_vectors, dtype=np.float32)

    q_gt_idx = index_distance_matrix(space, q, gt).reshape(-1)
    gt_pair_idx = upper_triangle_values(index_distance_matrix(space, gt, gt))
    q_gt_cos = cosine_distance_matrix(q, gt).reshape(-1)
    gt_pair_cos = upper_triangle_values(cosine_distance_matrix(gt, gt))
    gt_centroid = np.mean(gt, axis=0, keepdims=True)
    gt_centroid_cos = cosine_distance_matrix(gt_centroid, gt).reshape(-1)
    gt_centroid_idx = index_distance_matrix(space, gt_centroid, gt).reshape(-1)

    qgt_idx_p10 = safe_quantile(q_gt_idx, 0.10)
    qgt_idx_p50 = safe_quantile(q_gt_idx, 0.50)
    qgt_idx_p90 = safe_quantile(q_gt_idx, 0.90)
    qgt_cos_p10 = safe_quantile(q_gt_cos, 0.10)
    qgt_cos_p50 = safe_quantile(q_gt_cos, 0.50)
    qgt_cos_p90 = safe_quantile(q_gt_cos, 0.90)
    gt_pair_idx_mean = safe_mean(gt_pair_idx)
    gt_pair_cos_mean = safe_mean(gt_pair_cos)

    return {
        "dataset": dataset,
        "qid": int(qid),
        "cohort": cohort,
        "route": int(route),
        "drop": float(drop) if np.isfinite(drop) else np.nan,
        "classify_chr_mean": float(chr_value) if np.isfinite(chr_value) else np.nan,
        "classify_chr_ratio": float(chr_ratio) if np.isfinite(chr_ratio) else np.nan,
        "feature_first_final_step": float(first_step) if np.isfinite(first_step) else np.nan,
        "gt_k": k,
        "gt_pair_index_mean": gt_pair_idx_mean,
        "gt_pair_index_p90": safe_quantile(gt_pair_idx, 0.90),
        "gt_pair_index_max": safe_max(gt_pair_idx),
        "gt_pair_index_norm_by_qgt_p90": safe_ratio(gt_pair_idx_mean, qgt_idx_p90),
        "gt_pair_cos_mean": gt_pair_cos_mean,
        "gt_pair_cos_p90": safe_quantile(gt_pair_cos, 0.90),
        "gt_pair_cos_max": safe_max(gt_pair_cos),
        "gt_pair_cos_norm_by_qgt_p90": safe_ratio(gt_pair_cos_mean, qgt_cos_p90),
        "q_gt_index_p10": qgt_idx_p10,
        "q_gt_index_p50": qgt_idx_p50,
        "q_gt_index_p90": qgt_idx_p90,
        "q_gt_index_span_p90_p10": qgt_idx_p90 - qgt_idx_p10 if np.isfinite(qgt_idx_p90) and np.isfinite(qgt_idx_p10) else np.nan,
        "q_gt_index_radius_ratio_p90_p10": safe_ratio(qgt_idx_p90, qgt_idx_p10),
        "q_gt_cos_p10": qgt_cos_p10,
        "q_gt_cos_p50": qgt_cos_p50,
        "q_gt_cos_p90": qgt_cos_p90,
        "q_gt_cos_span_p90_p10": qgt_cos_p90 - qgt_cos_p10 if np.isfinite(qgt_cos_p90) and np.isfinite(qgt_cos_p10) else np.nan,
        "q_gt_cos_radius_ratio_p90_p10": safe_ratio(qgt_cos_p90, qgt_cos_p10),
        "gt_centroid_index_radius_mean": safe_mean(gt_centroid_idx),
        "gt_centroid_cos_radius_mean": safe_mean(gt_centroid_cos),
        "gt_component_count_cos_at_qgt_p90": component_count_from_threshold(
            cosine_distance_matrix(gt, gt),
            qgt_cos_p90,
        ),
        "gt_component_count_index_at_qgt_p90": component_count_from_threshold(
            index_distance_matrix(space, gt, gt),
            qgt_idx_p90,
        ),
    }


def full_pop_count_after(step: dict[str, Any], ef: int, fallback_count: int) -> int:
    raw_count = step.get("full_pop_count_after", 0)
    try:
        count = int(raw_count)
    except Exception:
        count = 0
    if count > 0:
        return count
    rs_size = step.get("rs_size_after", step.get("rs_size", np.nan))
    try:
        if np.isfinite(float(rs_size)) and float(rs_size) >= float(ef):
            return int(fallback_count + 1)
    except Exception:
        pass
    return 0


def extract_trace_metrics(path: list[dict[str, Any]], gt_labels: np.ndarray, ef: int) -> dict[str, Any]:
    gt_set = {int(x) for x in np.asarray(gt_labels, dtype=np.int64).tolist()}
    first_gt_step = math.nan
    first_gt_fullpop = math.nan
    total_gt_popped: set[int] = set()
    before16_gt_popped: set[int] = set()
    classify_gt_popped: set[int] = set()
    classify_labels: list[int] = []
    before16_labels: list[int] = []
    classify_chr_values: list[float] = []
    classify_popped_dist: list[float] = []
    classify_furthest_dist: list[float] = []
    fallback_full_count = 0
    max_full_count = 0

    for step_idx, step in enumerate(path):
        label = int(step.get("node_label", -1))
        count = full_pop_count_after(step, ef, fallback_full_count)
        if count > 0:
            fallback_full_count = count
            max_full_count = max(max_full_count, count)
        if label in gt_set:
            total_gt_popped.add(label)
            if not np.isfinite(first_gt_step):
                first_gt_step = float(step_idx + 1)
                first_gt_fullpop = float(count) if count > 0 else 0.0
        if count > 0 and count <= CLASSIFY_END:
            before16_labels.append(label)
            if label in gt_set:
                before16_gt_popped.add(label)
        if CLASSIFY_START <= count <= CLASSIFY_END:
            classify_labels.append(label)
            if label in gt_set:
                classify_gt_popped.add(label)
            popped = step.get("popped_query_dist", step.get("dist", step.get("query_dist", step.get("internal_dist", np.nan))))
            furthest = step.get("furthest_dist", step.get("lowerBound", np.nan))
            try:
                popped_f = abs(float(popped))
                furthest_f = abs(float(furthest))
            except Exception:
                popped_f = math.nan
                furthest_f = math.nan
            if np.isfinite(popped_f):
                classify_popped_dist.append(popped_f)
            if np.isfinite(furthest_f):
                classify_furthest_dist.append(furthest_f)
            if np.isfinite(popped_f) and np.isfinite(furthest_f) and furthest_f > EPS:
                classify_chr_values.append(popped_f / furthest_f)

    return {
        "trace_path_len": int(len(path)),
        "trace_full_pop_count": int(max_full_count),
        "trace_first_gt_pop_step": first_gt_step,
        "trace_first_gt_fullpop_count": first_gt_fullpop,
        "trace_gt_pop_count_total": int(len(total_gt_popped)),
        "trace_gt_pop_count_before_fullpop16": int(len(before16_gt_popped)),
        "trace_gt_pop_count_classify_window": int(len(classify_gt_popped)),
        "trace_classify_label_count": int(len(classify_labels)),
        "trace_classify_unique_label_count": int(len(set(classify_labels))),
        "trace_before16_label_count": int(len(before16_labels)),
        "trace_before16_unique_label_count": int(len(set(before16_labels))),
        "trace_classify_raw_chr_mean": safe_mean(np.asarray(classify_chr_values, dtype=np.float64)),
        "trace_classify_popped_dist_mean": safe_mean(np.asarray(classify_popped_dist, dtype=np.float64)),
        "trace_classify_furthest_dist_mean": safe_mean(np.asarray(classify_furthest_dist, dtype=np.float64)),
        "trace_classify_labels": classify_labels,
        "trace_before16_labels": before16_labels,
    }


def compute_early_basin_metrics(
    *,
    early_vectors: np.ndarray,
    query_vector: np.ndarray,
    gt_vectors: np.ndarray,
    gt_labels: np.ndarray,
    early_labels: list[int],
    space: str,
) -> dict[str, Any]:
    labels = np.asarray(early_labels, dtype=np.int64)
    if labels.size == 0:
        return {
            "early_pair_cos_mean": np.nan,
            "early_pair_index_mean": np.nan,
            "early_q_cos_mean": np.nan,
            "early_q_index_mean": np.nan,
            "early_to_gt_cos_min_mean": np.nan,
            "early_to_gt_index_min_mean": np.nan,
            "early_to_missing_gt_cos_min_mean": np.nan,
            "early_to_missing_gt_index_min_mean": np.nan,
            "early_gt_label_overlap": 0,
            "early_non_gt_fraction": np.nan,
            "early_q_vs_gt_cos_p50_gap": np.nan,
            "early_q_vs_gt_index_p50_gap": np.nan,
        }

    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    early = np.asarray(early_vectors, dtype=np.float32)
    gt = np.asarray(gt_vectors, dtype=np.float32)
    gt_set = {int(x) for x in np.asarray(gt_labels, dtype=np.int64).tolist()}
    early_set = {int(x) for x in labels.tolist()}
    missing_mask = np.asarray([int(x) not in early_set for x in gt_labels], dtype=bool)
    missing_gt = gt[missing_mask] if np.any(missing_mask) else gt

    early_pair_cos = upper_triangle_values(cosine_distance_matrix(early, early))
    early_pair_idx = upper_triangle_values(index_distance_matrix(space, early, early))
    early_q_cos = cosine_distance_matrix(q, early).reshape(-1)
    early_q_idx = index_distance_matrix(space, q, early).reshape(-1)
    early_gt_cos = cosine_distance_matrix(early, gt)
    early_gt_idx = index_distance_matrix(space, early, gt)
    early_missing_cos = cosine_distance_matrix(early, missing_gt)
    early_missing_idx = index_distance_matrix(space, early, missing_gt)
    q_gt_cos = cosine_distance_matrix(q, gt).reshape(-1)
    q_gt_idx = index_distance_matrix(space, q, gt).reshape(-1)
    overlap = len(early_set.intersection(gt_set))

    return {
        "early_pair_cos_mean": safe_mean(early_pair_cos),
        "early_pair_index_mean": safe_mean(early_pair_idx),
        "early_q_cos_mean": safe_mean(early_q_cos),
        "early_q_index_mean": safe_mean(early_q_idx),
        "early_to_gt_cos_min_mean": safe_mean(np.min(early_gt_cos, axis=1)),
        "early_to_gt_index_min_mean": safe_mean(np.min(early_gt_idx, axis=1)),
        "early_to_missing_gt_cos_min_mean": safe_mean(np.min(early_missing_cos, axis=1)),
        "early_to_missing_gt_index_min_mean": safe_mean(np.min(early_missing_idx, axis=1)),
        "early_gt_label_overlap": int(overlap),
        "early_non_gt_fraction": 1.0 - float(overlap) / float(len(early_set)) if early_set else np.nan,
        "early_q_vs_gt_cos_p50_gap": safe_mean(early_q_cos) - safe_quantile(q_gt_cos, 0.50),
        "early_q_vs_gt_index_p50_gap": safe_mean(early_q_idx) - safe_quantile(q_gt_idx, 0.50),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def auc_positive_higher(pos: np.ndarray, neg: np.ndarray) -> float:
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return float("nan")
    values = np.concatenate([p, n])
    ranks = average_ranks(values)
    rank_sum_pos = float(np.sum(ranks[: p.size]))
    return float((rank_sum_pos - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def ks_distance(pos: np.ndarray, neg: np.ndarray) -> float:
    p = np.sort(np.asarray(pos, dtype=np.float64))
    n = np.sort(np.asarray(neg, dtype=np.float64))
    p = p[np.isfinite(p)]
    n = n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([p, n])))
    cdf_p = np.searchsorted(p, values, side="right") / float(p.size)
    cdf_n = np.searchsorted(n, values, side="right") / float(n.size)
    return float(np.max(np.abs(cdf_p - cdf_n)))


def summarize_metrics(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, cohort), group in df.groupby(["dataset", "cohort"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "cohort": cohort,
            "n": int(len(group)),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = safe_mean(values)
            row[f"{metric}_p25"] = safe_quantile(values, 0.25)
            row[f"{metric}_p50"] = safe_quantile(values, 0.50)
            row[f"{metric}_p75"] = safe_quantile(values, 0.75)
            row[f"{metric}_finite_n"] = int(np.isfinite(values).sum())
        rows.append(row)

    test_rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        fe = group[group["cohort"].eq("hard_false_easy_loss")]
        no = group[group["cohort"].eq("hard_no_positive_loss")]
        for metric in metrics:
            pos = pd.to_numeric(fe[metric], errors="coerce").to_numpy(dtype=np.float64)
            neg = pd.to_numeric(no[metric], errors="coerce").to_numpy(dtype=np.float64)
            auc = auc_positive_higher(pos, neg)
            test_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "false_easy_n": int(np.isfinite(pos).sum()),
                    "no_loss_n": int(np.isfinite(neg).sum()),
                    "false_easy_p50": safe_quantile(pos, 0.50),
                    "no_loss_p50": safe_quantile(neg, 0.50),
                    "median_diff_fe_minus_no_loss": safe_quantile(pos, 0.50) - safe_quantile(neg, 0.50),
                    "auc_fe_higher": auc,
                    "directional_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                    "direction": "FE higher" if np.isfinite(auc) and auc >= 0.5 else "FE lower",
                    "ks": ks_distance(pos, neg),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(test_rows)


def plot_metric_panels(df: pd.DataFrame, metrics: list[str], output_dir: Path) -> None:
    plot_df = df[df["cohort"].isin(["hard_false_easy_loss", "hard_no_positive_loss", "hard_full_route_loss"])].copy()
    cohorts = ["hard_false_easy_loss", "hard_no_positive_loss", "hard_full_route_loss"]
    for dataset, group in plot_df.groupby("dataset", sort=True):
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, max(2.4 * len(metrics), 8)), constrained_layout=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            series = []
            labels = []
            for cohort in cohorts:
                values = pd.to_numeric(group[group["cohort"].eq(cohort)][metric], errors="coerce")
                values = values[np.isfinite(values)]
                if len(values):
                    series.append(values.to_numpy(dtype=np.float64))
                    labels.append(cohort.replace("hard_", "").replace("_", " "))
            if series:
                ax.boxplot(series, labels=labels, showfliers=False)
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{dataset}: false-easy first-pass diagnostics", fontsize=13, fontweight="bold")
        safe_name = str(dataset).replace("/", "_")
        fig.savefig(output_dir / f"{safe_name}_first_pass_diagnostics.png", dpi=180)
        fig.savefig(output_dir / f"{safe_name}_first_pass_diagnostics.pdf")
        plt.close(fig)


def write_summary(output_dir: Path, tests: pd.DataFrame, cohort_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    focus_metrics = [
        "gt_pair_cos_norm_by_qgt_p90",
        "gt_component_count_cos_at_qgt_p90",
        "trace_first_gt_pop_step",
        "trace_gt_pop_count_before_fullpop16",
        "early_to_gt_cos_min_mean",
        "early_to_missing_gt_cos_min_mean",
        "early_pair_cos_mean",
        "early_q_vs_gt_cos_p50_gap",
    ]
    lines = [
        "# First-Pass False-Easy GT Spread / Local-Minimum Diagnostics",
        "",
        f"- cohort CSV: `{args.cohort_csv}`",
        f"- ef: `{int(args.ef)}`",
        f"- k: `{int(args.k)}`",
        f"- num threads: `{int(args.num_threads)}`",
        f"- trace batch size: `{int(args.trace_batch_size)}`",
        "",
        "## Strongest FE-vs-No-Loss Separators",
        "",
    ]
    top = tests.sort_values("directional_auc", ascending=False).head(24)
    if top.empty:
        lines.append("No test rows were generated.")
    else:
        lines.append(
            "| dataset | metric | FE p50 | no-loss p50 | median diff | direction | directional AUC | KS |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
        for row in top.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | `{row.metric}` | {row.false_easy_p50:.4g} | "
                f"{row.no_loss_p50:.4g} | {row.median_diff_fe_minus_no_loss:.4g} | "
                f"{row.direction} | {row.directional_auc:.3f} | {row.ks:.3f} |"
            )
    lines.extend(["", "## Focus Metrics", ""])
    focus = tests[tests["metric"].isin(focus_metrics)].copy()
    if not focus.empty:
        lines.append(
            "| dataset | metric | FE p50 | no-loss p50 | median diff | direction | directional AUC | KS |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
        for row in focus.sort_values(["dataset", "metric"]).itertuples(index=False):
            lines.append(
                f"| {row.dataset} | `{row.metric}` | {row.false_easy_p50:.4g} | "
                f"{row.no_loss_p50:.4g} | {row.median_diff_fe_minus_no_loss:.4g} | "
                f"{row.direction} | {row.directional_auc:.3f} | {row.ks:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `per_query_gt_spread_local_minima_metrics.csv`: one row per hard query cohort.",
            "- `cohort_metric_summary.csv`: per-dataset/cohort aggregate quantiles.",
            "- `fe_vs_no_loss_metric_tests.csv`: false-easy vs hard no-positive-loss AUC/KS comparisons.",
            "- `*_first_pass_diagnostics.png/pdf`: dataset-level boxplot panels.",
            "",
            "## Notes",
            "",
            "- GT spread is measured with both index-space distance and cosine distance; cosine metrics are comparable across cosine/IP datasets.",
            "- Local-minimum behavior is approximated from the layer-0 pop trace and early classify-window popped nodes.",
            "- Full graph shortest-path distance is intentionally not computed in this first pass because exporting all level-0 edges is too large for the 10M-scale indexes.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_sample_cohorts(df: pd.DataFrame, max_per_cohort: int, seed: int) -> pd.DataFrame:
    if int(max_per_cohort) <= 0:
        return df
    rng = np.random.default_rng(int(seed))
    parts = []
    for _, group in df.groupby(["dataset", "cohort"], sort=False):
        if len(group) <= int(max_per_cohort):
            parts.append(group)
        else:
            positions = rng.choice(len(group), size=int(max_per_cohort), replace=False)
            parts.append(group.iloc[np.sort(positions)])
    return pd.concat(parts, ignore_index=True)


def run_dataset(dataset: str, cohort_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    dataset_file = dataset_file_from_stem(dataset)
    dataset_path = Path(args.dataset_root) / dataset_file
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    print(f"[DATASET] {dataset} rows={len(cohort_df)}", flush=True)
    with h5py.File(dataset_path, "r") as handle:
        train_ds = handle["train"]
        test_ds = handle["test"]
        neighbors_ds = handle["neighbors"]
        n_train = int(train_ds.shape[0])
        dim = int(train_ds.shape[1])
        space = resolve_space(dataset_file)

        qids = cohort_df["qid"].to_numpy(dtype=np.int64)
        query_vectors = read_rows(test_ds, qids)
        gt_labels_all = np.asarray(neighbors_ds[qids, : int(args.k)], dtype=np.int64)
        gt_vectors_flat = read_rows(train_ds, gt_labels_all.reshape(-1))
        gt_vectors_all = gt_vectors_flat.reshape(len(qids), int(args.k), dim)

        rows: list[dict[str, Any]] = []
        for local_idx, source_row in enumerate(cohort_df.itertuples(index=False)):
            rows.append(
                compute_gt_spread_metrics(
                    dataset=dataset,
                    cohort=str(source_row.cohort),
                    qid=int(source_row.qid),
                    route=int(source_row.route),
                    drop=float(source_row.drop),
                    chr_value=float(source_row.chr),
                    chr_ratio=float(source_row.ratio),
                    first_step=float(source_row.first_step),
                    query_vector=query_vectors[local_idx],
                    gt_labels=gt_labels_all[local_idx],
                    gt_vectors=gt_vectors_all[local_idx],
                    space=space,
                )
            )

        if args.skip_trace:
            return pd.DataFrame(rows)

        index, _index_path = load_index(
            Path(args.index_dir),
            dataset_file,
            n_train,
            dim,
            int(args.m),
            int(args.ef_construction),
            int(args.num_threads),
        )
        trace_fn = getattr(index, "search_layer0_path_with_dist_metrics_batch", None)
        if trace_fn is None:
            trace_fn = getattr(index, "search_layer0_cfr_trace_batch", None)
        if trace_fn is None:
            raise RuntimeError("Loaded hnswlib does not expose a layer-0 trace batch API.")

        trace_metrics: list[dict[str, Any]] = []
        early_labels_by_row: list[list[int]] = []
        start_time = time.time()
        for start in range(0, len(qids), int(args.trace_batch_size)):
            end = min(start + int(args.trace_batch_size), len(qids))
            paths, _, _ = trace_fn(
                query_vectors[start:end],
                k=int(args.k),
                ef=int(args.ef),
                num_threads=int(args.num_threads),
            )
            for offset, path in enumerate(paths):
                metrics = extract_trace_metrics(path, gt_labels_all[start + offset], int(args.ef))
                early_labels_by_row.append(metrics.pop("trace_classify_labels"))
                metrics.pop("trace_before16_labels", None)
                trace_metrics.append(metrics)
            elapsed = time.time() - start_time
            print(f"[TRACE] {dataset} {end}/{len(qids)} elapsed={elapsed:.1f}s", flush=True)

        for row, metrics in zip(rows, trace_metrics):
            row.update(metrics)

        unique_early = np.unique(
            np.asarray([label for labels in early_labels_by_row for label in labels], dtype=np.int64)
        )
        early_vector_map: dict[int, np.ndarray] = {}
        if unique_early.size:
            early_vectors_unique = read_rows(train_ds, unique_early)
            early_vector_map = {int(label): early_vectors_unique[i] for i, label in enumerate(unique_early.tolist())}

        for local_idx, labels in enumerate(early_labels_by_row):
            unique_labels = list(dict.fromkeys(int(x) for x in labels))
            if unique_labels:
                early_vectors = np.stack([early_vector_map[int(label)] for label in unique_labels], axis=0)
            else:
                early_vectors = np.empty((0, dim), dtype=np.float32)
            rows[local_idx].update(
                compute_early_basin_metrics(
                    early_vectors=early_vectors,
                    query_vector=query_vectors[local_idx],
                    gt_vectors=gt_vectors_all[local_idx],
                    gt_labels=gt_labels_all[local_idx],
                    early_labels=unique_labels,
                    space=space,
                )
            )

        del index
        gc.collect()
        return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = pd.read_csv(args.cohort_csv)
    cohorts = cohorts[cohorts["cohort"].isin(["hard_false_easy_loss", "hard_no_positive_loss", "hard_full_route_loss"])].copy()
    if str(args.datasets).strip():
        wanted = {dataset_stem(token.strip()) for token in str(args.datasets).split(",") if token.strip()}
        cohorts = cohorts[cohorts["dataset"].map(dataset_stem).isin(wanted)].copy()
    cohorts = maybe_sample_cohorts(cohorts, int(args.max_queries_per_cohort), int(args.random_seed))

    metric_frames: list[pd.DataFrame] = []
    for dataset, group in cohorts.groupby("dataset", sort=True):
        dataset_metrics = run_dataset(str(dataset), group.reset_index(drop=True), args)
        dataset_metrics.to_csv(output_dir / f"{dataset}_per_query_metrics.csv", index=False)
        metric_frames.append(dataset_metrics)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics.to_csv(output_dir / "per_query_gt_spread_local_minima_metrics.csv", index=False)

    metrics = [
        "classify_chr_mean",
        "classify_chr_ratio",
        "feature_first_final_step",
        "gt_pair_index_mean",
        "gt_pair_index_norm_by_qgt_p90",
        "gt_pair_cos_mean",
        "gt_pair_cos_norm_by_qgt_p90",
        "q_gt_index_span_p90_p10",
        "q_gt_cos_span_p90_p10",
        "gt_centroid_cos_radius_mean",
        "gt_component_count_cos_at_qgt_p90",
        "trace_first_gt_pop_step",
        "trace_first_gt_fullpop_count",
        "trace_gt_pop_count_total",
        "trace_gt_pop_count_before_fullpop16",
        "trace_gt_pop_count_classify_window",
        "trace_classify_raw_chr_mean",
        "trace_classify_popped_dist_mean",
        "early_pair_cos_mean",
        "early_q_cos_mean",
        "early_to_gt_cos_min_mean",
        "early_to_missing_gt_cos_min_mean",
        "early_gt_label_overlap",
        "early_non_gt_fraction",
        "early_q_vs_gt_cos_p50_gap",
    ]
    metrics = [metric for metric in metrics if metric in all_metrics.columns]
    cohort_summary, tests = summarize_metrics(all_metrics, metrics)
    cohort_summary.to_csv(output_dir / "cohort_metric_summary.csv", index=False)
    tests.to_csv(output_dir / "fe_vs_no_loss_metric_tests.csv", index=False)

    plot_metrics = [
        "gt_pair_cos_norm_by_qgt_p90",
        "gt_component_count_cos_at_qgt_p90",
        "trace_first_gt_pop_step",
        "trace_gt_pop_count_before_fullpop16",
        "early_to_gt_cos_min_mean",
        "early_to_missing_gt_cos_min_mean",
        "early_q_vs_gt_cos_p50_gap",
    ]
    plot_metrics = [metric for metric in plot_metrics if metric in all_metrics.columns]
    plot_metric_panels(all_metrics, plot_metrics, output_dir)
    write_summary(output_dir, tests, cohort_summary, args)

    print(f"[DONE] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
