#!/usr/bin/env python3
"""Compare FAISS calibration-probe pseudo GT with brute-force exact GT."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FAISS_IMPL_ROOT = EXPERIMENTS_ROOT / "faiss"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(FAISS_IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(FAISS_IMPL_ROOT))

from common.offline_calibration import compute_fixed_calibration_lid_pool  # noqa: E402
from common.projected_local_acceptable_runtime import _select_dummy_queries  # noqa: E402
import final_index_utils  # noqa: E402

DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HNSW_PLAYGROUND_ROOT", os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT)))
).expanduser()
DEFAULT_DATA_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get(
            "FAISS_INDEX_ROOT",
            str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8/index"),
        ),
    )
).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get(
        "FAISS_PYTHON_PATH",
        str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"),
    )
).expanduser()
DEFAULT_DATASETS = (
    "glove-100-angular.hdf5",
    "cohere-768-angular.hdf5",
)
DEFAULT_BASELINE_EF_SWEEP = (64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024)
BASELINE_RECOMMENDATION_EPS = 0.001


@dataclass
class DatasetAccess:
    name: str
    path: Path
    handle: h5py.File
    train_ds: h5py.Dataset
    n_train: int
    dim: int

    def close(self) -> None:
        self.handle.close()

    def read_train_rows(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids, kind="mergesort")
        sorted_ids = ids[order]
        values = np.asarray(self.train_ds[sorted_ids], dtype=np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return values[inverse]

    def iter_train_chunks(self, chunk_size: int) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, int(self.n_train), int(chunk_size)):
            stop = min(start + int(chunk_size), int(self.n_train))
            yield start, np.asarray(self.train_ds[start:stop], dtype=np.float32)


def parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list cannot be empty")
    return values


def parse_ints(value: str) -> list[int]:
    out = [int(part) for part in parse_csv(value)]
    if any(item < 1 for item in out):
        raise argparse.ArgumentTypeError("all integer values must be positive")
    return list(dict.fromkeys(out))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_csv, default=list(DEFAULT_DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--allow-system-faiss", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "probe_pseudo_gt_vs_exact_glove_cohere_faiss_p100_ef4096")
    parser.add_argument("--k-values", type=parse_ints, default=[10])
    parser.add_argument("--pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--baseline-ef-sweep", type=parse_ints, default=list(DEFAULT_BASELINE_EF_SWEEP))
    parser.add_argument("--baseline-recommendation-eps", type=float, default=BASELINE_RECOMMENDATION_EPS)
    parser.add_argument("--skip-baseline-recommendation", action="store_true")
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--internal-lid-k", type=int, default=15)
    parser.add_argument("--calibration-sample-seed", type=int, default=42)
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--exact-chunk-size", type=int, default=50000)
    parser.add_argument("--build-missing-indexes", action="store_true", default=True)
    parser.add_argument("--build-batch-size", type=int, default=int(os.environ.get("SAGE_FAISS_BUILD_BATCH_SIZE", "32768")))
    parser.add_argument("--index-build-threads", type=int, default=24)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", action="store_false", dest="continue_on_error")
    args = parser.parse_args(argv)
    for name in ("pseudo_gt_ef", "num_calibration_queries", "num_threads", "exact_chunk_size", "build_batch_size", "index_build_threads"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return args


def dataset_stem(dataset_name: str) -> str:
    return Path(str(dataset_name)).stem


def open_dataset(base_path: Path, dataset_name: str) -> DatasetAccess:
    path = Path(base_path).expanduser().resolve() / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    handle = h5py.File(path, "r")
    if "train" not in handle:
        handle.close()
        raise KeyError(f"{path} does not contain train")
    train_ds = handle["train"]
    if len(train_ds.shape) != 2:
        handle.close()
        raise ValueError(f"{path} train split must be 2-D, got shape={train_ds.shape!r}")
    return DatasetAccess(
        name=dataset_name,
        path=path,
        handle=handle,
        train_ds=train_ds,
        n_train=int(train_ds.shape[0]),
        dim=int(train_ds.shape[1]),
    )


def index_path_for(dataset_name: str, index_dir: Path, param_m: int, ef_construction: int) -> Path:
    spec = final_index_utils.resolve_dataset_spec(dataset_name)
    return Path(index_dir).expanduser().resolve() / spec.darth_name / f"{spec.darth_name}.M{int(param_m)}.efC{int(ef_construction)}.index"


def load_or_build_faiss_index(data: DatasetAccess, dataset_name: str, args: argparse.Namespace):
    final_index_utils.configure_faiss_loader(
        python_path=args.faiss_python_path,
        index_root=args.index_dir,
        allow_system_faiss=bool(args.allow_system_faiss),
    )
    spec = final_index_utils.resolve_dataset_spec(dataset_name)
    index_path = index_path_for(dataset_name, args.index_dir, int(args.param_m), int(args.ef_construction))
    index_class = final_index_utils.import_faiss_index_class()
    index = index_class(space=spec.space, dim=int(data.dim))
    if index_path.exists() and index_path.stat().st_size > 0:
        print(f"[FAISS] loading index dataset={dataset_name} space={spec.space} path={index_path}", flush=True)
        index.load_index(str(index_path), max_elements=int(data.n_train))
        index.set_num_threads(int(args.num_threads))
        count = int(index.get_current_count())
        if count != int(data.n_train):
            raise ValueError(f"Index count mismatch for {dataset_name}: index ntotal={count}, train rows={data.n_train}")
        return index, index_path, "loaded"
    if not bool(args.build_missing_indexes):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(index_path.name + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    print(
        f"[FAISS] building index dataset={dataset_name} rows={data.n_train} dim={data.dim} "
        f"M={int(args.param_m)} efC={int(args.ef_construction)} threads={int(args.index_build_threads)} path={index_path}",
        flush=True,
    )
    t0 = time.perf_counter()
    index.set_num_threads(int(args.index_build_threads))
    index.init_index(max_elements=int(data.n_train), M=int(args.param_m), ef_construction=int(args.ef_construction))
    last_report = 0
    for start, chunk in data.iter_train_chunks(int(args.build_batch_size)):
        stop = start + int(len(chunk))
        index.add_items(chunk, num_threads=int(args.index_build_threads))
        if stop - last_report >= 1_000_000 or stop == int(data.n_train):
            print(f"[FAISS] add progress {stop}/{data.n_train} elapsed_s={time.perf_counter() - t0:.1f}", flush=True)
            last_report = stop
    index.save_index(str(tmp_path))
    os.replace(tmp_path, index_path)
    index.set_num_threads(int(args.num_threads))
    count = int(index.get_current_count())
    if count != int(data.n_train):
        raise ValueError(f"Index count mismatch for {dataset_name}: index ntotal={count}, train rows={data.n_train}")
    print(f"[FAISS] built index path={index_path} wall_s={time.perf_counter() - t0:.1f}", flush=True)
    return index, index_path, "built"


def reconstruct_index_rows(index, ids: np.ndarray) -> np.ndarray:
    if hasattr(index, "_reconstruct_batch"):
        return np.asarray(index._reconstruct_batch(np.asarray(ids, dtype=np.int64)), dtype=np.float32)
    if hasattr(index, "reconstruct_batch"):
        return np.asarray(index.reconstruct_batch(np.asarray(ids, dtype=np.int64)), dtype=np.float32)
    raise RuntimeError("Loaded FAISS index does not expose vector reconstruction")


def iter_index_train_chunks(index, n_train: int, chunk_size: int) -> Iterator[tuple[int, np.ndarray]]:
    for start in range(0, int(n_train), int(chunk_size)):
        stop = min(start + int(chunk_size), int(n_train))
        ids = np.arange(start, stop, dtype=np.int64)
        yield start, reconstruct_index_rows(index, ids)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def exact_metric(dataset_name: str) -> str:
    spec = final_index_utils.resolve_dataset_spec(dataset_name)
    if spec.space == "cosine":
        return "cosine"
    if spec.space == "ip":
        return "ip"
    if spec.space == "l2":
        return "l2"
    raise ValueError(f"Unsupported FAISS space: {spec.space!r}")


def chunk_distances(queries: np.ndarray, train_chunk: np.ndarray, metric: str) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float32)
    t = np.asarray(train_chunk, dtype=np.float32)
    if metric == "cosine":
        return (1.0 - normalize_rows(q) @ normalize_rows(t).T).astype(np.float32, copy=False)
    if metric == "ip":
        return (-(q @ t.T)).astype(np.float32, copy=False)
    if metric == "l2":
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        t_sq = np.sum(t * t, axis=1, keepdims=True).T
        return np.maximum(q_sq + t_sq - 2.0 * (q @ t.T), 0.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported exact metric: {metric!r}")


def exact_train_topk(
    *,
    data: DatasetAccess,
    dataset_name: str,
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    k: int,
    chunk_size: int,
    index,
    use_index_vectors: bool,
) -> np.ndarray:
    queries = np.asarray(query_vectors, dtype=np.float32)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    best_dist = np.full((len(queries), int(k)), np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), int(k)), -1, dtype=np.int64)
    metric = exact_metric(dataset_name)
    chunk_iter = iter_index_train_chunks(index, data.n_train, chunk_size) if bool(use_index_vectors) else data.iter_train_chunks(chunk_size)
    for start, chunk in chunk_iter:
        stop = start + int(len(chunk))
        dist = chunk_distances(queries, chunk, metric)
        in_chunk = (query_ids >= start) & (query_ids < stop)
        if np.any(in_chunk):
            rows = np.flatnonzero(in_chunk)
            cols = query_ids[rows] - int(start)
            dist[rows, cols] = np.inf
        chunk_ids = np.arange(start, stop, dtype=np.int64)
        label_block = np.broadcast_to(chunk_ids.reshape(1, -1), dist.shape)
        combined_dist = np.concatenate([best_dist, dist], axis=1)
        combined_ids = np.concatenate([best_ids, label_block], axis=1)
        keep = np.argpartition(combined_dist, kth=int(k) - 1, axis=1)[:, : int(k)]
        best_dist = np.take_along_axis(combined_dist, keep, axis=1)
        best_ids = np.take_along_axis(combined_ids, keep, axis=1)
    order = np.argsort(best_dist, axis=1, kind="stable")
    return np.take_along_axis(best_ids, order, axis=1).astype(np.int64, copy=False)


def faiss_hide_node_topk(index, query_vectors: np.ndarray, query_ids: np.ndarray, *, k: int, ef: int, num_threads: int) -> np.ndarray:
    index.set_ef(int(ef))
    labels, _ = index.knn_query_hide_node(
        np.asarray(query_vectors, dtype=np.float32),
        k=int(k),
        hide_labels=np.asarray(query_ids, dtype=np.int64),
        num_threads=int(num_threads),
    )
    return np.asarray(labels, dtype=np.int64)


def hit_counts(labels: np.ndarray, exact_labels: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.int64)
    for row in range(len(labels)):
        out[row] = int(np.intersect1d(labels[row][: int(k)], exact_labels[row][: int(k)]).size)
    return out


def ef_sweep_signature(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def recommendation_from_curve(ef_values: np.ndarray, recalls: np.ndarray, eps: float) -> dict[str, object]:
    recalls = np.asarray(recalls, dtype=np.float64)
    ef_values = np.asarray(ef_values, dtype=np.int64)
    cumulative = np.maximum.accumulate(recalls)
    max_cumulative = float(cumulative[-1]) if cumulative.size else float("nan")
    remaining = max_cumulative - cumulative
    previous_gain = np.empty_like(cumulative)
    previous_gain[:] = np.nan
    if cumulative.size > 1:
        previous_gain[1:] = cumulative[1:] - cumulative[:-1]
    eligible = np.flatnonzero(remaining <= float(eps))
    selected_idx = int(eligible[0]) if eligible.size else int(len(ef_values) - 1)
    return {
        "recommended_ef": int(ef_values[selected_idx]),
        "recommended_recall": float(recalls[selected_idx]),
        "recommended_cumulative_recall": float(cumulative[selected_idx]),
        "max_cumulative_recall": max_cumulative,
        "remaining_cumulative_recall_gain": float(remaining[selected_idx]),
        "previous_step_cumulative_gain": float(previous_gain[selected_idx]),
    }


def compute_baseline_recommendation(
    *,
    index,
    dataset: str,
    k: int,
    query_vectors: np.ndarray,
    query_ids: np.ndarray,
    pseudo_labels: np.ndarray,
    exact_labels: np.ndarray,
    ef_values: Sequence[int],
    num_threads: int,
    eps: float,
) -> tuple[pd.DataFrame, dict[str, object], float]:
    valid_efs = sorted({int(ef) for ef in ef_values if int(ef) >= int(k)})
    if not valid_efs:
        raise ValueError(f"no baseline efSearch values >= k={int(k)}")
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    for ef in valid_efs:
        labels = faiss_hide_node_topk(index, query_vectors, query_ids, k=int(k), ef=int(ef), num_threads=int(num_threads))
        recall_vs_pseudo = hit_counts(labels, pseudo_labels, int(k)).astype(np.float64) / float(k)
        recall_vs_exact = hit_counts(labels, exact_labels, int(k)).astype(np.float64) / float(k)
        delta = recall_vs_pseudo - recall_vs_exact
        rows.append(
            {
                "dataset": dataset_stem(dataset),
                "dataset_file": dataset,
                "k": int(k),
                "ef": int(ef),
                "baseline_recall_vs_pseudo_gt": float(np.mean(recall_vs_pseudo)),
                "baseline_recall_vs_exact_gt": float(np.mean(recall_vs_exact)),
                "mean_query_recall_delta_pseudo_minus_exact": float(np.mean(delta)),
                "mean_abs_query_recall_delta": float(np.mean(np.abs(delta))),
                "max_abs_query_recall_delta": float(np.max(np.abs(delta))),
                "query_recall_match_rate": float(np.mean(np.isclose(recall_vs_pseudo, recall_vs_exact))),
                "recommendation_eps": float(eps),
            }
        )
    curve_df = pd.DataFrame(rows).sort_values("ef", kind="stable").reset_index(drop=True)
    ef_array = curve_df["ef"].to_numpy(dtype=np.int64)
    pseudo_rec = recommendation_from_curve(ef_array, curve_df["baseline_recall_vs_pseudo_gt"].to_numpy(dtype=np.float64), float(eps))
    exact_rec = recommendation_from_curve(ef_array, curve_df["baseline_recall_vs_exact_gt"].to_numpy(dtype=np.float64), float(eps))
    pseudo_cumulative = np.maximum.accumulate(curve_df["baseline_recall_vs_pseudo_gt"].to_numpy(dtype=np.float64))
    exact_cumulative = np.maximum.accumulate(curve_df["baseline_recall_vs_exact_gt"].to_numpy(dtype=np.float64))
    curve_df["baseline_cumulative_recall_vs_pseudo_gt"] = pseudo_cumulative
    curve_df["baseline_cumulative_recall_vs_exact_gt"] = exact_cumulative
    curve_df["pseudo_gt_remaining_cumulative_recall_gain"] = float(pseudo_cumulative[-1]) - pseudo_cumulative
    curve_df["exact_gt_remaining_cumulative_recall_gain"] = float(exact_cumulative[-1]) - exact_cumulative
    rec_row = {
        "dataset": dataset_stem(dataset),
        "dataset_file": dataset,
        "k": int(k),
        "baseline_ef_sweep": ef_sweep_signature(valid_efs),
        "recommendation_eps": float(eps),
        "pseudo_gt_recommended_ef": int(pseudo_rec["recommended_ef"]),
        "exact_gt_recommended_ef": int(exact_rec["recommended_ef"]),
        "recommended_ef_match": int(pseudo_rec["recommended_ef"]) == int(exact_rec["recommended_ef"]),
        "pseudo_gt_recommended_recall": float(pseudo_rec["recommended_recall"]),
        "exact_gt_recommended_recall": float(exact_rec["recommended_recall"]),
        "pseudo_gt_max_cumulative_recall": float(pseudo_rec["max_cumulative_recall"]),
        "exact_gt_max_cumulative_recall": float(exact_rec["max_cumulative_recall"]),
        "pseudo_gt_remaining_cumulative_recall_gain": float(pseudo_rec["remaining_cumulative_recall_gain"]),
        "exact_gt_remaining_cumulative_recall_gain": float(exact_rec["remaining_cumulative_recall_gain"]),
        "baseline_curve_wall_s": float(time.perf_counter() - t0),
    }
    return curve_df, rec_row, float(rec_row["baseline_curve_wall_s"])


def summarize_querywise(
    *,
    dataset: str,
    k: int,
    pseudo_gt_ef: int,
    data: DatasetAccess,
    selected_df: pd.DataFrame,
    pseudo_labels: np.ndarray,
    exact_labels: np.ndarray,
    index_path: Path,
    index_status: str,
    vector_source: str,
    pseudo_wall_s: float,
    exact_wall_s: float,
    dataset_load_wall_s: float,
    index_load_wall_s: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hits = hit_counts(pseudo_labels, exact_labels, int(k))
    recall = hits.astype(np.float64) / float(k)
    exact_top1 = exact_labels[:, 0]
    pseudo_top1 = pseudo_labels[:, 0]
    top1_match = pseudo_top1 == exact_top1
    exact_top1_in_pseudo = np.array([exact_top1[i] in set(pseudo_labels[i, : int(k)].tolist()) for i in range(len(exact_top1))], dtype=bool)
    pseudo_top1_in_exact = np.array([pseudo_top1[i] in set(exact_labels[i, : int(k)].tolist()) for i in range(len(pseudo_top1))], dtype=bool)
    ordered_match = np.all(pseudo_labels[:, : int(k)] == exact_labels[:, : int(k)], axis=1)
    qdf = pd.DataFrame(
        {
            "dataset": dataset_stem(dataset),
            "dataset_file": dataset,
            "backend": "faiss",
            "vector_source": vector_source,
            "k": int(k),
            "qid": selected_df["query_id"].to_numpy(dtype=np.int64),
            "selection_rank": selected_df["selection_rank"].to_numpy(dtype=np.int64),
            "lid": selected_df["lid"].to_numpy(dtype=np.float32),
            "pseudo_gt_ef": int(pseudo_gt_ef),
            "hit_count": hits,
            "recall_at_k": recall,
            "missing_exact_count": int(k) - hits,
            "exact_top1": exact_top1,
            "pseudo_top1": pseudo_top1,
            "top1_ordered_match": top1_match,
            "exact_top1_in_pseudo_topk": exact_top1_in_pseudo,
            "pseudo_top1_in_exact_topk": pseudo_top1_in_exact,
            "ordered_topk_match": ordered_match,
        }
    )
    summary = {
        "dataset": dataset_stem(dataset),
        "dataset_file": dataset,
        "status": "ok",
        "backend": "faiss",
        "vector_source": vector_source,
        "k": int(k),
        "probe_count": int(len(qdf)),
        "num_train": int(data.n_train),
        "dim": int(data.dim),
        "metric": exact_metric(dataset),
        "pseudo_gt_ef": int(pseudo_gt_ef),
        "mean_recall_at_k": float(np.mean(recall)),
        "p10_recall_at_k": float(np.percentile(recall, 10)),
        "p50_recall_at_k": float(np.percentile(recall, 50)),
        "min_recall_at_k": float(np.min(recall)),
        "mean_missing_exact_count": float(np.mean(int(k) - hits)),
        "queries_with_any_missing": int(np.sum(hits < int(k))),
        "full_set_match_rate": float(np.mean(hits == int(k))),
        "ordered_topk_match_rate": float(np.mean(ordered_match)),
        "top1_ordered_match_rate": float(np.mean(top1_match)),
        "exact_top1_in_pseudo_topk_rate": float(np.mean(exact_top1_in_pseudo)),
        "pseudo_top1_in_exact_topk_rate": float(np.mean(pseudo_top1_in_exact)),
        "dataset_load_wall_s": float(dataset_load_wall_s),
        "index_load_wall_s": float(index_load_wall_s),
        "pseudo_gt_wall_s": float(pseudo_wall_s),
        "exact_gt_wall_s": float(exact_wall_s),
        "index_status": str(index_status),
        "index_path": str(index_path),
    }
    return qdf, summary


def write_readme(output_dir: Path, args: argparse.Namespace, datasets: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(
        "# Probe Pseudo-GT vs Exact GT\n\n"
        "This compares the calibration-probe pseudo ground truth used by SAGE offline calibration "
        "against brute-force exact train-set neighbors for the same train-node probes.\n\n"
        f"- backend: `faiss`\n"
        f"- datasets: `{','.join(datasets)}`\n"
        f"- k values: `{','.join(str(k) for k in args.k_values)}`\n"
        f"- calibration probes: `{int(args.num_calibration_queries)}`\n"
        f"- pseudo GT: FAISS hide-node search at `efSearch={int(args.pseudo_gt_ef)}`\n"
        "- exact GT: brute-force scan over the train split, with the probe node removed\n"
        f"- baseline recommendation ef sweep: `{','.join(str(int(v)) for v in args.baseline_ef_sweep)}`\n"
        f"- baseline recommendation epsilon: `{float(args.baseline_recommendation_eps):g}`\n"
        f"- index root: `{Path(args.index_dir).expanduser()}`\n"
        f"- index: `M={int(args.param_m)}`, `efConstruction={int(args.ef_construction)}`\n"
        f"- threads: `{int(args.num_threads)}`\n"
        f"- LID sample seed: `{int(args.calibration_sample_seed)}`\n"
        f"- LID trim percentiles: `{float(args.trim_low_percentile):g}` / `{float(args.trim_high_percentile):g}`\n\n"
        "Primary files:\n\n"
        "- `summary.csv`: one row per dataset/k\n"
        "- `querywise.csv`: one row per probe query/k\n"
        "- `baseline_recommended_efsearch.csv`: pseudo-GT vs exact-GT baseline efSearch recommendations\n"
        "- `baseline_recommendation_curve.csv`: baseline recall curve used by the recommendation rule\n"
        "- `<dataset>/<dataset>__k<K>__querywise.csv`: dataset-local query details\n",
        encoding="utf-8",
        newline="\n",
    )


def error_summary(dataset: str, k: int, exc: BaseException) -> dict[str, object]:
    return {
        "dataset": dataset_stem(dataset),
        "dataset_file": dataset,
        "status": "error",
        "k": int(k),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(args.num_threads))
    output_dir = Path(args.output_dir).expanduser().resolve()
    datasets = tuple(args.datasets)
    write_readme(output_dir, args, datasets)
    summary_rows: list[dict[str, object]] = []
    querywise_frames: list[pd.DataFrame] = []
    baseline_curve_frames: list[pd.DataFrame] = []
    baseline_recommended_rows: list[dict[str, object]] = []

    for dataset in datasets:
        data: DatasetAccess | None = None
        try:
            t_dataset = time.perf_counter()
            print(f"[DATASET] {dataset}", flush=True)
            data = open_dataset(args.base_path, dataset)
            dataset_load_wall_s = time.perf_counter() - t_dataset
            t_index = time.perf_counter()
            index, index_path, index_status = load_or_build_faiss_index(data, dataset, args)
            index_load_wall_s = time.perf_counter() - t_index
            for k in args.k_values:
                try:
                    print(f"[SELECT] {dataset} k={int(k)} calibration probes", flush=True)
                    lid_df = compute_fixed_calibration_lid_pool(
                        index,
                        internal_lid_k=int(args.internal_lid_k),
                        num_nodes=int(data.n_train),
                        lid_sample_seed=int(args.calibration_sample_seed),
                        num_threads=int(args.num_threads),
                        dataset_name=dataset,
                    )
                    selected_df = _select_dummy_queries(
                        index=index,
                        num_nodes=int(data.n_train),
                        lid_df=lid_df,
                        num_calibration_queries=int(args.num_calibration_queries),
                        selection_mode="quantile",
                        trim_low_percentile=float(args.trim_low_percentile),
                        trim_high_percentile=float(args.trim_high_percentile),
                    )
                    query_ids = selected_df["query_id"].to_numpy(dtype=np.int64)
                    vector_source = "dataset"
                    exact_use_index_vectors = False
                    try:
                        query_vectors = data.read_train_rows(query_ids)
                    except OSError as exc:
                        print(f"[FALLBACK] {dataset} HDF5 row read failed ({exc}); using FAISS reconstruct", flush=True)
                        query_vectors = reconstruct_index_rows(index, query_ids)
                        vector_source = "faiss_reconstruct"
                        exact_use_index_vectors = True

                    print(f"[PSEUDO] {dataset} k={int(k)} ef={int(args.pseudo_gt_ef)}", flush=True)
                    t_pseudo = time.perf_counter()
                    pseudo_labels = faiss_hide_node_topk(
                        index,
                        query_vectors,
                        query_ids,
                        k=int(k),
                        ef=int(args.pseudo_gt_ef),
                        num_threads=int(args.num_threads),
                    )
                    pseudo_wall_s = time.perf_counter() - t_pseudo

                    print(f"[EXACT] {dataset} k={int(k)} brute force", flush=True)
                    t_exact = time.perf_counter()
                    exact_labels = exact_train_topk(
                        data=data,
                        dataset_name=dataset,
                        query_vectors=query_vectors,
                        query_ids=query_ids,
                        k=int(k),
                        chunk_size=int(args.exact_chunk_size),
                        index=index,
                        use_index_vectors=bool(exact_use_index_vectors),
                    )
                    exact_wall_s = time.perf_counter() - t_exact

                    baseline_curve_df = None
                    baseline_rec_row = None
                    baseline_wall_s = 0.0
                    if not bool(args.skip_baseline_recommendation):
                        print(f"[BASELINE-REC] {dataset} k={int(k)} ef_sweep={ef_sweep_signature(args.baseline_ef_sweep)}", flush=True)
                        baseline_curve_df, baseline_rec_row, baseline_wall_s = compute_baseline_recommendation(
                            index=index,
                            dataset=dataset,
                            k=int(k),
                            query_vectors=query_vectors,
                            query_ids=query_ids,
                            pseudo_labels=pseudo_labels,
                            exact_labels=exact_labels,
                            ef_values=args.baseline_ef_sweep,
                            num_threads=int(args.num_threads),
                            eps=float(args.baseline_recommendation_eps),
                        )

                    qdf, summary = summarize_querywise(
                        dataset=dataset,
                        k=int(k),
                        pseudo_gt_ef=int(args.pseudo_gt_ef),
                        data=data,
                        selected_df=selected_df,
                        pseudo_labels=pseudo_labels,
                        exact_labels=exact_labels,
                        index_path=index_path,
                        index_status=index_status,
                        vector_source=vector_source,
                        pseudo_wall_s=pseudo_wall_s,
                        exact_wall_s=exact_wall_s,
                        dataset_load_wall_s=dataset_load_wall_s,
                        index_load_wall_s=index_load_wall_s,
                    )
                    if baseline_rec_row is not None:
                        summary.update(baseline_rec_row)
                    stem = dataset_stem(dataset)
                    dataset_dir = output_dir / stem
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    qdf.to_csv(dataset_dir / f"{stem}__k{int(k)}__querywise.csv", index=False)
                    if baseline_curve_df is not None and baseline_rec_row is not None:
                        baseline_curve_df.to_csv(dataset_dir / f"{stem}__k{int(k)}__baseline_recommendation_curve.csv", index=False)
                        pd.DataFrame([baseline_rec_row]).to_csv(dataset_dir / f"{stem}__k{int(k)}__baseline_recommended_efsearch.csv", index=False)
                        baseline_curve_frames.append(baseline_curve_df)
                        baseline_recommended_rows.append(baseline_rec_row)
                    summary_rows.append(summary)
                    querywise_frames.append(qdf)
                    rec_suffix = ""
                    if baseline_rec_row is not None:
                        rec_suffix = (
                            f" pseudo_rec_ef={int(baseline_rec_row['pseudo_gt_recommended_ef'])}"
                            f" exact_rec_ef={int(baseline_rec_row['exact_gt_recommended_ef'])}"
                            f" rec_match={bool(baseline_rec_row['recommended_ef_match'])}"
                            f" baseline_wall_s={baseline_wall_s:.1f}"
                        )
                    print(
                        f"[DONE] {dataset} k={int(k)} pseudo_exact_recall={summary['mean_recall_at_k']:.5f} "
                        f"exact_wall_s={exact_wall_s:.1f}{rec_suffix}",
                        flush=True,
                    )
                except BaseException as exc:
                    summary_rows.append(error_summary(dataset, int(k), exc))
                    print(f"[ERROR] {dataset} k={int(k)} {type(exc).__name__}: {exc}", flush=True)
                    if not bool(args.continue_on_error):
                        raise
                finally:
                    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
                    if querywise_frames:
                        pd.concat(querywise_frames, ignore_index=True).to_csv(output_dir / "querywise.csv", index=False)
                    if baseline_curve_frames:
                        pd.concat(baseline_curve_frames, ignore_index=True).to_csv(output_dir / "baseline_recommendation_curve.csv", index=False)
                    if baseline_recommended_rows:
                        pd.DataFrame(baseline_recommended_rows).to_csv(output_dir / "baseline_recommended_efsearch.csv", index=False)
        finally:
            if data is not None:
                data.close()
    return 0 if summary_rows and all(row.get("status") == "ok" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
