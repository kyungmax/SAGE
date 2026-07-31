#!/usr/bin/env python3
"""Run RaBitQ recall/QPS sweep with an existing index and policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import time

import h5py
import numpy as np
import pandas as pd


def find_rabitq_root() -> Path:
    configured = os.environ.get("RABITQ_REPO_DIR")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    script = Path(__file__).resolve()
    if len(script.parents) > 2:
        candidates.append(script.parents[2] / "rabitq")
        candidates.append(script.parents[2])
    if len(script.parents) > 3:
        candidates.append(script.parents[3] / "rabitq")

    for candidate in candidates:
        if (candidate / "sample/python").exists() and (candidate / "python_bindings").exists():
            return candidate.resolve()
    return script.parents[2]


REPO = find_rabitq_root()
sys.path.insert(0, str(REPO / "sample/python"))
sys.path.insert(0, str(REPO / "python_bindings"))

from utils import cluster_data, compute_recall, l2_normalize_rows  # noqa: E402

try:
    from rabitqlib import HnswIndex  # noqa: E402
except ModuleNotFoundError as exc:
    HnswIndex = None
    HNSW_IMPORT_ERROR = exc
else:
    HNSW_IMPORT_ERROR = None


TOPK = 10
TMIN_POPS = 25
PAPER_BUCKET_COUNT = 4
FIXED_LID_POOL_SIZE = 10000
DEFAULT_EFS = [64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024]
DEFAULT_RECOMMENDATION_EPS = 0.001


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


DATA_DIR = env_path("RABITQ_DATA_DIR", REPO / "datasets")
MSSPACEV_RAW = env_path("RABITQ_MSSPACEV_RAW", DATA_DIR / "spacev100m_raw/spacev100m_base.i8bin")


def configure_dataset_paths(data_dir: Path, msspacev_raw: Path) -> None:
    global DATA_DIR, MSSPACEV_RAW
    DATA_DIR = Path(data_dir).expanduser()
    MSSPACEV_RAW = Path(msspacev_raw).expanduser()
    for cfg in DATASETS.values():
        cfg["h5"] = DATA_DIR / Path(cfg["h5"]).name
    if "msspacev-100M-i8-euclidean" in DATASETS:
        DATASETS["msspacev-100M-i8-euclidean"]["raw_i8bin"] = MSSPACEV_RAW

DATASETS = {
    "agnews-mxbai-1024-euclidean": {
        "h5": DATA_DIR / "agnews-mxbai-1024-euclidean.hdf5",
        "metric": "l2",
        "normalize": False,
    },
    "cohere-768-angular": {
        "h5": DATA_DIR / "cohere-768-angular.hdf5",
        "metric": "ip",
        "normalize": True,
    },
    "msspacev-100M-i8-euclidean": {
        "h5": DATA_DIR / "msspacev-100M-i8-euclidean.hdf5",
        "raw_i8bin": MSSPACEV_RAW,
        "metric": "l2",
        "normalize": False,
    },
    "youtube-15M-angular": {
        "h5": DATA_DIR / "youtube-15M-angular.hdf5",
        "metric": "l2",
        "normalize": True,
    },
}

ALIASES = {
    "agnews": "agnews-mxbai-1024-euclidean",
    "cohere": "cohere-768-angular",
    "msspacev": "msspacev-100M-i8-euclidean",
    "youtube": "youtube-15M-angular",
}


def canonical_dataset(name: str) -> str:
    return ALIASES.get(name, name)


def require_hnsw_index():
    if HnswIndex is None:
        raise RuntimeError(
            "Could not import rabitqlib.HnswIndex. Build/install the RaBitQ Python "
            "bindings or run with the correct PYTHONPATH before executing build, "
            "calibration, or sweep steps."
        ) from HNSW_IMPORT_ERROR
    return HnswIndex


def default_index_dir(m: int, ef_construction: int) -> Path:
    configured = os.environ.get("RABITQ_INDEX_DIR")
    if configured:
        return Path(configured).expanduser()
    return REPO / "artifacts" / f"rabitq_m{int(m)}_efc{int(ef_construction)}"


def default_output_path() -> str:
    return str(env_path("RABITQ_OUT_DIR", REPO / "artifacts") / "rabitq_paper_lid_hide_node_recall_qps.csv")


def index_path(index_dir: Path, name: str, args: argparse.Namespace) -> Path:
    return index_dir / f"{name}-M{int(args.degree)}-efC{int(args.ef_construction)}-hnsw-rabitq.index"


def artifact_stem(name: str, args: argparse.Namespace) -> str:
    return (
        f"{name}-M{int(args.degree)}-efC{int(args.ef_construction)}-"
        f"adaptive-calibration-lid{int(args.lid_pool_size)}-probe{int(args.num_calibration_queries)}-s{int(args.calibration_seed)}-"
        f"hide-node-paper-bucket-{int(args.calibration_threads)}thread"
    )


def policy_path(index_dir: Path, name: str, args: argparse.Namespace) -> Path:
    return index_dir / f"{artifact_stem(name, args)}.json"


def train_meta(name: str, cfg: dict) -> tuple[int, int, str]:
    del name
    if "raw_i8bin" in cfg:
        with open(cfg["raw_i8bin"], "rb") as handle:
            header = np.fromfile(handle, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Bad i8bin header: {cfg['raw_i8bin']}")
        return int(header[0]), int(header[1]), "int8"

    with h5py.File(cfg["h5"], "r") as handle:
        ds = handle["train"]
        return int(ds.shape[0]), int(ds.shape[1]), str(ds.dtype)


def read_hdf5_train(path: Path) -> np.ndarray:
    print(f"Reading HDF5 train - {path}", flush=True)
    start = time()
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["train"], dtype=np.float32)
    print(f"Read train shape={data.shape}, dtype={data.dtype}, time={time() - start:.2f}s", flush=True)
    return np.ascontiguousarray(data, dtype=np.float32)


def read_i8bin_as_float32(path: Path) -> np.ndarray:
    print(f"Reading i8bin - {path}", flush=True)
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Bad i8bin header: {path}")
        n, dim = int(header[0]), int(header[1])
        data = np.fromfile(handle, dtype=np.int8, count=n * dim)
    if data.size != n * dim:
        raise ValueError(f"Bad i8bin payload: expected {n * dim}, got {data.size}")
    data = data.reshape(n, dim).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def load_build_data(name: str, cfg: dict) -> np.ndarray:
    if "raw_i8bin" in cfg:
        data = read_i8bin_as_float32(cfg["raw_i8bin"])
    else:
        data = read_hdf5_train(cfg["h5"])
    if cfg.get("normalize"):
        print("L2-normalizing train vectors", flush=True)
        data = l2_normalize_rows(data)
    print(f"Build data for {name}: shape={data.shape}, metric={cfg['metric']}", flush=True)
    return np.ascontiguousarray(data, dtype=np.float32)


def restore_requested_order(sorted_ids: np.ndarray, requested_ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
    order = {int(value): idx for idx, value in enumerate(sorted_ids.tolist())}
    take = [order[int(value)] for value in requested_ids.tolist()]
    return np.asarray(rows[take], dtype=np.float32)


def read_train_rows(name: str, cfg: dict, ids: np.ndarray) -> np.ndarray:
    requested = np.asarray(ids, dtype=np.int64).reshape(-1)
    sorted_ids = np.sort(requested)
    if "raw_i8bin" in cfg:
        n, dim, _ = train_meta(name, cfg)
        mmap = np.memmap(cfg["raw_i8bin"], dtype=np.int8, mode="r", offset=8, shape=(n, dim))
        rows = np.asarray(mmap[sorted_ids]).astype(np.float32)
        del mmap
    else:
        with h5py.File(cfg["h5"], "r") as handle:
            rows = np.asarray(handle["train"][sorted_ids]).astype(np.float32)
    rows = restore_requested_order(sorted_ids, requested, rows)
    if cfg.get("normalize"):
        rows = l2_normalize_rows(rows)
    return np.ascontiguousarray(rows, dtype=np.float32)


def load_queries_gt(cfg: dict, query_key: str, gt_key: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"Reading queries/GT - {cfg['h5']}:{query_key},{gt_key}", flush=True)
    with h5py.File(cfg["h5"], "r") as handle:
        queries = np.asarray(handle[query_key], dtype=np.float32)
        gt = np.asarray(handle[gt_key])
    if cfg.get("normalize"):
        print("L2-normalizing queries", flush=True)
        queries = l2_normalize_rows(queries)
    return np.ascontiguousarray(queries, dtype=np.float32), np.asarray(gt, dtype=np.int64)


def build_or_load_index(name: str, cfg: dict, args: argparse.Namespace, index_dir: Path) -> Path:
    out = index_path(index_dir, name, args)
    if out.exists() and not args.force_build:
        print(f"BUILD_SKIP existing {out}", flush=True)
        return out
    if args.skip_build:
        raise FileNotFoundError(f"Missing index and --skip-build was set: {out}")

    data = load_build_data(name, cfg)
    centroids, cluster_ids = cluster_data(data, int(args.num_clusters), cfg["metric"])
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}", flush=True)

    n, dim = data.shape
    hnsw_index = require_hnsw_index()
    idx = hnsw_index(
        dim=dim,
        max_elements=n,
        M=int(args.degree),
        ef_construction=int(args.ef_construction),
        nbits=int(args.total_bits),
        metric=cfg["metric"],
    )
    start = time()
    idx.build(
        data,
        centroids,
        cluster_ids,
        num_threads=int(args.build_threads),
        fast_quantization=bool(args.faster_quant),
    )
    print(f"Indexing time: {time() - start:.2f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(out))
    print(f"Index saved -> {out}", flush=True)
    return out


def trim_lid_outliers(lid_df: pd.DataFrame, low_percentile: float, high_percentile: float):
    low_value = float(np.percentile(lid_df["lid"].to_numpy(dtype=np.float64), low_percentile))
    high_value = float(np.percentile(lid_df["lid"].to_numpy(dtype=np.float64), high_percentile))
    trimmed = lid_df[(lid_df["lid"] >= low_value) & (lid_df["lid"] <= high_value)].copy()
    if trimmed.empty:
        raise ValueError("Outlier trimming removed all LID candidates.")
    return trimmed.sort_values(["lid", "query_id"]).reset_index(drop=True), low_value, high_value


def select_lid_representatives(
    trimmed_df: pd.DataFrame,
    num_samples: int,
    selection_mode: str,
    lid_min: float,
    lid_max: float,
) -> pd.DataFrame:
    selection_count = min(int(num_samples), len(trimmed_df))
    if selection_count <= 0:
        raise ValueError("No trimmed LID candidates available.")

    if selection_mode == "quantile":
        ordered = trimmed_df.sort_values(["lid", "query_id"]).reset_index(drop=True)
        chunks = np.array_split(np.arange(len(ordered), dtype=np.int64), selection_count)
        rows = []
        for rank, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            chosen = ordered.iloc[int(chunk[len(chunk) // 2])]
            chunk_lids = ordered.iloc[chunk]["lid"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "selection_rank": int(rank),
                    "selection_mode": "quantile",
                    "query_id": int(chosen["query_id"]),
                    "query_source": "train",
                    "lid": float(chosen["lid"]),
                    "trimmed_lid_min": float(lid_min),
                    "trimmed_lid_max": float(lid_max),
                    "selection_bucket_id": int(rank),
                    "selection_bucket_lower": float(np.min(chunk_lids)),
                    "selection_bucket_upper": float(np.max(chunk_lids)),
                    "selection_target_lid": float(np.median(chunk_lids)),
                    "selection_used_fallback": False,
                }
            )
        return pd.DataFrame(rows).sort_values("selection_rank").reset_index(drop=True)

    if selection_mode == "uniform":
        selected_rows = []
        remaining = trimmed_df.copy()
        edges = (
            np.array([lid_min, lid_max], dtype=np.float64)
            if selection_count == 1
            else np.linspace(lid_min, lid_max, selection_count + 1, dtype=np.float64)
        )
        for rank in range(selection_count):
            lower, upper = float(edges[rank]), float(edges[rank + 1])
            target = (lower + upper) / 2.0
            if rank == selection_count - 1:
                mask = (remaining["lid"] >= lower) & (remaining["lid"] <= upper)
            else:
                mask = (remaining["lid"] >= lower) & (remaining["lid"] < upper)
            pool = remaining[mask]
            fallback = pool.empty
            if fallback:
                pool = remaining
            chosen = pool.iloc[int(np.argmin(np.abs(pool["lid"].to_numpy(dtype=np.float64) - target)))]
            selected_rows.append(
                {
                    "selection_rank": int(rank),
                    "selection_mode": "uniform",
                    "query_id": int(chosen["query_id"]),
                    "query_source": "train",
                    "lid": float(chosen["lid"]),
                    "trimmed_lid_min": float(lid_min),
                    "trimmed_lid_max": float(lid_max),
                    "selection_bucket_id": int(rank),
                    "selection_bucket_lower": lower,
                    "selection_bucket_upper": upper,
                    "selection_target_lid": target,
                    "selection_used_fallback": bool(fallback),
                }
            )
            remaining = remaining[remaining["query_id"] != int(chosen["query_id"])].reset_index(drop=True)
        return pd.DataFrame(selected_rows).sort_values("selection_rank").reset_index(drop=True)

    raise ValueError(f"Unsupported selection_mode={selection_mode!r}")


def mle_lid_from_distances(distances: np.ndarray) -> float:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[values > 1e-12]
    if values.size < 2:
        return float("nan")
    values = np.sort(values)
    radius = float(values[-1])
    if radius <= 1e-12:
        return float("nan")
    denom = float(np.sum(np.log(np.maximum(values[:-1], 1e-12) / radius)))
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        return float("nan")
    return float(-float(values.size - 1) / denom)


def lid_neighbor_distances(labels: np.ndarray, dists: np.ndarray, query_ids: np.ndarray, k: int) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for row_labels, row_dists, qid in zip(labels, dists, np.asarray(query_ids, dtype=np.int64)):
        keep = np.asarray(row_labels, dtype=np.int64) != int(qid)
        rows.append(np.asarray(row_dists, dtype=np.float64)[keep][: int(k)])
    return rows


def build_lid_pool(idx: HnswIndex, name: str, cfg: dict, args: argparse.Namespace) -> pd.DataFrame:
    n, _, _ = train_meta(name, cfg)
    pool_size = min(int(args.lid_pool_size), int(n))
    rng = np.random.default_rng(int(args.calibration_seed))
    ids = np.sort(rng.choice(int(n), size=pool_size, replace=False).astype(np.int64))

    frames: list[pd.DataFrame] = []
    for start in range(0, len(ids), int(args.lid_batch_size)):
        stop = min(start + int(args.lid_batch_size), len(ids))
        batch_ids = ids[start:stop]
        vectors = read_train_rows(name, cfg, batch_ids)
        labels, dists = idx.search(
            vectors,
            k=int(args.internal_lid_k) + 1,
            ef=int(args.lid_ef),
            num_threads=int(args.calibration_threads),
        )
        local_dists = lid_neighbor_distances(labels, dists, batch_ids, int(args.internal_lid_k))
        lids = np.asarray([mle_lid_from_distances(row) for row in local_dists], dtype=np.float32)
        frames.append(
            pd.DataFrame(
                {
                    "query_id": batch_ids.astype(np.int64),
                    "query_source": "train",
                    "lid": lids,
                }
            )
        )
        print(f"  LID pool {stop}/{len(ids)}", flush=True)

    lid_df = pd.concat(frames, ignore_index=True)
    lid_df = lid_df[np.isfinite(lid_df["lid"].to_numpy(dtype=np.float64))].copy()
    if len(lid_df) < int(args.num_calibration_queries):
        raise RuntimeError(f"Usable LID pool too small: {len(lid_df)}")
    return lid_df.sort_values("query_id").reset_index(drop=True)


def select_probe_df(lid_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    trimmed, lid_min, lid_max = trim_lid_outliers(
        lid_df,
        float(args.trim_low_percentile),
        float(args.trim_high_percentile),
    )
    selected = select_lid_representatives(
        trimmed,
        int(args.num_calibration_queries),
        str(args.selection_mode),
        float(lid_min),
        float(lid_max),
    )
    selected["query_id"] = pd.to_numeric(selected["query_id"], errors="raise").astype(np.int64)
    selected["selection_rank"] = pd.to_numeric(selected["selection_rank"], errors="raise").astype(np.int64)
    return selected.sort_values("selection_rank").reset_index(drop=True)


def route_efs_for_paper_floor(selection_ef: int, bucket_count: int) -> list[int]:
    routes: list[int] = []
    for bucket_idx in range(1, int(bucket_count)):
        routed = max(1, (int(selection_ef) * bucket_idx) // int(bucket_count))
        routed = min(routed, int(selection_ef) - 1)
        if not routes or routed != routes[-1]:
            routes.append(int(routed))
    if not routes:
        raise ValueError(f"Could not build route efs for ef={selection_ef}")
    return routes


def pair_targets_for_paper_floor(route_efs: list[int]) -> list[int]:
    return [max(int(route_ef) // 2, 1) for route_ef in route_efs]


def finite_quantile(values: np.ndarray, mass: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("No finite CFR values available for calibration.")
    return float(np.quantile(finite, float(np.clip(mass, 0.0, 1.0))))


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "cfr_mean": float("nan"),
            "cfr_std": float("nan"),
            "cfr_p10": float("nan"),
            "cfr_p50": float("nan"),
            "cfr_p90": float("nan"),
            "cfr_p95": float("nan"),
        }
    return {
        "cfr_mean": float(np.mean(finite)),
        "cfr_std": float(np.std(finite)),
        "cfr_p10": float(np.quantile(finite, 0.10)),
        "cfr_p50": float(np.quantile(finite, 0.50)),
        "cfr_p90": float(np.quantile(finite, 0.90)),
        "cfr_p95": float(np.quantile(finite, 0.95)),
    }


def monotone_non_decreasing(values: list[float]) -> list[float]:
    out = []
    running = 0.0
    for value in values:
        running = max(running, float(value))
        out.append(running)
    return out


def format_int_signature(values) -> str:
    return ",".join(str(int(value)) for value in values)


def format_float_signature(values) -> str:
    return ",".join(f"{float(value):.8g}" for value in values)


def route_count_signature(routed_efs: np.ndarray) -> str:
    values, counts = np.unique(np.asarray(routed_efs, dtype=np.int64), return_counts=True)
    return ";".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def route_ef_for_cfr_ratio(
    *,
    selection_ef: int,
    k: int,
    route_efs: tuple[int, ...],
    bucket_gamma_ratios: tuple[float, ...],
    cfr_ratio: float,
) -> int:
    if not np.isfinite(cfr_ratio):
        return int(selection_ef)
    for route_ef, gamma in zip(route_efs, bucket_gamma_ratios):
        if float(cfr_ratio) <= float(gamma) + 1e-12:
            return max(int(k), int(route_ef))
    return int(selection_ef)


def per_query_recall(ids: np.ndarray, gt: np.ndarray, topk: int) -> np.ndarray:
    recalls = np.zeros(ids.shape[0], dtype=np.float32)
    for idx in range(ids.shape[0]):
        gt_set = set(np.asarray(gt[idx, :topk], dtype=np.int64).tolist())
        hits = 0
        for rank in range(topk):
            if int(ids[idx, rank]) in gt_set:
                hits += 1
        recalls[idx] = hits / float(topk)
    return recalls


def search_hide(
    idx: HnswIndex,
    vectors: np.ndarray,
    ids: np.ndarray,
    ef: int,
    topk: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels, dists = idx.search_adaptive_light(
        vectors,
        k=int(topk),
        ef_init=int(ef),
        enable_stop=False,
        num_threads=int(threads),
        early_stop_ratio=float("nan"),
        tmin_pops=TMIN_POPS,
        ef_max=int(ef),
        hide_labels=np.asarray(ids, dtype=np.uint32),
    )
    return np.asarray(labels, dtype=np.int64), np.asarray(dists, dtype=np.float32)


def collect_cfr_hide(
    idx: HnswIndex,
    vectors: np.ndarray,
    ids: np.ndarray,
    ef: int,
    topk: int,
    threads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels, _, stats = idx.search_adaptive_light_with_stats(
        vectors,
        k=int(topk),
        ef_init=int(ef),
        enable_stop=False,
        num_threads=int(threads),
        early_stop_ratio=-1.0,
        tmin_pops=TMIN_POPS,
        ef_max=int(ef),
        hide_labels=np.asarray(ids, dtype=np.uint32),
    )
    return np.asarray(labels, dtype=np.int64), {key: np.asarray(value) for key, value in stats.items()}


def records_for_json(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, np.integer):
                cleaned[key] = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
                cleaned[key] = None if not np.isfinite(value) else value
            elif isinstance(value, float):
                cleaned[key] = None if not np.isfinite(value) else value
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def build_policy(
    idx: HnswIndex,
    name: str,
    probe_vectors: np.ndarray,
    probe_ids: np.ndarray,
    selected_df: pd.DataFrame,
    lid_pool_count: int,
    args: argparse.Namespace,
):
    print(f"Building hide-node pseudo-GT with ef={int(args.gt_ef)}", flush=True)
    gt, _ = search_hide(idx, probe_vectors, probe_ids, int(args.gt_ef), TOPK, int(args.calibration_threads))

    ef_values = sorted({int(value) for value in args.efs})
    recall_cache: dict[int, np.ndarray] = {}
    cfr_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def recall_for(ef_value: int) -> np.ndarray:
        resolved_ef = max(TOPK, int(ef_value))
        if resolved_ef not in recall_cache:
            pred, _ = search_hide(idx, probe_vectors, probe_ids, resolved_ef, TOPK, int(args.calibration_threads))
            recall_cache[resolved_ef] = per_query_recall(pred, gt, TOPK)
        return recall_cache[resolved_ef]

    recall_curve: dict[int, float] = {}
    for ef in ef_values:
        recall_curve[ef] = float(np.mean(recall_for(ef)))
        for target_ef in pair_targets_for_paper_floor(route_efs_for_paper_floor(ef, PAPER_BUCKET_COUNT)):
            recall_for(target_ef)

    policy: dict[int, dict[str, object]] = {}
    for ef in ef_values:
        routes = route_efs_for_paper_floor(ef, PAPER_BUCKET_COUNT)
        pair_targets = pair_targets_for_paper_floor(routes)
        _, stats = collect_cfr_hide(idx, probe_vectors, probe_ids, ef, TOPK, int(args.calibration_threads))
        cfr_values = np.asarray(stats["classify_cfr_mean"], dtype=np.float32)
        usable = np.isfinite(cfr_values)
        cfr_cache[ef] = (cfr_values, usable)
        if not np.any(usable):
            raise RuntimeError(f"No usable hide-node CFR values for ef={ef}")

        route_thetas: list[float] = []
        acceptable_rates: list[float] = []
        for target_ef in pair_targets:
            ok = recall_for(target_ef)[usable] + 1e-12 >= float(args.acceptable_recall)
            acceptable_rate = float(np.mean(ok))
            theta = max(finite_quantile(cfr_values[usable], acceptable_rate), 1e-6)
            route_thetas.append(theta)
            acceptable_rates.append(acceptable_rate)

        route_thetas = monotone_non_decreasing(route_thetas)
        tau = float(route_thetas[-1])
        gammas = [float(min(1.0, max(0.0, theta / max(tau, 1e-6)))) for theta in route_thetas]
        policy[ef] = {
            "tau": tau,
            "route_efs": [int(value) for value in routes],
            "pair_target_efs": [int(value) for value in pair_targets],
            "target_acceptable_rates": [float(value) for value in acceptable_rates],
            "bucket_gamma_ratios": gammas,
            "super_easy_gamma_ratio": gammas[0] if gammas else float("nan"),
            "mid_easy_upper_gamma_ratio": gammas[1] if len(gammas) > 1 else float("nan"),
            "calibration_cfr_mean": float(np.mean(cfr_values[usable])),
            "calibration_cfr_p50": float(np.quantile(cfr_values[usable], 0.50)),
            "calibration_cfr_p95": float(np.quantile(cfr_values[usable], 0.95)),
            "usable_probe_count": int(np.count_nonzero(usable)),
            "baseline_recall_at_ef": float(recall_curve[ef]),
        }
        print(
            f"ef={ef} recall={recall_curve[ef]:.6f} tau={tau:.6f} "
            f"gammas={','.join(f'{value:.6f}' for value in gammas)}",
            flush=True,
        )

    curve_rows: list[dict[str, object]] = []
    for ef in ef_values:
        full_recalls = recall_for(ef)
        routed_recalls = np.asarray(full_recalls, dtype=np.float64).copy()
        routed_efs = np.full(len(selected_df), int(ef), dtype=np.int64)
        entry = policy[ef]
        routes = tuple(int(value) for value in entry["route_efs"])
        gammas = tuple(float(value) for value in entry["bucket_gamma_ratios"])
        tau = float(entry["tau"])
        cfr_values, usable = cfr_cache[ef]
        finite_cfr_values = cfr_values[usable & np.isfinite(cfr_values)]

        if routes and gammas and len(routes) == len(gammas):
            for query_idx, cfr_value in enumerate(cfr_values):
                if not usable[query_idx] or not np.isfinite(cfr_value):
                    continue
                cfr_ratio = float(cfr_value) / max(tau, 1e-6)
                routed_ef = route_ef_for_cfr_ratio(
                    selection_ef=int(ef),
                    k=TOPK,
                    route_efs=routes,
                    bucket_gamma_ratios=gammas,
                    cfr_ratio=cfr_ratio,
                )
                routed_efs[query_idx] = int(routed_ef)
                if int(routed_ef) != int(ef):
                    routed_recalls[query_idx] = float(recall_for(routed_ef)[query_idx])

        curve_rows.append(
            {
                "dataset": name,
                "k": TOPK,
                "groundtruth_k": TOPK,
                "ef": int(ef),
                "offline_predicted_recall": float(np.mean(routed_recalls)),
                "offline_vanilla_recall": float(np.mean(full_recalls)),
                "calibration_query_count": int(len(selected_df)),
                "calibration_lid_pool_count": int(lid_pool_count),
                "usable_cfr_query_count": int(finite_cfr_values.size),
                "cfr_metric": "classify_cfr_mean",
                **finite_stats(finite_cfr_values),
                "offline_num_threads": int(args.calibration_threads),
                "mixed_threshold_mode": "paper_floor_half",
                "mixed_bucket_count": PAPER_BUCKET_COUNT,
                "route_signature": format_int_signature(routes + (int(ef),)) if routes else "",
                "bucket_gamma_signature": format_float_signature(gammas) if gammas else "",
                "routed_ef_count_signature": route_count_signature(routed_efs),
                "early_stop_ratio": tau,
                "stop_config_source": "lid_hide_node_train_probe_policy",
                "recommendation_source": "offline_calibration_proxy",
                "calibration_probe_routing": "hide_node",
            }
        )

    curve_df = pd.DataFrame(curve_rows).sort_values("ef", kind="stable").reset_index(drop=True)
    cumulative = np.maximum.accumulate(curve_df["offline_predicted_recall"].to_numpy(dtype=np.float64))
    max_cumulative = float(cumulative[-1]) if cumulative.size else float("nan")
    remaining = max_cumulative - cumulative
    previous_gain = np.empty_like(cumulative)
    previous_gain[:] = np.nan
    if cumulative.size > 1:
        previous_gain[1:] = cumulative[1:] - cumulative[:-1]
    curve_df["offline_cumulative_recall"] = cumulative
    curve_df["max_cumulative_recall"] = max_cumulative
    curve_df["remaining_cumulative_recall_gain"] = remaining
    curve_df["previous_step_cumulative_gain"] = previous_gain
    curve_df["recommendation_eps"] = float(args.recommendation_eps)

    eligible = np.flatnonzero(remaining <= float(args.recommendation_eps))
    selected_idx = int(eligible[0]) if eligible.size else int(len(curve_df) - 1)
    selected = curve_df.iloc[selected_idx]
    recommended_df = pd.DataFrame(
        [
            {
                "dataset": name,
                "k": TOPK,
                "recommended_ef": int(selected["ef"]),
                "offline_predicted_recall": float(selected["offline_predicted_recall"]),
                "offline_cumulative_recall": float(selected["offline_cumulative_recall"]),
                "max_cumulative_recall": float(selected["max_cumulative_recall"]),
                "remaining_cumulative_recall_gain": float(selected["remaining_cumulative_recall_gain"]),
                "previous_step_cumulative_gain": float(selected["previous_step_cumulative_gain"]),
                "recommendation_eps": float(args.recommendation_eps),
                "selection_rule": (
                    "first ef where max cumulative offline calibration-proxy Recall@10 minus "
                    f"current cumulative Recall@10 <= {float(args.recommendation_eps):g}"
                    if eligible.size
                    else "fallback last ef; no offline saturation point found"
                ),
                "calibration_query_count": int(len(selected_df)),
                "calibration_lid_pool_count": int(lid_pool_count),
                "recommendation_source": "offline_calibration_proxy",
                "calibration_probe_routing": "hide_node",
            }
        ]
    )
    print(
        f"offline recommended ef={int(selected['ef'])} "
        f"predicted_recall={float(selected['offline_predicted_recall']):.6f} "
        f"remaining_gain={float(selected['remaining_cumulative_recall_gain']):.6f}",
        flush=True,
    )
    return policy, recall_curve, curve_df, recommended_df


def policy_payload(
    name: str,
    idx_path: Path,
    cfg: dict,
    lid_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    policy: dict[int, dict],
    recall_curve: dict[int, float],
    offline_curve_df: pd.DataFrame,
    offline_recommended_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    n, dim, dtype = train_meta(name, cfg)
    return {
        "format": "rabitq-paper-source-lid-hide-node-adaptive-v1",
        "dataset": name,
        "index_file": str(idx_path),
        "train_file": str(cfg.get("raw_i8bin") or cfg["h5"]),
        "train_hdf5_key": "train" if "raw_i8bin" not in cfg else "",
        "normalize": bool(cfg.get("normalize")),
        "metric": cfg["metric"],
        "degree": int(args.degree),
        "ef_construction": int(args.ef_construction),
        "efs": [int(value) for value in args.efs],
        "topk": TOPK,
        "tmin_pops": TMIN_POPS,
        "calibration_source": "lid_local_neighbor_train_probe_hide_node",
        "calibration_lid_source": "local_neighbor_distance_self_removed",
        "calibration_probe_routing": "hide_node",
        "lid_pool_size_requested": int(args.lid_pool_size),
        "lid_pool_size_usable": int(len(lid_df)),
        "num_calibration_queries": int(len(selected_df)),
        "selection_mode": str(args.selection_mode),
        "trim_low_percentile": float(args.trim_low_percentile),
        "trim_high_percentile": float(args.trim_high_percentile),
        "internal_lid_k": int(args.internal_lid_k),
        "lid_ef": int(args.lid_ef),
        "calibration_seed": int(args.calibration_seed),
        "acceptable_recall": float(args.acceptable_recall),
        "gt_source": "rabitq_hnsw_hide_node",
        "gt_ef": int(args.gt_ef),
        "paper_bucket_count": PAPER_BUCKET_COUNT,
        "eval_mode": "paper_bucket",
        "calibration_threads": int(args.calibration_threads),
        "recommendation_source": "offline_calibration_proxy",
        "recommendation_eps": float(args.recommendation_eps),
        "sample_meta": {
            "train_count": int(n),
            "dim": int(dim),
            "train_dtype": dtype,
            "probe_min_id": int(selected_df["query_id"].min()),
            "probe_max_id": int(selected_df["query_id"].max()),
        },
        "baseline_recall_curve": {str(key): float(value) for key, value in sorted(recall_curve.items())},
        "offline_predicted_recall_curve": records_for_json(offline_curve_df),
        "offline_recommended_efsearch": records_for_json(offline_recommended_df),
        "policy": {str(key): value for key, value in sorted(policy.items())},
    }


def calibrate_or_load_policy(
    idx: HnswIndex,
    name: str,
    cfg: dict,
    idx_path: Path,
    args: argparse.Namespace,
    index_dir: Path,
) -> tuple[dict[int, dict], Path]:
    out = policy_path(index_dir, name, args)
    if out.exists() and not args.force_calibrate:
        print(f"CALIB_SKIP existing {out}", flush=True)
        payload = json.loads(out.read_text())
        return {int(key): value for key, value in payload["policy"].items()}, out
    if args.skip_calibrate:
        if not out.exists():
            raise FileNotFoundError(f"Missing policy and --skip-calibrate was set: {out}")
        payload = json.loads(out.read_text())
        return {int(key): value for key, value in payload["policy"].items()}, out

    start = time()
    lid_df = build_lid_pool(idx, name, cfg, args)
    print(f"LID pool usable={len(lid_df)} time={time() - start:.2f}s", flush=True)
    selected_df = select_probe_df(lid_df, args)
    probe_ids = selected_df["query_id"].to_numpy(dtype=np.int64)
    probe_vectors = read_train_rows(name, cfg, probe_ids)
    print(f"Selected probes={probe_vectors.shape}, id_range={probe_ids.min()}..{probe_ids.max()}", flush=True)

    policy, recall_curve, offline_curve_df, offline_recommended_df = build_policy(
        idx,
        name,
        probe_vectors,
        probe_ids,
        selected_df,
        len(lid_df),
        args,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            policy_payload(
                name,
                idx_path,
                cfg,
                lid_df,
                selected_df,
                policy,
                recall_curve,
                offline_curve_df,
                offline_recommended_df,
                args,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    lid_path = out.with_suffix(".lid_pool.csv")
    selected_path = out.with_suffix(".selected_probes.csv")
    offline_curve_path = out.with_suffix(".offline_predicted_recall_curve.csv")
    offline_recommended_path = out.with_suffix(".offline_recommended_efsearch.csv")
    lid_df.to_csv(lid_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    offline_curve_df.to_csv(offline_curve_path, index=False)
    offline_recommended_df.to_csv(offline_recommended_path, index=False)
    print(f"Policy saved -> {out}", flush=True)
    print(f"LID pool saved -> {lid_path}", flush=True)
    print(f"Selected probes saved -> {selected_path}", flush=True)
    print(f"Offline recall curve saved -> {offline_curve_path}", flush=True)
    print(f"Offline recommended efSearch saved -> {offline_recommended_path}", flush=True)
    return policy, out


def measure_one(
    idx: HnswIndex,
    queries: np.ndarray,
    gt: np.ndarray,
    method: str,
    ef: int,
    policy: dict[int, dict],
    args: argparse.Namespace,
) -> tuple[float, float]:
    qps_values = []
    recall_values = []
    for round_id in range(int(args.rounds)):
        start = time()
        if method == "RaBitQ":
            ids, _ = idx.search(
                queries,
                k=TOPK,
                ef=int(ef),
                num_threads=int(args.query_threads),
            )
        elif method == "RaBitQ+SAGE":
            p = policy[int(ef)]
            ids, _ = idx.search_adaptive_light(
                queries,
                k=TOPK,
                ef_init=int(ef),
                enable_stop=True,
                num_threads=int(args.query_threads),
                early_stop_ratio=float(p["tau"]),
                tmin_pops=TMIN_POPS,
                paper_bucket_mode=True,
                paper_bucket_count=PAPER_BUCKET_COUNT,
                bucket_gamma_ratios=[float(value) for value in p["bucket_gamma_ratios"]],
            )
        else:
            raise ValueError(f"Unsupported method={method!r}")
        elapsed = time() - start
        qps = len(queries) / elapsed
        recall = compute_recall(np.asarray(ids), gt, TOPK)
        qps_values.append(qps)
        recall_values.append(recall)
        print(
            f"  {method} ef={ef} round={round_id + 1}/{int(args.rounds)} "
            f"recall={recall:.6f} qps={qps:.2f}",
            flush=True,
        )
    return float(np.mean(recall_values)), float(np.mean(qps_values))


def run_sweep(
    idx: HnswIndex,
    name: str,
    cfg: dict,
    idx_path: Path,
    policy: dict[int, dict],
    policy_file: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    queries, gt = load_queries_gt(cfg, args.query_key, args.gt_key)
    print(f"Queries={queries.shape}, GT={gt.shape}, index dim={idx.dim}", flush=True)
    rows: list[dict[str, object]] = []
    for ef in [int(value) for value in args.efs]:
        if ef not in policy:
            print(f"  skip ef={ef}: not in policy", flush=True)
            continue
        for method in ("RaBitQ", "RaBitQ+SAGE"):
            recall, qps = measure_one(idx, queries, gt, method, ef, policy, args)
            row = {
                "dataset": name,
                "method": method,
                "ef": int(ef),
                "recall": recall,
                "qps": qps,
                "latency_per_query_ms": 1000.0 / qps,
                "build_threads": int(args.build_threads),
                "calibration_threads": int(args.calibration_threads),
                "query_threads": int(args.query_threads),
                "rounds": int(args.rounds),
                "index_path": str(idx_path),
                "policy_path": str(policy_file) if method == "RaBitQ+SAGE" else "",
                "calibration_probe_routing": "hide_node" if method == "RaBitQ+SAGE" else "",
                "calibration_lid_source": "local_neighbor_distance_self_removed" if method == "RaBitQ+SAGE" else "",
            }
            rows.append(row)
            print(f"{name},{method},ef={ef},recall={recall:.6f},qps={qps:.2f}", flush=True)
    return rows


def write_sweep_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No sweep rows to write.", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "ef",
        "recall",
        "qps",
        "latency_per_query_ms",
        "build_threads",
        "calibration_threads",
        "query_threads",
        "rounds",
        "index_path",
        "policy_path",
        "calibration_probe_routing",
        "calibration_lid_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV saved -> {path}", flush=True)


def process_dataset(raw_name: str, args: argparse.Namespace) -> list[dict[str, object]]:
    name = canonical_dataset(raw_name)
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {raw_name}")
    cfg = DATASETS[name]
    index_dir = Path(args.index_dir) if args.index_dir else default_index_dir(args.degree, args.ef_construction)

    print(f"\n=== {name} ===", flush=True)
    idx_path = build_or_load_index(name, cfg, args, index_dir)
    idx = require_hnsw_index().load(str(idx_path))
    policy, pol_path = calibrate_or_load_policy(idx, name, cfg, idx_path, args, index_dir)
    if args.skip_sweep:
        return []
    return run_sweep(idx, name, cfg, idx_path, policy, pol_path, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RaBitQ, calibrate LID hide-node adaptive policy, and run Recall/QPS sweeps."
    )
    parser.add_argument("--datasets", nargs="+", default=["agnews-mxbai-1024-euclidean"])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--msspacev-raw", default=str(MSSPACEV_RAW))
    parser.add_argument("--index-dir", default="")
    parser.add_argument("--degree", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--total-bits", type=int, default=8)
    parser.add_argument("--num-clusters", type=int, default=128)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--calibration-threads", type=int, default=24)
    parser.add_argument("--query-threads", type=int, default=24)
    parser.add_argument("--efs", type=int, nargs="+", default=DEFAULT_EFS)
    parser.add_argument("--lid-pool-size", type=int, default=FIXED_LID_POOL_SIZE)
    parser.add_argument("--lid-batch-size", type=int, default=256)
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--selection-mode", choices=("quantile", "uniform"), default="quantile")
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--internal-lid-k", type=int, default=15)
    parser.add_argument("--lid-ef", type=int, default=4096)
    parser.add_argument("--gt-ef", type=int, default=4096)
    parser.add_argument("--acceptable-recall", type=float, default=1.0)
    parser.add_argument("--recommendation-eps", type=float, default=DEFAULT_RECOMMENDATION_EPS)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--query-key", default="test")
    parser.add_argument("--gt-key", default="neighbors")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", default=default_output_path())
    parser.add_argument("--faster-quant", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-calibrate", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-calibrate", action="store_true")
    args = parser.parse_args()
    if int(args.num_calibration_queries) <= 0:
        raise ValueError("--num-calibration-queries must be positive.")
    if int(args.internal_lid_k) < 2:
        raise ValueError("--internal-lid-k must be at least 2.")
    if float(args.trim_low_percentile) >= float(args.trim_high_percentile):
        raise ValueError("--trim-low-percentile must be smaller than --trim-high-percentile.")
    configure_dataset_paths(Path(args.data_dir), Path(args.msspacev_raw))
    return args


def main() -> None:
    args = parse_args()
    args.skip_build = True
    args.skip_calibrate = True
    args.skip_sweep = False
    args.force_build = False
    args.force_calibrate = False

    all_rows: list[dict[str, object]] = []
    for raw_name in args.datasets:
        all_rows.extend(process_dataset(raw_name, args))
    write_sweep_csv(Path(args.out), all_rows)


if __name__ == "__main__":
    main()
