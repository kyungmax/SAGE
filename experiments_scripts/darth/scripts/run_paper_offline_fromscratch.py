#!/usr/bin/env python3
"""Run DARTH paper-style offline calibration from fresh artifacts.

The measured stages are:
  LVec:  write DARTH fvecs inputs from source HDF5.
  GT:    build a fresh brute-force index and write ground-truth files.
  TData: generate DARTH early-stop-training observations with hnsw_test.
  Train: train the LightGBM recall predictor.
  Online: run DARTH testing with the freshly trained model.

The common HNSW index is treated as a fixed benchmark input and is only read
through --index-filepath. No previous processed data, training logs, intervals,
or predictor models are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import faiss
import hnswlib
import lightgbm as lgb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
COMMON_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        str(PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index"),
    )
).expanduser()
DEFAULT_RUN_ROOT = PROJECT_ROOT / "index/darth_m32_efc500_target095_paper_fromscratch"
DARTH_ROOT = Path(os.environ.get("SAGE_DARTH_ROOT", str(REPO_ROOT / "baselines/darth/benchmarking-darth"))).expanduser()
DARTH_BIN = Path(os.environ.get("SAGE_DARTH_BIN", str(DARTH_ROOT / "build-simd-avx512/hnsw-test/hnsw_test"))).expanduser()

FEATURE_COLUMNS = [
    "step",
    "dists",
    "inserts",
    "first_nn_dist",
    "nn_dist",
    "furthest_dist",
    "avg_dist",
    "variance",
    "percentile_25",
    "percentile_50",
    "percentile_75",
]
TARGET_COLUMN = "r"
ONLINE_METRIC_RE = re.compile(
    r"Index\[M=(?P<m>\d+), efC=(?P<efc>\d+), efS=(?P<efs>\d+)\]"
    r"IndexTime: (?P<index_time>[0-9.eE+-]+)s, "
    r"SearchTime: (?P<search_time>[0-9.eE+-]+)s, "
    r"TotalTime: (?P<total_time>[0-9.eE+-]+)s, "
    r"Avg_Recall@(?P<k>\d+): (?P<avg_recall>[0-9.eE+-]+), "
    r"P1_Recall@\d+: (?P<p1_recall>[0-9.eE+-]+), "
    r"P5_Recall@\d+: (?P<p5_recall>[0-9.eE+-]+)"
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    darth_name: str
    hdf5: Path
    source_metric: str
    hnsw_metric: str

    @property
    def hnsw_space(self) -> str:
        return "l2" if self.source_metric == "euclidean" else "ip"


DATASETS: dict[str, DatasetSpec] = {
    "sift-100M": DatasetSpec(
        label="sift-100M",
        darth_name="sift-100M-euclidean",
        hdf5=DATASET_ROOT / "sift-100M-euclidean.hdf5",
        source_metric="euclidean",
        hnsw_metric="l2",
    ),
    "deep-100M": DatasetSpec(
        label="deep-100M",
        darth_name="deep-100M-angular",
        hdf5=DATASET_ROOT / "deep-100M.hdf5",
        source_metric="angular",
        hnsw_metric="ip",
    ),
    "spacev-100M": DatasetSpec(
        label="spacev-100M",
        darth_name="msspacev-100M-i8-euclidean",
        hdf5=DATASET_ROOT / "msspacev-100M-i8-euclidean.hdf5",
        source_metric="euclidean",
        hnsw_metric="l2",
    ),
    "msmarco": DatasetSpec(
        label="msmarco",
        darth_name="msmarco-v1-openai-ada2-full-ip",
        hdf5=DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5",
        source_metric="ip",
        hnsw_metric="ip",
    ),
    "glove-100": DatasetSpec(
        label="glove-100",
        darth_name="glove-100-angular",
        hdf5=DATASET_ROOT / "glove-100-angular.hdf5",
        source_metric="angular",
        hnsw_metric="ip",
    ),
    "nytimes": DatasetSpec(
        label="nytimes",
        darth_name="nytimes-256-angular",
        hdf5=DATASET_ROOT / "nytimes-256-angular.hdf5",
        source_metric="angular",
        hnsw_metric="ip",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--datasets",
        default="sift-100M,deep-100M,msmarco,glove-100,nytimes",
        help="Comma-separated dataset labels.",
    )
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--online-query-num", type=int, default=1000)
    parser.add_argument("--learn-queries", type=int, default=10000)
    parser.add_argument("--validation-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--base-batch-size", type=int, default=65536)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--train-threads", type=int, default=24)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--allow-existing-run-root",
        action="store_true",
        help="Allow writing into an existing run root. Default aborts to avoid mixing artifacts.",
    )
    parser.add_argument(
        "--keep-base-after-db-load",
        action="store_true",
        help="Keep base.fvecs after hnsw_test has loaded it. This needs much more disk.",
    )
    parser.add_argument(
        "--keep-training-log",
        action="store_true",
        help="Keep the raw TData CSV after interval extraction and model training.",
    )
    parser.add_argument(
        "--keep-base-after-online",
        action="store_true",
        help="Keep base.fvecs after online testing. Default deletes it to save disk.",
    )
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def monotonic() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return float(time.perf_counter() - start)


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataset_payload(spec: DatasetSpec) -> dict:
    payload = asdict(spec)
    payload["hdf5"] = str(spec.hdf5)
    return payload


def append_fvecs_rows(fh, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    d = int(matrix.shape[1])
    dims = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    packed = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    packed[:, :4] = dims.view(np.uint8).reshape(-1, 4)
    packed[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    fh.write(packed.tobytes())


def write_fvecs_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    with path.open("wb") as fh:
        append_fvecs_rows(fh, matrix)


def write_ivecs_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.int32, order="C")
    d = int(matrix.shape[1])
    dims = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    packed = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    packed[:, :4] = dims.view(np.uint8).reshape(-1, 4)
    packed[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    path.write_bytes(packed.tobytes())


def load_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    rows = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return rows[inverse]


def normalize_for_metric(matrix: np.ndarray, source_metric: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if source_metric != "angular":
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def score_from_hnsw_distance(distances: np.ndarray, source_metric: str) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    if source_metric in {"angular", "ip"}:
        return 1.0 - distances
    if source_metric == "euclidean":
        return distances
    raise ValueError(f"unsupported metric: {source_metric}")


def index_path_for(spec: DatasetSpec, common_index_root: Path, m: int, efc: int) -> Path:
    return common_index_root / spec.darth_name / f"{spec.darth_name}.M{m}.efC{efc}.index"


def parse_labels(value: str) -> list[str]:
    labels = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(labels) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; choices={sorted(DATASETS)}")
    return labels


def query_file_rows(path: Path) -> int:
    with path.open("rb") as fh:
        dim = int.from_bytes(fh.read(4), "little", signed=True)
    return int(path.stat().st_size // ((dim + 1) * 4))


def write_base_fvecs(train_dataset: h5py.Dataset, base_path: Path, batch_size: int) -> int:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = int(train_dataset.shape[0])
    with base_path.open("wb") as fh:
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            append_fvecs_rows(fh, np.asarray(train_dataset[start:end], dtype=np.float32))
            if start == 0 or end == total_rows or end % (batch_size * 32) == 0:
                log(f"LVec base rows [{start}:{end})")
    return total_rows


def run_lvec(
    spec: DatasetSpec,
    processed_dir: Path,
    *,
    learn_queries: int,
    validation_queries: int,
    seed: int,
    base_batch_size: int,
) -> dict:
    started = monotonic()
    processed_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(spec.hdf5, "r") as h5f:
        train_dataset = h5f["train"]
        query_vectors = np.asarray(h5f["test"], dtype=np.float32)
        train_size = int(train_dataset.shape[0])
        dim = int(train_dataset.shape[1])
        learn_total = int(learn_queries + validation_queries)
        if learn_total > train_size:
            raise ValueError(f"{spec.label}: learn+validation exceeds train size")

        rng = np.random.default_rng(seed)
        sampled = rng.choice(train_size, size=learn_total, replace=False)
        learn_indices = sampled[:learn_queries]
        validation_indices = sampled[learn_queries:]
        learn_vectors = load_rows(train_dataset, learn_indices)
        validation_vectors = load_rows(train_dataset, validation_indices)

        write_fvecs_matrix(processed_dir / "learn.fvecs", learn_vectors)
        write_fvecs_matrix(processed_dir / "validation.fvecs", validation_vectors)
        write_fvecs_matrix(processed_dir / "query.fvecs", query_vectors)
        base_rows = write_base_fvecs(
            train_dataset,
            processed_dir / "base.fvecs",
            batch_size=base_batch_size,
        )

    metadata = {
        "dataset_label": spec.label,
        "darth_name": spec.darth_name,
        "dimension": dim,
        "train_vectors": base_rows,
        "learn_queries": int(learn_queries),
        "validation_queries": int(validation_queries),
        "test_queries": int(query_vectors.shape[0]),
        "seed": int(seed),
        "source_hdf5": str(spec.hdf5),
        "source_metric": spec.source_metric,
        "hnsw_metric": spec.hnsw_metric,
        "train_sample_origin": "hdf5/train",
        "query_origin": "hdf5/test",
        "from_scratch_offline_artifacts": True,
        "common_hnsw_index_is_fixed_input": True,
    }
    write_json(processed_dir / "metadata.json", metadata)
    np.save(processed_dir / "learn_indices.npy", learn_indices)
    np.save(processed_dir / "validation_indices.npy", validation_indices)
    return {
        "lvec_s": elapsed_since(started),
        "base_bytes": int((processed_dir / "base.fvecs").stat().st_size),
        "metadata": metadata,
    }



def build_faiss_flat_index(
    spec: DatasetSpec,
    train_dataset: h5py.Dataset,
    *,
    threads: int,
    batch_size: int,
) -> faiss.Index:
    dim = int(train_dataset.shape[1])
    total_rows = int(train_dataset.shape[0])
    faiss.omp_set_num_threads(int(threads))
    # Avoid FAISS BLAS path here because the local BLAS build underutilizes cores
    # for large query batches. Direct FAISS distance computation uses OpenMP/AVX.
    faiss.cvar.distance_compute_blas_threshold = 1_000_000_000
    if spec.source_metric == "euclidean":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        chunk = np.asarray(train_dataset[start:end], dtype=np.float32, order="C")
        chunk = normalize_for_metric(chunk, spec.source_metric)
        index.add(chunk)
        if start == 0 or end == total_rows or end % (batch_size * 32) == 0:
            log(f"GT indexed base rows [{start}:{end})")
    return index


def search_groundtruth(
    index: faiss.Index,
    vectors: np.ndarray,
    spec: DatasetSpec,
    *,
    gt_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    ids_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        query = normalize_for_metric(vectors[start:end], spec.source_metric)
        scores, ids = index.search(query, gt_k)
        ids_all.append(np.asarray(ids, dtype=np.int32))
        scores_all.append(np.asarray(scores, dtype=np.float32))
        log(f"GT searched rows [{start}:{end})")
    return np.vstack(ids_all), np.vstack(scores_all)

def read_fvecs_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        dim = int.from_bytes(fh.read(4), "little", signed=True)
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, dim + 1)
    return np.asarray(raw[:, 1:], dtype=np.float32, order="C")


def write_gt_pair(processed_dir: Path, stem: str, ids: np.ndarray, scores: np.ndarray) -> None:
    write_ivecs_matrix(processed_dir / f"{stem}.groundtruth.ivecs", ids)
    write_fvecs_matrix(processed_dir / f"{stem}.groundtruth.fvecs", scores)


def compute_scores_from_neighbors_hdf5(
    spec: DatasetSpec,
    train_dataset: h5py.Dataset,
    query_dataset: h5py.Dataset,
    neighbors: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    scores = np.empty(neighbors.shape, dtype=np.float32)
    for start in range(0, int(neighbors.shape[0]), batch_size):
        end = min(start + batch_size, int(neighbors.shape[0]))
        batch_neighbors = np.asarray(neighbors[start:end], dtype=np.int64)
        unique_ids = np.unique(batch_neighbors)
        base_vectors = load_rows(train_dataset, unique_ids)
        query_vectors = np.asarray(query_dataset[start:end], dtype=np.float32)
        if spec.source_metric == "angular":
            base_vectors = normalize_for_metric(base_vectors, spec.source_metric)
            query_vectors = normalize_for_metric(query_vectors, spec.source_metric)
        positions = np.searchsorted(unique_ids, batch_neighbors.reshape(-1)).reshape(batch_neighbors.shape)
        neighbor_vectors = base_vectors[positions]
        if spec.source_metric == "euclidean":
            diff = query_vectors[:, None, :] - neighbor_vectors
            scores[start:end] = np.einsum("bkd,bkd->bk", diff, diff, optimize=True)
        else:
            scores[start:end] = np.einsum("bd,bkd->bk", query_vectors, neighbor_vectors, optimize=True)
        if start == 0 or end == int(neighbors.shape[0]):
            log(f"GT query scores from neighbors [{start}:{end})")
    return scores


def write_query_groundtruth_from_hdf5(spec: DatasetSpec, processed_dir: Path) -> dict:
    with h5py.File(spec.hdf5, "r") as h5f:
        neighbors = np.asarray(h5f["neighbors"], dtype=np.int32)
        if spec.source_metric == "euclidean":
            if "distances_sq" in h5f or "distances" in h5f:
                dist_key = "distances_sq" if "distances_sq" in h5f else "distances"
                scores = np.asarray(h5f[dist_key], dtype=np.float32)
                score_source = f"hdf5/{dist_key}"
            else:
                scores = compute_scores_from_neighbors_hdf5(
                    spec,
                    h5f["train"],
                    h5f["test"],
                    neighbors,
                )
                score_source = "sq_l2(hdf5/test, hdf5/train[neighbors])"
        elif "distances" in h5f:
            scores = 1.0 - np.asarray(h5f["distances"], dtype=np.float32)
            score_source = "1 - hdf5/distances"
        else:
            scores = compute_scores_from_neighbors_hdf5(
                spec,
                h5f["train"],
                h5f["test"],
                neighbors,
            )
            score_source = "dot(hdf5/test, hdf5/train[neighbors])"
    write_ivecs_matrix(processed_dir / "query.groundtruth.ivecs", neighbors)
    write_fvecs_matrix(processed_dir / "query.groundtruth.fvecs", scores)
    return {
        "queries": int(neighbors.shape[0]),
        "gt_k": int(neighbors.shape[1]),
        "score_source": score_source,
    }


def run_gt(
    spec: DatasetSpec,
    processed_dir: Path,
    *,
    gt_k: int,
    threads: int,
    base_batch_size: int,
    query_batch_size: int,
) -> dict:
    started = monotonic()
    with h5py.File(spec.hdf5, "r") as h5f:
        flat_index = build_faiss_flat_index(
            spec,
            h5f["train"],
            threads=threads,
            batch_size=base_batch_size,
        )

    split_shapes = {}
    # TData uses --query-type training, so only learn GT is generated by BF search.
    for stem in ["learn"]:
        vectors = read_fvecs_matrix(processed_dir / f"{stem}.fvecs")
        gt_ids, gt_scores = search_groundtruth(
            flat_index,
            vectors,
            spec,
            gt_k=gt_k,
            batch_size=query_batch_size,
        )
        write_gt_pair(processed_dir, stem, gt_ids, gt_scores)
        split_shapes[stem] = {
            "queries": int(vectors.shape[0]),
            "gt_k": int(gt_ids.shape[1]),
        }
        del vectors, gt_ids, gt_scores
    del flat_index
    split_shapes["query"] = write_query_groundtruth_from_hdf5(spec, processed_dir)
    return {"gt_s": elapsed_since(started), "groundtruth_k": int(gt_k), "splits": split_shapes, "bf_gt_splits": ["learn"], "hdf5_gt_splits": ["query"]}


def run_tdata(
    spec: DatasetSpec,
    run_root: Path,
    processed_dir: Path,
    *,
    m: int,
    efc: int,
    efs: int,
    k: int,
    logging_interval: int,
    threads: int,
    keep_base_after_db_load: bool,
) -> dict:
    query_num = query_file_rows(processed_dir / "learn.fvecs")
    index_path = index_path_for(spec, COMMON_INDEX_ROOT, m, efc)
    if not index_path.exists():
        raise FileNotFoundError(f"missing common HNSW index: {index_path}")
    output_path = (
        run_root
        / "darth/training"
        / spec.darth_name
        / f"k{k}"
        / f"M{m}_efC{efc}_efS{efs}_qs{query_num}_li{logging_interval}.csv"
    )
    process_log_path = run_root / "logs/darth" / f"{spec.label}.tdata.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DARTH_BIN),
        "--dataset",
        spec.darth_name,
        "--M",
        str(m),
        "--efConstruction",
        str(efc),
        "--efSearch",
        str(efs),
        "--query-num",
        str(query_num),
        "--k",
        str(k),
        "--mode",
        "early-stop-training",
        "--logging-interval",
        str(logging_interval),
        "--index-filepath",
        str(index_path),
        "--dataset-dir-prefix",
        str(run_root / "darth/processed") + "/",
        "--query-type",
        "training",
        "--output",
        str(output_path),
    ]
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd = [stdbuf, "-oL", "-eL", *cmd]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    base_path = processed_dir / "base.fvecs"
    base_deleted = False
    base_bytes = int(base_path.stat().st_size) if base_path.exists() else 0
    started = monotonic()
    with process_log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            if (
                not keep_base_after_db_load
                and not base_deleted
                and "DB loaded:" in line
                and base_path.exists()
            ):
                base_path.unlink()
                base_deleted = True
                msg = f"[{now()}] deleted base.fvecs after DB load ({base_bytes} bytes)\n"
                log_fh.write(msg)
                log_fh.flush()
        returncode = proc.wait()
    tdata_s = elapsed_since(started)
    if returncode != 0:
        raise RuntimeError(f"{spec.label} TData failed rc={returncode}; see {process_log_path}")
    return {
        "tdata_s": tdata_s,
        "training_log": str(output_path),
        "training_log_bytes": int(output_path.stat().st_size),
        "process_log": str(process_log_path),
        "base_deleted_after_db_load": bool(base_deleted),
        "base_bytes": int(base_bytes),
        "cmd": cmd,
    }


def compute_first_reaching_dists(
    training_csv: Path,
    *,
    target_recall: float,
    chunksize: int,
) -> tuple[dict[int, float], int]:
    best_by_qid: dict[int, float] = {}
    total_rows = 0
    for chunk in pd.read_csv(
        training_csv,
        usecols=["qid", "dists", "r"],
        chunksize=chunksize,
    ):
        total_rows += int(len(chunk))
        filtered = chunk.loc[chunk["r"] >= target_recall, ["qid", "dists"]]
        if filtered.empty:
            continue
        chunk_best = filtered.groupby("qid", sort=False)["dists"].min()
        for qid, dists in chunk_best.items():
            qid_int = int(qid)
            dists_float = float(dists)
            current = best_by_qid.get(qid_int)
            if current is None or dists_float < current:
                best_by_qid[qid_int] = dists_float
    return best_by_qid, total_rows


def round_interval(value: float) -> int:
    return int(max(1, round(value)))


def run_interval(
    spec: DatasetSpec,
    run_root: Path,
    training_csv: Path,
    *,
    target_recall: float,
    chunksize: int,
    k: int,
    efc: int,
    efs: int,
    query_num: int,
) -> dict:
    started = monotonic()
    best_by_qid, total_rows = compute_first_reaching_dists(
        training_csv,
        target_recall=target_recall,
        chunksize=chunksize,
    )
    if not best_by_qid:
        raise ValueError(f"{spec.label}: no training queries reached target recall {target_recall}")
    dists = list(best_by_qid.values())
    avg_dists_rt = float(sum(dists) / len(dists))
    ipi = round_interval(avg_dists_rt / 2.0)
    mpi = round_interval(avg_dists_rt / 10.0)
    if mpi > ipi:
        mpi = ipi
    interval_path = (
        run_root
        / "darth/intervals"
        / f"{spec.darth_name}_k{k}_efC{efc}_efS{efs}_rt{target_recall:.2f}_qs{query_num}.json"
    )
    payload = {
        "dataset": spec.darth_name,
        "target_recall": float(target_recall),
        "avg_dists_rt": avg_dists_rt,
        "initial_prediction_interval": int(ipi),
        "min_prediction_interval": int(mpi),
        "queries_reaching_target": int(len(best_by_qid)),
        "training_rows": int(total_rows),
        "training_csv": str(training_csv),
        "interval_s": elapsed_since(started),
    }
    write_json(interval_path, payload)
    payload["interval_json"] = str(interval_path)
    return payload


def train_predictor(
    spec: DatasetSpec,
    run_root: Path,
    training_csv: Path,
    *,
    m: int,
    efc: int,
    efs: int,
    k: int,
    query_num: int,
    logging_interval: int,
    n_estimators: int,
    learning_rate: float,
    train_threads: int,
) -> dict:
    started = monotonic()
    usecols = [*FEATURE_COLUMNS, TARGET_COLUMN]
    load_started = monotonic()
    df = pd.read_csv(training_csv, usecols=usecols)
    load_s = elapsed_since(load_started)
    y = df[TARGET_COLUMN]
    x = df[FEATURE_COLUMNS]
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        random_state=42,
        n_jobs=int(train_threads),
        verbose=-1,
    )
    fit_started = monotonic()
    model.fit(x, y)
    fit_s = elapsed_since(fit_started)
    model_path = (
        run_root
        / "darth/models"
        / (
            f"{spec.darth_name}_M{m}_efC{efc}_efS{efs}_s{query_num}_k{k}"
            f"_nestim{n_estimators}_li{logging_interval}_all_feats.txt"
        )
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_started = monotonic()
    model.booster_.save_model(str(model_path))
    save_s = elapsed_since(save_started)
    feature_importances = [
        {"feature": feature, "importance": int(importance)}
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
    ]
    payload = {
        "train_s": fit_s,
        "train_fit_s": fit_s,
        "train_load_s": load_s,
        "train_save_s": save_s,
        "train_wall_s": elapsed_since(started),
        "training_rows": int(len(df)),
        "model_path": str(model_path),
        "model_bytes": int(model_path.stat().st_size),
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "feature_importances": feature_importances,
    }
    del df, x, y, model
    return payload


def ensure_base_fvecs(
    spec: DatasetSpec,
    processed_dir: Path,
    *,
    base_batch_size: int,
) -> dict:
    base_path = processed_dir / "base.fvecs"
    if base_path.exists():
        return {"online_base_restore_s": 0.0, "base_restored": False, "base_bytes": int(base_path.stat().st_size)}
    started = monotonic()
    with h5py.File(spec.hdf5, "r") as h5f:
        write_base_fvecs(h5f["train"], base_path, batch_size=base_batch_size)
    return {
        "online_base_restore_s": elapsed_since(started),
        "base_restored": True,
        "base_bytes": int(base_path.stat().st_size),
    }


def parse_online_metrics(process_log_path: Path) -> dict:
    metrics: dict[str, float | int] = {}
    if not process_log_path.exists():
        return metrics
    for line in process_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ONLINE_METRIC_RE.search(line)
        if not match:
            continue
        metrics = {
            "online_index_time_s": float(match.group("index_time")),
            "online_search_time_s": float(match.group("search_time")),
            "online_total_time_s": float(match.group("total_time")),
            "online_avg_recall": float(match.group("avg_recall")),
            "online_p1_recall": float(match.group("p1_recall")),
            "online_p5_recall": float(match.group("p5_recall")),
            "online_recall_k": int(match.group("k")),
        }
    return metrics


def summarize_online_output(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    df = pd.read_csv(output_path, usecols=["dists", "elaps_ms", "r_actual", "r_predictor_calls", "r_predictor_time_ms"])
    return {
        "online_output_rows": int(len(df)),
        "online_avg_dists": float(df["dists"].mean()),
        "online_avg_query_ms_from_csv": float(df["elaps_ms"].mean()),
        "online_avg_actual_recall_from_csv": float(df["r_actual"].mean()),
        "online_avg_predictor_calls": float(df["r_predictor_calls"].mean()),
        "online_avg_predictor_time_ms": float(df["r_predictor_time_ms"].mean()),
    }


def run_online(
    spec: DatasetSpec,
    run_root: Path,
    processed_dir: Path,
    *,
    m: int,
    efc: int,
    efs: int,
    k: int,
    target_recall: float,
    online_query_num: int,
    interval: dict,
    train: dict,
    base_batch_size: int,
    keep_base_after_online: bool,
) -> dict:
    restore = ensure_base_fvecs(spec, processed_dir, base_batch_size=base_batch_size)
    index_path = index_path_for(spec, COMMON_INDEX_ROOT, m, efc)
    output_path = run_root / "darth/results" / spec.label / "online.darth.txt"
    process_log_path = run_root / "logs/darth" / f"{spec.label}.online.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DARTH_BIN),
        "--dataset",
        spec.darth_name,
        "--M",
        str(m),
        "--efConstruction",
        str(efc),
        "--efSearch",
        str(efs),
        "--query-num",
        str(online_query_num),
        "--k",
        str(k),
        "--output",
        str(output_path),
        "--mode",
        "early-stop-testing",
        "--index-filepath",
        str(index_path),
        "--dataset-dir-prefix",
        str(run_root / "darth/processed") + "/",
        "--target-recall",
        str(target_recall),
        "--initial-prediction-interval",
        str(interval["initial_prediction_interval"]),
        "--min-prediction-interval",
        str(interval["min_prediction_interval"]),
        "--query-type",
        "testing",
        "--predictor-model-path",
        str(train["model_path"]),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    started = monotonic()
    with process_log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
    online_s = elapsed_since(started)
    if proc.returncode != 0:
        raise RuntimeError(f"{spec.label} online failed rc={proc.returncode}; see {process_log_path}")
    base_deleted_after_online = False
    if not keep_base_after_online:
        base_deleted_after_online = remove_if_exists(processed_dir / "base.fvecs")
    payload = {
        **restore,
        "online_s": online_s,
        "online_query_num": int(online_query_num),
        "online_output": str(output_path),
        "online_output_bytes": int(output_path.stat().st_size),
        "online_log": str(process_log_path),
        "online_threads": 1,
        "base_deleted_after_online": bool(base_deleted_after_online),
        "cmd": cmd,
    }
    payload.update(parse_online_metrics(process_log_path))
    payload.update(summarize_online_output(output_path))
    return payload


def remove_if_exists(path: Path) -> bool:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    return False


def append_summary_row(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "darth_name",
        "learn_queries",
        "validation_queries",
        "query_queries",
        "groundtruth_k",
        "lvec_s",
        "gt_s",
        "tdata_s",
        "train_s",
        "train_fit_s",
        "train_load_s",
        "train_save_s",
        "train_wall_s",
        "offline_total_s",
        "offline_wall_total_s",
        "interval_s",
        "training_rows",
        "training_log_bytes",
        "training_log_deleted",
        "base_deleted_after_db_load",
        "model_path",
        "interval_json",
        "online_s",
        "online_search_time_s",
        "online_avg_recall",
        "online_p1_recall",
        "online_p5_recall",
        "online_avg_dists",
        "online_avg_query_ms_from_csv",
        "online_avg_predictor_calls",
        "online_base_restore_s",
        "online_threads",
        "online_output",
    ]
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_gt_k(hdf5_path: Path, minimum_k: int) -> int:
    with h5py.File(hdf5_path, "r") as h5f:
        if "neighbors" in h5f:
            return max(int(minimum_k), int(h5f["neighbors"].shape[1]))
    return max(int(minimum_k), 100)


def run_dataset(spec: DatasetSpec, args: argparse.Namespace, run_root: Path) -> dict:
    dataset_started = monotonic()
    processed_dir = run_root / "darth/processed" / spec.darth_name
    if processed_dir.exists():
        raise FileExistsError(f"refusing to reuse existing processed dir: {processed_dir}")
    log(f"{spec.label}: LVec start")
    lvec = run_lvec(
        spec,
        processed_dir,
        learn_queries=int(args.learn_queries),
        validation_queries=int(args.validation_queries),
        seed=int(args.seed),
        base_batch_size=int(args.base_batch_size),
    )
    log(f"{spec.label}: LVec done {lvec['lvec_s']:.3f}s")

    gt_k = infer_gt_k(spec.hdf5, int(args.k))
    log(f"{spec.label}: GT start k={gt_k}")
    gt = run_gt(
        spec,
        processed_dir,
        gt_k=gt_k,
        threads=int(args.threads),
        base_batch_size=int(args.base_batch_size),
        query_batch_size=int(args.query_batch_size),
    )
    log(f"{spec.label}: GT done {gt['gt_s']:.3f}s")

    log(f"{spec.label}: TData start")
    tdata = run_tdata(
        spec,
        run_root,
        processed_dir,
        m=int(args.m),
        efc=int(args.ef_construction),
        efs=int(args.ef_search),
        k=int(args.k),
        logging_interval=int(args.logging_interval),
        threads=int(args.threads),
        keep_base_after_db_load=bool(args.keep_base_after_db_load),
    )
    log(f"{spec.label}: TData done {tdata['tdata_s']:.3f}s")

    training_csv = Path(str(tdata["training_log"]))
    query_num = query_file_rows(processed_dir / "learn.fvecs")
    log(f"{spec.label}: interval extraction start")
    interval = run_interval(
        spec,
        run_root,
        training_csv,
        target_recall=float(args.target_recall),
        chunksize=int(args.chunksize),
        k=int(args.k),
        efc=int(args.ef_construction),
        efs=int(args.ef_search),
        query_num=query_num,
    )
    log(f"{spec.label}: interval extraction done {interval['interval_s']:.3f}s")

    log(f"{spec.label}: Train start")
    train = train_predictor(
        spec,
        run_root,
        training_csv,
        m=int(args.m),
        efc=int(args.ef_construction),
        efs=int(args.ef_search),
        k=int(args.k),
        query_num=query_num,
        logging_interval=int(args.logging_interval),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        train_threads=int(args.train_threads),
    )
    log(f"{spec.label}: Train done {train['train_s']:.3f}s")

    training_log_deleted = False
    if not args.keep_training_log:
        training_log_deleted = remove_if_exists(training_csv)
        log(f"{spec.label}: deleted training log={training_log_deleted}")

    log(f"{spec.label}: Online start (1 thread)")
    online = run_online(
        spec,
        run_root,
        processed_dir,
        m=int(args.m),
        efc=int(args.ef_construction),
        efs=int(args.ef_search),
        k=int(args.k),
        target_recall=float(args.target_recall),
        online_query_num=int(args.online_query_num),
        interval=interval,
        train=train,
        base_batch_size=int(args.base_batch_size),
        keep_base_after_online=bool(args.keep_base_after_online),
    )
    log(f"{spec.label}: Online done {online['online_s']:.3f}s recall={online.get('online_avg_recall', float('nan')):.4f}")

    row = {
        "dataset": spec.label,
        "darth_name": spec.darth_name,
        "learn_queries": int(args.learn_queries),
        "validation_queries": int(args.validation_queries),
        "query_queries": int(lvec["metadata"]["test_queries"]),
        "groundtruth_k": int(gt["groundtruth_k"]),
        "lvec_s": float(lvec["lvec_s"]),
        "gt_s": float(gt["gt_s"]),
        "tdata_s": float(tdata["tdata_s"]),
        "train_s": float(train["train_s"]),
        "train_fit_s": float(train["train_fit_s"]),
        "train_load_s": float(train["train_load_s"]),
        "train_save_s": float(train["train_save_s"]),
        "train_wall_s": float(train["train_wall_s"]),
        "offline_total_s": float(lvec["lvec_s"] + gt["gt_s"] + tdata["tdata_s"] + train["train_s"]),
        "offline_wall_total_s": float(lvec["lvec_s"] + gt["gt_s"] + tdata["tdata_s"] + train["train_wall_s"]),
        "interval_s": float(interval["interval_s"]),
        "training_rows": int(train["training_rows"]),
        "training_log_bytes": int(tdata["training_log_bytes"]),
        "training_log_deleted": bool(training_log_deleted),
        "base_deleted_after_db_load": bool(tdata["base_deleted_after_db_load"]),
        "model_path": str(train["model_path"]),
        "interval_json": str(interval["interval_json"]),
        "online_s": float(online["online_s"]),
        "online_search_time_s": online.get("online_search_time_s", ""),
        "online_avg_recall": online.get("online_avg_recall", ""),
        "online_p1_recall": online.get("online_p1_recall", ""),
        "online_p5_recall": online.get("online_p5_recall", ""),
        "online_avg_dists": online.get("online_avg_dists", ""),
        "online_avg_query_ms_from_csv": online.get("online_avg_query_ms_from_csv", ""),
        "online_avg_predictor_calls": online.get("online_avg_predictor_calls", ""),
        "online_base_restore_s": online.get("online_base_restore_s", ""),
        "online_threads": int(online.get("online_threads", 1)),
        "online_output": str(online["online_output"]),
        "dataset_wall_s": elapsed_since(dataset_started),
    }
    payload = {
        "status": "ok",
        "dataset": dataset_payload(spec),
        "lvec": lvec,
        "gt": gt,
        "tdata": tdata,
        "interval": interval,
        "train": train,
        "online": online,
        "summary_row": row,
    }
    write_json(run_root / "darth/results" / spec.label / "offline.wrapper.json", payload)
    append_summary_row(run_root / "darth/results/offline_cost_summary.csv", row)
    return row


def validate_inputs(labels: Iterable[str], args: argparse.Namespace) -> None:
    if not DARTH_BIN.exists():
        raise FileNotFoundError(f"missing DARTH binary: {DARTH_BIN}")
    for label in labels:
        spec = DATASETS[label]
        if not spec.hdf5.exists():
            raise FileNotFoundError(f"missing HDF5 for {label}: {spec.hdf5}")
        index_path = index_path_for(
            spec,
            COMMON_INDEX_ROOT,
            int(args.m),
            int(args.ef_construction),
        )
        if not index_path.exists():
            raise FileNotFoundError(f"missing common HNSW index for {label}: {index_path}")


def main() -> int:
    args = parse_args()
    labels = parse_labels(args.datasets)
    validate_inputs(labels, args)
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() and not args.allow_existing_run_root:
        raise FileExistsError(
            f"run root already exists: {run_root}; use a fresh path or --allow-existing-run-root"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "RUN_MANIFEST.json",
        {
            "created_at": now(),
            "purpose": "DARTH paper-style offline costs from fresh artifacts",
            "datasets": [dataset_payload(DATASETS[label]) for label in labels],
            "common_index_root": str(COMMON_INDEX_ROOT),
            "darth_bin": str(DARTH_BIN),
            "parameters": vars(args),
            "offline_total_definition": "LVec + GT + TData + Train, where Train follows the DARTH notebook timer and measures model.fit only; offline_wall_total_s includes CSV load/save in train_wall_s. interval_s and online_s reported separately",
            "reuse_policy": (
                "No processed data, training logs, interval JSON, or predictor models are reused. "
                "Common HNSW indexes are fixed benchmark inputs."
            ),
        },
    )
    rows = []
    try:
        for label in labels:
            rows.append(run_dataset(DATASETS[label], args, run_root))
    except Exception as exc:
        write_json(
            run_root / "darth/results/offline_failure.json",
            {
                "failed_at": now(),
                "error": repr(exc),
                "completed_rows": rows,
            },
        )
        raise
    write_json(
        run_root / "darth/results/offline_cost_summary.json",
        {
            "status": "ok",
            "completed_at": now(),
            "rows": rows,
        },
    )
    log(f"summary: {run_root / 'darth/results/offline_cost_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
