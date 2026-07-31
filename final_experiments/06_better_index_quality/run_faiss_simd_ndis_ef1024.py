#!/usr/bin/env python3
"""FAISS SIMD-on index-quality distance-computation experiment.

This script records actual FAISS HNSW distance computations (ndis), not QPS.
It compares weak and strong HNSW builds, typically M16/efC200 and M32/efC500,
at efSearch=1024 and writes recall-loss and ndis-reduction summaries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent
REPO_ROOT = ROOT.parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FAISS_ROOT = EXPERIMENTS_ROOT / "faiss"

DEFAULT_PYTHON_FAISS = REPO_ROOT / "faiss/build_sage_avx512/faiss/python"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))
).expanduser()
DEFAULT_DATA_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(DEFAULT_PROJECT_ROOT / "index"))).expanduser()
DEFAULT_M32_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get("FAISS_INDEX_ROOT", str(DEFAULT_INDEX_DIR / "faiss_m32_efc500_main8_20260707/darth/index")),
    )
).expanduser()
DEFAULT_INDEX_ROOT_BASE = DEFAULT_INDEX_DIR / "faiss_graph_quality_ndis_20260730/darth/index"
DEFAULT_POLICY_CSV = ROOT / "policy_source" / "combined_faiss_main_qps_latency_sweep.csv"
DEFAULT_OUT_DIR = ROOT / "faiss_simd_ndis_ef1024"
DEFAULT_EFS = (1024,)


@dataclass(frozen=True)
class DatasetSpec:
    stem: str
    file_name: str
    darth_name: str
    space: str
    dim: int


DATASETS: dict[str, DatasetSpec] = {
    "glove-100-angular": DatasetSpec("glove-100-angular", "glove-100-angular.hdf5", "glove-100-angular", "cosine", 100),
    "nytimes-256-angular": DatasetSpec("nytimes-256-angular", "nytimes-256-angular.hdf5", "nytimes-256-angular", "cosine", 256),
    "msmarco-v1-openai-ada2-full-ip": DatasetSpec("msmarco-v1-openai-ada2-full-ip", "msmarco-v1-openai-ada2-full-ip.hdf5", "msmarco-v1-openai-ada2-full-ip", "ip", 1536),
    "msspacev-100M-i8-euclidean": DatasetSpec("msspacev-100M-i8-euclidean", "msspacev-100M-i8-euclidean.hdf5", "msspacev-100M-i8-euclidean", "l2", 100),
    "cohere-768-angular": DatasetSpec("cohere-768-angular", "cohere-768-angular.hdf5", "cohere-768-angular", "cosine", 768),
    "youtube-15M-angular": DatasetSpec("youtube-15M-angular", "youtube-15M-angular.hdf5", "youtube-15M-angular", "cosine", 1024),
    "agnews-mxbai-1024-euclidean": DatasetSpec("agnews-mxbai-1024-euclidean", "agnews-mxbai-1024-euclidean.hdf5", "agnews-mxbai-1024-euclidean", "l2", 1024),
    "landmark-nomic-768-angular": DatasetSpec("landmark-nomic-768-angular", "landmark-nomic-768-angular.hdf5", "landmark-nomic-768-angular", "cosine", 768),
}
PAPER_DATASETS = (
    "glove-100-angular",
    "nytimes-256-angular",
    "msmarco-v1-openai-ada2-full-ip",
    "msspacev-100M-i8-euclidean",
    "cohere-768-angular",
    "youtube-15M-angular",
)
DEFAULT_SETTINGS = ((16, 200), (32, 500))


def parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list cannot be empty")
    return values


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in parse_csv(value)]


def parse_settings(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in parse_csv(value):
        cleaned = (
            token.replace("M", "")
            .replace("m", "")
            .replace("efC", "")
            .replace("efc", "")
            .replace("_", ":")
            .replace("/", ":")
        )
        parts = [part for part in cleaned.split(":") if part]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"bad setting token {token!r}; use 16:200 or M16_efC200")
        out.append((int(parts[0]), int(parts[1])))
    return list(dict.fromkeys(out))


def parse_float_signature(value: Any) -> tuple[float, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return tuple()
    text = str(value).strip().replace(",", "/").replace(";", "/")
    return tuple(float(part) for part in text.split("/") if part.strip())


def normalize_dataset_token(value: str) -> str:
    stem = Path(str(value).strip()).stem
    if stem in DATASETS:
        return stem
    raise argparse.ArgumentTypeError(f"unknown dataset {value!r}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def import_faiss_index(args: argparse.Namespace):
    faiss_python = Path(args.faiss_python_path).expanduser().resolve()
    if str(faiss_python) not in sys.path:
        sys.path.insert(0, str(faiss_python))
    if str(FAISS_ROOT) not in sys.path:
        sys.path.insert(0, str(FAISS_ROOT))
    import faiss  # type: ignore  # noqa: F401
    from faiss_sage_index import Index  # type: ignore

    return Index


def index_root_for_setting(args: argparse.Namespace, m: int, efc: int) -> Path:
    if int(m) == 32 and int(efc) == 500:
        return Path(args.m32_index_root).expanduser().resolve()
    return Path(args.index_root_base).expanduser().resolve() / f"M{int(m)}_efC{int(efc)}"


def index_path_for_setting(args: argparse.Namespace, spec: DatasetSpec, m: int, efc: int) -> Path:
    root = index_root_for_setting(args, int(m), int(efc))
    return root / spec.darth_name / f"{spec.darth_name}.M{int(m)}.efC{int(efc)}.index"


def load_queries(args: argparse.Namespace, spec: DatasetSpec, count: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    path = Path(args.base_path).expanduser().resolve() / spec.file_name
    with h5py.File(path, "r") as handle:
        test_count = int(handle["test"].shape[0])
        take = test_count if int(count) <= 0 else min(int(count), test_count)
        queries = np.asarray(handle["test"][:take], dtype=np.float32)
        gt = None
        if "neighbors" in handle:
            gt = np.asarray(handle["neighbors"][:take, :10], dtype=np.int64)
    return queries, gt, test_count


def train_shape(args: argparse.Namespace, spec: DatasetSpec) -> tuple[int, int, str]:
    path = Path(args.base_path).expanduser().resolve() / spec.file_name
    with h5py.File(path, "r") as handle:
        train = handle["train"]
        return int(train.shape[0]), int(train.shape[1]), str(train.dtype)


def build_index(args: argparse.Namespace, spec: DatasetSpec, m: int, efc: int, index_path: Path) -> None:
    if not bool(args.build_missing_indexes):
        raise FileNotFoundError(
            f"Missing FAISS index {index_path}. Pass --build-missing-indexes or provide the correct index root."
        )
    Index = import_faiss_index(args)
    n, dim, dtype = train_shape(args, spec)
    if dim != spec.dim:
        raise ValueError(f"{spec.stem}: expected dim {spec.dim}, got {dim}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[BUILD] {spec.stem} M={m} efC={efc} rows={n} dim={dim} dtype={dtype} path={index_path}", flush=True)
    index = Index(space=spec.space, dim=dim)
    index.set_num_threads(int(args.build_threads))
    index.init_index(max_elements=n, M=int(m), ef_construction=int(efc))
    batch_size = int(args.build_batch_size)
    t0 = time.perf_counter()
    last_report = 0
    with h5py.File(Path(args.base_path).expanduser().resolve() / spec.file_name, "r") as handle:
        train = handle["train"]
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = np.asarray(train[start:stop], dtype=np.float32)
            index.add_items(batch, num_threads=int(args.build_threads))
            del batch
            if stop - last_report >= 1_000_000 or stop == n:
                print(f"[BUILD-PROGRESS] {spec.stem} {stop}/{n} elapsed_s={time.perf_counter() - t0:.1f}", flush=True)
                last_report = stop
    index.save_index(str(tmp_path))
    os.replace(tmp_path, index_path)
    print(f"[BUILD-DONE] {spec.stem} path={index_path} wall_s={time.perf_counter() - t0:.1f}", flush=True)
    del index
    gc.collect()


def load_index(args: argparse.Namespace, spec: DatasetSpec, m: int, efc: int):
    index_path = index_path_for_setting(args, spec, int(m), int(efc))
    if not index_path.exists() or index_path.stat().st_size == 0:
        build_index(args, spec, int(m), int(efc), index_path)
    Index = import_faiss_index(args)
    index = Index(space=spec.space, dim=spec.dim)
    index.set_num_threads(int(args.num_threads))
    index.load_index(str(index_path), max_elements=0)
    return index, index_path


def recall_at_10(labels: np.ndarray, gt: np.ndarray | None) -> float:
    if gt is None:
        return float("nan")
    labels = np.asarray(labels)
    gt = np.asarray(gt)
    total = min(int(labels.shape[0]), int(gt.shape[0]))
    if total == 0:
        return float("nan")
    hits = 0
    for qid in range(total):
        hits += len(set(map(int, labels[qid, :10])).intersection(map(int, gt[qid, :10])))
    return float(hits / (total * 10.0))


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def load_policy(args: argparse.Namespace, dataset: str, m: int, efc: int, ef: int) -> dict[str, Any]:
    path = Path(args.policy_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing policy CSV: {path}. It must contain Ours rows with early_stop_ratio, "
            "route_signature, and bucket_gamma_signature from the matching FAISS graph-quality calibration run."
        )
    df = pd.read_csv(path)
    part = df[
        (df["dataset"].astype(str) == str(dataset))
        & (df["M"].astype(int) == int(m))
        & (df["efConstruction"].astype(int) == int(efc))
        & (df["ef"].astype(int) == int(ef))
        & (df["method"].astype(str) == "Ours")
    ]
    if part.empty:
        raise ValueError(f"No Ours policy row for dataset={dataset} M={m} efC={efc} ef={ef} in {path}")
    row = part.iloc[0]
    route_signature = str(row["route_signature"])
    bucket_gamma_signature = str(row["bucket_gamma_signature"])
    route_parts = [int(float(part)) for part in route_signature.replace(",", "/").split("/") if part.strip()]
    bucket_gammas = parse_float_signature(bucket_gamma_signature)
    return {
        "policy_source_csv": str(path),
        "early_stop_ratio": float(row["early_stop_ratio"]),
        "route_signature": route_signature,
        "bucket_gamma_signature": bucket_gamma_signature,
        "bucket_gamma_ratios": bucket_gammas,
        "paper_bucket_count": len(route_parts),
        "sweep_recall": float(row.get("recall", np.nan)),
        "sweep_recall_loss_vs_vanilla_pp": float(row.get("recall_loss_vs_vanilla_pp", np.nan)),
    }


def call_chr_summary(index: Any, queries: np.ndarray, args: argparse.Namespace, ef: int) -> dict[str, np.ndarray]:
    return index.search_layer0_chr_summary(
        queries,
        k=10,
        ef=int(ef),
        num_threads=int(args.num_threads),
        classify_start=int(args.classify_start),
        classify_end=int(args.classify_end),
        chr_ema_decay=float(args.chr_ema_decay),
    )


def adaptive_analysis(index: Any, queries: np.ndarray, args: argparse.Namespace, ef: int, policy: dict[str, Any]):
    kwargs = {
        "k": 10,
        "ef_init": int(ef),
        "ef_max": int(ef),
        "enable_stop": True,
        "num_threads": int(args.num_threads),
        "early_stop_ratio": float(policy["early_stop_ratio"]),
        "tmin_pops": int(args.tmin_pops),
        "paper_bucket_count": int(policy["paper_bucket_count"]),
        "bucket_gamma_ratios": list(policy["bucket_gamma_ratios"]),
        "classify_start": int(args.classify_start),
        "classify_end": int(args.classify_end),
        "chr_ema_decay": float(args.chr_ema_decay),
    }
    return index.knn_query_adaptive_analysis_paper_bucket(queries, **kwargs)


def adaptive_query(index: Any, queries: np.ndarray, args: argparse.Namespace, ef: int, policy: dict[str, Any]) -> np.ndarray:
    labels, _dists = index.knn_query_sage(
        queries,
        k=10,
        ef_init=int(ef),
        enable_stop=True,
        num_threads=int(args.num_threads),
        early_stop_ratio=float(policy["early_stop_ratio"]),
        tmin_pops=int(args.tmin_pops),
        paper_bucket_count=int(policy["paper_bucket_count"]),
        bucket_gamma_ratios=list(policy["bucket_gamma_ratios"]),
        classify_start=int(args.classify_start),
        classify_end=int(args.classify_end),
        chr_ema_decay=float(args.chr_ema_decay),
    )
    return np.asarray(labels, dtype=np.int64)


def measure_recall(index: Any, args: argparse.Namespace, spec: DatasetSpec, ef: int, policy: dict[str, Any]) -> dict[str, Any]:
    if bool(args.skip_full_recall):
        return {
            "full_recall_query_count": 0,
            "full_vanilla_recall_at_10": np.nan,
            "full_ours_recall_at_10": np.nan,
            "full_recall_loss_pp": np.nan,
            "full_recall_wall_s": np.nan,
        }
    queries, gt, test_count = load_queries(args, spec, int(args.recall_num_queries))
    t0 = time.perf_counter()
    index.set_ef(int(ef))
    vanilla_labels, _vanilla_dists = index.knn_query(queries, k=10, num_threads=int(args.num_threads))
    ours_labels = adaptive_query(index, queries, args, int(ef), policy)
    wall = time.perf_counter() - t0
    vanilla_recall = recall_at_10(np.asarray(vanilla_labels), gt)
    ours_recall = recall_at_10(ours_labels, gt)
    return {
        "full_recall_query_count": int(queries.shape[0]),
        "full_test_query_count": int(test_count),
        "full_vanilla_recall_at_10": float(vanilla_recall),
        "full_ours_recall_at_10": float(ours_recall),
        "full_recall_loss_pp": float((vanilla_recall - ours_recall) * 100.0),
        "full_recall_wall_s": float(wall),
    }


def measure_one(args: argparse.Namespace, spec: DatasetSpec, m: int, efc: int, ef: int) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    setting_dir = out_dir / spec.stem / f"M{int(m)}_efC{int(efc)}" / f"ef{int(ef)}"
    summary_path = setting_dir / "summary_actual_ndis.csv"
    if bool(args.reuse_existing) and summary_path.exists():
        with summary_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            print(f"[REUSE] {summary_path}", flush=True)
            return dict(rows[0])

    policy = load_policy(args, spec.stem, int(m), int(efc), int(ef))
    index, index_path = load_index(args, spec, int(m), int(efc))
    queries, gt, test_count = load_queries(args, spec, int(args.ndis_num_queries))

    print(f"[NDIS] {spec.stem} M={m} efC={efc} ef={ef} q={len(queries)} index={index_path}", flush=True)
    t0 = time.perf_counter()
    vanilla_trace = call_chr_summary(index, queries, args, int(ef))
    vanilla_wall = time.perf_counter() - t0
    vanilla_ndis = np.asarray(vanilla_trace["distance_counts"], dtype=np.float64)
    vanilla_pop = np.asarray(vanilla_trace.get("step_counts", np.full(vanilla_ndis.shape, int(ef))), dtype=np.float64)

    t1 = time.perf_counter()
    labels, dists, pop_steps, stop_flags, distance_counts = adaptive_analysis(index, queries, args, int(ef), policy)
    del dists
    ours_wall = time.perf_counter() - t1
    ours_ndis = np.asarray(distance_counts, dtype=np.float64)
    ours_pop = np.asarray(pop_steps, dtype=np.float64)
    stop_flags_arr = np.asarray(stop_flags, dtype=np.int64)
    saved_ndis = vanilla_ndis - ours_ndis

    index.set_ef(int(ef))
    vanilla_labels, _ = index.knn_query(queries, k=10, num_threads=int(args.num_threads))
    sample_vanilla_recall = recall_at_10(np.asarray(vanilla_labels), gt)
    sample_ours_recall = recall_at_10(np.asarray(labels), gt)
    recall_fields = measure_recall(index, args, spec, int(ef), policy)

    per_query = []
    for qid in range(int(queries.shape[0])):
        row_vanilla = float(vanilla_ndis[qid])
        row_saved = float(saved_ndis[qid])
        per_query.append(
            {
                "backend": "faiss",
                "simd": "on",
                "dataset": spec.stem,
                "M": int(m),
                "efConstruction": int(efc),
                "ef": int(ef),
                "qid": int(qid),
                "vanilla_pop_count": int(vanilla_pop[qid]),
                "vanilla_ndis": int(vanilla_ndis[qid]),
                "ours_pop_count": int(ours_pop[qid]),
                "ours_ndis": int(ours_ndis[qid]),
                "stop_flag": int(stop_flags_arr[qid]),
                "saved_ndis": int(row_saved),
                "saved_ndis_pct": row_saved / row_vanilla * 100.0 if row_vanilla > 0 else np.nan,
            }
        )

    vanilla_summary = summarize(vanilla_ndis)
    ours_summary = summarize(ours_ndis)
    saved_summary = summarize(saved_ndis)
    summary = {
        "backend": "faiss",
        "simd": "on",
        "dataset": spec.stem,
        "dataset_file": spec.file_name,
        "M": int(m),
        "efConstruction": int(efc),
        "ef": int(ef),
        "ndis_query_count": int(queries.shape[0]),
        "test_query_count": int(test_count),
        "num_threads": int(args.num_threads),
        "index_path": str(index_path),
        "index_size_bytes": int(index_path.stat().st_size) if index_path.exists() else 0,
        "vanilla_mean_pop": float(np.mean(vanilla_pop)),
        "ours_mean_pop": float(np.mean(ours_pop)),
        "vanilla_mean_ndis": vanilla_summary["mean"],
        "vanilla_p50_ndis": vanilla_summary["p50"],
        "vanilla_p95_ndis": vanilla_summary["p95"],
        "ours_mean_ndis": ours_summary["mean"],
        "ours_p50_ndis": ours_summary["p50"],
        "ours_p95_ndis": ours_summary["p95"],
        "saved_mean_ndis": saved_summary["mean"],
        "saved_p50_ndis": saved_summary["p50"],
        "saved_p95_ndis": saved_summary["p95"],
        "saved_ndis_pct": float(np.mean(saved_ndis) / np.mean(vanilla_ndis) * 100.0),
        "ndis_speedup": float(np.mean(vanilla_ndis) / np.mean(ours_ndis)),
        "stop_rate": float(np.mean(stop_flags_arr > 0)),
        "sample_vanilla_recall_at_10": float(sample_vanilla_recall),
        "sample_ours_recall_at_10": float(sample_ours_recall),
        "sample_recall_loss_pp": float((sample_vanilla_recall - sample_ours_recall) * 100.0),
        "vanilla_ndis_wall_s": float(vanilla_wall),
        "ours_ndis_wall_s": float(ours_wall),
        "policy_source_csv": str(policy["policy_source_csv"]),
        "early_stop_ratio": float(policy["early_stop_ratio"]),
        "route_signature": str(policy["route_signature"]),
        "bucket_gamma_signature": str(policy["bucket_gamma_signature"]),
        "sweep_recall": float(policy.get("sweep_recall", np.nan)),
        "sweep_recall_loss_vs_vanilla_pp": float(policy.get("sweep_recall_loss_vs_vanilla_pp", np.nan)),
        **recall_fields,
    }

    write_csv(setting_dir / "per_query_actual_ndis.csv", per_query)
    write_csv(summary_path, [summary])
    del index, queries, labels, vanilla_labels, vanilla_trace
    gc.collect()
    print(
        f"[DONE] {spec.stem} M={m} efC={efc} ef={ef} "
        f"saved_ndis_pct={summary['saved_ndis_pct']:.3f} "
        f"full_loss_pp={summary['full_recall_loss_pp']:.4f}",
        flush=True,
    )
    return summary


def make_comparison(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row["dataset"]), int(row["M"]), int(row["efConstruction"]), int(row["ef"]))] = row
    out: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    efs = sorted({int(row["ef"]) for row in rows})
    for dataset in datasets:
        for ef in efs:
            low = by_key.get((dataset, 16, 200, ef))
            high = by_key.get((dataset, 32, 500, ef))
            if low is None or high is None:
                continue
            low_loss = float(low.get("full_recall_loss_pp", low.get("sample_recall_loss_pp", np.nan)))
            high_loss = float(high.get("full_recall_loss_pp", high.get("sample_recall_loss_pp", np.nan)))
            out.append(
                {
                    "backend": "faiss",
                    "simd": "on",
                    "dataset": dataset,
                    "ef": int(ef),
                    "m16_efc200_saved_ndis_pct": float(low["saved_ndis_pct"]),
                    "m32_efc500_saved_ndis_pct": float(high["saved_ndis_pct"]),
                    "m32_minus_m16_saved_ndis_pct": float(high["saved_ndis_pct"]) - float(low["saved_ndis_pct"]),
                    "m16_efc200_ndis_speedup": float(low["ndis_speedup"]),
                    "m32_efc500_ndis_speedup": float(high["ndis_speedup"]),
                    "m32_minus_m16_ndis_speedup": float(high["ndis_speedup"]) - float(low["ndis_speedup"]),
                    "m16_efc200_recall_loss_pp": low_loss,
                    "m32_efc500_recall_loss_pp": high_loss,
                    "m32_minus_m16_recall_loss_pp": high_loss - low_loss,
                    "m32_saved_ndis_pct_bigger": bool(float(high["saved_ndis_pct"]) > float(low["saved_ndis_pct"])),
                    "m32_recall_loss_lower": bool(high_loss < low_loss),
                    "m16_index_path": str(low.get("index_path", "")),
                    "m32_index_path": str(high.get("index_path", "")),
                }
            )
    return out


def write_readme(out_dir: Path, args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> None:
    manifest = {
        "script": str(SCRIPT_PATH),
        "backend": "faiss",
        "simd": "on",
        "datasets": parse_csv(args.datasets),
        "settings": [f"M{m}/efC{efc}" for m, efc in parse_settings(args.settings)],
        "efs": parse_ints(args.efs),
        "ndis_num_queries": int(args.ndis_num_queries),
        "recall_num_queries": int(args.recall_num_queries),
        "num_threads": int(args.num_threads),
        "policy_csv": str(Path(args.policy_csv).expanduser().resolve()),
        "output_rows": len(rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# FAISS SIMD-On Index-Quality NDIS Results",
        "",
        "This run records actual FAISS distance computations, not QPS.",
        "",
        "Primary CSVs:",
        "",
        "- `combined_summary_actual_ndis.csv`",
        "- `m16_m32_ndis_reduction_comparison.csv`",
        "- per-dataset `per_query_actual_ndis.csv` files",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    parser.add_argument("--settings", default=",".join(f"{m}:{efc}" for m, efc in DEFAULT_SETTINGS))
    parser.add_argument("--efs", default=",".join(str(ef) for ef in DEFAULT_EFS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--m32-index-root", type=Path, default=DEFAULT_M32_INDEX_ROOT)
    parser.add_argument("--index-root-base", type=Path, default=DEFAULT_INDEX_ROOT_BASE)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_PYTHON_FAISS)
    parser.add_argument("--policy-csv", type=Path, default=DEFAULT_POLICY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ndis-num-queries", type=int, default=200)
    parser.add_argument("--recall-num-queries", type=int, default=0, help="0 means all test queries")
    parser.add_argument("--skip-full-recall", action="store_true")
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--build-batch-size", type=int, default=100000)
    parser.add_argument("--build-missing-indexes", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--classify-start", type=int, default=4)
    parser.add_argument("--classify-end", type=int, default=16)
    parser.add_argument("--chr-ema-decay", type=float, default=0.8)
    parser.add_argument("--tmin-pops", type=int, default=25)
    args = parser.parse_args(argv)

    os.environ.setdefault("SAGE_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
    os.environ["FAISS_PYTHON_PATH"] = str(Path(args.faiss_python_path).expanduser().resolve())
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(args.num_threads))

    datasets = [normalize_dataset_token(value) for value in parse_csv(args.datasets)]
    settings = parse_settings(args.settings)
    efs = parse_ints(args.efs)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        spec = DATASETS[dataset]
        for m, efc in settings:
            for ef in efs:
                rows.append(measure_one(args, spec, int(m), int(efc), int(ef)))
                write_csv(out_dir / "combined_summary_actual_ndis.csv", rows)
                write_csv(out_dir / "m16_m32_ndis_reduction_comparison.csv", make_comparison(rows))

    write_csv(out_dir / "combined_summary_actual_ndis.csv", rows)
    write_csv(out_dir / "m16_m32_ndis_reduction_comparison.csv", make_comparison(rows))
    write_readme(out_dir, args, rows)
    print(f"[RESULT] {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
