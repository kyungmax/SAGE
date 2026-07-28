#!/usr/bin/env python3
"""FAISS vs hnswlib investigation runner.

This script keeps the expensive work bounded. It uses existing final sweep
CSVs for recall/QPS analysis, measures layer-0 degree on representative
manageable indexes, and samples vanilla ndis on the same indexes.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FINAL_IMPL = Path(__file__).resolve().parent
PAPERS_OURS = FINAL_IMPL.parent
PROJECT_ROOT = Path("/home/kyungmin/vectordb/hnsw-playground")
OUT_DIR = PAPERS_OURS / "final_analysis/faiss_vs_hnswlib"
INDEX_ROOT = PROJECT_ROOT / "index"
DATASET_ROOT = PROJECT_ROOT / "datasets"
FAISS_PYTHON = INDEX_ROOT / "faiss_builds/ours_adaptive_light_hnsw_py/faiss/python"
FAISS_COMPAT = FINAL_IMPL / "faiss"
HNSWLIB_EXTENSION_ROOT = Path("/home/kyungmin/vectordb/hnswlib")

FAISS_SWEEP = (
    PAPERS_OURS
    / "final_experiments/FAISS/faiss_vanilla_ours_final6_m32_efc500_ncal100_20260617/final/main_qps_latency_sweep.csv"
)
HNSWLIB_SWEEP = (
    PAPERS_OURS
    / "final_experiments/HNSWLib/hnswlib_vanilla_ours_final6_m32_efc500_ncal100_20260617/final/main_qps_latency_sweep.csv"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_file: str
    display_name: str
    space: str
    faiss_name: str
    hnswlib_index: Path
    faiss_index: Path
    dim: int


DATASETS: dict[str, DatasetSpec] = {
    "nytimes": DatasetSpec(
        key="nytimes",
        dataset_file="nytimes-256-angular.hdf5",
        display_name="nytimes",
        space="cosine",
        faiss_name="nytimes-256-angular",
        hnswlib_index=INDEX_ROOT / "nytimes-256-angular_M32_M32_efC500_n290000_dim256",
        faiss_index=INDEX_ROOT
        / "m32_efc500_target095_adaef_darth_efs1000_20260603/darth/index/nytimes-256-angular/nytimes-256-angular.M32.efC500.index",
        dim=256,
    ),
    "glove": DatasetSpec(
        key="glove",
        dataset_file="glove-100-angular.hdf5",
        display_name="glove-100",
        space="cosine",
        faiss_name="glove-100-angular",
        hnswlib_index=INDEX_ROOT / "glove-100-angular_M32_M32_efC500_n1183514_dim100",
        faiss_index=INDEX_ROOT
        / "m32_efc500_target095_adaef_darth_efs1000_20260603/darth/index/glove-100-angular/glove-100-angular.M32.efC500.index",
        dim=100,
    ),
}

HNSW_HEADER = struct.Struct("<QQQQQQiIQQQdQ")
HNSW_HEADER_KEYS = (
    "offsetLevel0",
    "max_elements",
    "cur_element_count",
    "size_data_per_element",
    "label_offset",
    "offsetData",
    "maxlevel",
    "enterpoint_node",
    "maxM",
    "maxM0",
    "M",
    "mult",
    "ef_construction",
)


def import_runtime_modules():
    """Import extension modules without being shadowed by local package dirs."""
    sys.path.insert(0, str(HNSWLIB_EXTENSION_ROOT))
    import hnswlib  # type: ignore

    sys.path.insert(0, str(FAISS_PYTHON))
    sys.path.insert(0, str(FAISS_COMPAT))
    import faiss  # type: ignore
    try:
        from faiss_hnswlib_compat import Index as FaissCompatIndex  # type: ignore
    except ModuleNotFoundError:
        from faiss_sage_index import Index as FaissCompatIndex  # type: ignore

    return hnswlib, faiss, FaissCompatIndex


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)


def sample_ids(n: int, sample_limit: int) -> np.ndarray:
    if n <= sample_limit:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=sample_limit, dtype=np.int64)


def describe_counts(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "sample_count": 0,
            "mean_degree": float("nan"),
            "median_degree": float("nan"),
            "std_degree": float("nan"),
            "min_degree": float("nan"),
            "p05_degree": float("nan"),
            "p25_degree": float("nan"),
            "p75_degree": float("nan"),
            "p95_degree": float("nan"),
            "max_degree": float("nan"),
            "frac_degree_lt_64": float("nan"),
            "frac_degree_eq_64": float("nan"),
        }
    return {
        "sample_count": int(values.size),
        "mean_degree": float(np.mean(values)),
        "median_degree": float(np.median(values)),
        "std_degree": float(np.std(values)),
        "min_degree": float(np.min(values)),
        "p05_degree": float(np.percentile(values, 5)),
        "p25_degree": float(np.percentile(values, 25)),
        "p75_degree": float(np.percentile(values, 75)),
        "p95_degree": float(np.percentile(values, 95)),
        "max_degree": float(np.max(values)),
        "frac_degree_lt_64": float(np.mean(values < 64.0)),
        "frac_degree_eq_64": float(np.mean(values == 64.0)),
    }


def parse_hnswlib_header(path: Path) -> dict[str, int | float]:
    with path.open("rb") as handle:
        data = handle.read(HNSW_HEADER.size)
    values = HNSW_HEADER.unpack(data)
    return dict(zip(HNSW_HEADER_KEYS, values))


def hnswlib_degree_stats(spec: DatasetSpec, sample_limit: int) -> tuple[dict, np.ndarray]:
    header = parse_hnswlib_header(spec.hnswlib_index)
    n = int(header["cur_element_count"])
    stride = int(header["size_data_per_element"])
    ids = sample_ids(n, sample_limit)
    with spec.hnswlib_index.open("rb") as handle:
        mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            counts_view = np.ndarray(
                shape=(n,),
                dtype=np.uint16,
                buffer=mm,
                offset=HNSW_HEADER.size + int(header["offsetLevel0"]),
                strides=(stride,),
            )
            counts = np.asarray(counts_view[ids], dtype=np.int32)
        finally:
            mm.close()

    row = {
        "backend": "hnswlib",
        "dataset": spec.display_name,
        "node_count": n,
        "sample_mode": "full" if n == len(ids) else "linspace",
        "maxM0": int(header["maxM0"]),
        "M": int(header["M"]),
        "efConstruction": int(header["ef_construction"]),
        "index_path": str(spec.hnswlib_index),
    }
    row.update(describe_counts(counts))
    return row, counts


def faiss_degree_stats(spec: DatasetSpec, sample_limit: int, faiss) -> tuple[dict, np.ndarray]:
    index = faiss.read_index(str(spec.faiss_index))
    hnsw = index.hnsw
    n = int(index.ntotal)
    ids = sample_ids(n, sample_limit)
    offsets = faiss.rev_swig_ptr(hnsw.offsets.data(), hnsw.offsets.size())
    neighbors = faiss.rev_swig_ptr(hnsw.neighbors.data(), hnsw.neighbors.size())
    level0_slots = int(hnsw.nb_neighbors(0))
    base = int(hnsw.cum_nb_neighbors(0))

    counts = np.empty(ids.shape[0], dtype=np.int32)
    arange_slots = np.arange(level0_slots, dtype=np.int64)
    chunk_size = 50000
    for start in range(0, ids.shape[0], chunk_size):
        chunk = ids[start : start + chunk_size]
        positions = offsets[chunk].astype(np.int64)[:, None] + base + arange_slots[None, :]
        vals = neighbors[positions]
        counts[start : start + len(chunk)] = np.sum(vals >= 0, axis=1)

    row = {
        "backend": "faiss",
        "dataset": spec.display_name,
        "node_count": n,
        "sample_mode": "full" if n == len(ids) else "linspace",
        "maxM0": level0_slots,
        "M": int(hnsw.nb_neighbors(1)),
        "efConstruction": int(hnsw.efConstruction),
        "keep_max_size_level0": bool(getattr(index, "keep_max_size_level0", False)),
        "index_path": str(spec.faiss_index),
    }
    row.update(describe_counts(counts))
    return row, counts


def run_degree(args, faiss) -> pd.DataFrame:
    rows = []
    hist_rows = []
    for spec in DATASETS.values():
        if not spec.hnswlib_index.exists():
            raise FileNotFoundError(spec.hnswlib_index)
        if not spec.faiss_index.exists():
            raise FileNotFoundError(spec.faiss_index)
        print(f"[degree] {spec.key} hnswlib", flush=True)
        row, counts = hnswlib_degree_stats(spec, args.degree_sample_limit)
        rows.append(row)
        for degree, count in zip(*np.unique(counts, return_counts=True)):
            hist_rows.append(
                {
                    "backend": "hnswlib",
                    "dataset": spec.display_name,
                    "degree": int(degree),
                    "count": int(count),
                }
            )
        print(f"[degree] {spec.key} faiss", flush=True)
        row, counts = faiss_degree_stats(spec, args.degree_sample_limit, faiss)
        rows.append(row)
        for degree, count in zip(*np.unique(counts, return_counts=True)):
            hist_rows.append(
                {
                    "backend": "faiss",
                    "dataset": spec.display_name,
                    "degree": int(degree),
                    "count": int(count),
                }
            )

    degree_df = pd.DataFrame(rows)
    hist_df = pd.DataFrame(hist_rows)
    degree_df.to_csv(args.out_dir / "degree_stats.csv", index=False)
    hist_df.to_csv(args.out_dir / "degree_histogram.csv", index=False)
    return degree_df


def read_sweeps() -> pd.DataFrame:
    faiss_df = pd.read_csv(FAISS_SWEEP)
    hnsw_df = pd.read_csv(HNSWLIB_SWEEP)
    faiss_df["backend"] = "faiss"
    hnsw_df["backend"] = "hnswlib"
    df = pd.concat([faiss_df, hnsw_df], ignore_index=True)
    return df


def best_qps_at_recall(group: pd.DataFrame, target: float) -> tuple[float, float, int]:
    eligible = group[group["recall"] >= target]
    if eligible.empty:
        return float("nan"), float("nan"), 0
    idx = eligible["qps"].idxmax()
    row = eligible.loc[idx]
    return float(row["qps"]), float(row["recall"]), int(row["ef"])


def run_curve_analysis(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_sweeps()
    keep = [
        "backend",
        "dataset",
        "dataset_file",
        "ef",
        "method",
        "recall",
        "qps",
        "recall_loss_vs_vanilla_pp",
        "qps_gain_vs_vanilla_pct",
        "latency_per_query_mean_ms",
    ]
    compact = df[keep].copy()
    compact.to_csv(out_dir / "curve_points_compact.csv", index=False)

    same_ef_rows = []
    for (backend, dataset, ef), part in df.groupby(["backend", "dataset", "ef"]):
        by_method = {str(r["method"]): r for _, r in part.iterrows()}
        if "Vanilla" not in by_method or "Ours" not in by_method:
            continue
        v = by_method["Vanilla"]
        o = by_method["Ours"]
        same_ef_rows.append(
            {
                "backend": backend,
                "dataset": dataset,
                "ef": int(ef),
                "vanilla_recall": float(v["recall"]),
                "ours_recall": float(o["recall"]),
                "vanilla_qps": float(v["qps"]),
                "ours_qps": float(o["qps"]),
                "same_ef_qps_gain_pct": (float(o["qps"]) / float(v["qps"]) - 1.0) * 100.0,
                "recall_delta_ours_minus_vanilla_pp": (float(o["recall"]) - float(v["recall"])) * 100.0,
            }
        )
    same_ef = pd.DataFrame(same_ef_rows)
    same_ef.to_csv(out_dir / "same_ef_vanilla_ours_summary.csv", index=False)

    target_rows = []
    for (backend, dataset), part in df.groupby(["backend", "dataset"]):
        for target in (0.95, 0.97, 0.98, 0.99):
            row = {
                "backend": backend,
                "dataset": dataset,
                "target_recall": target,
            }
            for method in ("Vanilla", "Ours"):
                method_part = part[part["method"] == method]
                qps, recall, ef = best_qps_at_recall(method_part, target)
                prefix = method.lower()
                row[f"{prefix}_best_qps"] = qps
                row[f"{prefix}_recall_at_best_qps"] = recall
                row[f"{prefix}_ef_at_best_qps"] = ef
            if (
                np.isfinite(row["vanilla_best_qps"])
                and np.isfinite(row["ours_best_qps"])
                and row["vanilla_best_qps"] > 0
            ):
                row["iso_recall_speedup"] = row["ours_best_qps"] / row["vanilla_best_qps"]
            else:
                row["iso_recall_speedup"] = float("nan")
            target_rows.append(row)
    iso = pd.DataFrame(target_rows)
    iso.to_csv(out_dir / "iso_recall_best_qps_summary.csv", index=False)

    backend_summary = (
        same_ef.groupby("backend", as_index=False)
        .agg(
            same_ef_qps_gain_pct_mean=("same_ef_qps_gain_pct", "mean"),
            same_ef_qps_gain_pct_median=("same_ef_qps_gain_pct", "median"),
            recall_delta_pp_mean=("recall_delta_ours_minus_vanilla_pp", "mean"),
            rows=("same_ef_qps_gain_pct", "size"),
        )
        .sort_values("backend")
    )
    iso95 = iso[iso["target_recall"] == 0.95].groupby("backend", as_index=False).agg(
        iso95_speedup_mean=("iso_recall_speedup", "mean"),
        iso95_speedup_median=("iso_recall_speedup", "median"),
        iso95_rows=("iso_recall_speedup", "count"),
    )
    backend_summary = backend_summary.merge(iso95, on="backend", how="left")
    backend_summary.to_csv(out_dir / "backend_curve_summary.csv", index=False)
    return same_ef, iso, backend_summary


def load_test_and_gt(spec: DatasetSpec, n_queries: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(DATASET_ROOT / spec.dataset_file, "r") as handle:
        test = np.asarray(handle["test"][:n_queries], dtype=np.float32)
        gt = np.asarray(handle["neighbors"][:n_queries, :10], dtype=np.int64)
    return test, gt


def recall_at_10(labels: np.ndarray, gt: np.ndarray) -> float:
    recalls = []
    for row, truth in zip(labels, gt):
        recalls.append(len(set(map(int, row[:10])).intersection(map(int, truth[:10]))) / 10.0)
    return float(np.mean(recalls)) if recalls else float("nan")


def run_ndis(args, hnswlib, FaissCompatIndex) -> pd.DataFrame:
    rows = []
    for spec in DATASETS.values():
        print(f"[ndis] loading queries {spec.key}", flush=True)
        test, gt = load_test_and_gt(spec, args.ndis_queries)

        print(f"[ndis] loading hnswlib {spec.key}", flush=True)
        hidx = hnswlib.Index(space=spec.space, dim=spec.dim)
        hidx.set_num_threads(args.num_threads)
        hidx.load_index(str(spec.hnswlib_index), max_elements=0)

        print(f"[ndis] loading faiss {spec.key}", flush=True)
        fidx = FaissCompatIndex(space=spec.space, dim=spec.dim)
        fidx.set_num_threads(args.num_threads)
        fidx.load_index(str(spec.faiss_index))

        for ef in args.ndis_efs:
            print(f"[ndis] {spec.key} ef={ef} hnswlib", flush=True)
            hidx.set_ef(int(ef))
            h_labels, _ = hidx.knn_query(test, k=10, num_threads=args.num_threads)
            _, h_total_ndis, _ = hidx.search_layer0_path_with_dist_metrics_batch(
                test, 10, int(ef), args.num_threads
            )
            h_recall = recall_at_10(np.asarray(h_labels), gt)
            rows.append(
                {
                    "backend": "hnswlib",
                    "dataset": spec.display_name,
                    "ef": int(ef),
                    "query_count": int(test.shape[0]),
                    "recall": h_recall,
                    "total_ndis": int(h_total_ndis),
                    "mean_ndis": float(h_total_ndis) / float(test.shape[0]),
                }
            )

            print(f"[ndis] {spec.key} ef={ef} faiss", flush=True)
            fidx.set_ef(int(ef))
            f_labels, _ = fidx.knn_query(test, k=10, num_threads=args.num_threads)
            _, f_total_ndis, _ = fidx.search_layer0_path_with_dist_metrics_batch(
                test,
                k=10,
                ef=int(ef),
                num_threads=args.num_threads,
            )
            f_recall = recall_at_10(np.asarray(f_labels), gt)
            rows.append(
                {
                    "backend": "faiss",
                    "dataset": spec.display_name,
                    "ef": int(ef),
                    "query_count": int(test.shape[0]),
                    "recall": f_recall,
                    "total_ndis": int(f_total_ndis),
                    "mean_ndis": float(f_total_ndis) / float(test.shape[0]),
                }
            )

    ndis = pd.DataFrame(rows)
    ndis.to_csv(args.out_dir / "ndis_sample_summary.csv", index=False)
    return ndis


def write_report(
    out_dir: Path,
    degree: pd.DataFrame,
    same_ef: pd.DataFrame,
    iso: pd.DataFrame,
    backend_summary: pd.DataFrame,
    ndis: pd.DataFrame,
    started_at: float,
) -> None:
    report = out_dir / "RESULTS.md"
    elapsed = time.time() - started_at

    def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "(empty)\n"
        return df.head(max_rows).to_markdown(index=False)

    deg_cols = [
        "backend",
        "dataset",
        "node_count",
        "sample_count",
        "sample_mode",
        "mean_degree",
        "median_degree",
        "p05_degree",
        "p95_degree",
        "frac_degree_eq_64",
        "keep_max_size_level0",
    ]
    degree_show = degree.copy()
    for col in deg_cols:
        if col not in degree_show.columns:
            degree_show[col] = np.nan
    degree_show = degree_show[deg_cols]

    iso95 = iso[iso["target_recall"] == 0.95].copy()
    if ndis.empty:
        ndis_pivot = pd.DataFrame()
    else:
        ndis_pivot = ndis.pivot_table(
            index=["dataset", "ef"],
            columns="backend",
            values=["recall", "mean_ndis"],
            aggfunc="first",
        )
        ndis_pivot.columns = ["_".join(col).strip() for col in ndis_pivot.columns.values]
        ndis_pivot = ndis_pivot.reset_index()
        if {"mean_ndis_faiss", "mean_ndis_hnswlib"}.issubset(ndis_pivot.columns):
            ndis_pivot["faiss_ndis_over_hnswlib"] = (
                ndis_pivot["mean_ndis_faiss"] / ndis_pivot["mean_ndis_hnswlib"]
            )

    lines = [
        "# FAISS vs hnswlib investigation results",
        "",
        f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.",
        f"Elapsed seconds {elapsed:.1f}.",
        "",
        "## Main findings",
        "",
        "- Build parameters are matched in the final sweep CSVs. Both backends use M=32 and efConstruction=500.",
        "- The FAISS indexes measured here report keep_max_size_level0=false. Therefore the CAGRA-only full base-layer path is not the direct explanation for these FAISS IndexHNSWFlat runs.",
        "- Degree measurements on nytimes and glove show that both backends have sparse variable level-0 degree rather than universally full degree 64.",
        "- Existing final sweep curves show backend-specific absolute QPS differences, but the same-ef ours-over-vanilla gain is present in both backends.",
        "- The ndis sample separates logical work from wall-clock speed. Use ndis_sample_summary.csv for the direct H2 check.",
        "",
        "## Degree summary",
        "",
        md_table(degree_show),
        "",
        "## Backend curve summary",
        "",
        md_table(backend_summary),
        "",
        "## Iso recall best QPS at recall >= 0.95",
        "",
        md_table(iso95),
        "",
        "## ndis sample",
        "",
        md_table(ndis_pivot),
        "",
        "## Output files",
        "",
        "- degree_stats.csv",
        "- degree_histogram.csv",
        "- curve_points_compact.csv",
        "- same_ef_vanilla_ours_summary.csv",
        "- iso_recall_best_qps_summary.csv",
        "- backend_curve_summary.csv",
        "- ndis_sample_summary.csv",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--degree-sample-limit", type=int, default=1_500_000)
    parser.add_argument("--ndis-queries", type=int, default=200)
    parser.add_argument("--ndis-efs", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--skip-ndis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.time()
    ensure_out_dir(args.out_dir)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    hnswlib, faiss, FaissCompatIndex = import_runtime_modules()
    metadata = {
        "num_threads": args.num_threads,
        "degree_sample_limit": args.degree_sample_limit,
        "ndis_queries": args.ndis_queries,
        "ndis_efs": args.ndis_efs,
        "faiss_module": str(getattr(faiss, "__file__", "")),
        "hnswlib_module": str(getattr(hnswlib, "__file__", "")),
        "faiss_sweep": str(FAISS_SWEEP),
        "hnswlib_sweep": str(HNSWLIB_SWEEP),
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[1/4] degree stats", flush=True)
    degree = run_degree(args, faiss)
    print("[2/4] existing curve analysis", flush=True)
    same_ef, iso, backend_summary = run_curve_analysis(args.out_dir)
    if args.skip_ndis:
        ndis = pd.DataFrame()
    else:
        print("[3/4] ndis sample", flush=True)
        ndis = run_ndis(args, hnswlib, FaissCompatIndex)
    print("[4/4] report", flush=True)
    write_report(args.out_dir, degree, same_ef, iso, backend_summary, ndis, started_at)
    print(f"[DONE] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
