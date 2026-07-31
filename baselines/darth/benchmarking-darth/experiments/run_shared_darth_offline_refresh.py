#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import h5py
import hnswlib
import numpy as np

from darth_shared_config import (
    LOCAL_PREDICTOR_ROOT,
    SHARED_DATASET_ROOT,
    SHARED_INTERVAL_ROOT,
    SHARED_TRAINING_ROOT,
    get_dataset_specs,
    predictor_model_path,
    shared_index_path,
    shared_interval_json_path,
    shared_training_csv_path,
)


SUMMARY_RE = re.compile(
    r"Index\[M=(?P<M>\d+), efC=(?P<efc>\d+), efS=(?P<efs>\d+)\]"
    r"IndexTime:\s*(?P<index_time>[0-9.]+)s,\s*"
    r"SearchTime:\s*(?P<search_time>[0-9.]+)s,\s*"
    r"TotalTime:\s*(?P<total_time>[0-9.]+)s,\s*"
    r"Avg_Recall@\d+:\s*(?P<avg_recall>[0-9.]+),\s*"
    r"P1_Recall@\d+:\s*(?P<p1_recall>[0-9.]+),\s*"
    r"P5_Recall@\d+:\s*(?P<p5_recall>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh shared-root DARTH offline artifacts for M=16 / efConstruction=500 / k=10, "
            "including stage timing summaries."
        )
    )
    parser.add_argument("--binary", required=True, help="Path to benchmarking-darth hnsw_test binary.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional subset of datasets.")
    parser.add_argument("--results-root", default="", help="Local result directory root.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=2000)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--training-query-num", type=int, default=10000)
    parser.add_argument("--index-build-query-num", type=int, default=1)
    parser.add_argument("--index-build-query-type", default="validation")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--gt-base-batch-size", type=int, default=65536)
    parser.add_argument("--gt-query-batch-size", type=int, default=2048)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--predictor-python",
        default="",
        help="Optional Python executable to use for predictor training. Defaults to the hnsw conda env if present.",
    )
    parser.add_argument("--skip-gt-timing", action="store_true")
    parser.add_argument("--skip-index-build", action="store_true")
    parser.add_argument("--skip-training-data", action="store_true")
    parser.add_argument("--skip-intervals", action="store_true")
    parser.add_argument("--skip-predictor-training", action="store_true")
    parser.add_argument("--force-index-build", action="store_true")
    parser.add_argument("--force-training-data", action="store_true")
    parser.add_argument("--force-intervals", action="store_true")
    return parser.parse_args()


def resolve_predictor_python(path_value: str) -> str:
    if path_value:
        return str(Path(path_value).expanduser().resolve())
    env_override = os.environ.get("DARTH_PREDICTOR_PYTHON", "") or os.environ.get("HNSW_PYTHON", "")
    if env_override:
        return str(Path(env_override).expanduser().resolve())
    return os.environ.get("SAGE_PYTHON", "python3")


def resolve_results_root(path_value: str, *, m: int, efc: int, k: int) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return (
        Path(__file__).resolve().parent
        / "results"
        / f"darth_offline_refresh_M{m}_efC{efc}_k{k}"
    )


def load_processed_metadata(dataset: str) -> dict:
    metadata_path = SHARED_DATASET_ROOT / dataset / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def normalize_for_metric(matrix: np.ndarray, source_metric: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if source_metric != "angular":
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def load_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    rows = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return rows[inverse]


def measure_learn_gt_timing(
    dataset: str,
    *,
    learn_queries: int,
    seed: int,
    threads: int,
    base_batch_size: int,
    query_batch_size: int,
) -> dict:
    metadata = load_processed_metadata(dataset)
    source_hdf5 = Path(metadata["source_hdf5"]).expanduser().resolve()
    source_metric = str(metadata["source_metric"])
    groundtruth_k = int(metadata["groundtruth_k"])

    with h5py.File(source_hdf5, "r") as h5f:
        train_dataset = h5f["train"]
        train_size = int(train_dataset.shape[0])
        dim = int(train_dataset.shape[1])

        rng = np.random.default_rng(seed)
        learn_indices = rng.choice(train_size, size=learn_queries, replace=False)
        learn_vectors = load_rows(train_dataset, learn_indices)

        space = "l2" if source_metric == "euclidean" else "ip"
        bf = hnswlib.BFIndex(space=space, dim=dim)
        bf.init_index(max_elements=train_size)
        bf.set_num_threads(threads)

        build_start = time.time()
        for start in range(0, train_size, base_batch_size):
            end = min(start + base_batch_size, train_size)
            chunk = np.asarray(train_dataset[start:end], dtype=np.float32)
            bf.add_items(
                normalize_for_metric(chunk, source_metric),
                ids=np.arange(start, end, dtype=np.int64),
            )
        exact_index_build_seconds = time.time() - build_start

        search_start = time.time()
        for start in range(0, len(learn_vectors), query_batch_size):
            end = min(start + query_batch_size, len(learn_vectors))
            bf.knn_query(
                normalize_for_metric(learn_vectors[start:end], source_metric),
                k=groundtruth_k,
            )
        learn_gt_search_seconds = time.time() - search_start

    return {
        "dataset": dataset,
        "source_hdf5": str(source_hdf5),
        "source_metric": source_metric,
        "groundtruth_k": groundtruth_k,
        "learn_queries": learn_queries,
        "seed": seed,
        "exact_index_build_seconds": exact_index_build_seconds,
        "learn_gt_search_seconds": learn_gt_search_seconds,
        "learn_gt_total_seconds": exact_index_build_seconds + learn_gt_search_seconds,
    }


def run_command(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return combined


def parse_hnsw_summary(text: str) -> dict:
    for line in reversed(text.splitlines()):
        match = SUMMARY_RE.search(line)
        if match:
            payload = match.groupdict()
            return {
                "index_time": float(payload["index_time"]),
                "search_time": float(payload["search_time"]),
                "total_time": float(payload["total_time"]),
                "avg_recall": float(payload["avg_recall"]),
                "p1_recall": float(payload["p1_recall"]),
                "p5_recall": float(payload["p5_recall"]),
            }
    raise ValueError("Could not parse hnsw_test summary line.")


def compute_first_reaching_dists(training_csv: Path, *, target_recall: float, chunksize: int = 1_000_000) -> dict[int, float]:
    import pandas as pd

    best_by_qid: dict[int, float] = {}
    for chunk in pd.read_csv(training_csv, usecols=["qid", "dists", "r"], chunksize=chunksize):
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
    return best_by_qid


def write_markdown(summary_csv: Path, rows: list[dict], *, args: argparse.Namespace, predictor_summary_json: Path) -> Path:
    md_path = summary_csv.with_suffix(".md")
    lines = [
        f"# DARTH Offline Refresh Summary (M={args.m}, efC={args.ef_construction}, efS={args.ef_search}, k={args.k})",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Dataset | HNSW Build (s) | Learn GT (s) | Train-Data Total (s) | Interval (s) | Predictor Fit (s) | Predictor Total (s) | IPI | MPI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {index_build_seconds} | {learn_gt_total_seconds} | {training_data_total_seconds} | "
            "{interval_compute_seconds} | {predictor_fit_seconds} | {predictor_total_seconds} | {ipi} | {mpi} |".format(
                dataset=row["dataset"],
                index_build_seconds=f"{row.get('index_build_seconds', ''):.6f}" if row.get("index_build_seconds") is not None else "",
                learn_gt_total_seconds=f"{row.get('learn_gt_total_seconds', ''):.6f}" if row.get("learn_gt_total_seconds") is not None else "",
                training_data_total_seconds=f"{row.get('training_data_total_seconds', ''):.6f}" if row.get("training_data_total_seconds") is not None else "",
                interval_compute_seconds=f"{row.get('interval_compute_seconds', ''):.6f}" if row.get("interval_compute_seconds") is not None else "",
                predictor_fit_seconds=f"{row.get('predictor_fit_seconds', ''):.6f}" if row.get("predictor_fit_seconds") is not None else "",
                predictor_total_seconds=f"{row.get('predictor_total_seconds', ''):.6f}" if row.get("predictor_total_seconds") is not None else "",
                ipi=row.get("ipi", ""),
                mpi=row.get("mpi", ""),
            )
        )
    lines += [
        "",
        "Sources:",
        f"- Summary CSV: `{summary_csv}`",
        f"- Predictor training summary JSON: `{predictor_summary_json}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def flush_summary(
    *,
    rows: list[dict],
    summary_csv: Path,
    args: argparse.Namespace,
    predictor_summary_json: Path,
) -> None:
    if not rows:
        return
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(summary_csv, rows, args=args, predictor_summary_json=predictor_summary_json)


def main() -> int:
    args = parse_args()
    binary = Path(args.binary).expanduser().resolve()
    results_root = resolve_results_root(args.results_root, m=args.m, efc=args.ef_construction, k=args.k)
    logs_root = results_root / "logs"
    results_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["NUMEXPR_NUM_THREADS"] = str(args.threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
    env["BLIS_NUM_THREADS"] = str(args.threads)

    rows: list[dict] = []
    specs = get_dataset_specs(args.datasets)

    for spec in specs:
        row: dict = {
            "dataset": spec.dataset,
            "k": int(args.k),
            "M": int(args.m),
            "efC": int(args.ef_construction),
            "efS": int(args.ef_search),
        }

        if not args.skip_gt_timing:
            gt_info = measure_learn_gt_timing(
                spec.dataset,
                learn_queries=int(args.training_query_num),
                seed=987,
                threads=int(args.threads),
                base_batch_size=int(args.gt_base_batch_size),
                query_batch_size=int(args.gt_query_batch_size),
            )
            row.update(
                {
                    "source_hdf5": gt_info["source_hdf5"],
                    "groundtruth_k": gt_info["groundtruth_k"],
                    "learn_gt_exact_index_build_seconds": gt_info["exact_index_build_seconds"],
                    "learn_gt_search_seconds": gt_info["learn_gt_search_seconds"],
                    "learn_gt_total_seconds": gt_info["learn_gt_total_seconds"],
                }
            )

        index_path = shared_index_path(spec.dataset, m=int(args.m), efc=int(args.ef_construction))
        build_output_csv = results_root / "index_build_outputs" / f"{spec.dataset}.csv"
        build_log = logs_root / "index_build" / f"{spec.dataset}.log"
        if not args.skip_index_build:
            if args.force_index_build or not index_path.exists():
                build_output_csv.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(binary),
                    "--dataset",
                    spec.dataset,
                    "--M",
                    str(args.m),
                    "--efConstruction",
                    str(args.ef_construction),
                    "--efSearch",
                    str(args.ef_search),
                    "--query-num",
                    str(args.index_build_query_num),
                    "--k",
                    str(args.k),
                    "--mode",
                    "no-early-stop",
                    "--index-filepath",
                    str(index_path),
                    "--dataset-dir-prefix",
                    str(SHARED_DATASET_ROOT) + "/",
                    "--query-type",
                    str(args.index_build_query_type),
                    "--output",
                    str(build_output_csv),
                    "--save-index",
                ]
                text = run_command(cmd, log_path=build_log, env=env)
                summary = parse_hnsw_summary(text)
                row["index_build_seconds"] = summary["index_time"]
                row["index_build_log"] = str(build_log)
            else:
                row["index_build_seconds"] = None
                row["index_build_log"] = ""

        training_csv = shared_training_csv_path(
            spec.dataset,
            k=int(args.k),
            m=int(args.m),
            efc=int(args.ef_construction),
            efs=int(args.ef_search),
            qs=int(args.training_query_num),
            li=int(args.logging_interval),
        )
        training_log = logs_root / "training_data" / f"{spec.dataset}.log"
        training_output_dir = training_csv.parent
        training_output_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_training_data:
            if args.force_training_data or not training_csv.exists():
                cmd = [
                    str(binary),
                    "--dataset",
                    spec.dataset,
                    "--M",
                    str(args.m),
                    "--efConstruction",
                    str(args.ef_construction),
                    "--efSearch",
                    str(args.ef_search),
                    "--query-num",
                    str(args.training_query_num),
                    "--k",
                    str(args.k),
                    "--mode",
                    "early-stop-training",
                    "--logging-interval",
                    str(args.logging_interval),
                    "--index-filepath",
                    str(index_path),
                    "--dataset-dir-prefix",
                    str(SHARED_DATASET_ROOT) + "/",
                    "--query-type",
                    "training",
                    "--output",
                    str(training_csv),
                ]
                text = run_command(cmd, log_path=training_log, env=env)
                summary = parse_hnsw_summary(text)
                row["training_data_index_load_seconds"] = summary["index_time"]
                row["training_data_search_seconds"] = summary["search_time"]
                row["training_data_total_seconds"] = summary["total_time"]
                row["training_data_log"] = str(training_log)
            else:
                row["training_data_log"] = ""

        interval_path = shared_interval_json_path(
            spec.dataset,
            k=int(args.k),
            efc=int(args.ef_construction),
            efs=int(args.ef_search),
            target_recall=float(args.target_recall),
            qs=int(args.training_query_num),
        )
        if not args.skip_intervals:
            if args.force_intervals or not interval_path.exists():
                started = time.time()
                best_by_qid = compute_first_reaching_dists(training_csv, target_recall=float(args.target_recall))
                if not best_by_qid:
                    raise ValueError(f"No queries reached target recall for {training_csv}")
                dists = list(best_by_qid.values())
                avg_dists = statistics.mean(dists)
                ipi = int(max(1, round(avg_dists / 2.0)))
                mpi = int(max(1, round(avg_dists / 10.0)))
                if mpi > ipi:
                    mpi = ipi
                interval_payload = {
                    "avg_dists_rt": avg_dists,
                    "initial_prediction_interval": ipi,
                    "min_prediction_interval": mpi,
                    "queries_reaching_target": len(best_by_qid),
                    "target_recall": float(args.target_recall),
                }
                interval_path.parent.mkdir(parents=True, exist_ok=True)
                interval_path.write_text(json.dumps(interval_payload, indent=2) + "\n", encoding="utf-8")
                row["interval_compute_seconds"] = time.time() - started
                row["ipi"] = ipi
                row["mpi"] = mpi
                row["interval_json"] = str(interval_path)
            else:
                payload = json.loads(interval_path.read_text(encoding="utf-8"))
                row["interval_compute_seconds"] = payload.get("compute_seconds")
                row["ipi"] = payload["initial_prediction_interval"]
                row["mpi"] = payload["min_prediction_interval"]
                row["interval_json"] = str(interval_path)

        rows.append(row)
        flush_summary(
            rows=rows,
            summary_csv=results_root / "offline_refresh_summary.csv",
            args=args,
            predictor_summary_json=results_root / "predictor_training_summary.json",
        )
        print(f"[DONE] {spec.dataset}")

    predictor_summary_json = results_root / "predictor_training_summary.json"
    if not args.skip_predictor_training:
        predictor_python = resolve_predictor_python(args.predictor_python)
        predictor_cmd = [
            predictor_python,
            str(
                Path(__file__).resolve().parent.parent
                / "notebooks_scripts"
                / "utils"
                / "train_predictors_from_intervals.py"
            ),
            "--interval-root",
            str(SHARED_INTERVAL_ROOT),
            "--training-root",
            str(SHARED_TRAINING_ROOT),
            "--output-root",
            str(LOCAL_PREDICTOR_ROOT),
            "--datasets",
            *[spec.dataset for spec in specs],
            "--k",
            str(args.k),
            "--m",
            str(args.m),
            "--ef-construction",
            str(args.ef_construction),
            "--ef-search",
            str(args.ef_search),
            "--li",
            str(args.logging_interval),
            "--n-estimators",
            str(args.n_estimators),
            "--learning-rate",
            str(args.learning_rate),
            "--num-threads",
            str(args.threads),
            "--summary-json",
            str(predictor_summary_json),
        ]
        run_command(predictor_cmd, log_path=logs_root / "predictor_training.log", env=env)
        predictor_payload = json.loads(predictor_summary_json.read_text(encoding="utf-8"))
        by_dataset = {item["dataset"]: item for item in predictor_payload["results"]}
        for row in rows:
            result = by_dataset.get(row["dataset"])
            if result is None:
                continue
            row["predictor_rows_used"] = result["rows_used"]
            row["predictor_fit_seconds"] = result["fit_seconds"]
            row["predictor_total_seconds"] = result["total_seconds"]
            row["predictor_model_path"] = result["model_path"]

    summary_csv = results_root / "offline_refresh_summary.csv"
    flush_summary(
        rows=rows,
        summary_csv=summary_csv,
        args=args,
        predictor_summary_json=predictor_summary_json,
    )
    md_path = summary_csv.with_suffix(".md")
    print(f"[SUMMARY] {summary_csv}")
    print(f"[SUMMARY] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
