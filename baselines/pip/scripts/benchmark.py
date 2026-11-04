#!/usr/bin/env python3
"""Compare vanilla HNSW and PiP on one index, with exact-neighbor recall."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from pip_runtime import ROOT, DEFAULT_PYTHON_PATH, STAT_FIELDS, load_faiss, parameters, search_with_stats


def csv_values(value, convert):
    return list(dict.fromkeys(convert(item) for item in value.split(",")))


def recall(ids, truth, k):
    return float(np.mean([
        len(set(row[row >= 0]) & set(target[:k])) / k
        for row, target in zip(ids, truth)
    ]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faiss-python-path", type=Path, default=DEFAULT_PYTHON_PATH)
    ap.add_argument("--dataset", type=Path, help="HDF5 with train/test and optional neighbors")
    ap.add_argument("--index", type=Path, help="Existing FAISS IndexHNSWFlat for this dataset")
    ap.add_argument("--metric", choices=("auto", "l2", "ip", "cosine"), default="auto")
    ap.add_argument("--nb", type=int, default=20000, help="Synthetic database size")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--nq", type=int, default=256)
    ap.add_argument("--max-base", type=int, default=0, help="Limit HDF5 train rows; recompute exact GT")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", default="64,128,256")
    ap.add_argument("--thresholds", default="0.95")
    ap.add_argument("--patiences", default="-1", help="Comma-separated; -1 means Lucene default")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--build-threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", type=Path, default=ROOT / "runs" / "smoke")
    args = ap.parse_args()
    for name in ("nb", "dim", "nq", "k", "threads", "build_threads", "repeats", "M", "ef_construction"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_base < 0:
        ap.error("--max-base must be non-negative")
    if args.index and not args.dataset:
        ap.error("--index requires --dataset")
    efs = csv_values(args.ef, int)
    thresholds = csv_values(args.thresholds, float)
    patiences = csv_values(args.patiences, int)
    if any(ef < args.k for ef in efs):
        ap.error("Use efSearch >= k for the comparison")
    if any(not 0 < t <= 1 for t in thresholds) or any(p != -1 and p < 1 for p in patiences):
        ap.error("Thresholds must be in (0,1]; patience must be -1 or a positive Delta")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        ap.error(f"Output directory is not empty: {output}; choose a new --output")
    faiss = load_faiss(args.faiss_python_path)
    rng = np.random.default_rng(args.seed)
    truth = None
    metric = args.metric
    if args.dataset:
        import h5py
        with h5py.File(args.dataset, "r") as data:
            name = data.attrs.get("distance", "")
            if isinstance(name, bytes):
                name = name.decode()
            dataset_metric = {"euclidean": "l2", "angular": "cosine", "ip": "ip", "inner_product": "ip"}.get(str(name))
            if metric == "auto":
                metric = dataset_metric
                if metric is None:
                    ap.error("Unknown HDF5 distance attribute; specify --metric")
            nb = min(args.max_base or len(data["train"]), len(data["train"]))
            xb = np.asarray(data["train"][:nb], dtype="float32", order="C")
            xq = np.asarray(data["test"][:args.nq], dtype="float32", order="C")
            if not args.max_base and metric == dataset_metric and "neighbors" in data and data["neighbors"].shape[1] >= args.k:
                truth = np.asarray(data["neighbors"][:len(xq), :args.k], dtype="int64")
    else:
        metric = "l2" if metric == "auto" else metric
        xb = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
        xq = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    if xb.ndim != 2 or xq.ndim != 2 or xb.shape[1] != xq.shape[1] or len(xq) == 0 or args.k > len(xb):
        ap.error("Invalid vector shapes, empty queries, or k exceeds database size")
    if metric == "cosine":
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
    faiss.omp_set_num_threads(args.build_threads)
    if args.index:
        index = faiss.read_index(str(args.index))
        if not isinstance(index, faiss.IndexHNSWFlat):
            ap.error("This runner supports IndexHNSWFlat")
        if (index.ntotal, index.d, index.metric_type) != (len(xb), xb.shape[1], faiss_metric):
            ap.error("Index size, dimension, or metric does not match the dataset")
        # Check row ordering/normalization on a deterministic sample as well.
        sample_ids = np.unique(np.linspace(0, len(xb) - 1, min(17, len(xb)), dtype="int64"))
        stored = index.reconstruct_batch(sample_ids)
        if not np.allclose(stored, xb[sample_ids], rtol=1e-5, atol=1e-6):
            ap.error("Stored vectors do not match dataset row IDs or normalization")
        # Reading a prebuilt graph preserves its exact edges and node ordering.
        build_seconds = None
    else:
        index = faiss.IndexHNSWFlat(xb.shape[1], args.M, faiss_metric)
        index.hnsw.efConstruction = args.ef_construction
        started = time.perf_counter()
        index.add(xb)
        build_seconds = time.perf_counter() - started
    faiss.omp_set_num_threads(args.threads)
    truth_source = "provided HDF5 neighbors"
    if truth is None:
        exact = faiss.IndexFlat(xb.shape[1], faiss_metric)
        exact.add(xb)
        _, truth = exact.search(xq, args.k)
        truth_source = "computed exact IndexFlat neighbors"
        del exact
    if truth.shape != (len(xq), args.k) or np.any(truth < 0) or np.any(truth >= len(xb)):
        ap.error("Ground-truth IDs must be valid row IDs for the supplied dataset")

    configs = []
    for ef in efs:
        configs.append({"method": "vanilla", "ef": ef, "threshold": None, "patience": None})
        for threshold, patience in itertools.product(thresholds, patiences):
            configs.append({"method": "pip", "ef": ef, "threshold": threshold, "patience": patience})
    params = [parameters(faiss, c["ef"], pip=c["method"] == "pip",
                         threshold=c["threshold"], patience=c["patience"]) for c in configs]

    # Gate the benchmark on identical off-path results and graph work.
    for ef in efs:
        vanilla = search_with_stats(faiss, index, xq, args.k, parameters(faiss, ef))
        disabled = search_with_stats(faiss, index, xq, args.k, parameters(faiss, ef, pip=True, enabled=False))
        np.testing.assert_array_equal(vanilla[0], disabled[0])
        np.testing.assert_array_equal(vanilla[1], disabled[1])
        assert vanilla[2] == disabled[2], "PiP-off changed graph work"

    # Warm each configuration, then shuffle configuration order each repetition.
    for p in params:
        index.search(xq, args.k, params=p)
    samples = [[] for _ in configs]
    records = []
    for repeat in range(args.repeats):
        for pos in rng.permutation(len(configs)):
            faiss.cvar.hnsw_stats.reset()
            started = time.perf_counter()
            _, ids = index.search(xq, args.k, params=params[pos])
            elapsed = time.perf_counter() - started
            stats = {key: int(getattr(faiss.cvar.hnsw_stats, key)) for key in STAT_FIELDS}
            result = {**configs[pos], "repeat": repeat, "seconds": elapsed,
                      "recall_at_k": recall(ids, truth, args.k), **stats}
            samples[pos].append(result)
            records.append(result)
    summary = []
    for config, values in zip(configs, samples):
        seconds = statistics.median(v["seconds"] for v in values)
        summary.append({**config, "k": args.k, "nq": len(xq), "threads": args.threads,
                        "resolved_patience": (max(7, int(args.k * 0.3)) if config["patience"] == -1 else config["patience"]),
                        "recall_at_k": statistics.mean(v["recall_at_k"] for v in values),
                        "qps": len(xq) / seconds, "median_batch_seconds": seconds,
                        "distance_computations_per_query": statistics.mean(v["ndis"] for v in values) / len(xq),
                        "level0_expansions_per_query": statistics.mean(v["nexpan"] for v in values) / len(xq),
                        "pip_stop_fraction": statistics.mean(v["npip_stops"] for v in values) / len(xq)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", summary), ("repetitions.csv", records)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "faiss_module": faiss.__file__, "faiss_version": faiss.__version__,
                "compile_options": faiss.get_compile_options(), "platform": platform.platform(),
                "metric": metric, "nb": len(xb), "dimension": xb.shape[1], "nq": len(xq),
                "M": index.hnsw.nb_neighbors(1), "ef_construction": index.hnsw.efConstruction,
                "build_seconds": build_seconds, "ground_truth": truth_source,
                "disabled_equivalence_passed": True,
                "timing": "Median warmed batch time; randomized config order; Python search call included"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"FAISS: {faiss.__file__}")
    print(f"Data: {len(xb)} x {xb.shape[1]}, {len(xq)} queries, {metric}, k={args.k}")
    print("method   ef threshold patience recall      QPS  distances/query expansions/query stop_fraction")
    for row in summary:
        print(f'{row["method"]:7s} {row["ef"]:3d} {str(row["threshold"]):>9s} '
              f'{str(row["resolved_patience"]):>8s} {row["recall_at_k"]:.4f} '
              f'{row["qps"]:8.1f} {row["distance_computations_per_query"]:16.1f} '
              f'{row["level0_expansions_per_query"]:16.1f} {row["pip_stop_fraction"]:.3f}')
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
