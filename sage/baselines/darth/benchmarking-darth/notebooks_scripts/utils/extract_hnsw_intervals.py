#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent.parent / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from darth_shared_config import (
    SHARED_INTERVAL_ROOT,
    SHARED_TRAINING_ROOT,
    get_dataset_specs,
    shared_interval_json_path,
    shared_training_csv_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DARTH ipi/mpi interval JSONs from shared HNSW early-stop-training CSVs."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional subset of dataset names.")
    parser.add_argument("--training-root", default=str(SHARED_TRAINING_ROOT), help="Shared training CSV root.")
    parser.add_argument("--output-root", default=str(SHARED_INTERVAL_ROOT), help="Shared interval JSON root.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=2000)
    parser.add_argument("--query-sample-size", type=int, default=10000)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--ipi-divisor", type=float, default=2.0)
    parser.add_argument("--mpi-divisor", type=float, default=10.0)
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional summary CSV. Defaults to <output-root>/summary_intervals_k<K>_efC<EFC>_efS<EFS>_rt<RT>_qs<QS>.csv",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def round_interval(value: float) -> int:
    return int(max(1, round(value)))


def compute_first_reaching_dists(training_csv: Path, *, target_recall: float, chunksize: int) -> dict[int, float]:
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


def default_summary_path(output_root: Path, *, k: int, efc: int, efs: int, target_recall: float, qs: int) -> Path:
    rt_tag = f"{target_recall:.2f}".replace(".", "p")
    return output_root / f"summary_intervals_k{k}_efC{efc}_efS{efs}_rt{rt_tag}_qs{qs}.csv"


def main() -> int:
    args = parse_args()
    training_root = Path(args.training_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    specs = get_dataset_specs(args.datasets)
    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else default_summary_path(
            output_root,
            k=int(args.k),
            efc=int(args.ef_construction),
            efs=int(args.ef_search),
            target_recall=float(args.target_recall),
            qs=int(args.query_sample_size),
        )
    )

    rows: list[dict] = []
    for spec in specs:
        training_csv = shared_training_csv_path(
            spec.dataset,
            k=int(args.k),
            m=int(args.m),
            efc=int(args.ef_construction),
            efs=int(args.ef_search),
            qs=int(args.query_sample_size),
            li=int(args.logging_interval),
        )
        if training_root != SHARED_TRAINING_ROOT:
            training_csv = (
                training_root
                / spec.dataset
                / f"k{int(args.k)}"
                / f"M{int(args.m)}_efC{int(args.ef_construction)}_efS{int(args.ef_search)}_qs{int(args.query_sample_size)}_li{int(args.logging_interval)}.csv"
            )
        if not training_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {training_csv}")

        started = time.time()
        best_by_qid = compute_first_reaching_dists(
            training_csv,
            target_recall=float(args.target_recall),
            chunksize=int(args.chunksize),
        )
        compute_seconds = time.time() - started

        if not best_by_qid:
            raise ValueError(
                f"No queries reached target recall {args.target_recall:.2f} in {training_csv}"
            )

        dists = list(best_by_qid.values())
        avg_dists = sum(dists) / len(dists)
        ipi = round_interval(avg_dists / float(args.ipi_divisor))
        mpi = round_interval(avg_dists / float(args.mpi_divisor))
        if mpi > ipi:
            mpi = ipi

        interval_json = shared_interval_json_path(
            spec.dataset,
            k=int(args.k),
            efc=int(args.ef_construction),
            efs=int(args.ef_search),
            target_recall=float(args.target_recall),
            qs=int(args.query_sample_size),
        )
        if output_root != SHARED_INTERVAL_ROOT:
            interval_json = (
                output_root
                / f"{spec.dataset}_k{int(args.k)}_efC{int(args.ef_construction)}_efS{int(args.ef_search)}_rt{float(args.target_recall):.2f}_qs{int(args.query_sample_size)}.json"
            )

        payload = {
            "avg_dists_rt": avg_dists,
            "initial_prediction_interval": ipi,
            "min_prediction_interval": mpi,
            "queries_reaching_target": len(best_by_qid),
            "target_recall": float(args.target_recall),
            "compute_seconds": compute_seconds,
            "training_csv": str(training_csv),
        }
        interval_json.parent.mkdir(parents=True, exist_ok=True)
        interval_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        row = {
            "dataset": spec.dataset,
            "k": int(args.k),
            "M": int(args.m),
            "efC": int(args.ef_construction),
            "efS": int(args.ef_search),
            "query_sample_size": int(args.query_sample_size),
            "target_recall": float(args.target_recall),
            "avg_dists_rt": avg_dists,
            "ipi": ipi,
            "mpi": mpi,
            "queries_reaching_target": len(best_by_qid),
            "compute_seconds": compute_seconds,
            "training_csv": str(training_csv),
            "interval_json": str(interval_json),
        }
        rows.append(row)
        print(
            f"[OK] {spec.dataset}: avg_dists_rt={avg_dists:.3f}, "
            f"ipi={ipi}, mpi={mpi}, compute={compute_seconds:.2f}s"
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SUMMARY] {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
