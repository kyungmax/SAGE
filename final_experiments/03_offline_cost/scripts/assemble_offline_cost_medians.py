#!/usr/bin/env python3
"""Assemble FAISS/hnswlib offline-cost median CSVs into one backend table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def default_backend_csv(root: Path, backend: str) -> Path:
    return root / f"offline_cost_main8_{backend}_SIMD_on_24t" / "final" / f"{backend}_offline_cost_median.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--faiss-csv", type=Path, default=None)
    parser.add_argument("--hnswlib-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def index_rows(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["dataset"]): dict(row) for row in rows}


def metric(row: dict[str, str] | None, name: str) -> str:
    if row is None:
        return ""
    return row.get(name, "")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    faiss_csv = args.faiss_csv.expanduser().resolve() if args.faiss_csv else default_backend_csv(root, "faiss")
    hnsw_csv = args.hnswlib_csv.expanduser().resolve() if args.hnswlib_csv else default_backend_csv(root, "hnswlib")
    out = args.out.expanduser().resolve() if args.out else root / "offline_cost_backend_medians.csv"
    missing = [str(path) for path in (faiss_csv, hnsw_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing backend median CSV(s): {missing}")

    faiss = index_rows(read_csv(faiss_csv))
    hnsw = index_rows(read_csv(hnsw_csv))
    datasets = list(dict.fromkeys([*faiss.keys(), *hnsw.keys()]))
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        frow = faiss.get(dataset)
        hrow = hnsw.get(dataset)
        rows.append(
            {
                "dataset": dataset,
                "dataset_file": metric(frow or hrow, "dataset_file"),
                "points": metric(frow or hrow, "points"),
                "dimensions": metric(frow or hrow, "dimensions"),
                "faiss_samp_s": metric(frow, "paper_samp_s"),
                "faiss_select_s": metric(frow, "paper_select_s"),
                "faiss_eval_s": metric(frow, "paper_eval_s"),
                "faiss_total_s": metric(frow, "paper_total_s"),
                "hnswlib_samp_s": metric(hrow, "paper_samp_s"),
                "hnswlib_select_s": metric(hrow, "paper_select_s"),
                "hnswlib_eval_s": metric(hrow, "paper_eval_s"),
                "hnswlib_total_s": metric(hrow, "paper_total_s"),
                "faiss_source": str(faiss_csv),
                "hnswlib_source": str(hnsw_csv),
            }
        )
    fieldnames = [
        "dataset",
        "dataset_file",
        "points",
        "dimensions",
        "faiss_samp_s",
        "faiss_select_s",
        "faiss_eval_s",
        "faiss_total_s",
        "hnswlib_samp_s",
        "hnswlib_select_s",
        "hnswlib_eval_s",
        "hnswlib_total_s",
        "faiss_source",
        "hnswlib_source",
    ]
    write_csv(out, rows, fieldnames)
    print(f"[DONE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
