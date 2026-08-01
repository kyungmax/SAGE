#!/usr/bin/env python3
"""Run DARTH target-0.99 on CohereWiki and MSMARCO with full online queries.

The underlying DARTH runner accepts a single ``--online-query-num`` value.  This
orchestrator therefore launches one subprocess per dataset and reads the full
query count from each HDF5 ``test`` matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parents[1]
SHIM = SCRIPT_DIR / "run_darth_target099_simd_current.py"
DEFAULT_PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/index"),
    )
).expanduser()

DATASETS = {
    "cohere": {
        "hdf5": DATASET_ROOT / "cohere-768-angular.hdf5",
        "darth_name": "cohere-768-angular",
    },
    "msmarco": {
        "hdf5": DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5",
        "darth_name": "msmarco-v1-openai-ada2-full-ip",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_labels(value: str) -> list[str]:
    labels = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(labels) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; choices={sorted(DATASETS)}")
    return labels


def hdf5_query_count(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as h5f:
        return int(h5f["test"].shape[0])


def read_summary_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_logged(cmd: list[str], log_path: Path, *, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(" ".join(cmd))
        return 0
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "24")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(EXP_ROOT / "darth_target099_simd_cohere_msmarco_fullquery"))
    parser.add_argument("--datasets", default="cohere,msmarco")
    parser.add_argument("--common-index-root", default=str(FAISS_INDEX_ROOT))
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--learn-queries", type=int, default=10000)
    parser.add_argument("--validation-queries", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=24, help="Offline/ground-truth/TData threads. Online is single-thread in DARTH.")
    parser.add_argument("--train-threads", type=int, default=24)
    parser.add_argument("--allow-existing-run-root", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = parse_labels(args.datasets)
    out_root = Path(args.out_root).expanduser().resolve()

    plan = []
    for label in labels:
        hdf5 = DATASETS[label]["hdf5"]
        if not hdf5.exists():
            raise FileNotFoundError(f"missing HDF5 for {label}: {hdf5}")
        query_count = hdf5_query_count(hdf5)
        run_root = out_root / "runs" / label
        cmd = [
            sys.executable,
            str(SHIM),
            "--datasets",
            label,
            "--run-root",
            str(run_root),
            "--common-index-root",
            str(Path(args.common_index_root).expanduser()),
            "--m",
            str(args.m),
            "--ef-construction",
            str(args.ef_construction),
            "--ef-search",
            str(args.ef_search),
            "--k",
            str(args.k),
            "--target-recall",
            str(args.target_recall),
            "--online-query-num",
            str(query_count),
            "--learn-queries",
            str(args.learn_queries),
            "--validation-queries",
            str(args.validation_queries),
            "--threads",
            str(args.threads),
            "--train-threads",
            str(args.train_threads),
        ]
        if args.force_index:
            cmd.append("--force-index")
        if args.allow_existing_run_root:
            cmd.append("--allow-existing-run-root")
        if args.dry_run:
            cmd.append("--dry-run")
        plan.append({"label": label, "hdf5": str(hdf5), "full_query_count": query_count, "run_root": str(run_root), "cmd": cmd})

    if args.dry_run:
        print(json.dumps({"status": "dry-run-ok", "out_root": str(out_root), "plan": plan}, indent=2))
        return 0

    if out_root.exists() and not args.allow_existing_run_root:
        raise FileExistsError(f"out root already exists: {out_root}; choose a fresh --out-root or pass --allow-existing-run-root")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "created_at": now(),
                "purpose": "DARTH target0.99 SIMD, cohere/msmarco only, full online query set",
                "paper_settings": {
                    "target_recall": 0.99,
                    "ef_search": 1000,
                    "offline_threads": 24,
                    "online_threads": 1,
                },
                "plan": plan,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = []
    for item in plan:
        label = item["label"]
        print(f"[{now()}] DARTH {label}: start full_query_count={item['full_query_count']}", flush=True)
        rc = run_logged(item["cmd"], out_root / "logs" / f"{label}.darth.log", dry_run=False)
        if rc != 0:
            raise RuntimeError(f"DARTH {label} failed rc={rc}; see {out_root / 'logs' / f'{label}.darth.log'}")
        summary_rows = read_summary_csv(Path(item["run_root"]) / "darth/results/offline_cost_summary.csv")
        for row in summary_rows:
            row["full_query_count_expected"] = str(item["full_query_count"])
            row["dataset_run_root"] = item["run_root"]
            rows.append(row)
        print(f"[{now()}] DARTH {label}: done", flush=True)

    write_summary(out_root / "summary/darth_target099_simd_fullquery_summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
