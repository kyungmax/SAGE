#!/usr/bin/env python3
"""Run DARTH early-stop-training with 10K training queries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
REUSE_ROOT = PROJECT_ROOT / "index/m32_efc500_target095_adaef_darth_efs1000_20260603"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "index/darth_m32_efc500_efs1000_training10k_5datasets"
DARTH_ROOT = Path(os.environ.get("SAGE_DARTH_ROOT", str(REPO_ROOT / "baselines/darth/benchmarking-darth"))).expanduser()
DARTH_BIN = Path(os.environ.get("SAGE_DARTH_BIN", str(DARTH_ROOT / "build-simd-avx512/hnsw-test/hnsw_test"))).expanduser()


DATASETS = {
    "sift-100M": ("sift-100M-euclidean", "l2"),
    "deep-100M": ("deep-100M-angular", "ip"),
    "msmarco": ("msmarco-v1-openai-ada2-full-ip", "ip"),
    "glove-100": ("glove-100-angular", "ip"),
    "nytimes": ("nytimes-256-angular", "ip"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--datasets",
        default="sift-100M,deep-100M,msmarco,glove-100,nytimes",
    )
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--logging-interval", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(target)


def training_queries(processed_dir: Path) -> int:
    learn = processed_dir / "learn.fvecs"
    with learn.open("rb") as fh:
        dim = int.from_bytes(fh.read(4), "little", signed=True)
    return learn.stat().st_size // ((dim + 1) * 4)


def link_index(run_root: Path, darth_name: str) -> Path:
    source = REUSE_ROOT / "darth/index" / darth_name / f"{darth_name}.M32.efC500.index"
    target = run_root / "darth/index" / darth_name / f"{darth_name}.M32.efC500.index"
    ensure_symlink(target, source)
    return target


def run_one(run_root: Path, label: str, args: argparse.Namespace) -> dict[str, object]:
    darth_name, metric = DATASETS[label]
    processed_dir = run_root / "darth/processed" / darth_name
    query_num = training_queries(processed_dir)
    if query_num != 10000:
        raise ValueError(f"{label} is not 10K learn: {query_num}")

    index_path = link_index(run_root, darth_name)
    output = (
        run_root
        / "darth/training"
        / darth_name
        / f"k{args.k}"
        / f"M{args.m}_efC{args.ef_construction}_efS{args.ef_search}_qs{query_num}_li{args.logging_interval}.txt"
    )
    wrapper_json = run_root / "darth/results" / label / "training.wrapper.json"
    log_path = run_root / "logs/darth" / f"{label}.training.log"
    if args.skip_existing and wrapper_json.exists() and output.exists():
        return json.loads(wrapper_json.read_text())

    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(DARTH_BIN),
        "--dataset",
        darth_name,
        "--M",
        str(args.m),
        "--efConstruction",
        str(args.ef_construction),
        "--efSearch",
        str(args.ef_search),
        "--query-num",
        str(query_num),
        "--k",
        str(args.k),
        "--mode",
        "early-stop-training",
        "--logging-interval",
        str(args.logging_interval),
        "--index-filepath",
        str(index_path),
        "--dataset-dir-prefix",
        str(run_root / "darth/processed"),
        "--query-type",
        "training",
        "--metric",
        metric,
        "--output",
        str(output),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "24"
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    wall_s = time.perf_counter() - started
    payload: dict[str, object] = {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "wall_s": float(wall_s),
        "training_queries": int(query_num),
        "training_log": str(output),
        "log_path": str(log_path),
        "cmd": cmd,
    }
    write_json(wrapper_json, payload)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} training failed rc={proc.returncode}; see {log_path}")
    return payload


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    if not DARTH_BIN.exists():
        raise FileNotFoundError(f"missing DARTH binary: {DARTH_BIN}")
    labels = [part.strip() for part in args.datasets.split(",") if part.strip()]
    summary = []
    for label in labels:
        print(f"[{now()}] training start: {label}", flush=True)
        payload = run_one(run_root, label, args)
        print(f"[{now()}] training done: {label} wall_s={payload['wall_s']:.3f}", flush=True)
        summary.append({"label": label, **payload})
    write_json(run_root / "darth/results/training10k_summary.json", {"status": "ok", "datasets": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
