#!/usr/bin/env python3
"""Run DARTH target-0.99 for the selected main4 datasets.

This wrapper reuses the preserved DARTH from-scratch runner, but fixes the
pieces needed for the current target-0.99 rerun:

* injects the selected main8 dataset specs;
* uses the metric-aware DARTH binary from the archived patched tree;
* adds an index-build stage before DARTH training when the fixed index is absent;
* keeps the original runner's LVec/GT/TData/Train/Online accounting.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
THIS_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = THIS_ROOT / "imported/final_implementation_reference/darth/scripts/run_paper_offline_fromscratch.py"
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
PATCHED_DARTH_ROOT = Path(
    os.environ.get("SAGE_DARTH_ROOT", str(REPO_ROOT / "baselines/darth/benchmarking-darth"))
).expanduser()
PATCHED_DARTH_BIN = Path(
    os.environ.get("SAGE_DARTH_BIN", str(PATCHED_DARTH_ROOT / "build-simd-avx512/hnsw-test/hnsw_test"))
).expanduser()
PATCHED_FAISS_LIB_DIR = Path(
    os.environ.get("SAGE_DARTH_FAISS_LIB_DIR", str(PATCHED_DARTH_ROOT / "build-simd-avx512/faiss"))
).expanduser()
VERIFIED_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        str(PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index"),
    )
).expanduser()
DEFAULT_RUN_ROOT = PROJECT_ROOT / "index/darth_m32_efc500_target099_main4"


DATASET_DEFS = {
    "agnews": {
        "label": "agnews",
        "darth_name": "agnews-mxbai-1024-euclidean",
        "hdf5": DATASET_ROOT / "agnews-mxbai-1024-euclidean.hdf5",
        "source_metric": "euclidean",
        "hnsw_metric": "l2",
    },
    "cohere": {
        "label": "cohere",
        "darth_name": "cohere-768-angular",
        "hdf5": DATASET_ROOT / "cohere-768-angular.hdf5",
        "source_metric": "angular",
        "hnsw_metric": "ip",
    },
    "landmark-nomic": {
        "label": "landmark-nomic",
        "darth_name": "landmark-nomic-768-angular",
        "hdf5": DATASET_ROOT / "landmark-nomic-768-angular.hdf5",
        "source_metric": "angular",
        "hnsw_metric": "ip",
    },
    "msmarco": {
        "label": "msmarco",
        "darth_name": "msmarco-v1-openai-ada2-full-ip",
        "hdf5": DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5",
        "source_metric": "ip",
        "hnsw_metric": "ip",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def disk_snapshot(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except FileNotFoundError:
                pass
    return int(total)


def prepend_env_path(name: str, value: Path) -> None:
    old = os.environ.get(name, "")
    value_s = str(value)
    if not old:
        os.environ[name] = value_s
    elif value_s not in old.split(":"):
        os.environ[name] = value_s + ":" + old


def load_runner():
    # The preserved runner imports hnswlib but does not use it. The local hnsw
    # env has a broken editable hnswlib install, so stub it before import.
    sys.modules.setdefault("hnswlib", types.ModuleType("hnswlib"))
    spec = importlib.util.spec_from_file_location("darth_fromscratch_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_labels(value: str) -> list[str]:
    labels = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(labels) - set(DATASET_DEFS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}; choices={sorted(DATASET_DEFS)}")
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--common-index-root",
        default=str(VERIFIED_FAISS_INDEX_ROOT),
        help="FAISS IndexHNSWFlat root reused by DARTH TData/Online. Defaults to the verified main8 M32 efC500 index set.",
    )
    parser.add_argument("--datasets", default="agnews,cohere,landmark-nomic,msmarco")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--ef-search", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.99)
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
    parser.add_argument("--allow-existing-run-root", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--keep-base-after-db-load", action="store_true")
    parser.add_argument("--keep-training-log", action="store_true")
    parser.add_argument("--keep-base-after-online", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the planned run without writing artifacts.",
    )
    return parser.parse_args()


def configure_runner(runner, args: argparse.Namespace, run_root: Path):
    prepend_env_path("LD_LIBRARY_PATH", PATCHED_FAISS_LIB_DIR)
    runner.DARTH_BIN = PATCHED_DARTH_BIN
    runner.COMMON_INDEX_ROOT = Path(args.common_index_root).expanduser().resolve()
    patch_euclidean_query_groundtruth(runner)
    for label, payload in DATASET_DEFS.items():
        runner.DATASETS[label] = runner.DatasetSpec(**payload)
    return [runner.DATASETS[label] for label in parse_labels(args.datasets)]


def patch_euclidean_query_groundtruth(runner) -> None:
    original = runner.write_query_groundtruth_from_hdf5

    def patched(spec, processed_dir: Path) -> dict:
        if spec.source_metric != "euclidean":
            return original(spec, processed_dir)

        with runner.h5py.File(spec.hdf5, "r") as h5f:
            neighbors = runner.np.asarray(h5f["neighbors"], dtype=runner.np.int32)
            if "distances_sq" in h5f:
                scores = runner.np.asarray(h5f["distances_sq"], dtype=runner.np.float32)
                score_source = "hdf5/distances_sq"
            elif "distances" in h5f:
                distances = runner.np.asarray(h5f["distances"], dtype=runner.np.float32)
                scores = runner.np.square(distances, dtype=runner.np.float32)
                score_source = "square(hdf5/distances)"
            else:
                scores = runner.compute_scores_from_neighbors_hdf5(
                    spec,
                    h5f["train"],
                    h5f["test"],
                    neighbors,
                )
                score_source = "sq_l2(hdf5/test, hdf5/train[neighbors])"

        runner.write_ivecs_matrix(processed_dir / "query.groundtruth.ivecs", neighbors)
        runner.write_fvecs_matrix(processed_dir / "query.groundtruth.fvecs", scores)
        return {
            "queries": int(neighbors.shape[0]),
            "gt_k": int(neighbors.shape[1]),
            "score_source": score_source,
        }

    runner.write_query_groundtruth_from_hdf5 = patched


def validate_preflight(runner, specs: list, args: argparse.Namespace, run_root: Path) -> dict:
    checks = {
        "runner_path": str(RUNNER_PATH),
        "darth_bin": str(PATCHED_DARTH_BIN),
        "faiss_lib_dir": str(PATCHED_FAISS_LIB_DIR),
        "run_root": str(run_root),
        "common_index_root": str(runner.COMMON_INDEX_ROOT),
        "datasets": [],
    }
    if not RUNNER_PATH.exists():
        raise FileNotFoundError(f"missing runner: {RUNNER_PATH}")
    if not PATCHED_DARTH_BIN.exists():
        raise FileNotFoundError(f"missing DARTH binary: {PATCHED_DARTH_BIN}")
    if not (PATCHED_FAISS_LIB_DIR / "libfaiss.so").exists():
        raise FileNotFoundError(f"missing libfaiss.so in {PATCHED_FAISS_LIB_DIR}")
    for spec in specs:
        if not spec.hdf5.exists():
            raise FileNotFoundError(f"missing HDF5 for {spec.label}: {spec.hdf5}")
        index_path = runner.index_path_for(spec, runner.COMMON_INDEX_ROOT, args.m, args.ef_construction)
        checks["datasets"].append(
            {
                "label": spec.label,
                "darth_name": spec.darth_name,
                "hdf5": str(spec.hdf5),
                "metric": spec.source_metric,
                "index_path": str(index_path),
                "index_exists": index_path.exists(),
            }
        )
    return checks


def run_subprocess(cmd: list[str], log_path: Path, *, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd = [stdbuf, "-oL", "-eL", *cmd]
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
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
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"command failed rc={returncode}; see {log_path}")


def build_common_index(runner, spec, run_root: Path, args: argparse.Namespace) -> dict:
    index_path = runner.index_path_for(spec, runner.COMMON_INDEX_ROOT, args.m, args.ef_construction)
    if index_path.exists() and not args.force_index:
        payload = {
            "status": "reused",
            "index_path": str(index_path),
            "index_bytes": index_path.stat().st_size,
            "source_common_index_root": str(runner.COMMON_INDEX_ROOT),
            "wall_s": 0.0,
        }
        runner.write_json(run_root / "darth/results" / spec.label / "index_build.wrapper.json", payload)
        return payload

    if index_path.exists() and args.force_index and not index_path.resolve().is_relative_to(run_root):
        raise RuntimeError("refusing to overwrite an index outside run_root with --force-index; pass --common-index-root inside the run root for rebuilds")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = run_root / "darth/results" / spec.label / "index_build.darth.txt"
    log_path = run_root / "logs/darth" / f"{spec.label}.index_build.log"
    cmd = [
        str(runner.DARTH_BIN),
        "--dataset",
        spec.darth_name,
        "--M",
        str(args.m),
        "--efConstruction",
        str(args.ef_construction),
        "--efSearch",
        str(args.ef_search),
        "--query-num",
        "1",
        "--k",
        str(args.k),
        "--output",
        str(output_path),
        "--mode",
        "no-early-stop",
        "--index-filepath",
        str(index_path),
        "--save-index",
        "--dataset-dir-prefix",
        str(run_root / "darth/processed") + "/",
        "--query-type",
        "testing",
        "--metric",
        "auto",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(args.threads)
    started = time.perf_counter()
    run_subprocess(cmd, log_path, env=env)
    payload = {
        "status": "built",
        "index_path": str(index_path),
        "index_bytes": index_path.stat().st_size,
        "output": str(output_path),
        "log_path": str(log_path),
        "source_common_index_root": str(runner.COMMON_INDEX_ROOT),
        "wall_s": time.perf_counter() - started,
        "cmd": cmd,
    }
    runner.write_json(run_root / "darth/results" / spec.label / "index_build.wrapper.json", payload)
    return payload


def append_target099_summary(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "darth_name",
        "target_recall",
        "learn_queries",
        "validation_queries",
        "groundtruth_k",
        "lvec_s",
        "gt_s",
        "index_status",
        "index_path",
        "tdata_s",
        "train_s",
        "interval_s",
        "training_log_deleted",
        "online_s",
        "online_avg_recall",
        "online_avg_query_ms_from_csv",
        "online_output",
        "dataset_wall_s",
        "disk_free_start_bytes",
        "disk_free_after_lvec_bytes",
        "disk_free_after_tdata_bytes",
        "disk_free_after_online_bytes",
        "processed_bytes_after_online",
    ]
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_dataset(runner, spec, args: argparse.Namespace, run_root: Path) -> dict:
    dataset_started = time.perf_counter()
    disk = {"start": disk_snapshot(run_root)}
    processed_dir = run_root / "darth/processed" / spec.darth_name
    if processed_dir.exists():
        raise FileExistsError(f"refusing to reuse existing processed dir: {processed_dir}")

    log(f"{spec.label}: LVec start")
    lvec = runner.run_lvec(
        spec,
        processed_dir,
        learn_queries=args.learn_queries,
        validation_queries=args.validation_queries,
        seed=args.seed,
        base_batch_size=args.base_batch_size,
    )
    log(f"{spec.label}: LVec done {lvec['lvec_s']:.3f}s")
    disk["after_lvec"] = disk_snapshot(run_root)
    log(f"{spec.label}: disk free after LVec {disk['after_lvec']['free_bytes'] / (1024 ** 3):.1f} GiB")

    gt_k = runner.infer_gt_k(spec.hdf5, args.k)
    log(f"{spec.label}: GT start k={gt_k}")
    gt = runner.run_gt(
        spec,
        processed_dir,
        gt_k=gt_k,
        threads=args.threads,
        base_batch_size=args.base_batch_size,
        query_batch_size=args.query_batch_size,
    )
    log(f"{spec.label}: GT done {gt['gt_s']:.3f}s")
    disk["after_gt"] = disk_snapshot(run_root)

    log(f"{spec.label}: index build/reuse start")
    index_build = build_common_index(runner, spec, run_root, args)
    log(f"{spec.label}: index {index_build['status']}")

    log(f"{spec.label}: TData start")
    tdata = runner.run_tdata(
        spec,
        run_root,
        processed_dir,
        m=args.m,
        efc=args.ef_construction,
        efs=args.ef_search,
        k=args.k,
        logging_interval=args.logging_interval,
        threads=args.threads,
        keep_base_after_db_load=args.keep_base_after_db_load,
    )
    log(f"{spec.label}: TData done {tdata['tdata_s']:.3f}s")
    disk["after_tdata"] = disk_snapshot(run_root)
    log(f"{spec.label}: disk free after TData {disk['after_tdata']['free_bytes'] / (1024 ** 3):.1f} GiB")

    training_csv = Path(str(tdata["training_log"]))
    query_num = runner.query_file_rows(processed_dir / "learn.fvecs")
    log(f"{spec.label}: interval extraction start")
    interval = runner.run_interval(
        spec,
        run_root,
        training_csv,
        target_recall=args.target_recall,
        chunksize=args.chunksize,
        k=args.k,
        efc=args.ef_construction,
        efs=args.ef_search,
        query_num=query_num,
    )
    log(f"{spec.label}: interval extraction done {interval['interval_s']:.3f}s")

    log(f"{spec.label}: Train start")
    train = runner.train_predictor(
        spec,
        run_root,
        training_csv,
        m=args.m,
        efc=args.ef_construction,
        efs=args.ef_search,
        k=args.k,
        query_num=query_num,
        logging_interval=args.logging_interval,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        train_threads=args.train_threads,
    )
    log(f"{spec.label}: Train done {train['train_s']:.3f}s")

    training_log_deleted = False
    if not args.keep_training_log:
        training_log_deleted = runner.remove_if_exists(training_csv)
        log(f"{spec.label}: deleted training log={training_log_deleted}")
    disk["after_training_cleanup"] = disk_snapshot(run_root)

    log(f"{spec.label}: Online start")
    online = runner.run_online(
        spec,
        run_root,
        processed_dir,
        m=args.m,
        efc=args.ef_construction,
        efs=args.ef_search,
        k=args.k,
        target_recall=args.target_recall,
        online_query_num=args.online_query_num,
        interval=interval,
        train=train,
        base_batch_size=args.base_batch_size,
        keep_base_after_online=args.keep_base_after_online,
    )
    log(f"{spec.label}: Online done {online['online_s']:.3f}s recall={online.get('online_avg_recall', float('nan')):.4f}")
    disk["after_online"] = disk_snapshot(run_root)
    processed_bytes_after_online = tree_size_bytes(processed_dir)
    log(f"{spec.label}: disk free after Online {disk['after_online']['free_bytes'] / (1024 ** 3):.1f} GiB; processed={processed_bytes_after_online / (1024 ** 2):.1f} MiB")

    row = {
        "dataset": spec.label,
        "darth_name": spec.darth_name,
        "target_recall": args.target_recall,
        "learn_queries": args.learn_queries,
        "validation_queries": args.validation_queries,
        "groundtruth_k": gt["groundtruth_k"],
        "lvec_s": lvec["lvec_s"],
        "gt_s": gt["gt_s"],
        "index_status": index_build["status"],
        "index_path": index_build["index_path"],
        "tdata_s": tdata["tdata_s"],
        "train_s": train["train_s"],
        "interval_s": interval["interval_s"],
        "training_log_deleted": training_log_deleted,
        "online_s": online["online_s"],
        "online_avg_recall": online.get("online_avg_recall", ""),
        "online_avg_query_ms_from_csv": online.get("online_avg_query_ms_from_csv", ""),
        "online_output": online["online_output"],
        "dataset_wall_s": time.perf_counter() - dataset_started,
        "disk_free_start_bytes": disk["start"]["free_bytes"],
        "disk_free_after_lvec_bytes": disk["after_lvec"]["free_bytes"],
        "disk_free_after_tdata_bytes": disk["after_tdata"]["free_bytes"],
        "disk_free_after_online_bytes": disk["after_online"]["free_bytes"],
        "processed_bytes_after_online": processed_bytes_after_online,
    }
    runner.write_json(
        run_root / "darth/results" / spec.label / "target099.wrapper.json",
        {
            "status": "ok",
            "dataset": runner.dataset_payload(spec),
            "lvec": lvec,
            "gt": gt,
            "index_build": index_build,
            "tdata": tdata,
            "interval": interval,
            "train": train,
            "online": online,
            "disk": disk,
            "processed_bytes_after_online": processed_bytes_after_online,
            "summary_row": row,
        },
    )
    runner.append_summary_row(run_root / "darth/results/offline_cost_summary.csv", row)
    return row


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    runner = load_runner()
    specs = configure_runner(runner, args, run_root)
    checks = validate_preflight(runner, specs, args, run_root)

    if args.dry_run:
        print(json.dumps({"status": "dry-run-ok", "checks": checks, "parameters": vars(args)}, indent=2))
        return 0

    if run_root.exists() and not args.allow_existing_run_root:
        raise FileExistsError(f"run root already exists: {run_root}; pass --allow-existing-run-root or choose a fresh --run-root")
    run_root.mkdir(parents=True, exist_ok=True)
    runner.write_json(
        run_root / "RUN_MANIFEST.json",
        {
            "created_at": now(),
            "purpose": "DARTH target0.99 main4 from scratch with metric-aware binary",
            "checks": checks,
            "parameters": vars(args),
            "darth_bin": str(runner.DARTH_BIN),
            "faiss_lib_dir": str(PATCHED_FAISS_LIB_DIR),
            "common_index_root": str(runner.COMMON_INDEX_ROOT),
        },
    )

    rows = []
    try:
        for spec in specs:
            rows.append(run_dataset(runner, spec, args, run_root))
    except Exception as exc:
        runner.write_json(
            run_root / "darth/results/offline_failure.json",
            {"failed_at": now(), "error": repr(exc), "completed_rows": rows},
        )
        raise

    runner.write_json(
        run_root / "darth/results/offline_cost_summary.json",
        {"status": "ok", "completed_at": now(), "rows": rows},
    )
    log(f"summary: {run_root / 'darth/results/offline_cost_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
