#!/usr/bin/env python3
"""Run Ada-EF target-0.99 on CohereWiki and MSMARCO with full online queries.

The current ``experiments_scripts/ada-ef`` backend does not expose an online
query-count option: its online phase loads the full HDF5 ``test`` and
``neighbors`` matrices.  This wrapper records that expected query count in the
manifest and summary so the full-query setting is auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_HNSWLIB_INDEX_ROOT",
        os.environ.get("SAGE_INDEX_DIR", str(DEFAULT_PROJECT_ROOT / "index")),
    )
).expanduser()
ADA_EF_ROOT = Path(os.environ.get("SAGE_ADAEF_ROOT", str(REPO_ROOT / "experiments_scripts/ada-ef"))).expanduser()
BACKEND_BIN = Path(
    os.environ.get("SAGE_ADAEF_BACKEND_BIN", str(ADA_EF_ROOT / "build-simd-avx512/backend_runner"))
).expanduser()
SIMD_FLAGS = "-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt"

DATASETS = {
    "cohere": {
        "label": "cohere",
        "dataset": "cohere-768-angular",
        "hdf5": DATASET_ROOT / "cohere-768-angular.hdf5",
        "metric": "cd",
        "existing_index": INDEX_ROOT / "cohere-768-angular_M32_M32_efC500_n10000000_dim768",
    },
    "msmarco": {
        "label": "msmarco",
        "dataset": "msmarco-v1-openai-ada2-full-ip",
        "hdf5": DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5",
        "metric": "ipd",
        "existing_index": INDEX_ROOT / "msmarco-v1-openai-ada2-full-ip_M32_M32_efC500_n8841823_dim1536",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def monotonic() -> float:
    return time.perf_counter()


def elapsed_since(started: float) -> float:
    return float(time.perf_counter() - started)


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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    if path.is_file() and not path.is_symlink():
        return int(path.stat().st_size)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                if not file_path.is_symlink():
                    total += file_path.stat().st_size
            except FileNotFoundError:
                pass
    return int(total)


def backend_index_path(run_root: Path, dataset: str, m: int, efc: int) -> Path:
    return run_root / "adaef/index" / f"{dataset}-M{m}-efc-{efc}-parallel.hnsw"


def experiments_root(run_root: Path) -> Path:
    return run_root / "adaef/experiments"


def result_dir(run_root: Path, label: str) -> Path:
    return run_root / "adaef/results" / label


def log_dir(run_root: Path) -> Path:
    return run_root / "logs/adaef"


def metric_value(payload: dict, key: str, default=""):
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return metrics.get(key, default)


def usable_existing_index(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def base_backend_cmd(spec: dict, args: argparse.Namespace, run_root: Path, phase: str, output_json: Path) -> list[str]:
    return [
        str(BACKEND_BIN),
        "--phase",
        phase,
        "--dataset",
        spec["dataset"],
        "--data-path",
        str(spec["hdf5"]),
        "--dataset-root",
        str(DATASET_ROOT),
        "--index-root",
        str(run_root / "adaef/index"),
        "--experiments-root",
        str(experiments_root(run_root)),
        "--output-json",
        str(output_json),
        "--m",
        str(args.m),
        "--ef-construction",
        str(args.ef_construction),
        "--k",
        str(args.k),
        "--metric",
        spec["metric"],
    ]


def run_subprocess(cmd: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out_fh, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err_fh:
        out_fh.write(f"[{now()}] CMD: {' '.join(cmd)}\n")
        out_fh.flush()
        proc = subprocess.run(cmd, stdout=out_fh, stderr=err_fh, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}; stdout={stdout_path}; stderr={stderr_path}")


def ensure_index(spec: dict, args: argparse.Namespace, run_root: Path) -> dict:
    target = backend_index_path(run_root, spec["dataset"], args.m, args.ef_construction)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = Path(spec["existing_index"])
    if target.exists() and not args.force_build_index:
        resolved = target.resolve() if target.is_symlink() else target
        return {
            "status": "existing_in_run_root",
            "index_path": str(target),
            "index_bytes": int(resolved.stat().st_size),
            "is_symlink": bool(target.is_symlink()),
            "symlink_target": str(resolved) if target.is_symlink() else "",
        }
    if target.exists() or target.is_symlink():
        target.unlink()

    if usable_existing_index(existing) and not args.force_build_index:
        target.symlink_to(existing)
        return {
            "status": "symlinked_existing",
            "index_path": str(target),
            "index_bytes": int(existing.stat().st_size),
            "is_symlink": True,
            "symlink_target": str(existing.resolve()),
        }

    output_json = result_dir(run_root, spec["label"]) / "build.backend.json"
    cmd = base_backend_cmd(spec, args, run_root, "build", output_json)
    cmd += ["--num-threads", str(args.offline_threads)]
    started = monotonic()
    run_subprocess(
        cmd,
        log_dir(run_root) / f"{spec['label']}.build.stdout.log",
        log_dir(run_root) / f"{spec['label']}.build.stderr.log",
    )
    return {
        "status": "built",
        "index_path": str(target),
        "index_bytes": int(target.stat().st_size),
        "is_symlink": False,
        "symlink_target": "",
        "build_s": elapsed_since(started),
        "backend_json": str(output_json),
    }


def run_offline(spec: dict, args: argparse.Namespace, run_root: Path) -> dict:
    output_json = result_dir(run_root, spec["label"]) / "offline.backend.json"
    cmd = base_backend_cmd(spec, args, run_root, "offline", output_json)
    cmd += [
        "--expected-recall",
        str(args.target_recall),
        "--num-threads",
        str(args.offline_threads),
        "--sample-size",
        str(args.sample_size),
        "--ef-upper-bound",
        str(args.ef_upper_bound),
        "--quantile-step",
        str(args.quantile_step),
    ]
    if args.force_offline:
        cmd.append("--force")
    started = monotonic()
    run_subprocess(
        cmd,
        log_dir(run_root) / f"{spec['label']}.offline.stdout.log",
        log_dir(run_root) / f"{spec['label']}.offline.stderr.log",
    )
    payload = json.loads(output_json.read_text())
    payload["wrapper_wall_s"] = elapsed_since(started)
    payload["cmd"] = cmd
    return payload


def run_online(spec: dict, args: argparse.Namespace, run_root: Path) -> dict:
    output_json = result_dir(run_root, spec["label"]) / "online.backend.json"
    cmd = base_backend_cmd(spec, args, run_root, "online", output_json)
    cmd += [
        "--expected-recall",
        str(args.target_recall),
        "--num-threads",
        str(args.online_threads),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measured-runs",
        str(args.measured_runs),
    ]
    if args.online_threads > 1:
        cmd.append("--parallel-queries")
    if args.per_query_csv:
        cmd += ["--per-query-csv", str(result_dir(run_root, spec["label"]) / "online.per_query.csv")]
    started = monotonic()
    run_subprocess(
        cmd,
        log_dir(run_root) / f"{spec['label']}.online.stdout.log",
        log_dir(run_root) / f"{spec['label']}.online.stderr.log",
    )
    payload = json.loads(output_json.read_text())
    payload["wrapper_wall_s"] = elapsed_since(started)
    payload["cmd"] = cmd
    return payload


def append_summary(path: Path, row: dict) -> None:
    fieldnames = [
        "dataset",
        "dataset_name",
        "target_recall",
        "metric",
        "full_query_count_expected",
        "online_query_count_reported",
        "offline_threads",
        "online_threads",
        "index_status",
        "index_path",
        "index_bytes",
        "offline_wall_s",
        "offline_total_ms",
        "offline_ef_adaptor_ms",
        "offline_extra_ms",
        "online_wall_s",
        "online_recall",
        "online_qps",
        "online_latency_ms",
        "weighted_average_ef",
        "dataset_wall_s",
        "run_root_bytes_after_online",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_dataset(spec: dict, args: argparse.Namespace, run_root: Path) -> dict:
    started = monotonic()
    query_count = hdf5_query_count(Path(spec["hdf5"]))
    disk = {"start": disk_snapshot(run_root)}
    print(f"[{now()}] Ada-EF {spec['label']}: index prepare start", flush=True)
    index_info = ensure_index(spec, args, run_root)
    disk["after_index"] = disk_snapshot(run_root)
    print(f"[{now()}] Ada-EF {spec['label']}: index {index_info['status']}", flush=True)

    print(f"[{now()}] Ada-EF {spec['label']}: offline start", flush=True)
    offline = run_offline(spec, args, run_root)
    disk["after_offline"] = disk_snapshot(run_root)
    print(f"[{now()}] Ada-EF {spec['label']}: offline done {offline['wrapper_wall_s']:.3f}s", flush=True)

    print(f"[{now()}] Ada-EF {spec['label']}: online start full_query_count={query_count}", flush=True)
    online = run_online(spec, args, run_root)
    disk["after_online"] = disk_snapshot(run_root)
    reported_query_count = metric_value(online, "query_count")
    if int(reported_query_count) != int(query_count):
        raise RuntimeError(
            f"{spec['label']}: online query_count mismatch; expected {query_count}, got {reported_query_count}"
        )
    row = {
        "dataset": spec["label"],
        "dataset_name": spec["dataset"],
        "target_recall": args.target_recall,
        "metric": spec["metric"],
        "full_query_count_expected": query_count,
        "online_query_count_reported": reported_query_count,
        "offline_threads": args.offline_threads,
        "online_threads": args.online_threads,
        "index_status": index_info["status"],
        "index_path": index_info["index_path"],
        "index_bytes": index_info["index_bytes"],
        "offline_wall_s": offline["wrapper_wall_s"],
        "offline_total_ms": metric_value(offline, "total_offline_ms"),
        "offline_ef_adaptor_ms": metric_value(offline, "ef_adaptor_ms"),
        "offline_extra_ms": metric_value(offline, "extra_offline_ms"),
        "online_wall_s": online["wrapper_wall_s"],
        "online_recall": metric_value(online, "achieved_recall"),
        "online_qps": metric_value(online, "qps"),
        "online_latency_ms": metric_value(online, "mean_query_latency_ms"),
        "weighted_average_ef": metric_value(online, "weighted_average_ef"),
        "dataset_wall_s": elapsed_since(started),
        "run_root_bytes_after_online": tree_size_bytes(run_root),
    }
    write_json(
        result_dir(run_root, spec["label"]) / "target099.wrapper.json",
        {
            "status": "ok",
            "dataset": {k: str(v) if isinstance(v, Path) else v for k, v in spec.items()},
            "full_query_count_expected": query_count,
            "index": index_info,
            "offline": offline,
            "online": online,
            "disk": disk,
            "summary_row": row,
        },
    )
    append_summary(run_root / "adaef/results/offline_cost_summary.csv", row)
    print(
        f"[{now()}] Ada-EF {spec['label']}: online done recall={row['online_recall']} "
        f"latency_ms={row['online_latency_ms']}",
        flush=True,
    )
    return row


def validate_preflight(specs: list[dict], args: argparse.Namespace, run_root: Path, *, require_backend: bool = True) -> dict:
    checks = {
        "run_root": str(run_root),
        "backend_bin": str(BACKEND_BIN),
        "backend_exists": bool(BACKEND_BIN.exists() and os.access(BACKEND_BIN, os.X_OK)),
        "adaef_source_root": str(ADA_EF_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "simd": {
            "build_dir": str(BACKEND_BIN.parent),
            "expected_cxx_flags": SIMD_FLAGS,
            "build_helper": str(EXP_ROOT / "scripts/build_adaef_simd_avx512.sh"),
        },
        "datasets": [],
    }
    if require_backend and not checks["backend_exists"]:
        raise FileNotFoundError(f"missing executable SIMD backend_runner: {BACKEND_BIN}")
    for spec in specs:
        hdf5 = Path(spec["hdf5"])
        if not hdf5.exists():
            raise FileNotFoundError(f"missing dataset for {spec['label']}: {hdf5}")
        existing = Path(spec["existing_index"])
        checks["datasets"].append(
            {
                "label": spec["label"],
                "dataset": spec["dataset"],
                "hdf5": str(hdf5),
                "metric": spec["metric"],
                "full_query_count_expected": hdf5_query_count(hdf5),
                "backend_index_path": str(backend_index_path(run_root, spec["dataset"], args.m, args.ef_construction)),
                "existing_index_candidate": str(existing),
                "existing_index_usable": usable_existing_index(existing),
                "existing_index_bytes": int(existing.stat().st_size) if existing.exists() else 0,
            }
        )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(EXP_ROOT / "adaef_target099_simd_cohere_msmarco_fullquery"))
    parser.add_argument("--datasets", default="cohere,msmarco")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--offline-threads", type=int, default=24)
    parser.add_argument("--online-threads", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--ef-upper-bound", type=int, default=5000)
    parser.add_argument("--quantile-step", type=float, default=0.001)
    parser.add_argument("--force-build-index", action="store_true")
    parser.add_argument("--force-offline", action="store_true")
    parser.add_argument("--per-query-csv", action="store_true")
    parser.add_argument("--allow-existing-run-root", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    specs = [DATASETS[label] for label in parse_labels(args.datasets)]
    checks = validate_preflight(specs, args, run_root, require_backend=not args.dry_run)
    if args.dry_run:
        print(json.dumps({"status": "dry-run-ok", "checks": checks, "parameters": vars(args)}, indent=2))
        return 0

    if run_root.exists() and not args.allow_existing_run_root:
        raise FileExistsError(f"run root already exists: {run_root}; choose a fresh --run-root")
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "RUN_MANIFEST.json",
        {
            "created_at": now(),
            "purpose": "Ada-EF target0.99 SIMD, cohere/msmarco only, full online query set",
            "checks": checks,
            "parameters": vars(args),
            "simd_flags": SIMD_FLAGS,
        },
    )

    rows = []
    try:
        for spec in specs:
            rows.append(run_dataset(spec, args, run_root))
    except Exception as exc:
        write_json(run_root / "adaef/results/offline_failure.json", {"failed_at": now(), "error": repr(exc), "completed_rows": rows})
        raise
    write_json(run_root / "adaef/results/offline_cost_summary.json", {"status": "ok", "completed_at": now(), "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
