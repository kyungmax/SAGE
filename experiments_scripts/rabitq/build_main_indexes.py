#!/usr/bin/env python3
"""Sequential, resumable RaBitQ index builds; no policy calibration or benchmark."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments_scripts.rabitq.final_policy import load_library
from experiments_scripts.rabitq.runner import Dataset, build, file_identity, validate_index


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity():
    paths = [ROOT / "experiments_scripts/rabitq/runner.py", Path(__file__).resolve()]
    paths += sorted((ROOT / "rabitq/include").rglob("*"))
    paths += sorted((ROOT / "rabitq/build_sage/python/rabitqlib").glob("*.so"))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def disk_free(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def prepare(plan_path, destination, datasets_config, selected):
    catalog = json.loads(datasets_config.read_text())["datasets"]
    available = {row["dataset"] for row in catalog}
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= available:
        raise ValueError("Select distinct datasets present in --datasets-config")
    entries = []
    for row in catalog:
        if row["dataset"] not in selected:
            continue
        name, n, dim, m = row["dataset"], row["train_count"], row["dimension"], row["M"]
        metric = "l2" if row["space"] == "l2" or name == "youtube-15M-angular" else "ip"
        normalize = row["space"] == "cosine"
        raw = None
        path = (Path(row["dataset_path"]) if "dataset_path" in row else
                Path(os.environ.get("SAGE_DATA_DIR", ROOT / "datasets")) / row["dataset_file"])
        if name == "msspacev-100M-i8-euclidean":
            import h5py
            with h5py.File(path, "r") as handle:
                raw = handle["train"].external[0][0]
        data = Dataset(path, metric=metric, normalize=normalize, raw=raw)
        if data.shape != (n, dim):
            raise ValueError(f"Catalog shape mismatch: {name}")
        padded = (dim + 63) // 64 * 64
        record_bytes = 8 * m + 12 + padded + 20
        expected = int(n * (record_bytes + 4 + (4*m+4)/(m-1)) + 128*padded*4 + 2048)
        stem = f"{name}-M{m}-efC{row['efConstruction']}-b8-{metric}"
        if normalize:
            stem += "-normalized"
        entries.append(dict(dataset=name, data_path=str(path), raw=raw,
            dataset_identity=data.identity(), shape=[n, dim], metric=metric, normalize=normalize,
            degree=m, ef_construction=row["efConstruction"], total_bits=8, num_clusters=128,
            expected_bytes=expected, minimum_bytes=n*(record_bytes+4),
            index_path=str(destination / (stem + "-hnsw-rabitq.index"))))
    # Finish the smaller datasets before the three long builds.
    entries.sort(key=lambda row: row["shape"][0])
    plan = dict(format="sage-rabitq-index-build-v1", created_at=time.time(),
        source_identity=identity(), destination=str(destination), threads=24,
        cpus=list(range(0,48,2)), numa_node=0, reserve_bytes=10*1024**3,
        selected_datasets=selected, entries=entries,
        estimated_total_bytes=sum(row["expected_bytes"] for row in entries),
        free_bytes_at_preflight=disk_free(destination),
        scope="Build and save/reload-verify indexes only; no calibration or performance measurement")
    check(plan)
    save(plan_path, plan)
    print(json.dumps(dict(plan=str(plan_path), estimated_GiB=plan["estimated_total_bytes"]/1024**3,
        free_GiB=plan["free_bytes_at_preflight"]/1024**3,
        datasets=[row["dataset"] for row in entries]), indent=2))


def check(plan):
    load_library()
    if identity() != plan["source_identity"]:
        raise ValueError("Build source/runtime changed: prepare a new plan")
    required = sum(row["expected_bytes"] for row in plan["entries"] if not Path(row["index_path"]).exists())
    if disk_free(plan["destination"]) < required * 1.05 + plan["reserve_bytes"]:
        raise RuntimeError("Insufficient disk space for pending indexes plus reserve")
    for row in plan["entries"]:
        data = Dataset(row["data_path"], metric=row["metric"], normalize=row["normalize"], raw=row["raw"])
        if data.identity() != row["dataset_identity"]:
            raise ValueError(f"Dataset changed: {row['dataset']}")


def worker(plan_path, name):
    import numpy as np
    plan = json.loads(plan_path.read_text())
    check(plan)
    entry = next(row for row in plan["entries"] if row["dataset"] == name)
    state_dir = plan_path.parent / name
    started = time.time()
    state = dict(dataset=name, pid=os.getpid(), started_at=started, index=entry["index_path"])

    def report(stage, **details):
        state.update(stage=stage, updated_at=time.time(), **details)
        save(state_dir / "status.json", state)
        print(json.dumps({"stage": stage, "elapsed_seconds": time.time()-started, **details}), flush=True)

    try:
        data = Dataset(entry["data_path"], metric=entry["metric"], normalize=entry["normalize"], raw=entry["raw"])
        args = SimpleNamespace(**{key:entry[key] for key in ("degree", "ef_construction", "total_bits", "num_clusters")},
                               force_build=False, skip_build=False, build_threads=plan["threads"])
        path = Path(entry["index_path"])
        existed = path.exists()
        library = load_library()
        index = build(data, path, args, library, progress=report)
        if path.stat().st_size < entry["minimum_bytes"]:
            raise RuntimeError("Serialized index is shorter than the minimum expected payload")
        validate_index(index, data, args)
        report("verifying_before_reload")
        probe_ids = np.linspace(0, data.shape[0]-1, 32, dtype=np.int64)
        probes = data.read(probe_ids)
        labels, distances = index.search(probes, k=10, ef=128, num_threads=1)
        if np.any(labels >= data.shape[0]) or not np.isfinite(distances).all():
            raise RuntimeError("Invalid native verification results")
        del index
        gc.collect()
        report("reloading_and_verifying")
        restored = library.HnswIndex.load(str(path))
        validate_index(restored, data, args)
        reloaded = restored.search(probes, k=10, ef=128, num_threads=1)
        np.testing.assert_array_equal(labels, reloaded[0])
        np.testing.assert_array_equal(distances, reloaded[1])
        local = ROOT / "index/rabitq" / path.name
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.is_symlink() or local.exists():
            if local.resolve() != path.resolve():
                raise FileExistsError(local)
        else:
            local.symlink_to(path.resolve())
        report("complete", complete=True, reused_existing=existed, index_identity=file_identity(path),
               local_link=str(local), verified_queries=32, saved_reloaded_exact_match=True,
               elapsed_seconds=time.time()-started)
        np.savez(state_dir / "verification.npz", probe_ids=probe_ids, labels=labels, distances=distances)
    except BaseException as exc:
        report("failed", complete=False, error=f"{type(exc).__name__}: {exc}")
        raise


def cpu_ticks(cpus):
    return {int(p[0][3:]): list(map(int,p[1:9]))
        for line in Path('/proc/stat').read_text().splitlines()
        if (p:=line.split()) and p[0][3:].isdigit() and int(p[0][3:]) in cpus}


def supervise(plan_path):
    plan = json.loads(plan_path.read_text())
    control = plan_path.parent
    with (control / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(supervisor_pid=os.getpid(), started_at=time.time(), completed=[], stage="starting")
        def report(stage, **details):
            state.update(stage=stage, updated_at=time.time(), **details)
            save(control / "status.json", state)
        try:
            check(plan)
            for row in plan["entries"]:
                name = row["dataset"]
                previous_path = control / name / "status.json"
                if previous_path.exists():
                    previous = json.loads(previous_path.read_text())
                    if (previous.get("complete") and Path(row["index_path"]).exists()
                            and previous.get("index_identity") == file_identity(row["index_path"])):
                        state["completed"].append(name)
                        continue
                # Keep these 24 physical cores clear of another active computation.
                watched = plan["cpus"] + [cpu+48 for cpu in plan["cpus"]]
                while True:
                    report("waiting_for_cpus", current_dataset=name)
                    before = cpu_ticks(watched)
                    time.sleep(2)
                    after = cpu_ticks(watched)
                    busy = []
                    for cpu in watched:
                        delta = [y-x for x,y in zip(before[cpu],after[cpu])]
                        busy.append(100*(1-(delta[3]+delta[4])/max(1,sum(delta))))
                    utilization = sum(busy)/len(plan["cpus"])
                    if utilization < 25:
                        break
                    report("waiting_for_cpus", current_dataset=name, cpu_busy_percent=utilization)
                    time.sleep(10)
                check(plan)
                folder = control / name
                folder.mkdir(exist_ok=True)
                command = ["numactl", "--physcpubind="+','.join(map(str,plan["cpus"])),
                    "--membind="+str(plan["numa_node"]), "bash", str(ROOT / "scripts/python.sh"),
                    str(Path(__file__).resolve()), "worker", "--plan", str(plan_path), "--dataset", name]
                with (folder / "build.log").open("a") as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, cwd=ROOT)
                    while child.poll() is None:
                        detail = json.loads(previous_path.read_text()) if previous_path.exists() else {}
                        report("building", current_dataset=name, worker_pid=child.pid,
                               worker_status=detail, log=str(folder / "build.log"))
                        time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"{name} worker exited {child.returncode}; see {folder / 'build.log'}")
                detail = json.loads(previous_path.read_text())
                if not detail.get("complete"):
                    raise RuntimeError(f"{name}: missing verification completion")
                state["completed"].append(name)
            report("complete", complete=True, current_dataset=None, worker_pid=None)
        except BaseException as exc:
            report("failed", complete=False, error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


def launch(plan_path):
    plan = json.loads(plan_path.read_text())
    check(plan)
    state_path = plan_path.parent / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("stage") not in ("complete", "failed") and state.get("supervisor_pid"):
            try:
                os.kill(state["supervisor_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("A supervisor is already running")
    Path(plan["destination"]).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(plan["threads"]),
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (plan_path.parent / "supervisor.log").open("a") as log:
        child = subprocess.Popen(["bash", str(ROOT / "scripts/python.sh"), str(Path(__file__).resolve()),
            "supervise", "--plan", str(plan_path)], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
    save(plan_path.parent / "launch.json", dict(supervisor_pid=child.pid, launched_at=time.time(), plan=str(plan_path)))
    print(json.dumps(dict(supervisor_pid=child.pid, status=str(state_path)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "supervise", "worker"))
    parser.add_argument("--plan", type=Path, default=ROOT / "build/rabitq_main6_20260928/plan.json")
    parser.add_argument("--destination", type=Path,
        default=ROOT / "index/main6/rabitq_b8")
    parser.add_argument("--dataset")
    parser.add_argument("--datasets", help="Comma-separated dataset names required for prepare")
    parser.add_argument("--datasets-config", type=Path,
        default=ROOT / "experiments_scripts/01_main_results/config/main6_online24/datasets.json")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    if args.action == "prepare":
        if not args.datasets:
            parser.error("prepare requires --datasets with explicit comma-separated dataset names")
        prepare(args.plan, args.destination.resolve(), args.datasets_config.resolve(),
                [name.strip() for name in args.datasets.split(",")])
    elif args.action == "worker":
        worker(args.plan, args.dataset)
    elif args.action == "supervise":
        supervise(args.plan)
    else:
        launch(args.plan)
