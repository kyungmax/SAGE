"""Build/calibrate and separately measure the RaBitQ-Library SAGE backend."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments_scripts.rabitq.final_policy import POLICY_VERSION, calibrate_policy, load_library, query
from experiments_scripts.common.final_policy import _grid_for_cap

ALIASES = {"agnews": "agnews-mxbai-1024-euclidean", "cohere": "cohere-768-angular",
           "msspacev": "msspacev-100M-i8-euclidean", "youtube": "youtube-15M-angular"}


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


class Dataset:
    def __init__(self, path, *, metric=None, normalize=None, raw=None):
        self.path = Path(path).resolve()
        self.raw = Path(raw).resolve() if raw else None
        self.metric = metric or ("ip" if "angular" in self.path.stem else "l2")
        self.normalize = self.metric == "ip" if normalize is None else normalize
        if self.raw:
            shape = np.fromfile(self.raw, dtype="<i4", count=2)
            if len(shape) != 2 or np.any(shape <= 0):
                raise ValueError("Invalid i8bin header")
            self.shape = tuple(int(v) for v in shape)
            if self.raw.stat().st_size != 8 + np.prod(self.shape):
                raise ValueError("Invalid i8bin length")
        else:
            with h5py.File(self.path, "r") as handle:
                self.shape = handle["train"].shape
        if len(self.shape) != 2:
            raise ValueError("train must be a matrix")

    def prepare(self, rows):
        rows = np.array(rows, dtype=np.float32, order="C", copy=True)
        if self.normalize:
            rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(rows).all():
            raise ValueError("Dataset contains nonfinite vectors")
        return rows

    def read(self, ids=None):
        if self.raw:
            data = np.memmap(self.raw, dtype="i1", mode="r", offset=8, shape=self.shape)
            return self.prepare(data if ids is None else data[ids])
        with h5py.File(self.path, "r") as handle:
            if ids is None:
                return self.prepare(handle["train"][:])
            unique, inverse = np.unique(ids, return_inverse=True)
            return self.prepare(handle["train"][unique][inverse])

    def identity(self):
        return dict(data=file_identity(self.path), raw=file_identity(self.raw) if self.raw else None,
                    shape=list(self.shape), metric=self.metric, normalize=self.normalize)


def parse_args(argv=None):
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--datasets", nargs="+", required=True, help="Aliases or HDF5 paths")
    shared.add_argument("--data-dir", type=Path, default=Path(os.getenv("RABITQ_DATA_DIR", ROOT / "datasets")))
    shared.add_argument("--index-dir", type=Path, default=Path(os.getenv("RABITQ_INDEX_DIR", ROOT / "index/rabitq_m32_efc500")))
    shared.add_argument("--index-path", type=Path, help="Reuse a specific RaBitQ-Library index (one dataset)")
    shared.add_argument("--msspacev-raw", type=Path, default=os.getenv("RABITQ_MSSPACEV_RAW"))
    shared.add_argument("--metric", choices=("l2", "ip"))
    shared.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    shared.add_argument("--degree", type=int, default=32)
    shared.add_argument("--ef-construction", type=int, default=500)
    shared.add_argument("--total-bits", type=int, default=8)
    shared.add_argument("--k", type=int, default=10)
    shared.add_argument("--preflight", action="store_true", help="Inspect files and local runtime only")

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    calibrate = actions.add_parser("calibrate", parents=[shared], help="Build or load indexes and freeze SAGE policies")
    calibrate.add_argument("--num-clusters", type=int, default=128)
    calibrate.add_argument("--build-threads", type=int, default=24)
    calibrate.add_argument("--calibration-threads", type=int, default=24)
    calibrate.add_argument("--skip-build", action="store_true")
    calibrate.add_argument("--force-build", action="store_true")
    calibrate.add_argument("--force-calibrate", action="store_true")
    calibrate.add_argument("--probe-ids", type=Path, help="JSON file containing 500 frozen train labels")
    calibrate.add_argument("--efs", nargs="+", type=int, help="Additional curve policies to fit")
    sweep = actions.add_parser("sweep", parents=[shared], help="Measure queries using frozen policies")
    sweep.add_argument("--query-threads", type=int, default=24)
    sweep.add_argument("--rounds", type=int, default=3)
    sweep.add_argument("--efs", nargs="+", type=int, help="Subset of already calibrated curve points")
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--force-output", action="store_true")
    args = parser.parse_args(argv)
    if args.index_path and len(args.datasets) != 1:
        parser.error("--index-path requires exactly one dataset")
    if not 1 <= args.k <= 128 or not 2 <= args.total_bits <= 9:
        parser.error("Require k in [1,128] and total-bits in [2,9] for the HNSW full-estimate path")
    if args.degree < 2 or args.ef_construction < args.degree:
        parser.error("Require degree >= 2 and ef-construction >= degree")
    if args.efs is not None and any(ef < max(args.k, 64) for ef in args.efs):
        parser.error("Curve ef values must be >= max(k,64)")
    if args.action == "calibrate":
        if args.skip_build and args.force_build:
            parser.error("--skip-build conflicts with --force-build")
        if min(args.build_threads, args.calibration_threads, args.num_clusters) < 1:
            parser.error("Threads and num-clusters must be positive")
    elif min(args.query_threads, args.rounds) < 1:
        parser.error("Threads and rounds must be positive")
    return args


def paths(raw_name, args):
    name = ALIASES.get(raw_name, raw_name)
    path = Path(name)
    if path.suffix not in (".hdf5", ".h5"):
        path = Path(name + ".hdf5")
    path = path.expanduser()
    if not path.is_absolute() and not path.exists():
        path = args.data_dir / path
    raw = args.msspacev_raw if path.stem == ALIASES["msspacev"] else None
    if path.stem == ALIASES["msspacev"] and raw is None:
        raise ValueError("MSSPACEV requires --msspacev-raw")
    metric = args.metric or ("l2" if path.stem == ALIASES["youtube"] else None)
    normalize = args.normalize
    if normalize is None and "angular" in path.stem:
        normalize = True
    data = Dataset(path, metric=metric, normalize=normalize, raw=raw)
    stem = f"{path.stem}-M{args.degree}-efC{args.ef_construction}-b{args.total_bits}-{data.metric}"
    if data.normalize:
        stem += "-normalized"
    index = args.index_path or args.index_dir / (stem + "-hnsw-rabitq.index")
    policy = args.index_dir / (stem + f"-k{args.k}-{POLICY_VERSION}.json")
    return data, index, policy


def validate_index(index, data, args):
    if ((index.num_points, index.dim) != data.shape or index.metric != data.metric
            or index.nbits != args.total_bits or index.M != args.degree
            or index.ef_construction != args.ef_construction):
        raise ValueError("Index metadata does not match dataset/build settings")


def configure_clustering_blas(threads):
    """Promote loaded split MKL libraries for its runtime SGEMM dispatcher.

    This affects only the index-building process. OpenBLAS builds do not need it.
    The host's other offline builders use the same RTLD_GLOBAL workaround.
    """
    import ctypes
    maps = Path("/proc/self/maps")
    handles = []
    if not maps.exists():
        return handles
    loaded = {line.split()[-1] for line in maps.read_text().splitlines()
              if "libmkl_" in line and line.split()[-1].startswith("/")}
    for prefix in ("libmkl_intel_lp64.so", "libmkl_intel_thread.so", "libmkl_core.so"):
        for path in sorted(loaded):
            if Path(path).name.startswith(prefix):
                handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
    for handle in handles:
        setter = getattr(handle, "MKL_Set_Num_Threads", None)
        if setter is not None:
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            setter(threads)
            break
    return handles


def build(data, path, args, library, progress=None):
    report = progress or (lambda stage, **details: None)
    if path.exists() and not args.force_build:
        report("loading_existing_index")
        index = library.HnswIndex.load(str(path))
        validate_index(index, data, args)
        return index
    if args.skip_build:
        raise FileNotFoundError(path)
    import faiss
    faiss.omp_set_num_threads(args.build_threads)
    blas_handles = configure_clustering_blas(args.build_threads)
    report("clustering_runtime", promoted_mkl_libraries=[handle._name for handle in blas_handles])
    report("reading_train", rows=data.shape[0], dimension=data.shape[1])
    vectors = data.read()
    if args.num_clusters > len(vectors):
        raise ValueError("num-clusters exceeds train size")
    report("clustering", clusters=args.num_clusters)
    if args.num_clusters == 1:
        centroids = vectors.mean(axis=0, keepdims=True)
        assignments = np.zeros(len(vectors), dtype=np.uint32)
    else:
        metric = faiss.METRIC_INNER_PRODUCT if data.metric == "ip" else faiss.METRIC_L2
        clusters = faiss.index_factory(data.shape[1], f"IVF{args.num_clusters},Flat", metric)
        clusters.train(vectors)
        centroids = clusters.quantizer.reconstruct_n(0, clusters.nlist)
        assignments = clusters.quantizer.search(vectors, 1)[1].ravel().astype(np.uint32)
    report("building_graph", rows=data.shape[0], threads=args.build_threads)
    index = library.HnswIndex(dim=data.shape[1], max_elements=data.shape[0], M=args.degree,
                              ef_construction=args.ef_construction, nbits=args.total_bits,
                              metric=data.metric)
    index.build(vectors, centroids, assignments, num_threads=args.build_threads)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".index.tmp")
    report("saving_index", temporary=str(temporary))
    index.save(str(temporary))
    temporary.replace(path)
    return index


def load_policy(path, index_path, data, args):
    payload = json.loads(path.read_text())
    if payload.get("index_identity") != file_identity(index_path) or payload.get("dataset_identity") != data.identity():
        raise ValueError("Index/dataset changed: recalibrate before measuring")
    policy = payload["policy"]
    if policy.get("version") != POLICY_VERSION or policy.get("backend") != "rabitq" or policy.get("k") != args.k:
        raise ValueError("Incompatible policy: recalibrate before measuring")
    return policy


def measure(index, data, policy, args):
    points = args.efs or sorted(int(e) for e in policy["curve_policies"])
    for ef in points:
        if str(ef) not in policy["curve_policies"]:
            raise ValueError(f"No frozen policy for ef={ef}; calibrate it first")
    # Evaluation data is read only after training policy has been frozen.
    with h5py.File(data.path, "r") as handle:
        vectors = data.prepare(handle["test"][:])
        truth = np.asarray(handle["neighbors"][:, :args.k], dtype=np.int64)
    if not len(vectors) or truth.shape != (len(vectors), args.k):
        raise ValueError("Invalid test/ground-truth shapes")
    rows = []
    for ef in points:
        selected = policy["curve_policies"][str(ef)]
        for method in ("RaBitQ", "RaBitQ+SAGE"):
            def search():
                if method == "RaBitQ":
                    return index.search(vectors, k=args.k, ef=ef, num_threads=args.query_threads)
                return query(index, vectors, selected, num_threads=args.query_threads)
            search()
            durations = []
            for _ in range(args.rounds):
                started = perf_counter()
                labels, _ = search()
                durations.append(perf_counter() - started)
            seconds = float(np.median(durations))
            recall = np.mean([len(set(pred.tolist()) & set(gt.tolist())) / args.k
                              for pred, gt in zip(labels, truth)])
            rows.append(dict(dataset=data.path.stem, method=method, k=args.k, ef=ef,
                             recommended_ef=policy["ef_init"], recall=float(recall),
                             qps=len(vectors) / seconds, latency_per_query_ms=seconds * 1000 / len(vectors),
                             num_threads=args.query_threads, rounds=args.rounds,
                             seconds_by_round=json.dumps(durations), policy_version=POLICY_VERSION))
    return rows


def main(argv=None):
    args = parse_args(argv)
    action = args.action
    library = load_library()
    if action == "sweep" and not args.preflight and args.out.exists() and not args.force_output:
        raise FileExistsError(f"Use --force-output to replace {args.out}")
    rows = []
    for raw_name in args.datasets:
        data, index_path, policy_path = paths(raw_name, args)
        if data.shape[0] < 102 or not 64 <= data.shape[1] < 8192:
            raise ValueError("Require >=102 train rows and FhtKacRotator dimension in [64,8191]")
        print(json.dumps(dict(dataset=data.path.stem, shape=data.shape, index=str(index_path),
                              policy=str(policy_path), index_exists=index_path.exists(),
                              policy_exists=policy_path.exists())), flush=True)
        if args.preflight:
            continue
        if action == "calibrate":
            if policy_path.exists() and not args.force_calibrate:
                raise FileExistsError(f"Use --force-calibrate to replace {policy_path}")
            index = build(data, index_path, args, library)
            probes = json.loads(args.probe_ids.read_text()) if args.probe_ids else None
            policy = calibrate_policy(index, train_vectors=data.read, probe_ids=probes, k=args.k,
                num_threads=args.calibration_threads,
                curve_efs=args.efs if args.efs is not None else lambda ef, cap: _grid_for_cap(cap, k=args.k))
            payload = dict(index_identity=file_identity(index_path), dataset_identity=data.identity(), policy=policy)
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = policy_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(policy_path)
            print(f"Calibrated ef={policy['ef_init']} -> {policy_path}", flush=True)
        else:
            policy = load_policy(policy_path, index_path, data, args)
            index = library.HnswIndex.load(str(index_path))
            validate_index(index, data, args)
            rows.extend(measure(index, data, policy, args))
    if action == "sweep" and not args.preflight:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
