"""Paper ABS with its own local FAISS source and binary."""
import ctypes as ct
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPT_LEVEL = os.environ.get("ABS_OPT_LEVEL", "avx512")
DEFAULT_PYTHON_PATH = ROOT / "build/faiss" / OPT_LEVEL / "faiss/python"
DEFAULT_FAISS_SOURCE = ROOT / "faiss"
STAT_COLUMNS = ("ndis", "ndis_upper", "ndis_level0", "nexpan", "npop",
                "frontier_peak", "stopped_by_bound", "entry_point")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_faiss(python_path=None):
    path = Path(python_path or os.environ.get("ABS_FAISS_PYTHON_PATH", DEFAULT_PYTHON_PATH)).resolve()
    if not (path / "faiss/__init__.py").exists():
        raise RuntimeError(f"ABS FAISS missing at {path}; run bash scripts/build.sh")
    sys.path.insert(0, str(path))
    import faiss
    if not Path(faiss.__file__).resolve().is_relative_to(path):
        raise RuntimeError(f"Wrong FAISS already imported: {faiss.__file__}; expected {path}")
    return faiss


def active_binary():
    modules = [Path(m.__file__).resolve() for name, m in sys.modules.items()
               if name.startswith("faiss._swigfaiss") and
               getattr(m, "__file__", "").endswith(".so")]
    if len(modules) != 1:
        raise RuntimeError(f"Expected one active FAISS binary, found {modules}")
    return modules[0]


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= value <= 2**31 - 1:
        raise ValueError(f"{name} is outside the supported range")
    return value


class ABSSearch:
    """Read-only, reusable workspaces. Keep the index unchanged until close().

    Unit-normalized IP requires an explicit declaration and checks ALL database
    vector norms once. Query norms are checked on each call. Raw IP is unsupported.
    """
    def __init__(self, index, threads=1, *, unit_normalized=False, build_dir=None):
        self._lock = threading.Lock()
        self.context = None
        faiss = sys.modules.get("faiss")
        if faiss is None or not isinstance(index, faiss.IndexHNSWFlat):
            raise TypeError("Expected an IndexHNSWFlat from the selected local FAISS")
        threads = _integer(threads, "threads")
        if index.ntotal < 1:
            raise ValueError("Paper ABS requires a nonempty index")
        if index.metric_type not in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
            raise ValueError("Only L2 and unit-normalized IP are supported")
        self.ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
        if self.ip and not unit_normalized:
            raise ValueError("IP requires unit_normalized=True and unit database/query vectors")
        directory = Path(build_dir or os.environ.get("ABS_BUILD_DIR", ROOT / "build")).resolve()
        try:
            build = json.loads((directory / "build.json").read_text())
        except FileNotFoundError as error:
            raise RuntimeError("ABS native library missing; run scripts/build.py") from error
        binary = active_binary()
        if binary != Path(build["faiss_binary"]).resolve() or digest(binary) != build["faiss_binary_sha256"]:
            raise RuntimeError("FAISS binary differs from the ABS build; rebuild ABS")
        for path, expected in build["source_sha256"].items():
            if digest(path) != expected:
                raise RuntimeError(f"ABS source/header changed: {path}; rebuild ABS")
        library = directory / "libpaper_abs.so"
        if digest(library) != build["library_sha256"]:
            raise RuntimeError("ABS library differs from its build manifest; rebuild ABS")
        if self.ip:
            for start in range(0, index.ntotal, 16384):
                vectors = index.reconstruct_n(start, min(16384, index.ntotal - start))
                self._check_unit(vectors, "database")
        self.index, self.threads = index, threads
        self.ntotal, self.dimension = index.ntotal, index.d
        self.build = build
        self.lib = ct.CDLL(str(library))
        self.lib.paper_abs_api_version.restype = ct.c_int
        if self.lib.paper_abs_api_version() != 1:
            raise RuntimeError("Unsupported ABS native API version")
        self.lib.paper_abs_last_error.restype = ct.c_char_p
        self.lib.paper_abs_create.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
        self.lib.paper_abs_create.restype = ct.c_void_p
        self.lib.paper_abs_destroy.argtypes = [ct.c_void_p]
        self.lib.paper_abs_destroy.restype = None
        self.lib.paper_abs_search.argtypes = [ct.c_void_p, ct.c_int64, ct.c_void_p,
            ct.c_int, ct.c_double, ct.c_int64, ct.c_void_p, ct.c_void_p, ct.c_void_p]
        self.lib.paper_abs_search.restype = ct.c_int
        self.context = self.lib.paper_abs_create(int(index.this), threads, int(self.ip))
        if not self.context:
            raise RuntimeError(self.lib.paper_abs_last_error().decode())

    @staticmethod
    def _check_unit(vectors, kind):
        squared_norms = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
        if not np.all(np.isfinite(squared_norms)) or not np.allclose(
                squared_norms, 1.0, rtol=0, atol=2e-5):
            raise ValueError(f"IP {kind} vectors must have unit norm (zero vectors are unsupported)")

    def close(self):
        with self._lock:
            if self.context:
                self.lib.paper_abs_destroy(self.context)
                self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "context", None):
            self.close()

    def search(self, queries, k, *, gamma, entry=-1):
        """Return (FAISS distances, labels, per-query counter arrays).

        gamma is Euclidean-distance slack, >=0, with no implicit default.
        entry=-1 uses normal upper-layer descent; a nonnegative entry starts
        Algorithm 2 directly at that layer-0 node. No efSearch setting is used.
        """
        with self._lock:
            if not self.context:
                raise RuntimeError("ABS context is closed")
            if (self.index.ntotal, self.index.d) != (self.ntotal, self.dimension):
                raise RuntimeError("Index changed; recreate the ABS context")
            k = _integer(k, "k")
            entry = _integer(entry, "entry", -1)
            gamma = float(gamma)
            if k > self.ntotal or entry >= self.ntotal:
                raise ValueError("Require k <= ntotal and entry in [-1, ntotal)")
            if not math.isfinite(gamma) or gamma < 0 or not math.isfinite((1 + gamma) * (1 + gamma)):
                raise ValueError("gamma must be finite, nonnegative, and have a finite squared multiplier")
            q = np.ascontiguousarray(queries, dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != self.dimension or not np.isfinite(q).all():
                raise ValueError("Queries must be finite and shaped (n, index.d)")
            if self.ip:
                self._check_unit(q, "query")
            distances = np.empty((len(q), k), np.float32)
            labels = np.empty((len(q), k), np.int64)
            counters = np.empty((len(q), len(STAT_COLUMNS)), np.uint64)
            status = self.lib.paper_abs_search(self.context, len(q), q.ctypes.data, k,
                gamma, entry, distances.ctypes.data, labels.ctypes.data, counters.ctypes.data)
            if status:
                raise RuntimeError(self.lib.paper_abs_last_error().decode())
            return distances, labels, {name: counters[:, j] for j, name in enumerate(STAT_COLUMNS)}
