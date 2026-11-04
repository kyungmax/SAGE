"""Load the explicitly selected local FAISS build, without global installation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OPT_LEVEL = os.environ.get("PIP_OPT_LEVEL", "avx512")
BUILD_FLAVOR = "paper095" if OPT_LEVEL == "avx512" else f"paper095_{OPT_LEVEL}"
DEFAULT_PYTHON_PATH = Path(os.environ.get("PIP_BUILD_DIR", ROOT / "build" / BUILD_FLAVOR)) / "faiss" / "python"
STAT_FIELDS = ("n1", "n2", "ndis", "nhops", "nexpan", "npip_stops")


def load_faiss(python_path=DEFAULT_PYTHON_PATH):
    path = Path(python_path).resolve()
    if not (path / "faiss" / "__init__.py").exists():
        raise RuntimeError(f"Local FAISS package missing at {path}; run scripts/build.sh first")
    sys.path.insert(0, str(path))
    import faiss
    if not Path(faiss.__file__).resolve().is_relative_to(path):
        raise RuntimeError(f"Wrong FAISS imported: {faiss.__file__}; expected {path}")
    if not hasattr(faiss, "SearchParametersHNSWPiP"):
        raise RuntimeError(f"FAISS at {faiss.__file__} does not include PiP")
    return faiss


def parameters(faiss, ef, *, pip=False, threshold=0.95, patience=-1, enabled=True):
    params = faiss.SearchParametersHNSWPiP() if pip else faiss.SearchParametersHNSW()
    params.efSearch = int(ef)
    params.bounded_queue = True
    if pip:
        params.enable_pip = enabled
        params.saturation_threshold = threshold
        params.patience = patience
    return params


def search_with_stats(faiss, index, queries, k, params):
    # FAISS's global stats are suitable for a single calling Python thread;
    # C++ OpenMP workers reduce their query-local counters before returning.
    faiss.cvar.hnsw_stats.reset()
    distances, labels = index.search(queries, k, params=params)
    stats = {name: int(getattr(faiss.cvar.hnsw_stats, name)) for name in STAT_FIELDS}
    return distances, labels, stats
