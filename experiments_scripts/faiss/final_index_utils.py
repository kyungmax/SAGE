"""FAISS adapter loading and dataset metrics for exact/pseudo-GT comparisons."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))


DEFAULT_FAISS_PYTHON_PATH = os.environ.get("FAISS_PYTHON_PATH")


@dataclass(frozen=True)
class DatasetSpec:
    darth_name: str
    space: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "nytimes-256-angular.hdf5": DatasetSpec("nytimes-256-angular", "cosine"),
    "nytimes-256-angular": DatasetSpec("nytimes-256-angular", "cosine"),
    "glove-100-angular.hdf5": DatasetSpec("glove-100-angular", "cosine"),
    "glove-100-angular": DatasetSpec("glove-100-angular", "cosine"),
    "cohere-768-angular.hdf5": DatasetSpec("cohere-768-angular", "cosine"),
    "cohere-768-angular": DatasetSpec("cohere-768-angular", "cosine"),
    "msmarco-v1-openai-ada2-full-ip.hdf5": DatasetSpec("msmarco-v1-openai-ada2-full-ip", "ip"),
    "msmarco-v1-openai-ada2-full-ip": DatasetSpec("msmarco-v1-openai-ada2-full-ip", "ip"),
    "sift-100M-euclidean.hdf5": DatasetSpec("sift-100M-euclidean", "l2"),
    "sift-100M-euclidean": DatasetSpec("sift-100M-euclidean", "l2"),
    "deep-100M.hdf5": DatasetSpec("deep-100M-angular", "cosine"),
    "deep-100M": DatasetSpec("deep-100M-angular", "cosine"),
    "deep-100M-angular": DatasetSpec("deep-100M-angular", "cosine"),
    "deep-image-96-angular.hdf5": DatasetSpec("deep-image-96-angular", "cosine"),
    "deep-image-96-angular": DatasetSpec("deep-image-96-angular", "cosine"),
    "msspacev-100M-i8-euclidean.hdf5": DatasetSpec("msspacev-100M-i8-euclidean", "l2"),
    "msspacev-100M-i8-euclidean": DatasetSpec("msspacev-100M-i8-euclidean", "l2"),
    "youtube-15M-angular.hdf5": DatasetSpec("youtube-15M-angular", "cosine"),
    "youtube-15M-angular": DatasetSpec("youtube-15M-angular", "cosine"),
    "agnews-mxbai-1024-euclidean.hdf5": DatasetSpec("agnews-mxbai-1024-euclidean", "l2"),
    "agnews-mxbai-1024-euclidean": DatasetSpec("agnews-mxbai-1024-euclidean", "l2"),
    "landmark-nomic-768-angular.hdf5": DatasetSpec("landmark-nomic-768-angular", "cosine"),
    "landmark-nomic-768-angular": DatasetSpec("landmark-nomic-768-angular", "cosine"),
    "landmark-nomic-768-normalized.hdf5": DatasetSpec("landmark-nomic-768-angular", "cosine"),
    "landmark-nomic-768-normalized": DatasetSpec("landmark-nomic-768-angular", "cosine"),
}


@dataclass
class FaissLoaderConfig:
    python_path: Path | None = (
        Path(DEFAULT_FAISS_PYTHON_PATH).expanduser() if DEFAULT_FAISS_PYTHON_PATH else None
    )
    allow_system_faiss: bool = False


CONFIG = FaissLoaderConfig()
_FAISS_INDEX_CLASS = None


def configure_faiss_loader(
    *,
    python_path: str | Path | None,
    allow_system_faiss: bool,
) -> None:
    CONFIG.python_path = Path(python_path).expanduser() if python_path else None
    CONFIG.allow_system_faiss = bool(allow_system_faiss)


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = Path(dataset_name).name
    stem = Path(dataset_name).stem
    if key in DATASET_SPECS:
        return DATASET_SPECS[key]
    if stem in DATASET_SPECS:
        return DATASET_SPECS[stem]
    raise KeyError(
        f"No Faiss index mapping for dataset={dataset_name!r}. "
        "Add it to DATASET_SPECS or pass a known dataset."
    )


def import_faiss_index_class():
    global _FAISS_INDEX_CLASS
    if _FAISS_INDEX_CLASS is not None:
        return _FAISS_INDEX_CLASS

    python_path = CONFIG.python_path.expanduser().resolve() if CONFIG.python_path else None
    if python_path is not None:
        if not python_path.exists():
            raise FileNotFoundError(
                f"Missing Faiss Python package path: {python_path}. "
                "Build FAISS first or configure python_path=None to use the active Python environment."
            )
        sys.path.insert(0, str(python_path))

    import faiss  # type: ignore
    from faiss_sage_index import Index as FaissSageIndex  # type: ignore

    faiss_file = Path(getattr(faiss, "__file__", "")).resolve()
    if python_path is not None and not CONFIG.allow_system_faiss:
        try:
            faiss_file.relative_to(python_path)
        except ValueError as exc:
            raise RuntimeError(
                f"Imported faiss from {faiss_file}, not from {python_path}. "
                "Check python_path or configure allow_system_faiss=True intentionally."
            ) from exc

    if not hasattr(faiss, "SearchParametersHNSWAdaptiveLight"):
        raise RuntimeError(
            f"Imported faiss from {faiss_file}, but SearchParametersHNSWAdaptiveLight is missing."
        )

    _FAISS_INDEX_CLASS = FaissSageIndex
    print(f"[FAISS] module={faiss_file}")
    return _FAISS_INDEX_CLASS
