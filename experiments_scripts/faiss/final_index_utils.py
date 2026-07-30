"""Faiss index loading helpers for final SAGE experiments.

This module owns the Faiss backend setup. The adapter class exposes the
shared SAGE backend API while all index loading, search, trace, and LID work is
performed through Faiss.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))


def _find_default_project_root() -> Path:
    env_root = os.environ.get("HNSW_PLAYGROUND_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (EXP_ROOT, *EXP_ROOT.parents):
        if (candidate / "datasets").exists():
            return candidate
    return EXP_ROOT


PROJECT_ROOT = _find_default_project_root()
DEFAULT_FAISS_PYTHON_PATH = os.environ.get("FAISS_PYTHON_PATH")
DEFAULT_CONDA_ENV = os.environ.get("SAGE_FAISS_CONDA_ENV", "hnsw")
CONDA_REEXEC_SENTINEL = "SAGE_FAISS_CONDA_REEXECED"
DEFAULT_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "FAISS_INDEX_ROOT",
        str(PROJECT_ROOT / "index/m32_efc500_target095_adaef_darth_efs1000_20260603/darth/index"),
    )
)


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
    index_root: Path | None = None
    allow_system_faiss: bool = False


CONFIG = FaissLoaderConfig()
_FAISS_INDEX_CLASS = None
_FAISS_MODULE = None


def configure_faiss_loader(
    *,
    python_path: str | Path | None,
    index_root: str | Path,
    allow_system_faiss: bool,
) -> None:
    CONFIG.python_path = Path(python_path).expanduser() if python_path else None
    CONFIG.index_root = Path(index_root).expanduser()
    CONFIG.allow_system_faiss = bool(allow_system_faiss)


def maybe_reexec_in_conda_env(
    *,
    no_conda_reexec: bool,
    argv: Sequence[str],
    script_path: str | Path,
) -> None:
    target_env = str(DEFAULT_CONDA_ENV or "").strip()
    if bool(no_conda_reexec) or not target_env:
        return
    if os.environ.get(CONDA_REEXEC_SENTINEL) == "1":
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == target_env:
        return

    default_conda_exe = Path("/home/kyungmin/anaconda3/bin/conda")
    conda_exe = (
        os.environ.get("CONDA_EXE")
        or shutil.which("conda")
        or (str(default_conda_exe) if default_conda_exe.exists() else None)
    )
    if not conda_exe:
        print(
            f"[FAISS] conda env {target_env!r} requested but conda executable was not found; "
            "continuing in the current Python environment.",
            file=sys.stderr,
        )
        return

    env = os.environ.copy()
    env[CONDA_REEXEC_SENTINEL] = "1"
    cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        target_env,
        "python",
        str(Path(script_path).resolve()),
        *argv,
    ]
    print(f"[FAISS] re-execing under conda env {target_env}: {' '.join(cmd)}", flush=True)
    os.execvpe(conda_exe, cmd, env)


def resolve_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = Path(dataset_name).name
    stem = Path(dataset_name).stem
    if key in DATASET_SPECS:
        return DATASET_SPECS[key]
    if stem in DATASET_SPECS:
        return DATASET_SPECS[stem]
    raise KeyError(
        f"No Faiss/DARTH index mapping for dataset={dataset_name!r}. "
        "Add it to DATASET_SPECS or pass a known dataset."
    )


def import_faiss_index_class():
    global _FAISS_INDEX_CLASS, _FAISS_MODULE
    if _FAISS_INDEX_CLASS is not None:
        return _FAISS_INDEX_CLASS

    python_path = CONFIG.python_path.expanduser().resolve() if CONFIG.python_path else None
    if python_path is not None:
        if not python_path.exists():
            raise FileNotFoundError(
                f"Missing Faiss Python package path: {python_path}. "
                "Build FAISS first, install it into the configured conda env, or omit "
                "--faiss-python-path to use the active Python environment."
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
                "Check --faiss-python-path or use --allow-system-faiss intentionally."
            ) from exc

    if not hasattr(faiss, "SearchParametersHNSWAdaptiveLight"):
        raise RuntimeError(
            f"Imported faiss from {faiss_file}, but SearchParametersHNSWAdaptiveLight is missing."
        )

    _FAISS_MODULE = faiss
    _FAISS_INDEX_CLASS = FaissSageIndex
    print(f"[FAISS] module={faiss_file}")
    return _FAISS_INDEX_CLASS


def build_original_index(
    *,
    train,
    dataset_name: str,
    index_dir: str,
    param_m: int,
    ef_construction: int,
    num_threads: int,
):
    spec = resolve_dataset_spec(dataset_name)
    index_root = CONFIG.index_root if CONFIG.index_root is not None else Path(index_dir)
    index_path = (
        index_root.expanduser().resolve()
        / spec.darth_name
        / f"{spec.darth_name}.M{int(param_m)}.efC{int(ef_construction)}.index"
    )
    if not index_path.exists():
        raise FileNotFoundError(f"Missing Faiss/DARTH index for {dataset_name}: {index_path}")

    FaissSageIndex = import_faiss_index_class()
    dim = int(train.shape[1])
    index = FaissSageIndex(space=spec.space, dim=dim)
    print(f"[FAISS] loading index dataset={dataset_name} space={spec.space} path={index_path}")
    index.load_index(str(index_path), max_elements=int(len(train)))
    index.set_num_threads(int(num_threads))
    count = int(index.get_current_count())
    if count != int(len(train)):
        raise ValueError(
            f"Index count mismatch for {dataset_name}: index ntotal={count}, train rows={len(train)}"
        )
    return index, spec.space, spec.darth_name
