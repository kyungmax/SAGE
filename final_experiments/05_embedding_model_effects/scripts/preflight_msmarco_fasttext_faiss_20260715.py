#!/usr/bin/env python3
"""Preflight checks before running the MSMARCO fastText FAISS experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
EFFECTS_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
PROJECT_ROOT = Path(
    os.environ.get("HNSW_PLAYGROUND_ROOT", os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT)))
).expanduser()
DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
PYTHON = Path(os.environ.get("SAGE_PYTHON", "python3")).expanduser()

DEFAULT_COLLECTION_TSV = DATASET_DIR / "msmarco_passage_glove_static/raw/collection.tsv"
DEFAULT_QUERIES_TSV = DATASET_DIR / "msmarco_passage_glove_static/raw/queries.dev.tsv"
DEFAULT_FASTTEXT_BIN = DATASET_DIR / "msmarco_passage_fasttext_static/raw/cc.en.300.bin"
DEFAULT_MEAN_HDF5 = DATASET_DIR / "msmarco-v1-fasttext-cc300d-full-ip.hdf5"
DEFAULT_TFIDF_HDF5 = DATASET_DIR / "msmarco-v1-fasttext-cc300d-tfidf-full-ip.hdf5"
DEFAULT_IDF_CACHE = DATASET_DIR / "msmarco_passage_fasttext_static/fasttext-cc300d_msmarco_collection_token_idf.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--collection-tsv", type=Path, default=DEFAULT_COLLECTION_TSV)
    parser.add_argument("--queries-tsv", type=Path, default=DEFAULT_QUERIES_TSV)
    parser.add_argument("--fasttext-bin", type=Path, default=DEFAULT_FASTTEXT_BIN)
    parser.add_argument("--mean-hdf5", type=Path, default=DEFAULT_MEAN_HDF5)
    parser.add_argument("--tfidf-hdf5", type=Path, default=DEFAULT_TFIDF_HDF5)
    parser.add_argument("--idf-cache", type=Path, default=DEFAULT_IDF_CACHE)
    parser.add_argument(
        "--require-ready-for-faiss",
        action="store_true",
        help="Exit nonzero unless package, model, and both HDF5 files are present and readable.",
    )
    return parser.parse_args()


def format_size(path: Path) -> str:
    size = path.stat().st_size
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def check_file(label: str, path: Path, *, required: bool, min_bytes: int = 1) -> bool:
    path = path.expanduser()
    if path.exists() and path.is_file() and path.stat().st_size >= int(min_bytes):
        print(f"[OK]      {label}: {path} ({format_size(path)})")
        return True
    status = "MISSING" if required else "ABSENT"
    print(f"[{status}] {label}: {path}")
    return not required


def check_python_package(python: Path, package: str) -> bool:
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.find_spec({package!r}); "
        "sys.exit(0 if spec is not None else 1)"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        print(f"[MISSING] python: {python} ({exc})")
        return False
    ok = proc.returncode == 0
    marker = "OK" if ok else "MISSING"
    extra = "" if ok else f" stderr={proc.stderr.strip()!r}"
    print(f"[{marker}] python package {package}: {python}{extra}")
    return ok


def inspect_hdf5(label: str, path: Path, *, required: bool) -> bool:
    if not path.exists():
        status = "MISSING" if required else "ABSENT"
        print(f"[{status}] {label}: {path}")
        return not required
    try:
        import h5py
    except ModuleNotFoundError as exc:
        print(f"[MISSING] local h5py needed to inspect {label}: {exc}")
        return False
    try:
        with h5py.File(path, "r") as h5f:
            keys = set(h5f.keys())
            required_keys = {"train", "test", "neighbors", "distances"}
            missing = sorted(required_keys - keys)
            if missing:
                print(f"[BAD]     {label}: {path} missing keys={missing}")
                return False
            train_shape = tuple(int(v) for v in h5f["train"].shape)
            test_shape = tuple(int(v) for v in h5f["test"].shape)
            neigh_shape = tuple(int(v) for v in h5f["neighbors"].shape)
            dist_shape = tuple(int(v) for v in h5f["distances"].shape)
    except Exception as exc:  # noqa: BLE001
        print(f"[BAD]     {label}: {path} ({exc})")
        return False
    shape_note = f"train={train_shape} test={test_shape} neighbors={neigh_shape} distances={dist_shape}"
    if len(train_shape) != 2 or train_shape[1] != 300:
        print(f"[BAD]     {label}: {path} unexpected shape {shape_note}")
        return False
    if neigh_shape != dist_shape or len(neigh_shape) != 2:
        print(f"[BAD]     {label}: {path} invalid gt shapes {shape_note}")
        return False
    print(f"[OK]      {label}: {path} ({format_size(path)}) {shape_note}")
    return True


def print_next_steps(args: argparse.Namespace, *, basic_ready: bool, ready_for_faiss: bool) -> None:
    print("\n[NEXT]")
    if ready_for_faiss:
        print("  ./run_msmarco_embedding_models_faiss_24t.sh --datasets msmarco-v1-fasttext-cc300d-full-ip.hdf5")
        return
    if not basic_ready:
        print(f"  {args.python} -m pip install fasttext-wheel")
        print("  ./scripts/download_msmarco_fasttext_model.sh")
    print("  ./scripts/build_msmarco_fasttext_static_hdf5.sh")
    print("  ./scripts/preflight_msmarco_fasttext_faiss_20260715.py --require-ready-for-faiss")
    print("  ./run_msmarco_embedding_models_faiss_24t.sh --datasets msmarco-v1-fasttext-cc300d-full-ip.hdf5")


def main() -> int:
    args = parse_args()
    basic_checks: list[bool] = []
    basic_checks.append(check_file("python", args.python, required=True))
    basic_checks.append(check_python_package(args.python, "numpy"))
    basic_checks.append(check_python_package(args.python, "h5py"))
    basic_checks.append(check_python_package(args.python, "fasttext"))
    basic_checks.append(check_file("collection.tsv", args.collection_tsv, required=True, min_bytes=1024))
    basic_checks.append(check_file("queries.dev.tsv", args.queries_tsv, required=True, min_bytes=1024))
    basic_checks.append(check_file("fastText .bin", args.fasttext_bin, required=True, min_bytes=1024))

    mean_exists = args.mean_hdf5.expanduser().exists()
    tfidf_exists = args.tfidf_hdf5.expanduser().exists()
    mean_ok = inspect_hdf5("fastText mean HDF5", args.mean_hdf5, required=bool(args.require_ready_for_faiss))
    tfidf_ok = inspect_hdf5("fastText tfidf HDF5", args.tfidf_hdf5, required=bool(args.require_ready_for_faiss))
    check_file("fastText IDF cache", args.idf_cache, required=False, min_bytes=1024)

    basic_ready = all(basic_checks)
    ready_for_faiss = bool(basic_ready and mean_exists and tfidf_exists and mean_ok and tfidf_ok)
    print_next_steps(args, basic_ready=basic_ready, ready_for_faiss=ready_for_faiss)
    if args.require_ready_for_faiss and not ready_for_faiss:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
