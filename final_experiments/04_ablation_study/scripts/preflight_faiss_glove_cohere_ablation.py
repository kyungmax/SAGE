#!/usr/bin/env python3
"""Preflight checks for the FAISS GloVe/Cohere ablation scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import h5py

DEFAULT_LEGACY_ROOT = Path("/home/kyungmin/vectordb/hnsw-playground")
DEFAULT_PROJECT_ROOT = Path(os.environ.get("HNSW_PLAYGROUND_ROOT", str(DEFAULT_LEGACY_ROOT))).expanduser()
DEFAULT_DATA_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get(
            "FAISS_INDEX_ROOT",
            str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index"),
        ),
    )
).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get("FAISS_PYTHON_PATH", "/home/kyungmin/vectordb/faiss/build_hnsw_py312_avx512/faiss/python")
).expanduser()
DATASETS = ("glove-100-angular.hdf5", "cohere-768-angular.hdf5")
DARTH_NAME = {
    "glove-100-angular.hdf5": "glove-100-angular",
    "cohere-768-angular.hdf5": "cohere-768-angular",
}


def parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_csv, default=list(DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--require-indexes", action="store_true")
    parser.add_argument("--require-faiss-path", action="store_true")
    return parser.parse_args(argv)


def hdf5_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        print(f"[MISSING] hdf5 {path}")
        return False
    try:
        with h5py.File(path, "r") as handle:
            missing = sorted({"train", "test", "neighbors"} - set(handle.keys()))
            if missing:
                print(f"[BAD]     hdf5 {path} missing={missing}")
                return False
            train_shape = tuple(int(v) for v in handle["train"].shape)
            test_shape = tuple(int(v) for v in handle["test"].shape)
            neighbors_shape = tuple(int(v) for v in handle["neighbors"].shape)
    except Exception as exc:  # noqa: BLE001
        print(f"[BAD]     hdf5 {path} error={exc!r}")
        return False
    print(f"[OK]      hdf5 {path} train={train_shape} test={test_shape} neighbors={neighbors_shape}")
    return True


def index_path(index_root: Path, dataset: str, m: int, efc: int) -> Path:
    name = DARTH_NAME.get(Path(dataset).name, Path(dataset).stem)
    return index_root / name / f"{name}.M{int(m)}.efC{int(efc)}.index"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ok = True
    base = Path(args.base_path).expanduser().resolve()
    index_root = Path(args.index_root).expanduser().resolve()
    faiss_path = Path(args.faiss_python_path).expanduser()
    if faiss_path.exists():
        print(f"[OK]      faiss-python-path {faiss_path}")
    else:
        print(f"[MISSING] faiss-python-path {faiss_path} (system FAISS may still work if patched)")
        if bool(args.require_faiss_path):
            ok = False
    for dataset in args.datasets:
        ok = hdf5_ok(base / dataset) and ok
        idx = index_path(index_root, dataset, int(args.param_m), int(args.ef_construction))
        if idx.exists() and idx.stat().st_size > 0:
            print(f"[OK]      index {idx}")
        else:
            print(f"[MISSING] index {idx} (runners can build this on miss)")
            if bool(args.require_indexes):
                ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
