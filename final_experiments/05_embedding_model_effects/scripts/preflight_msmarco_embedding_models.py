#!/usr/bin/env python3
"""Preflight checks for the MSMARCO five-embedding FAISS experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import h5py

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATA_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(REPO_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_MSMARCO_EMBEDDING_FAISS_INDEX_ROOT",
        str(REPO_ROOT / "index/msmarco_embedding_models_faiss_m32_efc500/darth/index"),
    )
).expanduser()
DATASETS = (
    "msmarco-v1-glove6b300d-full-ip.hdf5",
    "msmarco-v1-fasttext-cc300d-full-ip.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "marco_embeddings/msmarco-v1-bge-m3-fp32-dev6980-ip.hdf5",
    "marco_embeddings/msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip.hdf5",
)
DARTH_NAME = {
    "msmarco-v1-glove6b300d-full-ip.hdf5": "msmarco-v1-glove6b300d-full-ip",
    "msmarco-v1-fasttext-cc300d-full-ip.hdf5": "msmarco-v1-fasttext-cc300d-full-ip",
    "msmarco-v1-openai-ada2-full-ip.hdf5": "msmarco-v1-openai-ada2-full-ip",
    "marco_embeddings/msmarco-v1-bge-m3-fp32-dev6980-ip.hdf5": "msmarco-v1-bge-m3-fp32-dev6980-ip",
    "marco_embeddings/msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip.hdf5": "msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip",
}


def parse_datasets(value: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_datasets, default=list(DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--require-indexes", action="store_true")
    return parser.parse_args()


def hdf5_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        print(f"[MISSING] hdf5 {path}")
        return False
    try:
        with h5py.File(path, "r") as handle:
            keys = set(handle.keys())
            missing = sorted({"train", "test", "neighbors"} - keys)
            if missing:
                print(f"[BAD]     hdf5 {path} missing={missing}")
                return False
            train_shape = tuple(int(v) for v in handle["train"].shape)
            test_shape = tuple(int(v) for v in handle["test"].shape)
            neigh_shape = tuple(int(v) for v in handle["neighbors"].shape)
    except Exception as exc:  # noqa: BLE001
        print(f"[BAD]     hdf5 {path} error={exc!r}")
        return False
    print(f"[OK]      hdf5 {path} train={train_shape} test={test_shape} neighbors={neigh_shape}")
    return True


def index_path(index_root: Path, dataset: str, m: int, efc: int) -> Path:
    name = DARTH_NAME.get(dataset, Path(dataset).stem)
    return index_root / name / f"{name}.M{int(m)}.efC{int(efc)}.index"


def main() -> int:
    args = parse_args()
    base = args.base_path.expanduser().resolve()
    index_root = args.index_root.expanduser().resolve()
    ok = True
    for dataset in args.datasets:
        ok = hdf5_ok(base / dataset) and ok
        idx = index_path(index_root, dataset, int(args.param_m), int(args.ef_construction))
        if idx.exists() and idx.stat().st_size > 0:
            print(f"[OK]      index {idx}")
        else:
            print(f"[MISSING] index {idx} (runner can build this on miss)")
            if args.require_indexes:
                ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
