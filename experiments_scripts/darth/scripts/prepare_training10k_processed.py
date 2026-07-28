#!/usr/bin/env python3
"""Prepare DARTH processed inputs for 10K training-query runs.

This script avoids rewriting 100M base vectors. For datasets that already have
10K DARTH learn splits it symlinks the existing processed directory. For
SIFT100M and DEEP100M it reuses the previous base/query files and rewrites
learn.fvecs + learn.groundtruth.* from the HDF5 10K test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path("/home/kyungmin/vectordb/hnsw-playground")
PAPERS_ROOT = PROJECT_ROOT / "trials_on_fixing_search_process/adaptive_efsearch/papers"
DATASET_ROOT = PROJECT_ROOT / "datasets"
GLOBAL_PROCESSED = PROJECT_ROOT / "datasets/processed/DARTH"
REUSE_ROOT = PROJECT_ROOT / "index/m32_efc500_target095_adaef_darth_efs1000_20260603"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "index/darth_m32_efc500_efs1000_training10k_5datasets_20260615"


DATASETS = {
    "nytimes": {
        "darth_name": "nytimes-256-angular",
        "source_processed": GLOBAL_PROCESSED / "nytimes-256-angular",
    },
    "glove-100": {
        "darth_name": "glove-100-angular",
        "source_processed": GLOBAL_PROCESSED / "glove-100-angular",
    },
    "msmarco": {
        "darth_name": "msmarco-v1-openai-ada2-full-ip",
        "source_processed": GLOBAL_PROCESSED / "msmarco-v1-openai-ada2-full-ip",
    },
    "deep-100M": {
        "darth_name": "deep-100M-angular",
        "hdf5": DATASET_ROOT / "deep-100M.hdf5",
        "source_processed": REUSE_ROOT / "darth/processed/deep-100M-angular",
        "metric": "angular",
    },
    "sift-100M": {
        "darth_name": "sift-100M-euclidean",
        "hdf5": DATASET_ROOT / "sift-100M-euclidean.hdf5",
        "source_processed": REUSE_ROOT / "darth/processed/sift-100M-euclidean",
        "metric": "euclidean",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument(
        "--datasets",
        default="sift-100M,deep-100M,msmarco,glove-100,nytimes",
        help="Comma-separated labels.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(target)


def write_fvecs(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix for {path}: {matrix.shape}")
    d = matrix.shape[1]
    dims = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    packed = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    packed[:, :4] = dims.view(np.uint8).reshape(-1, 4)
    packed[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed.tobytes())


def write_ivecs(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.int32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix for {path}: {matrix.shape}")
    d = matrix.shape[1]
    dims = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    packed = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    packed[:, :4] = dims.view(np.uint8).reshape(-1, 4)
    packed[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packed.tobytes())


def prepare_existing_10k(target_root: Path, label: str, spec: dict[str, object]) -> None:
    target = target_root / "darth/processed" / str(spec["darth_name"])
    source = Path(spec["source_processed"])
    meta = json.loads((source / "metadata.json").read_text())
    if int(meta.get("learn_queries", 0)) != 10000:
        raise ValueError(f"{label} source is not 10K learn: {source}")
    ensure_symlink(target, source)


def prepare_from_hdf5_10k(target_root: Path, label: str, spec: dict[str, object], force: bool) -> None:
    darth_name = str(spec["darth_name"])
    source = Path(spec["source_processed"])
    hdf5_path = Path(spec["hdf5"])
    metric = str(spec["metric"])
    target = target_root / "darth/processed" / darth_name
    target.mkdir(parents=True, exist_ok=True)

    for name in [
        "base.fvecs",
        "query.fvecs",
        "query.groundtruth.ivecs",
        "query.groundtruth.fvecs",
        "validation.fvecs",
        "validation.groundtruth.ivecs",
        "validation.groundtruth.fvecs",
    ]:
        src = source / name
        if src.exists():
            ensure_symlink(target / name, src)

    learn_files = [
        target / "learn.fvecs",
        target / "learn.groundtruth.ivecs",
        target / "learn.groundtruth.fvecs",
    ]
    if not force and all(path.exists() for path in learn_files):
        return

    with h5py.File(hdf5_path, "r") as h5f:
        queries = np.asarray(h5f["test"][:10000], dtype=np.float32)
        neighbors = np.asarray(h5f["neighbors"][:10000], dtype=np.int32)
        if metric == "euclidean":
            dist_key = "distances_sq" if "distances_sq" in h5f else "distances"
            gt_scores = np.asarray(h5f[dist_key][:10000], dtype=np.float32)
        elif metric == "angular":
            gt_scores = 1.0 - np.asarray(h5f["distances"][:10000], dtype=np.float32)
        else:
            gt_scores = np.asarray(h5f["distances"][:10000], dtype=np.float32)

    write_fvecs(target / "learn.fvecs", queries)
    write_ivecs(target / "learn.groundtruth.ivecs", neighbors)
    write_fvecs(target / "learn.groundtruth.fvecs", gt_scores)

    source_meta = json.loads((source / "metadata.json").read_text())
    metadata = dict(source_meta)
    metadata.update(
        {
            "learn_queries": int(queries.shape[0]),
            "learn_origin": "hdf5/test[:10000]",
            "learn_groundtruth_origin": "hdf5/neighbors + hdf5/distances",
            "paper_faithful_training_queries": True,
            "previous_source_processed": str(source.resolve()),
        }
    )
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"prepared {label}: {target}")


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    labels = [part.strip() for part in args.datasets.split(",") if part.strip()]
    for label in labels:
        spec = DATASETS[label]
        if "hdf5" in spec:
            prepare_from_hdf5_10k(run_root, label, spec, args.force)
        else:
            prepare_existing_10k(run_root, label, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
