#!/usr/bin/env python3
"""Preflight checks for final experiment 08 DARTH/Ada-EF reruns."""

from __future__ import annotations

import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))).expanduser()
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index"),
    )
).expanduser()
HNSWLIB_INDEX_ROOT = Path(os.environ.get("SAGE_HNSWLIB_INDEX_ROOT", str(DEFAULT_PROJECT_ROOT / "index"))).expanduser()

DARTH_SOURCE_ROOT = Path(
    os.environ.get("SAGE_DARTH_ROOT", str(REPO_ROOT / "baselines/darth/benchmarking-darth"))
).expanduser()
DARTH_BUILD_ROOT = Path(
    os.environ.get("SAGE_DARTH_SIMD_BUILD_ROOT", str(DARTH_SOURCE_ROOT / "build-simd-avx512"))
).expanduser()
DARTH_BIN = Path(os.environ.get("SAGE_DARTH_BIN", str(DARTH_BUILD_ROOT / "hnsw-test/hnsw_test"))).expanduser()
DARTH_FAISS_LIB_DIR = Path(
    os.environ.get("SAGE_DARTH_FAISS_LIB_DIR", str(DARTH_BUILD_ROOT / "faiss"))
).expanduser()
ADAEF_ROOT = Path(os.environ.get("SAGE_ADAEF_ROOT", str(REPO_ROOT / "experiments_scripts/ada-ef"))).expanduser()
ADAEF_BIN = Path(
    os.environ.get("SAGE_ADAEF_BACKEND_BIN", str(ADAEF_ROOT / "build-simd-avx512/backend_runner"))
).expanduser()

DATASETS = {
    "cohere": {
        "hdf5": DATASET_ROOT / "cohere-768-angular.hdf5",
        "darth_name": "cohere-768-angular",
        "darth_index": FAISS_INDEX_ROOT / "cohere-768-angular/cohere-768-angular.M32.efC500.index",
        "adaef_existing_index": HNSWLIB_INDEX_ROOT / "cohere-768-angular_M32_M32_efC500_n10000000_dim768",
        "paper_full_query_count": 1000,
    },
    "msmarco": {
        "hdf5": DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5",
        "darth_name": "msmarco-v1-openai-ada2-full-ip",
        "darth_index": FAISS_INDEX_ROOT
        / "msmarco-v1-openai-ada2-full-ip/msmarco-v1-openai-ada2-full-ip.M32.efC500.index",
        "adaef_existing_index": HNSWLIB_INDEX_ROOT
        / "msmarco-v1-openai-ada2-full-ip_M32_M32_efC500_n8841823_dim1536",
        "paper_full_query_count": 6980,
    },
}


def hdf5_query_count(path: Path):
    if not path.exists():
        return None
    import h5py

    with h5py.File(path, "r") as h5f:
        return int(h5f["test"].shape[0])


def file_status(path: Path, *, executable: bool = False) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(path.is_file()) if exists else False,
        "executable": bool(os.access(path, os.X_OK)) if exists and executable else None,
        "bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
    }


def main() -> int:
    checks = {
        "paper_scope": {
            "section": "paper_tex/6.Experimental Evaluation.tex:635",
            "target_recall": 0.99,
            "datasets": ["cohere", "msmarco"],
            "darth_ef_search": 1000,
            "offline_threads": 24,
            "query_latency_threads": 1,
            "query_scope": "full HDF5 test set",
        },
        "scripts": {
            "darth_shim": file_status(SCRIPT_DIR / "run_darth_target099_simd_current.py"),
            "darth_runner": file_status(SCRIPT_DIR / "run_darth_cohere_msmarco_simd_target099.py"),
            "adaef_runner": file_status(SCRIPT_DIR / "run_adaef_cohere_msmarco_simd_target099.py"),
        },
        "darth": {
            "source_root": str(DARTH_SOURCE_ROOT),
            "build_root": str(DARTH_BUILD_ROOT),
            "binary": file_status(DARTH_BIN, executable=True),
            "faiss_static_lib": file_status(DARTH_FAISS_LIB_DIR / "libfaiss.a"),
            "faiss_shared_lib": file_status(DARTH_FAISS_LIB_DIR / "libfaiss.so"),
            "build_helper": str(SCRIPT_DIR / "build_darth_simd_avx512.sh"),
        },
        "adaef": {
            "source_root": str(ADAEF_ROOT),
            "binary": file_status(ADAEF_BIN, executable=True),
            "build_helper": str(SCRIPT_DIR / "build_adaef_simd_avx512.sh"),
        },
        "datasets": [],
    }
    failures = []

    if not checks["darth"]["binary"]["exists"]:
        failures.append("missing DARTH SIMD binary")
    if not checks["darth"]["faiss_static_lib"]["exists"] and not checks["darth"]["faiss_shared_lib"]["exists"]:
        failures.append("missing DARTH FAISS lib in SIMD build")
    if not checks["adaef"]["binary"]["exists"]:
        failures.append("missing Ada-EF SIMD backend_runner")

    for label, spec in DATASETS.items():
        query_count = hdf5_query_count(spec["hdf5"])
        row = {
            "label": label,
            "hdf5": file_status(spec["hdf5"]),
            "hdf5_test_query_count": query_count,
            "paper_full_query_count": spec["paper_full_query_count"],
            "full_query_matches_paper": bool(query_count == spec["paper_full_query_count"]),
            "darth_faiss_index": file_status(spec["darth_index"]),
            "adaef_existing_index": file_status(spec["adaef_existing_index"]),
        }
        checks["datasets"].append(row)
        if not row["hdf5"]["exists"]:
            failures.append(f"missing HDF5 for {label}")
        if query_count is not None and query_count != spec["paper_full_query_count"]:
            failures.append(f"{label} query count differs from paper expectation: {query_count}")
        if not row["darth_faiss_index"]["exists"]:
            failures.append(f"missing reusable DARTH FAISS index for {label}")
        if not row["adaef_existing_index"]["exists"]:
            failures.append(f"missing reusable Ada-EF hnswlib index for {label}; wrapper can build it if needed")

    payload = {"status": "ok" if not failures else "needs-attention", "failures": failures, "checks": checks}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
