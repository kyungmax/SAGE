#!/usr/bin/env python3
"""Run the current-sage DARTH target-0.99 wrapper with the local SIMD build.

This is a thin compatibility shim around
``experiments_scripts/darth/scripts/run_darth_target099_main4.py``.  The
original 08 experiment pointed at an archived DARTH source tree inside the old
paper workspace; this shim redirects it to the source copy in this repository
and records the expected AVX512 build settings in preflight output.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parents[1]
BASE_WRAPPER = REPO_ROOT / "experiments_scripts/darth/scripts/run_darth_target099_main4.py"

DEFAULT_PROJECT_ROOT = Path(os.environ.get("SAGE_PROJECT_ROOT", "/home/kyungmin/vectordb/hnsw-playground"))
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index"),
    )
).expanduser()

SIMD_DARTH_ROOT = Path(
    os.environ.get("SAGE_DARTH_ROOT", str(REPO_ROOT / "baselines/darth/benchmarking-darth"))
).expanduser()
SIMD_BUILD_ROOT = Path(
    os.environ.get("SAGE_DARTH_SIMD_BUILD_ROOT", str(SIMD_DARTH_ROOT / "build-simd-avx512"))
).expanduser()
SIMD_DARTH_BIN = Path(
    os.environ.get("SAGE_DARTH_BIN", str(SIMD_BUILD_ROOT / "hnsw-test/hnsw_test"))
).expanduser()
SIMD_FAISS_LIB_DIR = Path(
    os.environ.get("SAGE_DARTH_FAISS_LIB_DIR", str(SIMD_BUILD_ROOT / "faiss"))
).expanduser()
SIMD_FLAGS = "-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt"


def load_base():
    sys.modules.setdefault("hnswlib", types.ModuleType("hnswlib"))
    spec = importlib.util.spec_from_file_location("darth_target099_current_base", BASE_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base wrapper: {BASE_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_query_groundtruth_from_hdf5(runner, *, query_gt_k: int) -> None:
    def patched(spec, processed_dir: Path) -> dict:
        with runner.h5py.File(spec.hdf5, "r") as h5f:
            available_k = int(h5f["neighbors"].shape[1])
            if available_k < query_gt_k:
                raise ValueError(
                    f"{spec.label}: HDF5 neighbors only has k={available_k}, "
                    f"cannot write query top{query_gt_k}"
                )
            neighbors = runner.np.asarray(h5f["neighbors"][:, :query_gt_k], dtype=runner.np.int32)
            if spec.source_metric == "euclidean":
                if "distances_sq" in h5f:
                    scores = runner.np.asarray(h5f["distances_sq"][:, :query_gt_k], dtype=runner.np.float32)
                    score_source = "hdf5/distances_sq"
                elif "distances" in h5f:
                    distances = runner.np.asarray(h5f["distances"][:, :query_gt_k], dtype=runner.np.float32)
                    scores = runner.np.asarray(distances * distances, dtype=runner.np.float32)
                    score_source = "square(hdf5/distances)"
                else:
                    scores = runner.compute_scores_from_neighbors_hdf5(spec, h5f["train"], h5f["test"], neighbors)
                    score_source = "sq_l2(hdf5/test, hdf5/train[neighbors])"
            elif "distances" in h5f:
                distances = runner.np.asarray(h5f["distances"][:, :query_gt_k], dtype=runner.np.float32)
                scores = runner.np.asarray(1.0 - distances, dtype=runner.np.float32)
                score_source = "1 - hdf5/distances"
            else:
                scores = runner.compute_scores_from_neighbors_hdf5(spec, h5f["train"], h5f["test"], neighbors)
                score_source = "dot(hdf5/test, hdf5/train[neighbors])"

        runner.write_ivecs_matrix(processed_dir / "query.groundtruth.ivecs", neighbors)
        runner.write_fvecs_matrix(processed_dir / "query.groundtruth.fvecs", scores)
        return {
            "queries": int(neighbors.shape[0]),
            "gt_k": int(neighbors.shape[1]),
            "score_source": score_source,
        }

    runner.write_query_groundtruth_from_hdf5 = patched


def configure_base(base) -> None:
    base.PROJECT_ROOT = DEFAULT_PROJECT_ROOT
    base.DATASET_ROOT = DATASET_ROOT
    base.PATCHED_DARTH_ROOT = SIMD_DARTH_ROOT
    base.PATCHED_DARTH_BIN = SIMD_DARTH_BIN
    base.PATCHED_FAISS_LIB_DIR = SIMD_FAISS_LIB_DIR
    base.VERIFIED_FAISS_INDEX_ROOT = FAISS_INDEX_ROOT
    base.DEFAULT_RUN_ROOT = DEFAULT_PROJECT_ROOT / "index/darth_m32_efc500_target099_simd_cohere_msmarco_fullquery"
    for label, hdf5_name in {
        "cohere": "cohere-768-angular.hdf5",
        "msmarco": "msmarco-v1-openai-ada2-full-ip.hdf5",
    }.items():
        base.DATASET_DEFS[label]["hdf5"] = DATASET_ROOT / hdf5_name

    def configure_runner(runner, args, run_root):
        base.prepend_env_path("LD_LIBRARY_PATH", SIMD_FAISS_LIB_DIR)
        runner.DARTH_BIN = SIMD_DARTH_BIN
        runner.COMMON_INDEX_ROOT = Path(args.common_index_root).expanduser().resolve()
        patch_query_groundtruth_from_hdf5(runner, query_gt_k=int(args.k))
        for label, payload in base.DATASET_DEFS.items():
            runner.DATASETS[label] = runner.DatasetSpec(**payload)
        return [runner.DATASETS[label] for label in base.parse_labels(args.datasets)]

    def validate_preflight(runner, specs, args, run_root):
        static_lib = SIMD_FAISS_LIB_DIR / "libfaiss.a"
        shared_lib = SIMD_FAISS_LIB_DIR / "libfaiss.so"
        checks = {
            "runner_path": str(base.RUNNER_PATH),
            "darth_bin": str(SIMD_DARTH_BIN),
            "faiss_lib_dir": str(SIMD_FAISS_LIB_DIR),
            "faiss_static_lib": str(static_lib),
            "faiss_shared_lib": str(shared_lib),
            "run_root": str(run_root),
            "common_index_root": str(runner.COMMON_INDEX_ROOT),
            "datasets": [],
            "simd": {
                "darth_source_root": str(SIMD_DARTH_ROOT),
                "build_dir": str(SIMD_BUILD_ROOT),
                "expected_cxx_flags": SIMD_FLAGS,
                "build_helper": str(EXP_ROOT / "scripts/build_darth_simd_avx512.sh"),
            },
        }
        if not base.RUNNER_PATH.exists():
            raise FileNotFoundError(f"missing runner: {base.RUNNER_PATH}")
        if not SIMD_DARTH_BIN.exists():
            raise FileNotFoundError(f"missing DARTH SIMD binary: {SIMD_DARTH_BIN}")
        if not static_lib.exists() and not shared_lib.exists():
            raise FileNotFoundError(f"missing libfaiss.a/libfaiss.so in {SIMD_FAISS_LIB_DIR}")
        for spec in specs:
            if not spec.hdf5.exists():
                raise FileNotFoundError(f"missing HDF5 for {spec.label}: {spec.hdf5}")
            index_path = runner.index_path_for(spec, runner.COMMON_INDEX_ROOT, args.m, args.ef_construction)
            checks["datasets"].append(
                {
                    "label": spec.label,
                    "darth_name": spec.darth_name,
                    "hdf5": str(spec.hdf5),
                    "metric": spec.source_metric,
                    "index_path": str(index_path),
                    "index_exists": index_path.exists(),
                }
            )
        return checks

    base.configure_runner = configure_runner
    base.validate_preflight = validate_preflight


def main() -> int:
    base = load_base()
    configure_base(base)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
