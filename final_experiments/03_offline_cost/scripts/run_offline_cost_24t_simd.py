#!/usr/bin/env python3
"""FAISS/hnswlib SIMD-on 24-thread SAGE offline-calibration cost runner.

This runner measures only the offline calibration added by SAGE. It excludes
HNSW index build/load time from the paper-facing timing columns, but records
load/build wall time separately for auditability.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FAISS_IMPL_ROOT = EXPERIMENTS_ROOT / "faiss"
HNSW_IMPL_ROOT = EXPERIMENTS_ROOT / "hnswlib"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HNSW_PLAYGROUND_ROOT", os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT)))
).expanduser()
DEFAULT_DATA_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_HNSW_INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(DEFAULT_PROJECT_ROOT / "index"))).expanduser()
DEFAULT_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get("FAISS_INDEX_ROOT", str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/darth/index")),
    )
).expanduser()
DEFAULT_HNSWLIB_EXTENSION_ROOT = Path(
    os.environ.get("SAGE_HNSWLIB_EXTENSION_ROOT", str(REPO_ROOT.parent / "hnswlib"))
).expanduser()


def _first_existing_path(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()
    return candidates[0].expanduser()


DEFAULT_FAISS_PYTHON_PATH = _first_existing_path(
    (
        Path(os.environ["FAISS_PYTHON_PATH"]) if os.environ.get("FAISS_PYTHON_PATH") else REPO_ROOT / "faiss/build_sage_avx512/faiss/python",
        REPO_ROOT / "faiss/build_sage_avx512/faiss/python",
    )
)

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.adaptive_runtime import (  # noqa: E402
    DEFAULT_CALIBRATION_SAMPLE_SEED,
    DEFAULT_EF_SWEEP_VALUES,
    DEFAULT_MIXED_GT_EF,
    DEFAULT_MIXED_GT_SOURCE,
    load_dataset_with_special_cases,
    parse_ef_sweep,
    resolve_runtime_threshold_scales,
)
from common import projected_local_acceptable_runtime as mixed_runtime  # noqa: E402
from common.offline_calibration import (  # noqa: E402
    FIXED_CALIBRATION_LID_POOL_SIZE,
    FIXED_CALIBRATION_SAMPLE_FRACTION,
    compute_fixed_calibration_lid_pool,
)

DATASETS = (
    "nytimes-256-angular.hdf5",
    "glove-100-angular.hdf5",
    "agnews-mxbai-1024-euclidean.hdf5",
    "landmark-nomic-768-angular.hdf5",
    "cohere-768-angular.hdf5",
    "youtube-15M-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
)

K = 10
OFFLINE_THREADS = 24
PARAM_M = 32
EF_CONSTRUCTION = 500
NUM_CALIBRATION_QUERIES = 100
INTERNAL_LID_K = 15
TMIN_POPS = 25
MIXED_THRESHOLD_MODE = "paper_floor_half"
MIXED_BUCKET_COUNT = 4
PAIR_GAP = 2
CLASSIFY_START = 4
CLASSIFY_END = 16
CFR_EMA_DECAY = 0.8

BACKEND = ""
BACKEND_LABEL = ""
BACKEND_BUILD_OR_LOAD_INDEX = None

FIELDNAMES = [
    "backend",
    "simd",
    "repeat",
    "dataset",
    "dataset_file",
    "points",
    "dimensions",
    "k",
    "groundtruth_k",
    "offline_num_threads",
    "M",
    "efConstruction",
    "num_calibration_queries",
    "calibration_lid_pool_count",
    "calibration_lid_pool_wall_s",
    "step1_lid_sampling_wall_s",
    "step2_query_selection_wall_s",
    "step2_pseudo_gt_wall_s",
    "step2_pre_evaluation_wall_s",
    "step3_eval_wall_s",
    "step3_config_wall_s",
    "step3_recall_curve_wall_s",
    "step3_cfr_distribution_wall_s",
    "step3_threshold_survival_wall_s",
    "offline_algorithm_wall_s",
    "threshold_calibration_wall_s",
    "offline_calibration_wall_s",
    "calibration_cache_load_wall_s",
    "calibration_cache_save_wall_s",
    "calibration_unattributed_wall_s",
    "calibration_query_selection_count",
    "step3_selection_ef_count",
    "step3_unique_recall_ef_count",
    "calibration_gt_source",
    "calibration_gt_ef",
    "dataset_load_wall_s",
    "index_load_wall_s",
    "calibration_cache_status",
    "calibration_cache_exists_before",
    "mixed_threshold_mode",
    "mixed_bucket_count",
    "pair_gap",
    "classify_start",
    "classify_end",
    "cfr_ema_decay",
    "cache_path",
    "policy_source_label",
    "log_path",
]

MEDIAN_FIELDNAMES = [
    "backend",
    "simd",
    "dataset",
    "dataset_file",
    "points",
    "dimensions",
    "k",
    "offline_num_threads",
    "M",
    "efConstruction",
    "num_calibration_queries",
    "repeats",
    "paper_samp_s",
    "paper_select_s",
    "paper_eval_s",
    "paper_total_s",
    "step1_lid_sampling_wall_s",
    "step2_pre_evaluation_wall_s",
    "step3_eval_wall_s",
    "offline_calibration_wall_s",
    "step2_query_selection_wall_s",
    "step2_pseudo_gt_wall_s",
    "step3_recall_curve_wall_s",
    "step3_cfr_distribution_wall_s",
    "step3_threshold_survival_wall_s",
    "threshold_calibration_wall_s",
]


def prepend_path(path: Path) -> None:
    text = str(path.expanduser().resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_dataset_list(value: str) -> tuple[str, ...]:
    datasets = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not datasets:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")
    return datasets


def append_csv(path: Path, fieldnames: list[str], row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, fieldnames: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


class IndexBackedTrainProxy:
    def __init__(self, *, shape: tuple[int, int], dtype: Any = np.float32):
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = np.dtype(dtype)
        self._index = None

    def __len__(self) -> int:
        return int(self.shape[0])

    def astype(self, dtype, copy: bool = False):
        del copy
        self.dtype = np.dtype(dtype)
        return self

    def bind_index(self, index) -> None:
        self._index = index

    def __getitem__(self, item):
        if self._index is None:
            raise RuntimeError("IndexBackedTrainProxy must be bound to an index before vector access.")
        raw_index = self._index._require_index() if hasattr(self._index, "_require_index") else self._index
        if isinstance(item, slice):
            ids = np.arange(*item.indices(len(self)), dtype=np.int64)
        else:
            ids = np.asarray(item, dtype=np.int64)
            if ids.ndim == 0:
                ids = ids.reshape(1)
        if hasattr(raw_index, "reconstruct_batch"):
            vectors = raw_index.reconstruct_batch(np.ascontiguousarray(ids, dtype=np.int64))
        elif hasattr(raw_index, "reconstruct"):
            vectors = np.vstack([raw_index.reconstruct(int(idx)) for idx in ids])
        else:
            raise RuntimeError("Bound index does not support vector reconstruction.")
        return np.asarray(vectors, dtype=np.float32)


def load_dataset_for_offline_cost(base_path: str, dataset_name: str):
    try:
        return load_dataset_with_special_cases(base_path, dataset_name)
    except OSError as exc:
        message = str(exc)
        if "msspacev" not in dataset_name.lower() or "external raw data file" not in message:
            raise
        file_path = Path(base_path) / dataset_name
        with h5py.File(file_path, "r") as handle:
            print(f"Keys in HDF5 file: {list(handle.keys())}")
            train_shape = tuple(int(value) for value in handle["train"].shape)
            test = np.asarray(handle["test"])
            neighbors = np.asarray(handle["neighbors"])
        train = IndexBackedTrainProxy(shape=train_shape, dtype=np.float32)
        return train, test, neighbors


def configure_backend(args: argparse.Namespace) -> None:
    global BACKEND, BACKEND_LABEL, BACKEND_BUILD_OR_LOAD_INDEX
    BACKEND = str(args.backend)
    if BACKEND == "hnswlib":
        prepend_path(Path(args.hnswlib_extension_root))
        stale = sys.modules.get("hnswlib")
        if stale is not None and not hasattr(stale, "Index"):
            del sys.modules["hnswlib"]
        import hnswlib as hnswlib_extension  # noqa: F401

        if not hasattr(hnswlib_extension, "Index"):
            raise RuntimeError(
                f"Imported hnswlib without Index from {getattr(hnswlib_extension, '__file__', None)!r}. "
                "Set --hnswlib-extension-root to the compiled SIMD-enabled hnswlib directory."
            )
        prepend_path(HNSW_IMPL_ROOT)
        module = load_module_from_path("offline_cost_hnswlib_final_index_utils", HNSW_IMPL_ROOT / "final_index_utils.py")
        BACKEND_LABEL = "HNSWLib"
        BACKEND_BUILD_OR_LOAD_INDEX = module.build_original_index
        return

    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(DEFAULT_PROJECT_ROOT))
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    os.environ["FAISS_PYTHON_PATH"] = str(Path(args.faiss_python_path).expanduser().resolve())
    os.environ["FAISS_INDEX_ROOT"] = str(Path(args.faiss_index_root).expanduser().resolve())
    prepend_path(FAISS_IMPL_ROOT)
    module = load_module_from_path("offline_cost_faiss_final_index_utils", FAISS_IMPL_ROOT / "final_index_utils.py")
    module.maybe_reexec_in_conda_env(
        no_conda_reexec=bool(args.no_conda_reexec),
        argv=sys.argv[1:],
        script_path=SCRIPT_PATH,
    )
    module.configure_faiss_loader(
        python_path=Path(args.faiss_python_path),
        index_root=Path(args.faiss_index_root),
        allow_system_faiss=bool(args.allow_system_faiss),
    )
    BACKEND_LABEL = "FAISS"
    BACKEND_BUILD_OR_LOAD_INDEX = module.build_original_index


def add_elapsed(measurements: dict[str, Any], key: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    measurements[key] = float(measurements.get(key, 0.0)) + float(elapsed)
    return float(elapsed)


def mixed_cache_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "calibration_graph_variant": "original",
        "calibration_lid_source_graph": "original",
        "calibration_probe_routing": "hide_node",
        "benchmark_family": "sage_offline_cost",
        "offline_num_threads": int(args.offline_num_threads),
        "ablation_name": "baseline",
        "ablation_value": "default",
        "classify_start": int(args.classify_start),
        "classify_end": int(args.classify_end),
        "cfr_ema_decay": float(args.cfr_ema_decay),
        "pair_gap": int(args.pair_gap),
    }


def resolve_mixed_policy_with_step_timings(
    *,
    index,
    train,
    dataset_name: str,
    ef_values: list[int],
    k: int,
    acceptable_recall_threshold: float,
    tmin_pops: int,
    num_calibration_queries: int,
    selection_mode: str,
    trim_low_percentile: float,
    trim_high_percentile: float,
    internal_lid_k: int,
    num_threads: int,
    easy_threshold_scale: float,
    mid_threshold_scale: float,
    super_threshold_scale: float,
    lid_df: pd.DataFrame,
    lid_sample_seed: int,
    cache_path: Path,
    mixed_threshold_mode: str,
    mixed_bucket_count: int,
    classify_start: int,
    classify_end: int,
    cfr_ema_decay: float,
    pair_gap: int,
    cache_context: dict[str, Any],
    emit=None,
):
    measurements: dict[str, Any] = {
        "step2_query_selection_wall_s": 0.0,
        "step2_pseudo_gt_wall_s": 0.0,
        "step2_pre_evaluation_wall_s": 0.0,
        "step3_config_wall_s": 0.0,
        "step3_recall_curve_wall_s": 0.0,
        "step3_cfr_distribution_wall_s": 0.0,
        "step3_threshold_survival_wall_s": 0.0,
        "step3_eval_wall_s": 0.0,
        "calibration_cache_load_wall_s": 0.0,
        "calibration_cache_save_wall_s": 0.0,
        "calibration_query_selection_count": 0,
        "step3_selection_ef_count": 0,
        "step3_unique_recall_ef_count": 0,
        "calibration_gt_source": str(DEFAULT_MIXED_GT_SOURCE),
        "calibration_gt_ef": int(DEFAULT_MIXED_GT_EF),
    }
    cache_settings = mixed_runtime._build_cache_settings(
        dataset_name=dataset_name,
        ef_values=ef_values,
        acceptable_recall_threshold=float(acceptable_recall_threshold),
        requested_tmin_pops=int(tmin_pops),
        num_calibration_queries=int(num_calibration_queries),
        selection_mode=selection_mode,
        trim_low_percentile=float(trim_low_percentile),
        trim_high_percentile=float(trim_high_percentile),
        gt_source=DEFAULT_MIXED_GT_SOURCE,
        gt_ef=int(DEFAULT_MIXED_GT_EF),
        internal_lid_k=int(internal_lid_k),
        k=int(k),
        easy_threshold_scale=float(easy_threshold_scale),
        mid_threshold_scale=float(mid_threshold_scale),
        super_threshold_scale=float(super_threshold_scale),
        lid_source_graph="original",
        mixed_threshold_mode=str(mixed_threshold_mode),
        mixed_bucket_count=int(mixed_bucket_count),
        classify_start=int(classify_start),
        classify_end=int(classify_end),
        cfr_ema_decay=float(cfr_ema_decay),
        paper_floor_pair_gap=int(pair_gap),
        lid_sampling_mode="sampled",
        lid_sample_fraction=FIXED_CALIBRATION_SAMPLE_FRACTION,
        lid_min_sample_size=FIXED_CALIBRATION_LID_POOL_SIZE,
        lid_sample_seed=int(lid_sample_seed),
        cache_context=cache_context,
    )

    rebuilt_due_to_mismatch = False
    if cache_path.exists():
        cache_load_start = time.perf_counter()
        try:
            tau_by_ef, policy = mixed_runtime.load_mixed_calibration_cache(
                cache_path=cache_path,
                cache_settings=cache_settings,
            )
        except mixed_runtime.MixedCalibrationCacheSettingsMismatchError as exc:
            rebuilt_due_to_mismatch = True
            if emit is not None:
                emit(f"[MIXED_CACHE] rebuild_due_to_settings_mismatch reason={exc}")
            else:
                print(f"[MIXED_CACHE] rebuild_due_to_settings_mismatch reason={exc}", flush=True)
        else:
            add_elapsed(measurements, "calibration_cache_load_wall_s", cache_load_start)
            return tau_by_ef, policy, "loaded", measurements

    resolved_ef_values = sorted({int(ef) for ef in ef_values})
    if not resolved_ef_values:
        raise ValueError("ef_values must be non-empty.")

    step2_start = time.perf_counter()
    selected_df = mixed_runtime._select_dummy_queries(
        index=index,
        num_nodes=len(train),
        lid_df=lid_df,
        num_calibration_queries=int(num_calibration_queries),
        selection_mode=str(selection_mode),
        trim_low_percentile=float(trim_low_percentile),
        trim_high_percentile=float(trim_high_percentile),
    )
    query_ids = selected_df["query_id"].to_numpy(dtype=np.int64)
    query_vectors = np.asarray(train[query_ids], dtype=np.float32)
    measurements["calibration_query_selection_count"] = int(len(selected_df))
    add_elapsed(measurements, "step2_query_selection_wall_s", step2_start)

    pseudo_start = time.perf_counter()
    gt_neighbors = mixed_runtime._compute_gt_neighbors(
        index=index,
        train=(
            np.empty((0, int(train.shape[1])), dtype=np.float32)
            if str(DEFAULT_MIXED_GT_SOURCE) == "hnsw"
            else np.asarray(train, dtype=np.float32)
        ),
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        query_vectors=query_vectors,
        gt_source=DEFAULT_MIXED_GT_SOURCE,
        gt_ef=int(DEFAULT_MIXED_GT_EF),
        num_threads=int(num_threads),
        k=int(k),
    )
    add_elapsed(measurements, "step2_pseudo_gt_wall_s", pseudo_start)
    measurements["step2_pre_evaluation_wall_s"] = float(
        measurements["step2_query_selection_wall_s"] + measurements["step2_pseudo_gt_wall_s"]
    )

    step3_start = time.perf_counter()
    config_start = time.perf_counter()
    configs = [
        mixed_runtime._build_mixed_threshold_config(
            ef,
            mixed_threshold_mode=str(mixed_threshold_mode),
            mixed_bucket_count=int(mixed_bucket_count),
            paper_floor_pair_gap=int(pair_gap),
        )
        for ef in resolved_ef_values
    ]
    unique_recall_efs = sorted(
        {
            int(pair_target_ef)
            for config in configs
            for pair_target_ef in config.resolved_pair_target_efs
        }
    )
    measurements["step3_selection_ef_count"] = int(len(configs))
    measurements["step3_unique_recall_ef_count"] = int(len(unique_recall_efs))
    add_elapsed(measurements, "step3_config_wall_s", config_start)

    recall_start = time.perf_counter()
    recall_cache: dict[int, np.ndarray] = {}
    for pair_target_ef in unique_recall_efs:
        recall_cache[int(pair_target_ef)] = mixed_runtime._compute_recall_by_ef(
            index=index,
            query_ids=query_ids,
            query_vectors=query_vectors,
            gt_neighbors=gt_neighbors,
            ef_value=int(pair_target_ef),
            num_threads=int(num_threads),
            k=int(k),
        )
    add_elapsed(measurements, "step3_recall_curve_wall_s", recall_start)

    cfr_start = time.perf_counter()
    anchor_by_ef: dict[int, pd.DataFrame] = {}
    for config in configs:
        anchor_by_ef[int(config.selection_ef)] = mixed_runtime._extract_cfr_mean_by_query(
            index=index,
            selected_df=selected_df,
            query_vectors=query_vectors,
            query_ids=query_ids,
            selection_ef=int(config.selection_ef),
            num_threads=int(num_threads),
            k=int(k),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            cfr_ema_decay=float(cfr_ema_decay),
        )
    add_elapsed(measurements, "step3_cfr_distribution_wall_s", cfr_start)

    survival_start = time.perf_counter()
    tau_by_ef: dict[int, float] = {}
    super_gamma_by_ef: dict[int, float] = {}
    mid_gamma_by_ef: dict[int, float] = {}
    route_efs_by_ef: dict[int, tuple[int, ...]] = {}
    bucket_gamma_ratios_by_ef: dict[int, tuple[float, ...]] = {}
    nan_by_ef: dict[int, float] = {}
    for config in configs:
        anchor_df = anchor_by_ef[int(config.selection_ef)]
        usable_mask = anchor_df["usable_for_mean_window_calibration"].astype(bool).to_numpy(dtype=bool)
        anchor_cfr_values = pd.to_numeric(
            anchor_df.loc[usable_mask, "mean_smoothed_cfr_classify_window"],
            errors="coerce",
        ).to_numpy(dtype=float)
        anchor_cfr_values = anchor_cfr_values[np.isfinite(anchor_cfr_values)]
        if anchor_cfr_values.size == 0:
            raise RuntimeError(f"No usable calibration CFR values for selection_ef={int(config.selection_ef)}.")

        route_thetas: list[float] = []
        route_efs = config.resolved_route_efs
        pair_target_efs = config.resolved_pair_target_efs
        for route_index, pair_target_ef in enumerate(pair_target_efs):
            acceptable_rate = float(
                np.mean(recall_cache[int(pair_target_ef)][usable_mask] + 1e-12 >= float(acceptable_recall_threshold))
            )
            route_theta = max(
                mixed_runtime._quantile_theta(anchor_cfr_values, acceptable_rate)
                * mixed_runtime._threshold_scale_for_route_index(
                    route_index=route_index,
                    route_count=len(route_efs),
                    easy_threshold_scale=float(easy_threshold_scale),
                    mid_threshold_scale=float(mid_threshold_scale),
                    super_threshold_scale=float(super_threshold_scale),
                ),
                1e-6,
            )
            route_thetas.append(float(route_theta))

        monotone_route_thetas: list[float] = []
        running_theta = 0.0
        for route_theta in route_thetas:
            running_theta = max(float(route_theta), float(running_theta))
            monotone_route_thetas.append(float(running_theta))

        tau_value = float(monotone_route_thetas[-1])
        gamma_ratios = tuple(
            float(min(1.0, max(0.0, route_theta / max(tau_value, 1e-6))))
            for route_theta in monotone_route_thetas
        )
        tau_by_ef[int(config.selection_ef)] = tau_value
        super_gamma_by_ef[int(config.selection_ef)] = float(gamma_ratios[0]) if gamma_ratios else float("nan")
        mid_gamma_by_ef[int(config.selection_ef)] = float(gamma_ratios[1]) if len(gamma_ratios) > 1 else float("nan")
        route_efs_by_ef[int(config.selection_ef)] = tuple(int(route_ef) for route_ef in route_efs)
        bucket_gamma_ratios_by_ef[int(config.selection_ef)] = tuple(float(value) for value in gamma_ratios)
        nan_by_ef[int(config.selection_ef)] = float("nan")

    policy = mixed_runtime.MixedProjectedLocalAcceptablePolicy(
        policy_name="mixed-dynamic",
        source_label=mixed_runtime._build_source_label(
            configs,
            mixed_threshold_mode=str(mixed_threshold_mode),
            mixed_bucket_count=int(mixed_bucket_count),
            paper_floor_pair_gap=int(pair_gap),
        ),
        enabled=False,
        tmin_pops=int(tmin_pops),
        accepted_threshold=float(acceptable_recall_threshold),
        accepted_patience=0,
        super_pct_by_ef=dict(nan_by_ef),
        gamma_ratio_by_ef=dict(super_gamma_by_ef),
        mid_easy_upper_pct_by_ef=dict(nan_by_ef),
        mid_easy_upper_gamma_ratio_by_ef=dict(mid_gamma_by_ef),
        route_efs_by_ef=dict(route_efs_by_ef),
        bucket_gamma_ratios_by_ef=dict(bucket_gamma_ratios_by_ef),
    )
    add_elapsed(measurements, "step3_threshold_survival_wall_s", survival_start)
    measurements["step3_eval_wall_s"] = float(time.perf_counter() - step3_start)

    cache_save_start = time.perf_counter()
    mixed_runtime.save_mixed_calibration_cache(
        cache_path=cache_path,
        tau_by_ef=tau_by_ef,
        policy=policy,
        cache_settings=cache_settings,
    )
    add_elapsed(measurements, "calibration_cache_save_wall_s", cache_save_start)
    cache_status = "rebuilt" if rebuilt_due_to_mismatch else "saved"
    return tau_by_ef, policy, cache_status, measurements


def run_dataset(args: argparse.Namespace, dataset: str, rows: list[dict[str, Any]]) -> None:
    if BACKEND_BUILD_OR_LOAD_INDEX is None:
        raise RuntimeError("Backend not configured")
    run_root = Path(args.output_dir).expanduser().resolve() / "run"
    final_dir = Path(args.output_dir).expanduser().resolve() / "final"
    stem = dataset_stem(dataset)
    dataset_dir = run_root / stem
    dataset_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "logs" / f"{stem}__offline_cost.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        def emit(message: str) -> None:
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()

        emit(f"# started_at={now_label()}")
        emit(f"[1/3] Loading dataset: {dataset}")
        dataset_load_start = time.perf_counter()
        train, test, neighbors = load_dataset_for_offline_cost(str(Path(args.base_path).expanduser().resolve()), dataset)
        del test
        train = train.astype("float32", copy=False)
        groundtruth_k = int(neighbors.shape[1])
        dataset_load_wall_s = time.perf_counter() - dataset_load_start
        emit(f"[1/3] train={train.shape} neighbors_k={groundtruth_k} load_wall_s={dataset_load_wall_s:.3f}")

        emit(
            f"[2/3] Loading original {BACKEND_LABEL} index "
            f"M={int(args.param_m)} efC={int(args.ef_construction)} offline_threads={int(args.offline_num_threads)}"
        )
        index_load_start = time.perf_counter()
        index_dir = args.faiss_index_root if BACKEND == "faiss" else args.index_dir
        index, space, index_dataset_name = BACKEND_BUILD_OR_LOAD_INDEX(
            train=train,
            dataset_name=dataset,
            index_dir=str(Path(index_dir).expanduser().resolve()),
            param_m=int(args.param_m),
            ef_construction=int(args.ef_construction),
            num_threads=int(args.offline_num_threads),
        )
        if hasattr(train, "bind_index"):
            train.bind_index(index)
        index_load_wall_s = time.perf_counter() - index_load_start
        emit(f"[2/3] Index ready space={space} index_dataset={index_dataset_name} index_load_wall_s={index_load_wall_s:.3f}")

        easy_scale, mid_scale, super_scale = resolve_runtime_threshold_scales(
            easy_threshold_scale=None,
            mid_threshold_scale=None,
            super_threshold_scale=None,
        )
        k_ef_values = [int(ef) for ef in args.ef_sweep if int(ef) >= int(args.k)]
        if not k_ef_values:
            raise ValueError(f"No efSearch values >= k={int(args.k)}")

        for repeat in range(1, int(args.repeats) + 1):
            emit(f"[3/3] repeat={repeat}/{int(args.repeats)} building sampled calibration LID pool")
            lid_pool_start = time.perf_counter()
            calibration_lid_df = compute_fixed_calibration_lid_pool(
                index,
                internal_lid_k=int(args.internal_lid_k),
                num_nodes=len(train),
                lid_sample_seed=int(args.calibration_sample_seed),
                num_threads=int(args.offline_num_threads),
                dataset_name=dataset,
            )
            probe_wall_s = time.perf_counter() - lid_pool_start
            emit(f"[3/3] repeat={repeat} LID pool count={len(calibration_lid_df)} probe_wall_s={probe_wall_s:.3f}")

            cache_path = (
                dataset_dir
                / f"repeat{repeat}"
                / f"{stem}__k{int(args.k)}__{args.mixed_threshold_mode}_b{int(args.mixed_bucket_count)}"
                / f"{stem}__k{int(args.k)}__mixed_original_M{int(args.param_m)}_efC{int(args.ef_construction)}.json"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_existed_before_unlink = cache_path.exists()
            if cache_path.exists() and not bool(args.reuse_cache):
                cache_path.unlink()
                emit(f"[3/3] repeat={repeat} removed stale calibration cache: {cache_path}")
            cache_exists_before = cache_path.exists()

            threshold_start = time.perf_counter()
            tau_by_ef, policy, cache_status, step_measurements = resolve_mixed_policy_with_step_timings(
                index=index,
                train=train,
                dataset_name=dataset,
                ef_values=k_ef_values,
                k=int(args.k),
                acceptable_recall_threshold=1.0,
                tmin_pops=int(args.tmin_pops),
                num_calibration_queries=int(args.num_calibration_queries),
                selection_mode="quantile",
                trim_low_percentile=float(args.trim_low_percentile),
                trim_high_percentile=float(args.trim_high_percentile),
                internal_lid_k=int(args.internal_lid_k),
                num_threads=int(args.offline_num_threads),
                easy_threshold_scale=easy_scale,
                mid_threshold_scale=mid_scale,
                super_threshold_scale=super_scale,
                lid_df=calibration_lid_df,
                lid_sample_seed=int(args.calibration_sample_seed),
                cache_path=cache_path,
                mixed_threshold_mode=str(args.mixed_threshold_mode),
                mixed_bucket_count=int(args.mixed_bucket_count),
                classify_start=int(args.classify_start),
                classify_end=int(args.classify_end),
                cfr_ema_decay=float(args.cfr_ema_decay),
                pair_gap=int(args.pair_gap),
                cache_context=mixed_cache_context(args),
                emit=emit,
            )
            del tau_by_ef
            sweep_wall_s = time.perf_counter() - threshold_start
            step1_wall_s = float(probe_wall_s)
            step2_selection_wall_s = float(step_measurements.get("step2_query_selection_wall_s", 0.0))
            step2_pseudo_gt_wall_s = float(step_measurements.get("step2_pseudo_gt_wall_s", 0.0))
            step2_wall_s = float(
                step_measurements.get("step2_pre_evaluation_wall_s", step2_selection_wall_s + step2_pseudo_gt_wall_s)
            )
            step3_wall_s = float(step_measurements.get("step3_eval_wall_s", 0.0))
            cache_load_wall_s = float(step_measurements.get("calibration_cache_load_wall_s", 0.0))
            cache_save_wall_s = float(step_measurements.get("calibration_cache_save_wall_s", 0.0))
            calibration_unattributed_wall_s = float(
                sweep_wall_s - step2_wall_s - step3_wall_s - cache_load_wall_s - cache_save_wall_s
            )
            offline_algorithm_wall_s = float(step1_wall_s + step2_wall_s + step3_wall_s)
            total_wall_s = float(probe_wall_s) + float(sweep_wall_s)
            emit(
                f"[3/3] repeat={repeat} done cache={cache_status or 'disabled'} "
                f"samp={step1_wall_s:.3f} select={step2_wall_s:.3f} eval={step3_wall_s:.3f} total={total_wall_s:.3f}"
            )

            row = {
                "backend": BACKEND,
                "simd": "on",
                "repeat": int(repeat),
                "dataset": stem,
                "dataset_file": dataset,
                "points": int(len(train)),
                "dimensions": int(train.shape[1]),
                "k": int(args.k),
                "groundtruth_k": groundtruth_k,
                "offline_num_threads": int(args.offline_num_threads),
                "M": int(args.param_m),
                "efConstruction": int(args.ef_construction),
                "num_calibration_queries": int(args.num_calibration_queries),
                "calibration_lid_pool_count": int(len(calibration_lid_df)),
                "calibration_lid_pool_wall_s": step1_wall_s,
                "step1_lid_sampling_wall_s": step1_wall_s,
                "step2_query_selection_wall_s": step2_selection_wall_s,
                "step2_pseudo_gt_wall_s": step2_pseudo_gt_wall_s,
                "step2_pre_evaluation_wall_s": step2_wall_s,
                "step3_eval_wall_s": step3_wall_s,
                "step3_config_wall_s": float(step_measurements.get("step3_config_wall_s", 0.0)),
                "step3_recall_curve_wall_s": float(step_measurements.get("step3_recall_curve_wall_s", 0.0)),
                "step3_cfr_distribution_wall_s": float(step_measurements.get("step3_cfr_distribution_wall_s", 0.0)),
                "step3_threshold_survival_wall_s": float(step_measurements.get("step3_threshold_survival_wall_s", 0.0)),
                "offline_algorithm_wall_s": offline_algorithm_wall_s,
                "threshold_calibration_wall_s": float(sweep_wall_s),
                "offline_calibration_wall_s": float(total_wall_s),
                "calibration_cache_load_wall_s": cache_load_wall_s,
                "calibration_cache_save_wall_s": cache_save_wall_s,
                "calibration_unattributed_wall_s": calibration_unattributed_wall_s,
                "calibration_query_selection_count": int(step_measurements.get("calibration_query_selection_count", 0)),
                "step3_selection_ef_count": int(step_measurements.get("step3_selection_ef_count", 0)),
                "step3_unique_recall_ef_count": int(step_measurements.get("step3_unique_recall_ef_count", 0)),
                "calibration_gt_source": str(step_measurements.get("calibration_gt_source", "")),
                "calibration_gt_ef": int(step_measurements.get("calibration_gt_ef", 0)),
                "dataset_load_wall_s": float(dataset_load_wall_s),
                "index_load_wall_s": float(index_load_wall_s),
                "calibration_cache_status": str(cache_status or ""),
                "calibration_cache_exists_before": bool(cache_existed_before_unlink),
                "mixed_threshold_mode": str(args.mixed_threshold_mode),
                "mixed_bucket_count": int(args.mixed_bucket_count),
                "pair_gap": int(args.pair_gap),
                "classify_start": int(args.classify_start),
                "classify_end": int(args.classify_end),
                "cfr_ema_decay": float(args.cfr_ema_decay),
                "cache_path": str(cache_path),
                "policy_source_label": str(getattr(policy, "source_label", "")),
                "log_path": str(log_path),
            }
            rows.append(row)
            append_csv(final_dir / f"{BACKEND}_offline_cost_raw_partial.csv", FIELDNAMES, row)
            write_csv(final_dir / f"{BACKEND}_offline_cost_raw.csv", FIELDNAMES, rows)
            write_csv(final_dir / f"{BACKEND}_offline_cost_median.csv", MEDIAN_FIELDNAMES, median_rows(rows))
            del calibration_lid_df, policy
            gc.collect()

        del index, train, neighbors
        gc.collect()


def median_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    keys = ["backend", "dataset", "k"]
    for _, group in df.groupby(keys, sort=False):
        first = group.iloc[0]
        def med(column: str) -> float:
            return float(pd.to_numeric(group[column], errors="coerce").median())
        out.append(
            {
                "backend": str(first["backend"]),
                "simd": str(first.get("simd", "on")),
                "dataset": str(first["dataset"]),
                "dataset_file": str(first["dataset_file"]),
                "points": int(first["points"]),
                "dimensions": int(first["dimensions"]),
                "k": int(first["k"]),
                "offline_num_threads": int(first["offline_num_threads"]),
                "M": int(first["M"]),
                "efConstruction": int(first["efConstruction"]),
                "num_calibration_queries": int(first["num_calibration_queries"]),
                "repeats": int(len(group)),
                "paper_samp_s": med("step1_lid_sampling_wall_s"),
                "paper_select_s": med("step2_pre_evaluation_wall_s"),
                "paper_eval_s": med("step3_eval_wall_s"),
                "paper_total_s": med("offline_calibration_wall_s"),
                "step1_lid_sampling_wall_s": med("step1_lid_sampling_wall_s"),
                "step2_pre_evaluation_wall_s": med("step2_pre_evaluation_wall_s"),
                "step3_eval_wall_s": med("step3_eval_wall_s"),
                "offline_calibration_wall_s": med("offline_calibration_wall_s"),
                "step2_query_selection_wall_s": med("step2_query_selection_wall_s"),
                "step2_pseudo_gt_wall_s": med("step2_pseudo_gt_wall_s"),
                "step3_recall_curve_wall_s": med("step3_recall_curve_wall_s"),
                "step3_cfr_distribution_wall_s": med("step3_cfr_distribution_wall_s"),
                "step3_threshold_survival_wall_s": med("step3_threshold_survival_wall_s"),
                "threshold_calibration_wall_s": med("threshold_calibration_wall_s"),
            }
        )
    return out


def write_manifest(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    payload = {
        "backend": str(args.backend),
        "simd": "on",
        "faiss_opt_level": os.environ.get("FAISS_OPT_LEVEL", "AVX512"),
        "datasets": list(args.datasets),
        "settings": {
            "k": int(args.k),
            "ef_values": [int(ef) for ef in args.ef_sweep],
            "offline_num_threads": int(args.offline_num_threads),
            "M": int(args.param_m),
            "efConstruction": int(args.ef_construction),
            "num_calibration_queries": int(args.num_calibration_queries),
            "internal_lid_k": int(args.internal_lid_k),
            "calibration_sample_seed": int(args.calibration_sample_seed),
            "calibration_gt_source": DEFAULT_MIXED_GT_SOURCE,
            "calibration_gt_ef": DEFAULT_MIXED_GT_EF,
            "mixed_threshold_mode": str(args.mixed_threshold_mode),
            "mixed_bucket_count": int(args.mixed_bucket_count),
            "pair_gap": int(args.pair_gap),
            "classify_start": int(args.classify_start),
            "classify_end": int(args.classify_end),
            "cfr_ema_decay": float(args.cfr_ema_decay),
            "repeats": int(args.repeats),
        },
        "paths": {
            "base_path": str(Path(args.base_path).expanduser().resolve()),
            "index_dir": str(Path(args.index_dir).expanduser().resolve()),
            "faiss_index_root": str(Path(args.faiss_index_root).expanduser().resolve()),
            "faiss_python_path": str(Path(args.faiss_python_path).expanduser().resolve()),
            "hnswlib_extension_root": str(Path(args.hnswlib_extension_root).expanduser().resolve()),
        },
        "started_at": now_label(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("faiss", "hnswlib"), required=True)
    parser.add_argument("--datasets", type=parse_dataset_list, default=parse_dataset_list(",".join(DATASETS)))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_HNSW_INDEX_DIR)
    parser.add_argument("--faiss-index-root", type=Path, default=DEFAULT_FAISS_INDEX_ROOT)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--hnswlib-extension-root", type=Path, default=DEFAULT_HNSWLIB_EXTENSION_ROOT)
    parser.add_argument("--allow-system-faiss", action="store_true")
    parser.add_argument("--no-conda-reexec", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--ef-sweep", type=parse_ef_sweep, default=DEFAULT_EF_SWEEP_VALUES)
    parser.add_argument("--offline-num-threads", type=int, default=OFFLINE_THREADS)
    parser.add_argument("--param-m", type=int, default=PARAM_M)
    parser.add_argument("--ef-construction", type=int, default=EF_CONSTRUCTION)
    parser.add_argument("--num-calibration-queries", type=int, default=NUM_CALIBRATION_QUERIES)
    parser.add_argument("--internal-lid-k", type=int, default=INTERNAL_LID_K)
    parser.add_argument("--calibration-sample-seed", type=int, default=DEFAULT_CALIBRATION_SAMPLE_SEED)
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--tmin-pops", type=int, default=TMIN_POPS)
    parser.add_argument("--mixed-threshold-mode", choices=("paper_floor_half",), default=MIXED_THRESHOLD_MODE)
    parser.add_argument("--mixed-bucket-count", type=int, default=MIXED_BUCKET_COUNT)
    parser.add_argument("--pair-gap", type=int, default=PAIR_GAP)
    parser.add_argument("--classify-start", type=int, default=CLASSIFY_START)
    parser.add_argument("--classify-end", type=int, default=CLASSIFY_END)
    parser.add_argument("--cfr-ema-decay", type=float, default=CFR_EMA_DECAY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Configure the backend and print resolved paths without running datasets.")
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = ROOT / f"offline_cost_main8_{args.backend}_SIMD_on_24t"
    if int(args.offline_num_threads) < 1:
        raise ValueError("--offline-num-threads must be positive")
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be positive")
    if int(args.mixed_bucket_count) < 2:
        raise ValueError("--mixed-bucket-count must be at least 2")
    if int(args.classify_end) < int(args.classify_start):
        raise ValueError("--classify-end must be >= --classify-start")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(DEFAULT_PROJECT_ROOT))
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(args.offline_num_threads))

    configure_backend(args)
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "backend": str(args.backend),
                    "simd": "on",
                    "offline_num_threads": int(args.offline_num_threads),
                    "datasets": list(args.datasets),
                    "base_path": str(Path(args.base_path).expanduser().resolve()),
                    "index_dir": str(Path(args.index_dir).expanduser().resolve()),
                    "faiss_index_root": str(Path(args.faiss_index_root).expanduser().resolve()),
                    "faiss_python_path": str(Path(args.faiss_python_path).expanduser().resolve()),
                    "hnswlib_extension_root": str(Path(args.hnswlib_extension_root).expanduser().resolve()),
                    "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    out_dir = Path(args.output_dir).expanduser().resolve()
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run" / "logs").mkdir(parents=True, exist_ok=True)
    if not bool(args.append):
        for name in (
            f"{args.backend}_offline_cost_raw.csv",
            f"{args.backend}_offline_cost_raw_partial.csv",
            f"{args.backend}_offline_cost_median.csv",
            "failures.csv",
        ):
            path = final_dir / name
            if path.exists():
                path.unlink()

    write_manifest(args)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset in args.datasets:
        try:
            run_dataset(args, dataset, rows)
        except Exception as exc:
            failure = {"backend": str(args.backend), "dataset_file": str(dataset), "error": repr(exc)}
            failures.append(failure)
            append_csv(final_dir / "failures.csv", ["backend", "dataset_file", "error"], failure)
            print(f"[ERROR] {dataset}: {exc!r}", flush=True)
            if bool(args.stop_on_error):
                break

    write_csv(final_dir / f"{args.backend}_offline_cost_raw.csv", FIELDNAMES, rows)
    write_csv(final_dir / f"{args.backend}_offline_cost_median.csv", MEDIAN_FIELDNAMES, median_rows(rows))
    if failures:
        write_csv(final_dir / "failures.csv", ["backend", "dataset_file", "error"], failures)
    print(f"[DONE] raw={final_dir / f'{args.backend}_offline_cost_raw.csv'}", flush=True)
    print(f"[DONE] median={final_dir / f'{args.backend}_offline_cost_median.csv'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
