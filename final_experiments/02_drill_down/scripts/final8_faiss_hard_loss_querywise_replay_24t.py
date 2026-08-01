#!/usr/bin/env python3
"""FAISS SIMD-on 24-thread querywise hard-loss replay for main8 drill-down."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
DRILLDOWN_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = DRILLDOWN_ROOT.parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FINAL_FAISS_ROOT = EXPERIMENTS_ROOT / "faiss"

if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_ROOT))


def _default_project_root() -> Path:
    for key in ("HNSW_PLAYGROUND_ROOT", "SAGE_PROJECT_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return REPO_ROOT


PROJECT_ROOT = _default_project_root()
DEFAULT_DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(PROJECT_ROOT / "index"))).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get("FAISS_PYTHON_PATH", str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"))
).expanduser()
DEFAULT_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get("FAISS_INDEX_ROOT", str(DEFAULT_INDEX_DIR / "faiss_m32_efc500_main8/index")),
    )
).expanduser()

from common.adaptive_runtime import evaluate_recall_per_query, load_dataset_with_special_cases  # noqa: E402
from common.projected_local_acceptable_runtime import _extract_cfr_mean_by_query  # noqa: E402


build_original_index = None
BACKEND_LABEL = "FAISS"


DEFAULT_DATASETS = (
    "glove-100-angular.hdf5",
    "nytimes-256-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
    "cohere-768-angular.hdf5",
    "youtube-15M-angular.hdf5",
    "agnews-mxbai-1024-euclidean.hdf5",
    "landmark-nomic-768-angular.hdf5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--query-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("faiss",), default="faiss")
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--faiss-index-root", type=Path, default=DEFAULT_FAISS_INDEX_ROOT)
    parser.add_argument("--allow-system-faiss", action="store_true")
    parser.add_argument("--efs", default="1024")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--eval-gt-source", choices=("exact", "pseudo_hnsw"), default="exact")
    parser.add_argument("--pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cfr-batch-size", type=int, default=2048)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument("--classify-start", type=int, default=4)
    parser.add_argument("--classify-end", type=int, default=16)
    parser.add_argument("--cfr-ema-decay", type=float, default=0.8)
    args = parser.parse_args()
    args.datasets = tuple(part.strip() for part in str(args.datasets).split(",") if part.strip())
    args.efs = tuple(int(part.strip()) for part in str(args.efs).split(",") if part.strip())
    if not args.datasets:
        raise ValueError("--datasets must not be empty")
    if not args.efs:
        raise ValueError("--efs must not be empty")
    return args


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_backend(args: argparse.Namespace) -> None:
    global build_original_index, BACKEND_LABEL
    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(PROJECT_ROOT))
    os.environ["FAISS_PYTHON_PATH"] = str(args.faiss_python_path.expanduser().resolve())
    os.environ["FAISS_INDEX_ROOT"] = str(args.faiss_index_root.expanduser().resolve())
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    if str(FINAL_FAISS_ROOT) not in sys.path:
        sys.path.insert(0, str(FINAL_FAISS_ROOT))
    module = load_module_from_path("drilldown_faiss_final_index_utils", FINAL_FAISS_ROOT / "final_index_utils.py")
    module.configure_faiss_loader(
        python_path=args.faiss_python_path,
        index_root=args.faiss_index_root,
        allow_system_faiss=bool(args.allow_system_faiss),
    )
    build_original_index = module.build_original_index
    BACKEND_LABEL = "FAISS"


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def load_policy(run_root: Path, stem: str, ef: int, k: int, m: int, ef_construction: int) -> tuple[float, tuple[int, ...], tuple[float, ...], Path]:
    filename = f"{stem}__k{int(k)}__mixed_original_M{int(m)}_efC{int(ef_construction)}.json"
    matches = sorted((run_root / stem).glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"No policy cache under {run_root / stem} matching */{filename}")
    path = matches[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    tau = float(payload["tau_by_ef"][str(int(ef))])
    policy = payload["policy"]
    route_efs = tuple(int(value) for value in policy["route_efs_by_ef"][str(int(ef))])
    gammas = tuple(float(value) for value in policy["bucket_gamma_ratios_by_ef"][str(int(ef))])
    if len(route_efs) != len(gammas):
        raise ValueError(f"{stem} ef={ef}: route/gamma length mismatch {route_efs} vs {gammas}")
    return tau, route_efs, gammas, path


def route_for_cfr(cfr_value: float, *, tau: float, selection_ef: int, route_efs: tuple[int, ...], gammas: tuple[float, ...], k: int) -> int:
    if not np.isfinite(cfr_value):
        return int(selection_ef)
    ratio = float(cfr_value) / max(float(tau), 1e-12)
    for route_ef, gamma in zip(route_efs, gammas):
        if ratio <= float(gamma) + 1e-12:
            return max(int(k), int(route_ef))
    return int(selection_ef)


def knn_labels(index: Any, queries: np.ndarray, *, ef: int, k: int, num_threads: int, batch_size: int) -> np.ndarray:
    index.set_ef(int(ef))
    parts: list[np.ndarray] = []
    for start in range(0, len(queries), int(batch_size)):
        end = min(start + int(batch_size), len(queries))
        labels, _ = index.knn_query(queries[start:end], k=int(k), num_threads=int(num_threads))
        parts.append(np.asarray(labels, dtype=np.int64))
    return np.vstack(parts) if parts else np.empty((0, int(k)), dtype=np.int64)


def sage_labels(
    index: Any,
    queries: np.ndarray,
    *,
    ef: int,
    k: int,
    tau: float,
    gammas: tuple[float, ...],
    tmin_pops: int,
    classify_start: int,
    classify_end: int,
    cfr_ema_decay: float,
    num_threads: int,
    batch_size: int,
) -> np.ndarray:
    query_sage = getattr(index, "knn_query_sage", None)
    if query_sage is None:
        query_sage = getattr(index, "knn_query_adaptive_light_paper_bucket")
    parts: list[np.ndarray] = []
    for start in range(0, len(queries), int(batch_size)):
        end = min(start + int(batch_size), len(queries))
        labels, _ = query_sage(
            queries[start:end],
            k=int(k),
            ef_init=int(ef),
            enable_stop=True,
            early_stop_ratio=float(tau),
            tmin_pops=int(tmin_pops),
            paper_bucket_count=int(len(gammas) + 1),
            bucket_gamma_ratios=[float(value) for value in gammas],
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            cfr_ema_decay=float(cfr_ema_decay),
            num_threads=int(num_threads),
        )
        parts.append(np.asarray(labels, dtype=np.int64))
    return np.vstack(parts) if parts else np.empty((0, int(k)), dtype=np.int64)


def extract_cfr(index: Any, queries: np.ndarray, qids: np.ndarray, *, ef: int, k: int, num_threads: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    parts: list[pd.DataFrame] = []
    for start in range(0, len(queries), int(batch_size)):
        end = min(start + int(batch_size), len(queries))
        selected = pd.DataFrame(
            {
                "selection_rank": np.arange(start, end, dtype=np.int64),
                "query_id": qids[start:end].astype(np.int64),
                "lid": np.full(end - start, np.nan, dtype=np.float64),
            }
        )
        part = _extract_cfr_mean_by_query(
            index=index,
            selected_df=selected,
            query_vectors=np.asarray(queries[start:end], dtype=np.float32),
            selection_ef=int(ef),
            num_threads=int(num_threads),
            k=int(k),
        )
        parts.append(part)
    df = pd.concat(parts, ignore_index=True).sort_values("selection_rank").reset_index(drop=True)
    usable = df["usable_for_mean_window_calibration"].astype(bool).to_numpy(dtype=bool)
    cfr_values = pd.to_numeric(df["mean_smoothed_cfr_classify_window"], errors="coerce").to_numpy(dtype=np.float64)
    return cfr_values, usable


def analysis_stop_one(index: Any, query: np.ndarray, gt: np.ndarray, *, ef: int, k: int, tau: float, gammas: tuple[float, ...], args: argparse.Namespace) -> tuple[float, int, int, int]:
    out = index.knn_query_adaptive_analysis_paper_bucket(
        query.reshape(1, -1),
        k=int(k),
        ef_init=int(ef),
        ef_max=int(ef),
        tmin_pops=int(args.tmin_pops),
        enable_stop=True,
        early_stop_ratio=float(tau),
        paper_bucket_count=int(len(gammas) + 1),
        bucket_gamma_ratios=[float(value) for value in gammas],
        classify_start=int(args.classify_start),
        classify_end=int(args.classify_end),
        cfr_ema_decay=float(args.cfr_ema_decay),
        num_threads=1,
    )
    labels = np.asarray(out[0][0], dtype=np.int64)
    recall = float(np.intersect1d(labels[: int(k)], gt[: int(k)]).size / float(k))
    steps = int(np.asarray(out[2], dtype=np.int64)[0])
    stop_flag = int(np.asarray(out[3], dtype=np.int64).reshape(-1)[0])
    distance_count = int(np.asarray(out[4], dtype=np.int64).reshape(-1)[0]) if len(out) > 4 else -1
    return recall, steps, stop_flag, distance_count


def no_stop_batch(index: Any, queries: np.ndarray, gt: np.ndarray, *, ef: int, k: int, tau: float, gammas: tuple[float, ...], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for start in range(0, len(queries), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(queries))
        out = index.knn_query_adaptive_analysis_paper_bucket(
            queries[start:end],
            k=int(k),
            ef_init=int(ef),
            ef_max=int(ef),
            tmin_pops=int(args.tmin_pops),
            enable_stop=False,
            early_stop_ratio=float(tau),
            paper_bucket_count=int(len(gammas) + 1),
            bucket_gamma_ratios=[float(value) for value in gammas],
            classify_start=int(args.classify_start),
            classify_end=int(args.classify_end),
            cfr_ema_decay=float(args.cfr_ema_decay),
            num_threads=int(args.num_threads),
        )
        labels_parts.append(np.asarray(out[0], dtype=np.int64))
        step_parts.append(np.asarray(out[2], dtype=np.int64))
        if len(out) > 4:
            distance_parts.append(np.asarray(out[4], dtype=np.int64))
    labels = np.vstack(labels_parts) if labels_parts else np.empty((0, int(k)), dtype=np.int64)
    steps = np.concatenate(step_parts) if step_parts else np.empty((0,), dtype=np.int64)
    distances = np.concatenate(distance_parts) if distance_parts else np.full((len(labels),), -1, dtype=np.int64)
    recalls = evaluate_recall_per_query(labels, gt, int(k)) if len(labels) else np.empty((0,), dtype=np.float64)
    return recalls.astype(np.float64), steps.astype(np.int64), distances.astype(np.int64)


def attach_stop_vs_nostop(index: Any, rows: pd.DataFrame, queries: np.ndarray, eval_gt: np.ndarray, *, ef: int, k: int, tau: float, gammas: tuple[float, ...], args: argparse.Namespace) -> pd.DataFrame:
    if rows.empty:
        return rows
    local = rows.reset_index(drop=True).copy()
    local_queries = queries[local["local_idx"].to_numpy(dtype=np.int64)]
    local_gt = eval_gt[local["local_idx"].to_numpy(dtype=np.int64)]

    index.set_num_threads(1)
    stop_recall = np.zeros(len(local), dtype=np.float64)
    stop_steps = np.zeros(len(local), dtype=np.int64)
    stop_flags = np.zeros(len(local), dtype=np.int64)
    stop_distances = np.zeros(len(local), dtype=np.int64)
    with ThreadPoolExecutor(max_workers=min(int(args.workers), max(1, len(local)))) as executor:
        futures = {
            executor.submit(
                analysis_stop_one,
                index,
                local_queries[idx],
                local_gt[idx],
                ef=ef,
                k=k,
                tau=tau,
                gammas=gammas,
                args=args,
            ): idx
            for idx in range(len(local))
        }
        for future in as_completed(futures):
            idx = futures[future]
            recall, steps, flag, distance_count = future.result()
            stop_recall[idx] = recall
            stop_steps[idx] = steps
            stop_flags[idx] = flag
            stop_distances[idx] = distance_count

    index.set_num_threads(int(args.num_threads))
    nostop_recall, nostop_steps, nostop_distances = no_stop_batch(
        index,
        local_queries,
        local_gt,
        ef=ef,
        k=k,
        tau=tau,
        gammas=gammas,
        args=args,
    )
    local["analysis_stop_recall"] = stop_recall
    local["analysis_pop_steps"] = stop_steps
    local["analysis_distance_computations"] = stop_distances
    local["adaptive_stop_flag"] = stop_flags
    local["adaptive_stop_fired"] = stop_flags > 0
    local["nostop_recall"] = nostop_recall
    local["nostop_pop_steps"] = nostop_steps
    local["nostop_distance_computations"] = nostop_distances
    local["loss_vs_nostop"] = np.maximum(local["nostop_recall"].astype(float) - local["analysis_stop_recall"].astype(float), 0.0)
    local["saved_steps_vs_nostop"] = local["nostop_pop_steps"].astype(int) - local["analysis_pop_steps"].astype(int)
    local["saved_distance_computations_vs_nostop"] = local["nostop_distance_computations"].astype(int) - local["analysis_distance_computations"].astype(int)
    local["adaptive_stop_loss"] = local["loss_vs_nostop"].astype(float) > 1e-12
    return local


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, ef), part in per_query.groupby(["dataset", "ef"], sort=True):
        loss = part[part["hard_positive_loss"].astype(float) > 1e-12]
        fe = part[part["false_easy_loss"].astype(bool)]
        full = part[part["full_route_loss"].astype(bool)]
        stag = part[part.get("adaptive_stop_loss", False).fillna(False).astype(bool)] if "adaptive_stop_loss" in part else part.iloc[0:0]
        total_loss_sum = float(loss["hard_positive_loss"].sum())
        fe_loss_sum = float(fe["hard_positive_loss"].sum())
        full_loss_sum = float(full["hard_positive_loss"].sum())
        stag_loss_sum = float(stag["loss_vs_nostop"].sum()) if "loss_vs_nostop" in stag else 0.0
        rows.append(
            {
                "dataset": dataset,
                "ef": int(ef),
                "hard_q": int(len(part)),
                "hard_positive_loss_q": int(len(loss)),
                "false_easy_loss_q": int(len(fe)),
                "full_route_loss_q": int(len(full)),
                "adaptive_stop_loss_q": int(len(stag)),
                "hard_vanilla_recall_mean": float(part["vanilla_recall"].mean()),
                "hard_exact_ours_recall_mean": float(part["exact_ours_recall"].mean()),
                "hard_net_drop_mean": float((part["vanilla_recall"].astype(float) - part["exact_ours_recall"].astype(float)).mean()),
                "hard_positive_drop_mean": float(part["hard_positive_loss"].mean()),
                "hard_positive_drop_sum": total_loss_sum,
                "false_easy_drop_sum": fe_loss_sum,
                "full_route_drop_sum": full_loss_sum,
                "adaptive_stop_loss_vs_nostop_sum": stag_loss_sum,
                "false_easy_share_of_positive_drop": fe_loss_sum / total_loss_sum if total_loss_sum > 0 else np.nan,
                "full_route_share_of_positive_drop": full_loss_sum / total_loss_sum if total_loss_sum > 0 else np.nan,
                "adaptive_stop_vs_nostop_share_of_positive_drop": stag_loss_sum / total_loss_sum if total_loss_sum > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_dataset_ef(dataset: str, ef: int, qgroups: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stem = dataset_stem(dataset)
    started = time.time()
    tau, route_efs, gammas, policy_path = load_policy(args.run_root, stem, ef, args.k, args.param_m, args.ef_construction)
    print(f"[DATASET] {stem} ef={ef} routes={route_efs + (ef,)} policy={policy_path}", flush=True)

    train, test, neighbors = load_dataset_with_special_cases(str(args.base_path.resolve()), dataset)
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if int(neighbors.shape[1]) < int(args.k):
        raise ValueError(f"{dataset}: groundtruth_k={int(neighbors.shape[1])} < k={int(args.k)}")
    index, _space, _ = build_original_index(
        train=train,
        dataset_name=dataset,
        index_dir=str(args.index_dir.resolve()),
        param_m=int(args.param_m),
        ef_construction=int(args.ef_construction),
        num_threads=int(args.num_threads),
    )
    index.set_num_threads(int(args.num_threads))

    hard = qgroups[(qgroups["dataset_file"].astype(str) == dataset) & (qgroups["easiness_group"].astype(str) == "hard")].copy()
    if hard.empty:
        raise ValueError(f"No hard-group rows for {dataset}")
    qids = hard["qid"].to_numpy(dtype=np.int64)
    queries = np.ascontiguousarray(test[qids])

    if str(args.eval_gt_source) == "exact":
        eval_gt = np.ascontiguousarray(np.asarray(neighbors[qids, : int(args.k)], dtype=np.int64))
        eval_gt_label = "exact"
    else:
        eval_gt = knn_labels(index, queries, ef=int(args.pseudo_gt_ef), k=int(args.k), num_threads=int(args.num_threads), batch_size=int(args.batch_size))
        eval_gt_label = f"{str(args.backend)}_ef{int(args.pseudo_gt_ef)}"
    vanilla_labels = knn_labels(index, queries, ef=int(ef), k=int(args.k), num_threads=int(args.num_threads), batch_size=int(args.batch_size))
    vanilla_recall = evaluate_recall_per_query(vanilla_labels, eval_gt, int(args.k)).astype(np.float64)
    cfr_values, usable = extract_cfr(index, queries, qids, ef=int(ef), k=int(args.k), num_threads=int(args.num_threads), batch_size=int(args.cfr_batch_size))
    routed = np.asarray(
        [
            route_for_cfr(value, tau=tau, selection_ef=ef, route_efs=route_efs, gammas=gammas, k=args.k)
            for value in cfr_values
        ],
        dtype=np.int64,
    )

    recall_by_ef: dict[int, np.ndarray] = {}
    for candidate_ef in sorted(set(route_efs + (int(ef),))):
        labels = knn_labels(index, queries, ef=int(candidate_ef), k=int(args.k), num_threads=int(args.num_threads), batch_size=int(args.batch_size))
        recall_by_ef[int(candidate_ef)] = evaluate_recall_per_query(labels, eval_gt, int(args.k)).astype(np.float64)
    route_recall = np.zeros(len(queries), dtype=np.float64)
    for candidate_ef, recalls in recall_by_ef.items():
        mask = routed == int(candidate_ef)
        route_recall[mask] = recalls[mask]

    exact_labels = sage_labels(
        index,
        queries,
        ef=int(ef),
        k=int(args.k),
        tau=float(tau),
        gammas=gammas,
        tmin_pops=int(args.tmin_pops),
        classify_start=int(args.classify_start),
        classify_end=int(args.classify_end),
        cfr_ema_decay=float(args.cfr_ema_decay),
        num_threads=int(args.num_threads),
        batch_size=int(args.batch_size),
    )
    exact_recall = evaluate_recall_per_query(exact_labels, eval_gt, int(args.k)).astype(np.float64)

    out = pd.DataFrame(
        {
            "dataset": stem,
            "dataset_file": dataset,
            "qid": qids,
            "local_idx": np.arange(len(qids), dtype=np.int64),
            "ef": int(ef),
            "evaluation_gt_source": eval_gt_label,
            "tau": float(tau),
            "route_efs": "/".join(str(value) for value in route_efs + (int(ef),)),
            "bucket_gammas": "/".join(f"{value:.6f}" for value in gammas),
            "cfr": cfr_values,
            "usable_cfr": usable,
            "routed_ef": routed,
            "vanilla_recall": vanilla_recall,
            "route_recall": route_recall,
            "exact_ours_recall": exact_recall,
        }
    )
    out["hard_signed_loss"] = out["vanilla_recall"].astype(float) - out["exact_ours_recall"].astype(float)
    out["hard_positive_loss"] = np.maximum(out["hard_signed_loss"].to_numpy(dtype=np.float64), 0.0)
    out["false_easy_loss"] = (out["hard_positive_loss"].astype(float) > 1e-12) & (out["routed_ef"].astype(int) < int(ef))
    out["full_route_loss"] = (out["hard_positive_loss"].astype(float) > 1e-12) & (out["routed_ef"].astype(int) >= int(ef))

    full_loss = out[out["full_route_loss"].astype(bool)].copy()
    if not full_loss.empty and hasattr(index, "knn_query_adaptive_analysis_paper_bucket"):
        print(f"[STOP] {stem} ef={ef} full_route_loss_q={len(full_loss)}", flush=True)
        stag = attach_stop_vs_nostop(index, full_loss, queries, eval_gt, ef=int(ef), k=int(args.k), tau=float(tau), gammas=gammas, args=args)
        cols = [col for col in stag.columns if col not in out.columns or col in {"dataset", "qid", "ef"}]
        out = out.merge(stag[["dataset", "qid", "ef"] + [col for col in cols if col not in {"dataset", "qid", "ef"}]], on=["dataset", "qid", "ef"], how="left")
    else:
        out["adaptive_stop_loss"] = False

    print(f"[DONE] {stem} ef={ef} hard_q={len(out)} elapsed={time.time() - started:.1f}s", flush=True)
    return out


def main() -> int:
    args = parse_args()
    configure_backend(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qgroups = pd.read_csv(args.query_groups)
    frames: list[pd.DataFrame] = []
    for dataset in args.datasets:
        for ef in args.efs:
            frame = run_dataset_ef(dataset, int(ef), qgroups, args)
            frame.to_csv(output_dir / f"{dataset_stem(dataset)}__ef{int(ef)}__hard_loss_querywise.csv", index=False)
            frames.append(frame)
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / "hard_loss_querywise.csv", index=False)
            summarize(combined).to_csv(output_dir / "hard_loss_summary.csv", index=False)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize(combined)
    combined.to_csv(output_dir / "hard_loss_querywise.csv", index=False)
    summary.to_csv(output_dir / "hard_loss_summary.csv", index=False)
    (output_dir / "README.md").write_text(
        f"# Main8 {BACKEND_LABEL} Hard-Loss Querywise Replay\n\n"
        f"Backend: {str(args.backend)}. "
        "Hard group is read from the new exact-GT/groupDef1024 drilldown query groups. "
        "False-easy loss means exact Ours has positive hard-group recall loss and the CFR route is below the selected efSearch. "
        "Adaptive stop loss is measured on full-route loss rows by comparing adaptive stop against the same analysis path with stop disabled.\n",
        encoding="utf-8",
    )
    print(f"[RESULT] {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
