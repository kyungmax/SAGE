#!/usr/bin/env python3
"""Main SAGE recall/QPS/latency sweep with separate offline and online threads."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
EXPERIMENTS_SCRIPT_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if parent.name == "experiments_scripts"
)
if str(EXPERIMENTS_SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_SCRIPT_ROOT))


def _find_default_project_root() -> Path:
    env_root = os.environ.get("HNSW_PLAYGROUND_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (EXP_ROOT, *EXP_ROOT.parents):
        if (candidate / "datasets").exists():
            return candidate
    return EXP_ROOT


PROJECT_ROOT = _find_default_project_root()

from common.adaptive_runtime import (  # noqa: E402
    DEFAULT_CALIBRATION_SAMPLE_SEED,
    DEFAULT_EF_SWEEP_VALUES,
    DEFAULT_MIXED_GT_EF,
    DEFAULT_MIXED_GT_SOURCE,
    dedupe_preserve_order,
    evaluate_recall_per_query,
    flush_cache,
    format_float_signature,
    format_int_signature,
    load_dataset_with_special_cases,
    parse_ef_sweep,
    resolve_runtime_threshold_scales,
    run_adaptive_query,
    validate_query_method,
)
from common.benchmark_utils import benchmark_query_batch  # noqa: E402
from common.drilldown_runtime import (  # noqa: E402
    drilldown_sweep_csv_for,
    run_drilldown_for_k,
)
from common.offline_calibration import (  # noqa: E402
    compute_fixed_calibration_lid_pool,
    resolve_mixed_policy_with_status,
)
from common.offline_recommendation import (  # noqa: E402
    compute_offline_recommended_efsearch,
    offline_curve_csv_for,
    offline_recommendation_outputs_current,
    offline_recommended_csv_for,
    write_offline_recommendation_summary,
)
from final_index_utils import (  # noqa: E402
    DEFAULT_FAISS_INDEX_ROOT,
    DEFAULT_FAISS_PYTHON_PATH,
    build_original_index,
    configure_faiss_loader,
    maybe_reexec_in_conda_env,
)


DATASETS = (
    "nytimes-256-angular.hdf5",
    "glove-100-angular.hdf5",
    "cohere-768-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
    "youtube-15M-angular.hdf5",
)
RUN_ROOT = EXP_ROOT / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1" / "run"
FINAL_DIR = EXP_ROOT / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1" / "final"
RESULT_CSV = FINAL_DIR / "main_qps_latency_sweep.csv"
RESULT_MD = FINAL_DIR / "main_qps_latency_sweep.md"


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty.")
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive.")
    return dedupe_preserve_order(values)


def safe_path_token(value: Any) -> str:
    token = str(value).strip().lower().replace(" ", "_")
    for char in "/\\:=,[](){}":
        token = token.replace(char, "_")
    token = token.replace(".", "p").replace("-", "m")
    token = "_".join(part for part in token.split("_") if part)
    return token or "default"


def ablation_signature(args: argparse.Namespace) -> str:
    alpha = safe_path_token(f"{float(args.chr_ema_decay):g}")
    return "__".join(
        [
            f"{safe_path_token(args.ablation_name)}_{safe_path_token(args.ablation_value)}",
            f"ncal{int(args.num_calibration_queries)}",
            f"cs{int(args.classify_start)}ce{int(args.classify_end)}",
            f"alpha{alpha}",
            f"g{int(args.pair_gap)}",
            f"b{int(args.mixed_bucket_count)}",
        ]
    )


def ablation_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ablation_name": str(args.ablation_name),
        "ablation_value": str(args.ablation_value),
        "ablation_signature": ablation_signature(args),
        "classify_start": int(args.classify_start),
        "classify_end": int(args.classify_end),
        "chr_ema_decay": float(args.chr_ema_decay),
        "pair_gap": int(args.pair_gap),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--base-path", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_FAISS_INDEX_ROOT),
        help="Root containing Faiss/DARTH HNSW index subdirectories.",
    )
    parser.add_argument(
        "--faiss-python-path",
        default=DEFAULT_FAISS_PYTHON_PATH,
        help=(
            "Optional path containing a built faiss Python package. When omitted, "
            "the faiss module installed in the active Python environment is used."
        ),
    )
    parser.add_argument(
        "--allow-system-faiss",
        action="store_true",
        help=(
            "When --faiss-python-path is provided, allow importing an already "
            "installed faiss module instead of requiring that exact path."
        ),
    )
    parser.add_argument(
        "--no-conda-reexec",
        action="store_true",
        help="Do not re-exec this runner with conda run -n hnsw when outside the target env.",
    )
    parser.add_argument("--run-root", default=str(RUN_ROOT))
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("10"))
    parser.add_argument("--ef-sweep", type=parse_ef_sweep, default=DEFAULT_EF_SWEEP_VALUES)
    parser.add_argument("--offline-num-threads", type=int, default=24)
    parser.add_argument("--online-num-threads", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--query-method", choices=("adaptive-light", "adaptive"), default="adaptive-light")
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--ablation-name", default="baseline")
    parser.add_argument("--ablation-value", default="default")
    parser.add_argument("--classify-start", type=int, default=4)
    parser.add_argument("--classify-end", type=int, default=16)
    parser.add_argument("--chr-ema-decay", type=float, default=0.8)
    parser.add_argument("--pair-gap", type=int, default=2)
    parser.add_argument("--internal-lid-k", type=int, default=15)
    parser.add_argument("--calibration-sample-seed", type=int, default=DEFAULT_CALIBRATION_SAMPLE_SEED)
    parser.add_argument("--trim-low-percentile", type=float, default=1.0)
    parser.add_argument("--trim-high-percentile", type=float, default=99.0)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument(
        "--mixed-threshold-mode",
        choices=("paper_floor_half",),
        default="paper_floor_half",
    )
    parser.add_argument("--mixed-bucket-count", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--skip-unavailable-k", action="store_true", default=True)
    parser.add_argument("--fail-unavailable-k", action="store_false", dest="skip_unavailable_k")
    parser.add_argument(
        "--enable-drilldown",
        action="store_true",
        help="Also run the easy/medium/hard group efSearch drilldown.",
    )
    parser.add_argument(
        "--drilldown-output-dir",
        default="",
        help="Directory for drilldown CSV/MD outputs. Defaults under --final-dir.",
    )
    parser.add_argument("--drilldown-pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--drilldown-group-def-ef", type=int, default=1024)
    parser.add_argument(
        "--drilldown-ef-sweep",
        default="",
        help="Optional efSearch list/range for drilldown. Defaults to the main ef sweep.",
    )
    args = parser.parse_args()
    (
        args.easy_threshold_scale,
        args.mid_threshold_scale,
        args.super_threshold_scale,
    ) = resolve_runtime_threshold_scales(
        easy_threshold_scale=None,
        mid_threshold_scale=None,
        super_threshold_scale=None,
    )
    args.datasets = tuple(part.strip() for part in str(args.datasets).split(",") if part.strip())
    if not args.datasets:
        raise ValueError("--datasets must not be empty.")
    if int(args.offline_num_threads) < 1:
        raise ValueError("--offline-num-threads must be positive.")
    if int(args.online_num_threads) < 1:
        raise ValueError("--online-num-threads must be positive.")
    if int(args.measured_runs) < 1:
        raise ValueError("--measured-runs must be positive.")
    if int(args.warmup_runs) < 0:
        raise ValueError("--warmup-runs must be >= 0.")
    if int(args.num_calibration_queries) < 1:
        raise ValueError("--num-calibration-queries must be positive.")
    if int(args.classify_start) < 0:
        raise ValueError("--classify-start must be >= 0.")
    if int(args.classify_end) < 1:
        raise ValueError("--classify-end must be >= 1.")
    if int(args.classify_end) < int(args.classify_start):
        raise ValueError("--classify-end must be >= --classify-start.")
    if not np.isfinite(float(args.chr_ema_decay)) or not (0.0 <= float(args.chr_ema_decay) <= 1.0):
        raise ValueError("--chr-ema-decay must lie in [0, 1].")
    if int(args.pair_gap) < 1:
        raise ValueError("--pair-gap must be >= 1.")
    if args.drilldown_ef_sweep:
        args.drilldown_ef_sweep = parse_ef_sweep(str(args.drilldown_ef_sweep))
    else:
        args.drilldown_ef_sweep = None
    if int(args.drilldown_pseudo_gt_ef) < 1:
        raise ValueError("--drilldown-pseudo-gt-ef must be positive.")
    if int(args.drilldown_group_def_ef) < 1:
        raise ValueError("--drilldown-group-def-ef must be positive.")
    if int(args.drilldown_pseudo_gt_ef) < int(args.drilldown_group_def_ef):
        raise ValueError("--drilldown-pseudo-gt-ef must be >= --drilldown-group-def-ef.")
    if int(args.mixed_bucket_count) < 2 and str(args.mixed_threshold_mode) == "paper_floor_half":
        raise ValueError("--mixed-bucket-count must be at least 2 for paper_floor_half.")
    return args


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = "" if pd.isna(value) else f"{float(value):.3f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def latency_metrics(benchmark: dict[str, Any], query_count: int) -> dict[str, float]:
    durations = np.asarray(benchmark["durations_s"], dtype=np.float64)
    query_count = int(query_count)
    mean_ms = float(benchmark["avg_duration_s"] * 1000.0)
    return {
        "query_count": query_count,
        "batch_latency_mean_ms": mean_ms,
        "batch_latency_p50_ms": float(np.percentile(durations, 50) * 1000.0),
        "batch_latency_p95_ms": float(np.percentile(durations, 95) * 1000.0),
        "batch_latency_min_ms": float(benchmark["min_duration_s"] * 1000.0),
        "batch_latency_max_ms": float(benchmark["max_duration_s"] * 1000.0),
        "latency_per_query_mean_ms": mean_ms / float(query_count),
    }


def benchmark_baseline_k(
    *,
    index,
    test: np.ndarray,
    neighbors: np.ndarray,
    ef: int,
    k: int,
    warmup_runs: int,
    measured_runs: int,
    num_threads: int,
) -> dict[str, float]:
    def run_once():
        index.set_ef(int(ef))
        return index.knn_query(test, k=int(k), num_threads=int(num_threads))

    flush_cache()
    benchmark = benchmark_query_batch(
        run_once=run_once,
        query_count=len(test),
        warmup_runs=int(warmup_runs),
        measured_runs=int(measured_runs),
    )
    labels, _ = benchmark["last_output"]
    recalls = evaluate_recall_per_query(labels, neighbors, int(k))
    metrics = {
        "recall": float(np.mean(recalls)),
        "qps": float(benchmark["qps"]),
    }
    metrics.update(latency_metrics(benchmark, query_count=len(test)))
    return metrics


def benchmark_ours_k(
    *,
    index,
    test: np.ndarray,
    neighbors: np.ndarray,
    ef: int,
    k: int,
    query_method: str,
    tau: float,
    super_gamma: float,
    mid_gamma: float,
    tmin_pops: int,
    mixed_threshold_mode: str,
    paper_bucket_count: int,
    paper_bucket_gamma_ratios: tuple[float, ...],
    classify_start: int,
    classify_end: int,
    chr_ema_decay: float,
    warmup_runs: int,
    measured_runs: int,
    num_threads: int,
) -> dict[str, float]:
    def run_once():
        return run_adaptive_query(
            index=index,
            test=test,
            k_search=int(k),
            ef=int(ef),
            query_method=str(query_method),
            enable_stop=True,
            early_stop_ratio=float(tau),
            super_easy_gamma_ratio=float(super_gamma),
            mid_easy_upper_gamma_ratio=float(mid_gamma),
            tmin_pops=int(tmin_pops),
            mixed_threshold_mode=str(mixed_threshold_mode),
            paper_bucket_count=int(paper_bucket_count),
            paper_bucket_gamma_ratios=paper_bucket_gamma_ratios,
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
            num_threads=int(num_threads),
        )

    flush_cache()
    benchmark = benchmark_query_batch(
        run_once=run_once,
        query_count=len(test),
        warmup_runs=int(warmup_runs),
        measured_runs=int(measured_runs),
    )
    adaptive_output = benchmark["last_output"]
    if str(query_method) == "adaptive-light":
        labels, dists = adaptive_output[:2]
        reduced_steps = None
        stop_count = np.nan
    else:
        labels, dists, reduced_steps, stop_count = adaptive_output
    recalls = evaluate_recall_per_query(labels, neighbors, int(k))
    metrics = {
        "recall": float(np.mean(recalls)),
        "qps": float(benchmark["qps"]),
        "adaptive_max_dist_mean": float(np.mean(np.max(dists, axis=1))),
        "stop_count": stop_count,
        "reduced_steps_mean": np.nan if reduced_steps is None else float(np.mean(reduced_steps)),
        "reduced_steps_max": np.nan if reduced_steps is None else int(np.max(reduced_steps)),
    }
    metrics.update(latency_metrics(benchmark, query_count=len(test)))
    return metrics



def write_summary(
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
    *,
    final_dir: Path,
    result_csv: Path,
    result_md: Path,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["dataset", "k", "ef", "method"]).reset_index(drop=True)
    result.to_csv(result_csv, index=False)

    with result_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Main SAGE Recall/QPS/Latency Sweep\n\n")
        if result.empty:
            setup_m = setup_efc = setup_ncal = setup_offline = setup_online = "?"
        else:
            first = result.iloc[0]
            setup_m = int(first["M"])
            setup_efc = int(first["efConstruction"])
            setup_ncal = int(first["num_calibration_queries"])
            setup_offline = int(first["offline_num_threads"])
            setup_online = int(first["online_num_threads"])
        handle.write(
            f"Setup: vanilla HNSW vs ours, M={setup_m}, efConstruction={setup_efc}, "
            f"calibration n={setup_ncal}, offline threads={setup_offline}, "
            f"online search threads={setup_online}.\n\n"
        )
        if result.empty:
            handle.write("No completed result rows yet.\n")
        else:
            ours = result[result["method"] == "Ours"].copy()
            handle.write(f"- Result rows: {len(result)}.\n")
            handle.write(f"- Completed ours cells: {len(ours)}.\n")
            handle.write(f"- Datasets: {', '.join(sorted(result['dataset'].unique()))}.\n")
            handle.write(f"- efSearch values completed: {', '.join(str(int(v)) for v in sorted(result['ef'].unique()))}.\n\n")
            if not ours.empty:
                latest = (
                    ours.sort_values(["dataset", "k", "ef"])
                    .groupby(["dataset", "k"], as_index=False)
                    .tail(1)
                    .sort_values(["dataset", "k"])
                )
                handle.write("## Highest ef per dataset/k\n\n")
                handle.write(
                    markdown_table(
                        latest[
                            [
                                "dataset",
                                "k",
                                "ef",
                                "recall",
                                "recall_loss_vs_vanilla_pp",
                                "qps",
                                "qps_gain_vs_vanilla_pct",
                                "batch_latency_mean_ms",
                                "offline_calibration_wall_s",
                            ]
                        ]
                    )
                )
                handle.write("\n")
        if skips:
            skip_df = pd.DataFrame(skips)
            handle.write("\n## Skips\n\n")
            handle.write(markdown_table(skip_df))
            handle.write("\n")


def dataset_stem(dataset: str) -> str:
    return Path(dataset).stem


def result_csv_for(run_root: Path, dataset: str, k: int) -> Path:
    return run_root / dataset_stem(dataset) / f"{dataset_stem(dataset)}__k{int(k)}__main_qps_latency.csv"


def append_status(status_path: Path, fields: list[Any]) -> None:
    with status_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(str(field) for field in fields) + "\n")


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    args = parse_args()
    maybe_reexec_in_conda_env(
        no_conda_reexec=bool(args.no_conda_reexec),
        argv=sys.argv[1:],
        script_path=SCRIPT_PATH,
    )
    configure_faiss_loader(
        python_path=args.faiss_python_path,
        index_root=args.index_dir,
        allow_system_faiss=bool(args.allow_system_faiss),
    )
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path(args.final_dir).expanduser().resolve()
    summary_csv = final_dir / RESULT_CSV.name
    summary_md = final_dir / RESULT_MD.name
    final_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    offline_curve_rows: list[dict[str, Any]] = []
    offline_recommended_rows: list[dict[str, Any]] = []
    offline_curve_summary_csv = final_dir / "offline_predicted_recall_curve.csv"
    offline_recommended_summary_csv = final_dir / "offline_recommended_efsearch.csv"
    if summary_csv.exists():
        rows.extend(pd.read_csv(summary_csv).to_dict("records"))
    if offline_curve_summary_csv.exists():
        offline_curve_rows.extend(pd.read_csv(offline_curve_summary_csv).to_dict("records"))
    if offline_recommended_summary_csv.exists():
        offline_recommended_rows.extend(pd.read_csv(offline_recommended_summary_csv).to_dict("records"))

    status_tsv = run_root / "status.tsv"
    if not status_tsv.exists():
        append_status(
            status_tsv,
            ["timestamp", "dataset", "k", "status", "result_csv", "log_path"],
        )
    main_log = run_root / "main_qps_latency_sweep.log"
    with main_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# Main sweep started_at={now_label()}\n")
        handle.write(f"# run_root={run_root}\n")
        handle.write(f"# datasets={args.datasets}\n")
        handle.write(f"# k_values={args.k_values}\n")
        handle.write(f"# ef_sweep={args.ef_sweep}\n")
        handle.write(
            f"# offline_num_threads={int(args.offline_num_threads)} "
            f"online_num_threads={int(args.online_num_threads)} "
            f"measured_runs={int(args.measured_runs)}\n\n"
        )

    completed = {
        (str(row.get("dataset_file")), int(row.get("k")), str(row.get("method")), int(row.get("ef")))
        for row in rows
        if "dataset_file" in row and "k" in row and "method" in row and "ef" in row
    }

    for dataset in args.datasets:
        stem = dataset_stem(dataset)
        dataset_dir = run_root / stem
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_log = log_dir / f"{stem}__main_qps_latency.log"
        append_status(
            status_tsv,
            [now_label(), dataset, "*", "dataset_start", "", dataset_log],
        )

        with dataset_log.open("a", encoding="utf-8", newline="\n") as log:
            def emit(message: str) -> None:
                print(message)
                log.write(message + "\n")
                log.flush()

            dataset_load_start = time.perf_counter()
            emit(f"[1/5] Loading dataset: {dataset}")
            train, test, neighbors = load_dataset_with_special_cases(
                str(Path(args.base_path).expanduser().resolve()),
                dataset,
            )
            train = train.astype("float32")
            test = test.astype("float32")
            gt_k = int(neighbors.shape[1])
            dataset_load_wall_s = time.perf_counter() - dataset_load_start
            emit(
                f"[1/5] shapes train={train.shape} test={test.shape} "
                f"neighbors={neighbors.shape} load_wall_s={dataset_load_wall_s:.3f}"
            )

            index_load_start = time.perf_counter()
            emit(
                "[2/5] Loading original index "
                f"M={int(args.param_m)} efC={int(args.ef_construction)} "
                f"offline_threads={int(args.offline_num_threads)}"
            )
            index, _space, _index_dataset_name = build_original_index(
                train=train,
                dataset_name=dataset,
                index_dir=str(Path(args.index_dir).expanduser().resolve()),
                param_m=int(args.param_m),
                ef_construction=int(args.ef_construction),
                num_threads=int(args.offline_num_threads),
            )
            index_load_wall_s = time.perf_counter() - index_load_start
            emit(f"[2/5] Index ready index_load_wall_s={index_load_wall_s:.3f}")

            lid_pool_start = time.perf_counter()
            emit("[2/5] Building sampled calibration LID pool")
            calibration_lid_df = compute_fixed_calibration_lid_pool(
                index,
                internal_lid_k=int(args.internal_lid_k),
                num_nodes=len(train),
                lid_sample_seed=int(args.calibration_sample_seed),
                num_threads=int(args.offline_num_threads),
                dataset_name=dataset,
            )
            calibration_lid_pool_wall_s = time.perf_counter() - lid_pool_start
            emit(
                f"[2/5] Calibration LID pool count={len(calibration_lid_df)} "
                f"wall_s={calibration_lid_pool_wall_s:.3f}"
            )

            for k in args.k_values:
                result_csv = result_csv_for(run_root, dataset, int(k))
                if int(k) > gt_k:
                    message = f"groundtruth_k={gt_k} < requested_k={int(k)}"
                    if not args.skip_unavailable_k:
                        raise ValueError(f"{dataset}: {message}")
                    emit(f"[SKIP] {dataset} k={int(k)} {message}")
                    skip_row = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "reason": message,
                    }
                    skips.append(skip_row)
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_unavailable_k", result_csv, dataset_log],
                    )
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    continue

                k_ef_values = [int(ef) for ef in args.ef_sweep if int(ef) >= int(k)]
                skipped_low_efs = [int(ef) for ef in args.ef_sweep if int(ef) < int(k)]
                if skipped_low_efs:
                    emit(
                        f"[SKIP-EF] {dataset} k={int(k)} skipping efSearch<k: "
                        f"{skipped_low_efs}"
                    )
                if not k_ef_values:
                    message = f"no efSearch values >= requested_k={int(k)}"
                    emit(f"[SKIP] {dataset} k={int(k)} {message}")
                    skip_row = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "reason": message,
                    }
                    skips.append(skip_row)
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_no_valid_ef", result_csv, dataset_log],
                    )
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    continue

                needed = {
                    (dataset, int(k), "Vanilla", int(ef)) for ef in k_ef_values
                } | {
                    (dataset, int(k), "Ours", int(ef)) for ef in k_ef_values
                }
                skip_main_sweep = bool(
                    args.skip_existing and result_csv.exists() and needed.issubset(completed)
                )
                offline_curve_csv = offline_curve_csv_for(run_root, dataset, int(k))
                offline_recommended_csv = offline_recommended_csv_for(run_root, dataset, int(k))
                offline_recommendation_complete = offline_recommendation_outputs_current(
                    offline_curve_csv,
                    offline_recommended_csv,
                )
                drilldown_csv = drilldown_sweep_csv_for(args, final_dir, dataset, int(k))
                drilldown_complete = bool(args.enable_drilldown and drilldown_csv.exists())
                if (
                    skip_main_sweep
                    and offline_recommendation_complete
                    and (not args.enable_drilldown or drilldown_complete)
                ):
                    emit(f"[SKIP] {dataset} k={int(k)} existing result={result_csv}")
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "skipped_existing", result_csv, dataset_log],
                    )
                    continue
                if skip_main_sweep:
                    pending_parts = []
                    if not offline_recommendation_complete:
                        pending_parts.append(f"offline_recommended={offline_recommended_csv}")
                    if args.enable_drilldown and not drilldown_complete:
                        pending_parts.append(f"drilldown={drilldown_csv}")
                    emit(
                        f"[SKIP-MAIN] {dataset} k={int(k)} existing main result; "
                        f"continuing for {'; '.join(pending_parts)}"
                    )

                emit(f"[3/5] Calibrating ours thresholds for k={int(k)}")
                cache_path = (
                    dataset_dir
                    / ablation_signature(args)
                    / f"{stem}__k{int(k)}__mixed_original_M{int(args.param_m)}_efC{int(args.ef_construction)}.json"
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_exists_before = cache_path.exists()
                threshold_start = time.perf_counter()
                tau_by_ef, policy, calibration_cache_status = resolve_mixed_policy_with_status(
                    index=index,
                    train=train,
                    dataset_name=dataset,
                    ef_values=k_ef_values,
                    k=int(k),
                    acceptable_recall_threshold=1.0,
                    tmin_pops=int(args.tmin_pops),
                    num_calibration_queries=int(args.num_calibration_queries),
                    selection_mode="quantile",
                    trim_low_percentile=float(args.trim_low_percentile),
                    trim_high_percentile=float(args.trim_high_percentile),
                    internal_lid_k=int(args.internal_lid_k),
                    num_threads=int(args.offline_num_threads),
                    easy_threshold_scale=float(args.easy_threshold_scale),
                    mid_threshold_scale=float(args.mid_threshold_scale),
                    super_threshold_scale=float(args.super_threshold_scale),
                    lid_df=calibration_lid_df,
                    lid_sample_seed=int(args.calibration_sample_seed),
                    cache_path=cache_path,
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    mixed_bucket_count=int(args.mixed_bucket_count),
                    ablation_name=str(args.ablation_name),
                    ablation_value=str(args.ablation_value),
                    classify_start=int(args.classify_start),
                    classify_end=int(args.classify_end),
                    chr_ema_decay=float(args.chr_ema_decay),
                    pair_gap=int(args.pair_gap),
                )
                threshold_calibration_wall_s = time.perf_counter() - threshold_start
                offline_calibration_wall_s = (
                    float(calibration_lid_pool_wall_s) + float(threshold_calibration_wall_s)
                )
                emit(
                    f"[3/5] Calibration done cache={calibration_cache_status or 'disabled'} "
                    f"threshold_wall_s={threshold_calibration_wall_s:.3f} "
                    f"offline_calibration_wall_s={offline_calibration_wall_s:.3f}"
                )

                offline_curve_df, offline_recommended_df = compute_offline_recommended_efsearch(
                    index=index,
                    train=train,
                    dataset=dataset,
                    stem=stem,
                    k=int(k),
                    gt_k=int(gt_k),
                    ef_values=k_ef_values,
                    tau_by_ef=tau_by_ef,
                    policy=policy,
                    calibration_lid_df=calibration_lid_df,
                    num_calibration_queries=int(args.num_calibration_queries),
                    trim_low_percentile=float(args.trim_low_percentile),
                    trim_high_percentile=float(args.trim_high_percentile),
                    num_threads=int(args.offline_num_threads),
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    mixed_bucket_count=int(args.mixed_bucket_count),
                    cache_path=cache_path,
                    classify_start=int(args.classify_start),
                    classify_end=int(args.classify_end),
                    chr_ema_decay=float(args.chr_ema_decay),
                )
                offline_curve_df.to_csv(offline_curve_csv, index=False)
                offline_recommended_df.to_csv(offline_recommended_csv, index=False)
                offline_curve_rows = [
                    old
                    for old in offline_curve_rows
                    if not (str(old.get("dataset_file")) == dataset and int(old.get("k")) == int(k))
                ]
                offline_curve_rows.extend(offline_curve_df.to_dict("records"))
                offline_recommended_rows = [
                    old
                    for old in offline_recommended_rows
                    if not (str(old.get("dataset_file")) == dataset and int(old.get("k")) == int(k))
                ]
                offline_recommended_rows.extend(offline_recommended_df.to_dict("records"))
                write_offline_recommendation_summary(
                    final_dir=final_dir,
                    curve_rows=offline_curve_rows,
                    recommended_rows=offline_recommended_rows,
                )
                offline_rec = offline_recommended_df.iloc[0]
                emit(
                    f"[3/5] Offline recommended efSearch={int(offline_rec['recommended_ef'])} "
                    f"pred_recall={float(offline_rec['offline_predicted_recall']):.5f} "
                    f"remaining_gain={float(offline_rec['remaining_cumulative_recall_gain']):.6f} "
                    f"curve={offline_curve_csv}"
                )
                append_status(
                    status_tsv,
                    [now_label(), dataset, int(k), "offline_recommended", offline_recommended_csv, dataset_log],
                )

                first_ef = int(k_ef_values[0])
                first_bucket_gammas = tuple(
                    float(value)
                    for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(first_ef, ())
                )
                first_paper_bucket_count = (
                    int(args.mixed_bucket_count)
                    if str(args.mixed_threshold_mode) == "paper_floor_half"
                    else 0
                )
                validate_query_method(
                    index,
                    query_method=str(args.query_method),
                    sample_query=test[:1],
                    enable_stop=True,
                    early_stop_ratio=float(tau_by_ef[first_ef]),
                    super_easy_gamma_ratio=float(policy.gamma_ratio_by_ef.get(first_ef, float("nan"))),
                    mid_easy_upper_gamma_ratio=float(
                        policy.mid_easy_upper_gamma_ratio_by_ef.get(first_ef, float("nan"))
                    ),
                    tmin_pops=int(args.tmin_pops),
                    mixed_threshold_mode=str(args.mixed_threshold_mode),
                    paper_bucket_count=first_paper_bucket_count,
                    paper_bucket_gamma_ratios=first_bucket_gammas,
                    classify_start=int(args.classify_start),
                    classify_end=int(args.classify_end),
                    chr_ema_decay=float(args.chr_ema_decay),
                    num_threads=int(args.online_num_threads),
                )

                main_ef_values = [] if skip_main_sweep else k_ef_values
                if skip_main_sweep:
                    emit(
                        f"[4/5] Main efSearch sweep already complete for k={int(k)}; "
                        "skipping main timing rows"
                    )
                else:
                    emit(
                        f"[4/5] Running efSearch sweep for k={int(k)} "
                        f"online_threads={int(args.online_num_threads)}"
                    )
                k_rows: list[dict[str, Any]] = []
                for ef in main_ef_values:
                    ef = int(ef)
                    vanilla_metrics = benchmark_baseline_k(
                        index=index,
                        test=test,
                        neighbors=neighbors,
                        ef=ef,
                        k=int(k),
                        warmup_runs=int(args.warmup_runs),
                        measured_runs=int(args.measured_runs),
                        num_threads=int(args.online_num_threads),
                    )
                    route_efs = tuple(
                        int(value)
                        for value in getattr(policy, "route_efs_by_ef", {}).get(ef, ())
                    )
                    bucket_gammas = tuple(
                        float(value)
                        for value in getattr(policy, "bucket_gamma_ratios_by_ef", {}).get(ef, ())
                    )
                    paper_bucket_count = (
                        int(args.mixed_bucket_count)
                        if str(args.mixed_threshold_mode) == "paper_floor_half"
                        else 0
                    )
                    ours_metrics = benchmark_ours_k(
                        index=index,
                        test=test,
                        neighbors=neighbors,
                        ef=ef,
                        k=int(k),
                        query_method=str(args.query_method),
                        tau=float(tau_by_ef[ef]),
                        super_gamma=float(policy.gamma_ratio_by_ef.get(ef, float("nan"))),
                        mid_gamma=float(policy.mid_easy_upper_gamma_ratio_by_ef.get(ef, float("nan"))),
                        tmin_pops=int(args.tmin_pops),
                        mixed_threshold_mode=str(args.mixed_threshold_mode),
                        paper_bucket_count=paper_bucket_count,
                        paper_bucket_gamma_ratios=bucket_gammas,
                        classify_start=int(args.classify_start),
                        classify_end=int(args.classify_end),
                        chr_ema_decay=float(args.chr_ema_decay),
                        warmup_runs=int(args.warmup_runs),
                        measured_runs=int(args.measured_runs),
                        num_threads=int(args.online_num_threads),
                    )
                    recall_loss = float(vanilla_metrics["recall"] - ours_metrics["recall"])
                    qps_gain_pct = (
                        float((ours_metrics["qps"] / vanilla_metrics["qps"] - 1.0) * 100.0)
                        if float(vanilla_metrics["qps"]) > 0.0
                        else float("nan")
                    )
                    base_common = {
                        "dataset": stem,
                        "dataset_file": dataset,
                        "k": int(k),
                        "groundtruth_k": gt_k,
                        "ef": ef,
                        "offline_num_threads": int(args.offline_num_threads),
                        "online_num_threads": int(args.online_num_threads),
                        "num_threads": int(args.online_num_threads),
                        "warmup_runs": int(args.warmup_runs),
                        "measured_runs": int(args.measured_runs),
                        "M": int(args.param_m),
                        "efConstruction": int(args.ef_construction),
                        "query_method": str(args.query_method),
                        "num_calibration_queries": int(args.num_calibration_queries),
                        "calibration_lid_pool_count": int(len(calibration_lid_df)),
                        "calibration_lid_pool_wall_s": float(calibration_lid_pool_wall_s),
                        "threshold_calibration_wall_s": float(threshold_calibration_wall_s),
                        "offline_calibration_wall_s": float(offline_calibration_wall_s),
                        "calibration_cache_status": str(calibration_cache_status or ""),
                        "calibration_cache_exists_before": bool(cache_exists_before),
                        "dataset_load_wall_s": float(dataset_load_wall_s),
                        "index_load_wall_s": float(index_load_wall_s),
                        "mixed_threshold_mode": str(args.mixed_threshold_mode),
                        "mixed_bucket_count": (
                            int(args.mixed_bucket_count)
                            if str(args.mixed_threshold_mode) == "paper_floor_half"
                            else np.nan
                        ),
                        "route_signature": (
                            format_int_signature(route_efs + (ef,)) if route_efs else ""
                        ),
                        "bucket_gamma_signature": (
                            format_float_signature(bucket_gammas) if bucket_gammas else ""
                        ),
                        "early_stop_ratio": float(tau_by_ef[ef]),
                        "stop_config_source": str(policy.source_label),
                        "cache_path": str(cache_path),
                    }
                    base_common.update(ablation_metadata(args))
                    vanilla_row = {
                        **base_common,
                        "method": "Vanilla",
                        "enable_stop": False,
                        "recall": float(vanilla_metrics["recall"]),
                        "qps": float(vanilla_metrics["qps"]),
                        "recall_loss_vs_vanilla_pp": 0.0,
                        "qps_gain_vs_vanilla_pct": 0.0,
                        "adaptive_max_dist_mean": np.nan,
                        "stop_count": np.nan,
                        "reduced_steps_mean": np.nan,
                        "reduced_steps_max": np.nan,
                    }
                    vanilla_row.update(
                        {
                            key: vanilla_metrics[key]
                            for key in [
                                "query_count",
                                "batch_latency_mean_ms",
                                "batch_latency_p50_ms",
                                "batch_latency_p95_ms",
                                "batch_latency_min_ms",
                                "batch_latency_max_ms",
                                "latency_per_query_mean_ms",
                            ]
                        }
                    )
                    ours_row = {
                        **base_common,
                        "method": "Ours",
                        "enable_stop": True,
                        "recall": float(ours_metrics["recall"]),
                        "qps": float(ours_metrics["qps"]),
                        "recall_loss_vs_vanilla_pp": recall_loss * 100.0,
                        "qps_gain_vs_vanilla_pct": qps_gain_pct,
                        "adaptive_max_dist_mean": float(ours_metrics["adaptive_max_dist_mean"]),
                        "stop_count": ours_metrics["stop_count"],
                        "reduced_steps_mean": ours_metrics["reduced_steps_mean"],
                        "reduced_steps_max": ours_metrics["reduced_steps_max"],
                    }
                    ours_row.update(
                        {
                            key: ours_metrics[key]
                            for key in [
                                "query_count",
                                "batch_latency_mean_ms",
                                "batch_latency_p50_ms",
                                "batch_latency_p95_ms",
                                "batch_latency_min_ms",
                                "batch_latency_max_ms",
                                "latency_per_query_mean_ms",
                            ]
                        }
                    )
                    k_rows.extend([vanilla_row, ours_row])
                    rows = [
                        old
                        for old in rows
                        if not (
                            str(old.get("dataset_file")) == dataset
                            and int(old.get("k")) == int(k)
                            and int(old.get("ef")) == ef
                        )
                    ]
                    rows.extend([vanilla_row, ours_row])
                    completed.add((dataset, int(k), "Vanilla", ef))
                    completed.add((dataset, int(k), "Ours", ef))
                    emit(
                        f"[k={int(k)} ef={ef}] "
                        f"vanilla={vanilla_metrics['recall']:.5f}/{vanilla_metrics['qps']:.1f}qps "
                        f"ours={ours_metrics['recall']:.5f}/{ours_metrics['qps']:.1f}qps "
                        f"loss={recall_loss * 100.0:+.3f}pp gain={qps_gain_pct:+.1f}% "
                        f"lat_mean_ms(v/o)={vanilla_metrics['batch_latency_mean_ms']:.1f}/"
                        f"{ours_metrics['batch_latency_mean_ms']:.1f}"
                    )

                if k_rows:
                    pd.DataFrame(k_rows).to_csv(result_csv, index=False)
                    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
                    emit(f"[5/5] Wrote {result_csv}")
                else:
                    emit(f"[5/5] Main sweep already complete; kept {result_csv}")

                if args.enable_drilldown:
                    run_drilldown_for_k(
                        args=args,
                        final_dir=final_dir,
                        index=index,
                        test=test,
                        dataset=dataset,
                        k=int(k),
                        gt_k=int(gt_k),
                        k_ef_values=k_ef_values,
                        tau_by_ef=tau_by_ef,
                        policy=policy,
                        calibration_lid_df=calibration_lid_df,
                        calibration_lid_pool_wall_s=float(calibration_lid_pool_wall_s),
                        threshold_calibration_wall_s=float(threshold_calibration_wall_s),
                        offline_calibration_wall_s=float(offline_calibration_wall_s),
                        dataset_load_wall_s=float(dataset_load_wall_s),
                        index_load_wall_s=float(index_load_wall_s),
                        cache_path=cache_path,
                        benchmark_baseline_fn=benchmark_baseline_k,
                        benchmark_ours_fn=benchmark_ours_k,
                        emit=emit,
                        backend_label="Faiss",
                    )
                    append_status(
                        status_tsv,
                        [now_label(), dataset, int(k), "drilldown_done", drilldown_csv, dataset_log],
                    )

                append_status(
                    status_tsv,
                    [now_label(), dataset, int(k), "done", result_csv, dataset_log],
                )

        append_status(
            status_tsv,
            [now_label(), dataset, "*", "dataset_done", "", dataset_log],
        )

    write_summary(rows, skips, final_dir=final_dir, result_csv=summary_csv, result_md=summary_md)
    with main_log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[COMPLETE] {now_label()} main sweep finished.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
