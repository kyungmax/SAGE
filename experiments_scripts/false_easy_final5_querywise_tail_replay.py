#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import false_easy_first_pass_gt_spread_local_minima as first_pass  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(ROOT / "datasets"))).expanduser()
INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(ROOT / "index"))).expanduser()
OUTPUT_DIR = ROOT / "final_analysis/false_easy_analysis/final5_querywise_tail_replay"
DEFAULT_DATASETS = (
    "cohere-768-angular",
    "glove-100-angular",
    "msspacev-100M-i8-euclidean",
    "nytimes-256-angular",
    "youtube-15M-angular",
)
DEFAULT_SPACEV_QUERY_CAP = 10000
CLASSIFY_START = 4
CLASSIFY_END = 16
CFR_EMA_DECAY = 0.8
CFR_EMA_UPDATE = 1.0 - CFR_EMA_DECAY
ROUTE_EFS = (256, 512, 768, 1024)
TAIL_BUDGETS = (16, 32, 64, 128, 256, 384, 512, 768)


POLICY_PATHS = {
    "cohere-768-angular": ROOT
    / "final_experiments/final6_ncal100_m32_efc500_target095/hnswlib_run/run/cohere-wiki/"
    "cohere-768-angular__k10__paper_floor_half_b4/cohere-768-angular__k10__mixed_original_M32_efC500.json",
    "glove-100-angular": ROOT
    / "final_experiments/final6_ncal100_m32_efc500_target095/hnswlib_run/run/glove-100/"
    "glove-100-angular__k10__paper_floor_half_b4/glove-100-angular__k10__mixed_original_M32_efC500.json",
    "msspacev-100M-i8-euclidean": ROOT
    / "final_experiments/final6_ncal100_m32_efc500_target095/hnswlib_run/run/spacev/"
    "msspacev-100M-i8-euclidean__k10__paper_floor_half_b4/"
    "msspacev-100M-i8-euclidean__k10__mixed_original_M32_efC500.json",
    "nytimes-256-angular": ROOT
    / "final_experiments/final6_ncal100_m32_efc500_target095/hnswlib_run/run/nytimes/"
    "nytimes-256-angular__k10__paper_floor_half_b4/nytimes-256-angular__k10__mixed_original_M32_efC500.json",
    "youtube-15M-angular": ROOT
    / "final_experiments/youtube15m_target095_full_rerun_20260622/hnswlib/run/youtube-15M-angular/"
    "youtube-15M-angular__k10__current_b4/youtube-15M-angular__k10__mixed_original_M32_efC500.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast querywise false-easy tail replay for the final five datasets."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=80)
    parser.add_argument("--search-batch-size", type=int, default=512)
    parser.add_argument("--trace-batch-size", type=int, default=96)
    parser.add_argument(
        "--spacev-query-cap",
        type=int,
        default=DEFAULT_SPACEV_QUERY_CAP,
        help="Use the first N SpaceV test queries for the fast pass. Set 0 for all.",
    )
    parser.add_argument("--max-queries-per-dataset", type=int, default=0)
    return parser.parse_args()


def dataset_file(stem: str) -> str:
    return stem if stem.endswith(".hdf5") else f"{stem}.hdf5"


def load_policy(stem: str, ef: int) -> tuple[float, tuple[int, ...], tuple[float, ...], str]:
    path = POLICY_PATHS[stem]
    payload = json.loads(path.read_text(encoding="utf-8"))
    tau = float(payload["tau_by_ef"][str(int(ef))])
    route_efs = tuple(int(v) for v in payload["policy"]["route_efs_by_ef"][str(int(ef))])
    gammas = tuple(float(v) for v in payload["policy"]["bucket_gamma_ratios_by_ef"][str(int(ef))])
    mode = str(payload.get("settings", {}).get("mixed_threshold_mode", ""))
    if len(route_efs) != len(gammas):
        raise ValueError(f"Route/gamma length mismatch for {stem}: {route_efs} vs {gammas}")
    return tau, route_efs, gammas, mode


def evaluate_recall(labels: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(labels.shape[0], dtype=np.float64)
    for idx in range(labels.shape[0]):
        out[idx] = np.intersect1d(labels[idx, :k], gt[idx, :k]).size / float(k)
    return out


def search_recall_by_ef(
    *,
    index: Any,
    query_vectors: np.ndarray,
    gt: np.ndarray,
    ef_values: tuple[int, ...],
    k: int,
    num_threads: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    recalls: dict[int, np.ndarray] = {}
    for ef in ef_values:
        start_time = time.time()
        labels_parts = []
        index.set_ef(int(ef))
        for start in range(0, len(query_vectors), int(batch_size)):
            end = min(start + int(batch_size), len(query_vectors))
            labels, _ = index.knn_query(
                query_vectors[start:end],
                k=int(k),
                num_threads=int(num_threads),
            )
            labels_parts.append(np.asarray(labels, dtype=np.int64))
        labels_all = np.vstack(labels_parts)
        recalls[int(ef)] = evaluate_recall(labels_all, gt, int(k))
        print(
            f"[RECALL] ef={int(ef)} queries={len(query_vectors)} mean={recalls[int(ef)].mean():.6f} "
            f"elapsed={time.time() - start_time:.1f}s",
            flush=True,
        )
    return recalls


def route_for_ratio(
    *,
    selection_ef: int,
    k: int,
    route_efs: tuple[int, ...],
    gammas: tuple[float, ...],
    ratio: float,
) -> int:
    if not np.isfinite(ratio):
        return int(selection_ef)
    for route_ef, gamma in zip(route_efs, gammas):
        if float(ratio) <= float(gamma) + 1e-12:
            return max(int(k), int(route_ef))
    return int(selection_ef)


def first_safe_route(recalls: dict[int, np.ndarray], idx: int, target: float) -> int:
    for ef in ROUTE_EFS:
        if float(recalls[int(ef)][idx]) + 1e-12 >= float(target):
            return int(ef)
    return int(ROUTE_EFS[-1])


def hit_steps_from_path(path: list[dict[str, Any]], gt_labels: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    gt_set = {int(x) for x in np.asarray(gt_labels, dtype=np.int64).tolist()}
    seen: set[int] = set()
    labels: list[int] = []
    path_steps: list[int] = []
    fullpop_steps: list[int] = []
    for step_idx, step in enumerate(path):
        label = int(step.get("node_label", -1))
        if label not in gt_set or label in seen:
            continue
        seen.add(label)
        labels.append(label)
        path_steps.append(step_idx + 1)
        try:
            fullpop = int(step.get("full_pop_count_after", 0) or 0)
        except Exception:
            fullpop = 0
        fullpop_steps.append(fullpop)
    return labels, path_steps, fullpop_steps


def classify_cfr_from_path(path: list[dict[str, Any]], selection_ef: int) -> tuple[float, int, int]:
    ema = float("nan")
    observed_full_pop = 0
    window_values: list[float] = []
    for step in path:
        rs_size_after = step.get("rs_size_after", step.get("rs_size", np.nan))
        if pd.isna(rs_size_after) or int(rs_size_after) < int(selection_ef):
            continue
        observed_full_pop += 1
        raw_popped = step.get(
            "popped_query_dist",
            step.get("dist", step.get("query_dist", step.get("internal_dist", np.nan))),
        )
        raw_furthest = step.get(
            "furthest_dist",
            step.get("lowerBound", step.get("internal_dist", np.nan)),
        )
        popped = float(raw_popped) if pd.notna(raw_popped) else float("nan")
        furthest = float(raw_furthest) if pd.notna(raw_furthest) else float("nan")
        cfr_value = float("nan")
        if np.isfinite(popped) and np.isfinite(furthest) and abs(furthest) > 1e-12:
            cfr_value = float(abs(popped) / abs(furthest))
        if np.isfinite(cfr_value):
            if np.isnan(ema):
                ema = float(cfr_value)
            else:
                ema = float(CFR_EMA_DECAY * ema + CFR_EMA_UPDATE * cfr_value)
        if CLASSIFY_START <= observed_full_pop <= CLASSIFY_END and np.isfinite(ema):
            window_values.append(float(ema))
        if observed_full_pop >= CLASSIFY_END:
            # The path is still needed for GT-hit metrics, but CFR classification is done.
            break
    mean_window = float(np.mean(window_values)) if window_values else float("nan")
    return mean_window, int(observed_full_pop), int(len(window_values))


def finite_quantile(values: pd.Series, quantile: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, quantile)) if arr.size else float("nan")


def pct(mask: pd.Series) -> float:
    return float(mask.fillna(False).mean() * 100.0) if len(mask) else float("nan")


def tail_recovery_fields(hit_steps: list[int], route: int, drop: float, k: int) -> dict[str, Any]:
    hit_drop = max(0, int(round(float(drop) * int(k))))
    post_route = [step for step in hit_steps if int(step) > int(route)]
    needed = post_route[:hit_drop]
    covered = hit_drop == 0 or len(post_route) >= hit_drop
    recovery_step = float(needed[-1]) if hit_drop > 0 and covered else float("nan")
    extra = recovery_step - float(route) if np.isfinite(recovery_step) else float("nan")
    out: dict[str, Any] = {
        "hit_drop_at_k": int(hit_drop),
        "post_route_gt_hits": int(len(post_route)),
        "trace_covers_recall_drop": bool(covered),
        "recovery_step_for_drop_hits": recovery_step,
        "extra_budget_to_recover_drop_hits": extra,
        "post_route_steps_covering_drop": "|".join(str(step) for step in needed),
    }
    for budget in TAIL_BUDGETS:
        out[f"recovered_by_route_plus_{budget}"] = bool(np.isfinite(extra) and extra <= float(budget))
        out[f"post_route_gt_hits_within_{budget}"] = int(
            sum(1 for step in post_route if int(step) <= int(route) + int(budget))
        )
    return out


def assign_groups(vanilla_recall: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = sorted(range(len(vanilla_recall)), key=lambda idx: (-float(vanilla_recall[idx]), int(idx)))
    n = len(order)
    easy_end = int(math.ceil(n * 0.30))
    medium_end = int(math.ceil(n * 0.70))
    groups = np.empty(n, dtype=object)
    ranks = np.zeros(n, dtype=np.int64)
    percentiles = np.zeros(n, dtype=np.float64)
    for rank_idx, qid in enumerate(order):
        if rank_idx < easy_end:
            group = "easy"
        elif rank_idx < medium_end:
            group = "medium"
        else:
            group = "hard"
        groups[qid] = group
        ranks[qid] = rank_idx + 1
        percentiles[qid] = (rank_idx + 1) / float(n)
    return groups, ranks, percentiles


def max_queries_for_dataset(stem: str, total: int, args: argparse.Namespace) -> int:
    if int(args.max_queries_per_dataset) > 0:
        return min(int(total), int(args.max_queries_per_dataset))
    if stem == "msspacev-100M-i8-euclidean" and int(args.spacev_query_cap) > 0:
        return min(int(total), int(args.spacev_query_cap))
    return int(total)


def run_dataset(stem: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset_file(stem)
    tau, route_efs, gammas, mode = load_policy(stem, int(args.ef))
    dataset_path = Path(args.dataset_root) / dataset
    print(
        f"[DATASET] {stem} ef={int(args.ef)} tau={tau:.6f} routes={route_efs} "
        f"gammas={tuple(round(g, 6) for g in gammas)} mode={mode}",
        flush=True,
    )

    with h5py.File(dataset_path, "r") as handle:
        test_ds = handle["test"]
        neighbors_ds = handle["neighbors"]
        n_train = int(handle["train"].shape[0])
        dim = int(handle["train"].shape[1])
        n_query = max_queries_for_dataset(stem, int(test_ds.shape[0]), args)
        qids = np.arange(n_query, dtype=np.int64)
        query_vectors = first_pass.read_rows(test_ds, qids)
        gt = np.asarray(neighbors_ds[qids, : int(args.k)], dtype=np.int64)

    index, index_path = first_pass.load_index(
        Path(args.index_dir),
        dataset,
        n_train,
        dim,
        int(args.m),
        int(args.ef_construction),
        int(args.num_threads),
    )
    index.set_num_threads(int(args.num_threads))
    print(f"[INDEX] {stem} loaded {index_path}", flush=True)

    recalls = search_recall_by_ef(
        index=index,
        query_vectors=query_vectors,
        gt=gt,
        ef_values=ROUTE_EFS,
        k=int(args.k),
        num_threads=int(args.num_threads),
        batch_size=int(args.search_batch_size),
    )
    vanilla = recalls[int(args.ef)]
    groups, ranks, percentiles = assign_groups(vanilla)

    trace_fn = getattr(index, "search_layer0_path_with_dist_metrics_batch", None)
    if trace_fn is None:
        raise RuntimeError("Loaded hnswlib does not expose search_layer0_path_with_dist_metrics_batch.")

    all_rows: list[dict[str, Any]] = []
    fe_rows: list[dict[str, Any]] = []
    start_time = time.time()
    for start in range(0, len(qids), int(args.trace_batch_size)):
        end = min(start + int(args.trace_batch_size), len(qids))
        paths, _, _ = trace_fn(
            query_vectors[start:end],
            k=int(args.k),
            ef=int(args.ef),
            num_threads=int(args.num_threads),
        )
        for offset, path in enumerate(paths):
            idx = start + offset
            qid = int(qids[idx])
            cfr_mean, observed_full_pop, window_count = classify_cfr_from_path(path, int(args.ef))
            ratio = float(cfr_mean) / max(float(tau), 1e-12) if np.isfinite(cfr_mean) else float("nan")
            route = route_for_ratio(
                selection_ef=int(args.ef),
                k=int(args.k),
                route_efs=route_efs,
                gammas=gammas,
                ratio=ratio,
            )
            route_recall = float(recalls[int(route)][idx])
            vanilla_recall = float(vanilla[idx])
            drop = max(0.0, vanilla_recall - route_recall)
            safe_route = first_safe_route(recalls, idx, vanilla_recall)
            labels, hit_steps, fullpop_steps = hit_steps_from_path(path, gt[idx])
            vanilla_hits = int(round(vanilla_recall * int(args.k)))
            first_final_step = (
                float(hit_steps[vanilla_hits - 1])
                if vanilla_hits > 0 and len(hit_steps) >= vanilla_hits
                else float("nan")
            )
            cohort = "other"
            if groups[idx] == "hard" and drop > 1e-12 and int(route) < int(args.ef):
                cohort = "hard_false_easy_loss"
            elif groups[idx] == "hard" and drop > 1e-12:
                cohort = "hard_full_route_loss"
            elif groups[idx] == "hard":
                cohort = "hard_no_positive_loss"

            row: dict[str, Any] = {
                "dataset": stem,
                "dataset_file": dataset,
                "qid": qid,
                "cohort": cohort,
                "easiness_group": groups[idx],
                "easiness_rank": int(ranks[idx]),
                "easiness_percentile": float(percentiles[idx]),
                "ef": int(args.ef),
                "route": int(route),
                "first_safe_route": int(safe_route),
                "drop": float(drop),
                "vanilla_recall": vanilla_recall,
                "route_recall": route_recall,
                "recall_256": float(recalls[256][idx]),
                "recall_512": float(recalls[512][idx]),
                "recall_768": float(recalls[768][idx]),
                "recall_1024": float(recalls[1024][idx]),
                "cfr": float(cfr_mean),
                "tau": float(tau),
                "ratio": float(ratio),
                "gammas": "/".join(f"{gamma:.6f}" for gamma in gammas),
                "classify_cfr_mean": float(cfr_mean),
                "classify_cfr_ratio": float(ratio),
                "classify_observed_full_pop_count": int(observed_full_pop),
                "classify_window_obs_count": int(window_count),
                "feature_first_final_step": first_final_step,
                "first_step": first_final_step,
                "gt_k": int(args.k),
                "trace_path_len": int(len(path)),
                "gt_hit_unique_count": int(len(hit_steps)),
                "gt_hit_label_order": "|".join(str(label) for label in labels),
                "gt_hit_path_steps": "|".join(str(step) for step in hit_steps),
                "gt_hit_fullpop_steps": "|".join(str(step) for step in fullpop_steps),
                "first_gt_hit_path_step": float(hit_steps[0]) if hit_steps else float("nan"),
                "last_gt_hit_path_step": float(hit_steps[-1]) if hit_steps else float("nan"),
                "gt_hit_path_span": float(hit_steps[-1] - hit_steps[0]) if hit_steps else float("nan"),
            }
            row.update(tail_recovery_fields(hit_steps, int(route), float(drop), int(args.k)))
            all_rows.append(row)
            if cohort == "hard_false_easy_loss":
                fe_rows.append(row)
        print(
            f"[TRACE] {stem} {end}/{len(qids)} false_easy={len(fe_rows)} "
            f"elapsed={time.time() - start_time:.1f}s",
            flush=True,
        )
    del index
    return pd.DataFrame(all_rows), pd.DataFrame(fe_rows)


def summarize(all_rows: pd.DataFrame, fe_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dataset, group in all_rows.groupby("dataset", sort=True):
        if fe_rows.empty or "dataset" not in fe_rows.columns:
            fe = pd.DataFrame(columns=all_rows.columns)
        else:
            fe = fe_rows[fe_rows["dataset"].astype(str).eq(str(dataset))]
        hard = group[group["easiness_group"].astype(str).eq("hard")]
        row = {
            "dataset": dataset,
            "query_count": int(len(group)),
            "hard_count": int(len(hard)),
            "false_easy_loss_count": int(len(fe)),
            "false_easy_loss_pct_of_all": len(fe) / len(group) * 100.0 if len(group) else 0.0,
            "false_easy_loss_pct_of_hard": len(fe) / len(hard) * 100.0 if len(hard) else 0.0,
            "vanilla_recall_mean": float(group["vanilla_recall"].mean()),
            "hard_vanilla_recall_mean": float(hard["vanilla_recall"].mean()) if len(hard) else float("nan"),
            "false_easy_drop_mean": float(fe["drop"].mean()) if len(fe) else float("nan"),
            "false_easy_drop_p50": finite_quantile(fe["drop"], 0.50) if len(fe) else float("nan"),
            "false_easy_drop_max": float(fe["drop"].max()) if len(fe) else float("nan"),
            "false_easy_hit_drop_1_pct": pct(fe["hit_drop_at_k"].eq(1)) if len(fe) else float("nan"),
            "trace_covers_drop_pct": pct(fe["trace_covers_recall_drop"]) if len(fe) else float("nan"),
            "extra_to_recover_p50": finite_quantile(fe["extra_budget_to_recover_drop_hits"], 0.50)
            if len(fe)
            else float("nan"),
            "extra_to_recover_p90": finite_quantile(fe["extra_budget_to_recover_drop_hits"], 0.90)
            if len(fe)
            else float("nan"),
            "recovered_by_route_plus_128_pct": pct(fe["recovered_by_route_plus_128"]) if len(fe) else float("nan"),
            "recovered_by_route_plus_256_pct": pct(fe["recovered_by_route_plus_256"]) if len(fe) else float("nan"),
        }
        out.append(row)
    return pd.DataFrame(out)


def write_summary(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Final-5 False-Easy Querywise Tail Replay",
        "",
        "Definition: HDF5 test queries, `ef=1024`, `k=10`, hard group is bottom 30% by vanilla recall. A hard false-easy loss has assigned route `<1024` and positive recall drop versus vanilla `ef=1024`.",
        "",
        "| dataset | queries | hard | false-easy loss | FE / hard | mean drop | hit drop=1 | trace covers | extra p50 | extra p90 | <=+128 | <=+256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.query_count} | {row.hard_count} | {row.false_easy_loss_count} | "
            f"{row.false_easy_loss_pct_of_hard:.1f}% | {row.false_easy_drop_mean:.4f} | "
            f"{row.false_easy_hit_drop_1_pct:.1f}% | {row.trace_covers_drop_pct:.1f}% | "
            f"{row.extra_to_recover_p50:.0f} | {row.extra_to_recover_p90:.0f} | "
            f"{row.recovered_by_route_plus_128_pct:.1f}% | {row.recovered_by_route_plus_256_pct:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- SpaceV uses the first 10,000 test queries in this fast pass unless rerun with `--spacev-query-cap 0`.",
            "- Youtube has 1,000 test queries and is fully covered.",
            "- Recovery budget is measured from the assigned route cutoff to the GT hit that covers the observed hit drop in the layer-0 pop trace.",
            "",
            "Files:",
            "",
            "- `per_query_route_replay.csv`: all replayed query rows.",
            "- `per_query_gt_hit_burst_metrics.csv`: hard false-easy loss rows, compatible with the tail recovery script.",
            "- `hard_false_easy_loss_cohort.csv`: compact cohort columns.",
            "- `dataset_summary.csv`: table backing this summary.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [first_pass.dataset_stem(part.strip()) for part in str(args.datasets).split(",") if part.strip()]

    all_frames = []
    fe_frames = []
    for dataset in datasets:
        all_rows, fe_rows = run_dataset(dataset, args)
        all_rows.to_csv(output_dir / f"{dataset}_querywise_replay.csv", index=False)
        fe_rows.to_csv(output_dir / f"{dataset}_gt_hit_burst_metrics.csv", index=False)
        all_frames.append(all_rows)
        fe_frames.append(fe_rows)

    all_combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    fe_combined = pd.concat(fe_frames, ignore_index=True) if fe_frames else pd.DataFrame()
    all_combined.to_csv(output_dir / "per_query_route_replay.csv", index=False)
    fe_combined.to_csv(output_dir / "per_query_gt_hit_burst_metrics.csv", index=False)
    compact_cols = [
        "dataset",
        "qid",
        "cohort",
        "route",
        "drop",
        "cfr",
        "tau",
        "ratio",
        "gammas",
        "first_step",
    ]
    if fe_combined.empty:
        pd.DataFrame(columns=compact_cols).to_csv(output_dir / "hard_false_easy_loss_cohort.csv", index=False)
    else:
        fe_combined[compact_cols].to_csv(output_dir / "hard_false_easy_loss_cohort.csv", index=False)
    summary = summarize(all_combined, fe_combined)
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    write_summary(output_dir, summary)
    print(f"[DONE] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
