#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

import false_easy_final5_querywise_tail_replay as replay
import false_easy_first_pass_gt_spread_local_minima as first_pass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "final_analysis/false_easy_analysis/final5_querywise_tail_replay"
    / "paper_difficulty_full_ladder/per_query_route_replay_with_paper_difficulty.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_analysis/false_easy_analysis/final5_querywise_tail_replay"
    / "paper_difficulty_full_ladder/exact_ours_replay"
)
OLD_TRUE_GT = ROOT / "final_analysis/false_easy_analysis/hard_group_drilldown_true_gt.csv"

DATASET_LABELS = {
    "cohere-768-angular": "Cohere",
    "glove-100-angular": "GloVe",
    "msspacev-100M-i8-euclidean": "SpaceV",
    "nytimes-256-angular": "NYTimes",
    "youtube-15M-angular": "YouTube",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach exact SAGE/Ours recall to false-easy replay rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=replay.DATASET_ROOT)
    parser.add_argument("--index-dir", type=Path, default=replay.INDEX_DIR)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def evaluate_recall(labels: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(labels.shape[0], dtype=np.float64)
    for idx in range(labels.shape[0]):
        out[idx] = np.intersect1d(labels[idx, :k], gt[idx, :k]).size / float(k)
    return out


def run_sage_batches(
    *,
    index: Any,
    query_vectors: np.ndarray,
    k: int,
    ef: int,
    tau: float,
    gammas: tuple[float, ...],
    tmin_pops: int,
    num_threads: int,
    batch_size: int,
) -> np.ndarray:
    labels_parts: list[np.ndarray] = []
    bucket_count = len(gammas) + 1
    start_time = time.time()
    for start in range(0, len(query_vectors), int(batch_size)):
        end = min(start + int(batch_size), len(query_vectors))
        labels, _dists = index.knn_query_sage(
            query_vectors[start:end],
            k=int(k),
            ef_init=int(ef),
            enable_stop=True,
            early_stop_ratio=float(tau),
            tmin_pops=int(tmin_pops),
            paper_bucket_count=int(bucket_count),
            bucket_gamma_ratios=list(float(v) for v in gammas),
            num_threads=int(num_threads),
        )
        labels_parts.append(np.asarray(labels, dtype=np.int64))
        print(f"[SAGE] {end}/{len(query_vectors)} elapsed={time.time() - start_time:.1f}s", flush=True)
    return np.vstack(labels_parts)


def run_dataset(stem: str, rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    dataset_file = replay.dataset_file(stem)
    tau, route_efs, gammas, mode = replay.load_policy(stem, int(args.ef))
    if str(mode) != "paper_floor_half" and stem != "youtube-15M-angular":
        print(f"[WARN] {stem}: policy mode={mode!r}", flush=True)
    print(
        f"[DATASET] {stem} q={len(rows)} ef={int(args.ef)} tau={tau:.6f} "
        f"routes={route_efs} gammas={tuple(round(v, 6) for v in gammas)}",
        flush=True,
    )

    qids = rows["qid"].to_numpy(dtype=np.int64)
    with h5py.File(Path(args.dataset_root) / dataset_file, "r") as handle:
        n_train = int(handle["train"].shape[0])
        dim = int(handle["train"].shape[1])
        query_vectors = first_pass.read_rows(handle["test"], qids)
        gt = np.asarray(handle["neighbors"][qids, : int(args.k)], dtype=np.int64)

    index, _index_path = first_pass.load_index(
        Path(args.index_dir),
        dataset_file,
        n_train,
        dim,
        int(args.m),
        int(args.ef_construction),
        int(args.num_threads),
    )
    index.set_num_threads(int(args.num_threads))
    labels = run_sage_batches(
        index=index,
        query_vectors=query_vectors,
        k=int(args.k),
        ef=int(args.ef),
        tau=float(tau),
        gammas=gammas,
        tmin_pops=int(args.tmin_pops),
        num_threads=int(args.num_threads),
        batch_size=int(args.batch_size),
    )
    exact_recall = evaluate_recall(labels, gt, int(args.k))
    out = rows.copy()
    out["exact_ours_recall"] = exact_recall
    out["exact_ours_signed_recall_loss"] = out["vanilla_recall"].astype(float) - out["exact_ours_recall"].astype(float)
    out["exact_ours_positive_recall_loss"] = np.maximum(
        out["exact_ours_signed_recall_loss"].to_numpy(dtype=np.float64),
        0.0,
    )
    out["exact_is_paper_hard_loss"] = (
        out["paper_difficulty_group"].astype(str).eq("hard")
        & (out["exact_ours_positive_recall_loss"].astype(float) > 1e-12)
    )
    out["exact_is_paper_hard_false_easy_loss"] = (
        out["exact_is_paper_hard_loss"]
        & (out["route"].astype(int) < int(args.ef))
    )
    out["exact_is_paper_hard_full_route_loss"] = (
        out["exact_is_paper_hard_loss"]
        & (out["route"].astype(int) >= int(args.ef))
    )
    return out


def bucket_counts(values: pd.Series) -> dict[str, int]:
    rounded = np.round(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64), 1)
    out: dict[str, int] = {}
    for bucket in [round(i / 10.0, 1) for i in range(1, 11)]:
        out[f"drop_{bucket:.1f}_count"] = int(np.sum(np.isclose(rounded, bucket, atol=1e-9)))
    return out


def summarize_exact(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    compact_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for dataset, group in rows.groupby("dataset", sort=True):
        hard = group[group["paper_difficulty_group"].astype(str).eq("hard")].copy()
        hard_loss = hard[hard["exact_is_paper_hard_loss"].astype(bool)].copy()
        fe = hard[hard["exact_is_paper_hard_false_easy_loss"].astype(bool)].copy()
        full_route = hard[hard["exact_is_paper_hard_full_route_loss"].astype(bool)].copy()
        row: dict[str, Any] = {
            "dataset": dataset,
            "hard_q": int(len(hard)),
            "hard_positive_loss_q": int(len(hard_loss)),
            "fe_q": int(len(fe)),
            "full_route_loss_q": int(len(full_route)),
            "fe_pct_of_hard": float(len(fe) / len(hard) * 100.0) if len(hard) else np.nan,
            "hard_net_mean_drop_all": float(hard["exact_ours_signed_recall_loss"].mean()) if len(hard) else np.nan,
            "hard_positive_mean_drop_all": float(hard["exact_ours_positive_recall_loss"].mean()) if len(hard) else np.nan,
            "fe_mean_drop": float(fe["exact_ours_positive_recall_loss"].mean()) if len(fe) else np.nan,
        }
        row.update(bucket_counts(fe["exact_ours_positive_recall_loss"] if len(fe) else pd.Series(dtype=float)))
        compact_rows.append(row)
        if len(fe):
            route_part = (
                fe.groupby("route", as_index=False)
                .agg(
                    fe_q=("qid", "count"),
                    fe_mean_drop=("exact_ours_positive_recall_loss", "mean"),
                    fe_drop_sum=("exact_ours_positive_recall_loss", "sum"),
                )
            )
            route_part.insert(0, "dataset", dataset)
            route_rows.extend(route_part.to_dict("records"))
    return pd.DataFrame(compact_rows), pd.DataFrame(route_rows)


def compare_old(summary: pd.DataFrame) -> pd.DataFrame:
    if not OLD_TRUE_GT.exists():
        return pd.DataFrame()
    old = pd.read_csv(OLD_TRUE_GT)
    old = old[old["group"].astype(str).str.lower().eq("hard")].copy()
    label_to_stem = {label: stem for stem, label in DATASET_LABELS.items()}
    old["dataset"] = old["dataset"].map(label_to_stem)
    old = old.dropna(subset=["dataset"])
    old = old[["dataset", "ef1024_recall_loss", "ef1024_recall_loss_pp"]].rename(
        columns={
            "ef1024_recall_loss": "old_final_hard_loss",
            "ef1024_recall_loss_pp": "old_final_hard_loss_pp",
        }
    )
    merged = summary.merge(old, on="dataset", how="left")
    merged["exact_minus_old_hard_loss"] = (
        merged["hard_net_mean_drop_all"].astype(float) - merged["old_final_hard_loss"].astype(float)
    )
    return merged[
        [
            "dataset",
            "hard_q",
            "hard_net_mean_drop_all",
            "hard_positive_mean_drop_all",
            "old_final_hard_loss",
            "exact_minus_old_hard_loss",
            "fe_q",
            "full_route_loss_q",
        ]
    ]


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("" if pd.isna(value) else f"{float(value):.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.input)
    datasets = [part.strip() for part in str(args.datasets).split(",") if part.strip()]
    if not datasets:
        datasets = sorted(rows["dataset"].astype(str).unique().tolist())
    frames = []
    for dataset in datasets:
        stem = first_pass.dataset_stem(dataset)
        part = rows[rows["dataset"].astype(str).eq(stem)].copy()
        if part.empty:
            print(f"[SKIP] {stem}: no input rows", flush=True)
            continue
        frames.append(run_dataset(stem, part, args))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(out_dir / "per_query_exact_ours_replay_with_paper_difficulty.csv", index=False)
    summary, route_summary = summarize_exact(combined)
    summary.to_csv(out_dir / "exact_paper_hard_overall_vs_fe_drop_by_bucket_compact.csv", index=False)
    route_summary.to_csv(out_dir / "exact_paper_hard_false_easy_by_route.csv", index=False)
    old_compare = compare_old(summary)
    old_compare.to_csv(out_dir / "exact_vs_old_final_hard_loss.csv", index=False)
    lines = [
        "# Exact Ours False-Easy Replay",
        "",
        "This rerun uses `knn_query_sage`, i.e. the actual adaptive-light paper-bucket/SAGE online path, not fixed-route vanilla recall.",
        "",
        "## Compact",
        "",
        md_table(summary),
        "",
        "## Old Final Hard-Loss Comparison",
        "",
        md_table(old_compare),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
