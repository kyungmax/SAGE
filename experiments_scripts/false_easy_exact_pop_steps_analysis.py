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
    / "paper_difficulty_full_ladder/exact_ours_replay"
    / "per_query_exact_ours_replay_with_paper_difficulty.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "final_analysis/false_easy_analysis/final5_querywise_tail_replay"
    / "paper_difficulty_full_ladder/exact_ours_replay"
    / "pop_steps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure actual paper-bucket adaptive pop steps.")
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
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def run_analysis_batches(
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
    steps_parts: list[np.ndarray] = []
    bucket_count = len(gammas) + 1
    start_time = time.time()
    for start in range(0, len(query_vectors), int(batch_size)):
        end = min(start + int(batch_size), len(query_vectors))
        out = index.knn_query_adaptive_analysis_paper_bucket(
            query_vectors[start:end],
            k=int(k),
            ef_init=int(ef),
            ef_max=int(ef),
            tmin_pops=int(tmin_pops),
            enable_stop=True,
            early_stop_ratio=float(tau),
            paper_bucket_count=int(bucket_count),
            bucket_gamma_ratios=list(float(v) for v in gammas),
            num_threads=int(num_threads),
        )
        steps_parts.append(np.asarray(out[2], dtype=np.int64))
        print(f"[STEPS] {end}/{len(query_vectors)} elapsed={time.time() - start_time:.1f}s", flush=True)
    return np.concatenate(steps_parts)


def run_dataset(stem: str, rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    dataset_file = replay.dataset_file(stem)
    tau, route_efs, gammas, _mode = replay.load_policy(stem, int(args.ef))
    print(
        f"[DATASET] {stem} q={len(rows)} tau={tau:.6f} routes={route_efs} "
        f"gammas={tuple(round(v, 6) for v in gammas)}",
        flush=True,
    )
    qids = rows["qid"].to_numpy(dtype=np.int64)
    with h5py.File(Path(args.dataset_root) / dataset_file, "r") as handle:
        n_train = int(handle["train"].shape[0])
        dim = int(handle["train"].shape[1])
        query_vectors = first_pass.read_rows(handle["test"], qids)
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
    actual_steps = run_analysis_batches(
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
    out = rows.copy()
    out["exact_ours_pop_steps"] = actual_steps
    out["exact_ours_steps_lt_route"] = out["exact_ours_pop_steps"].astype(int) < out["route"].astype(int)
    out["exact_ours_steps_lt_256"] = out["exact_ours_pop_steps"].astype(int) < 256
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dataset, group in rows.groupby("dataset", sort=True):
        route256 = group[group["route"].astype(int).eq(256)]
        fe = group[group["exact_is_paper_hard_false_easy_loss"].astype(bool)]
        fe256 = fe[fe["route"].astype(int).eq(256)]
        for name, part in [
            ("all_route256", route256),
            ("hard_fe_route256", fe256),
            ("hard_fe_all_routes", fe),
        ]:
            steps = part["exact_ours_pop_steps"].astype(float) if len(part) else pd.Series(dtype=float)
            out.append(
                {
                    "dataset": dataset,
                    "cohort": name,
                    "q": int(len(part)),
                    "steps_lt_256_q": int((steps < 256).sum()) if len(part) else 0,
                    "steps_lt_256_pct": float((steps < 256).mean() * 100.0) if len(part) else np.nan,
                    "steps_lt_route_q": int(part["exact_ours_steps_lt_route"].sum()) if len(part) else 0,
                    "steps_lt_route_pct": float(part["exact_ours_steps_lt_route"].mean() * 100.0) if len(part) else np.nan,
                    "steps_min": float(steps.min()) if len(part) else np.nan,
                    "steps_p25": float(steps.quantile(0.25)) if len(part) else np.nan,
                    "steps_p50": float(steps.quantile(0.50)) if len(part) else np.nan,
                    "steps_p90": float(steps.quantile(0.90)) if len(part) else np.nan,
                    "steps_max": float(steps.max()) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(out)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.input)
    datasets = [part.strip() for part in str(args.datasets).split(",") if part.strip()]
    if not datasets:
        datasets = sorted(rows["dataset"].astype(str).unique().tolist())
    frames = []
    for dataset in datasets:
        stem = first_pass.dataset_stem(dataset)
        part = rows[rows["dataset"].astype(str).eq(stem)].copy()
        if part.empty:
            print(f"[SKIP] {stem}: no rows", flush=True)
            continue
        frames.append(run_dataset(stem, part, args))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output_dir / "per_query_exact_ours_with_pop_steps.csv", index=False)
    summary = summarize(combined)
    summary.to_csv(output_dir / "pop_steps_summary.csv", index=False)
    print(f"[DONE] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
