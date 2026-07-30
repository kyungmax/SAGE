#!/usr/bin/env python3
"""Summarize iso-recall speedups for the MSMARCO embedding-model sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_LABEL = {
    "msmarco-v1-glove6b300d-full-ip": "GloVe mean",
    "msmarco-v1-fasttext-cc300d-full-ip": "FastText mean",
    "msmarco-v1-openai-ada2-full-ip": "OpenAI ada-002",
    "msmarco-v1-bge-m3-fp32-dev6980-ip": "BGE-M3",
    "msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip": "EmbeddingGemma-300M",
}
MODEL_ORDER = list(MODEL_LABEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-prefix", default="msmarco_embedding_model_iso_speedup")
    return parser.parse_args()


def latency_ms(row: pd.Series) -> float:
    value = row.get("latency_per_query_mean_ms", np.nan)
    if pd.notna(value):
        return float(value)
    return 1000.0 / float(row["qps"])


def interp_vanilla_at_recall(vanilla: pd.DataFrame, target_recall: float) -> tuple[float | None, float | None, str]:
    pts = sorted(
        [(float(row["recall"]), latency_ms(row), int(row["ef"])) for _, row in vanilla.iterrows()],
        key=lambda item: (item[0], item[2]),
    )
    if not pts:
        return None, None, "no vanilla rows"
    if target_recall < pts[0][0] - 1e-12:
        return None, None, f"target below vanilla range [{pts[0][0]:.6f}, {pts[-1][0]:.6f}]"
    if target_recall > pts[-1][0] + 1e-12:
        return None, None, f"target above vanilla range [{pts[0][0]:.6f}, {pts[-1][0]:.6f}]"
    for recall, lat, ef in pts:
        if abs(recall - target_recall) <= 1e-12:
            return lat, float(ef), "exact"
    for (r0, l0, e0), (r1, l1, e1) in zip(pts, pts[1:]):
        lo, hi = min(r0, r1), max(r0, r1)
        if lo - 1e-12 <= target_recall <= hi + 1e-12 and abs(r1 - r0) > 1e-12:
            t = (target_recall - r0) / (r1 - r0)
            return l0 + t * (l1 - l0), e0 + t * (e1 - e0), "linear"
    return None, None, "not bracketed"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, sub in df.groupby(df["dataset"].astype(str), sort=False):
        vanilla = sub[sub["method"].astype(str).eq("Vanilla")].sort_values("recall")
        ours = sub[sub["method"].astype(str).eq("Ours")].sort_values("ef")
        for _, row in ours.iterrows():
            ours_latency = latency_ms(row)
            target_recall = float(row["recall"])
            vanilla_latency, vanilla_ef, status = interp_vanilla_at_recall(vanilla, target_recall)
            speedup = np.nan
            if vanilla_latency is not None and vanilla_latency > 0 and ours_latency > 0:
                speedup = float(vanilla_latency / ours_latency)
            rows.append(
                {
                    "dataset": dataset,
                    "model_label": MODEL_LABEL.get(dataset, dataset),
                    "ours_ef": int(row["ef"]),
                    "ours_recall": target_recall,
                    "ours_latency_per_query_ms": ours_latency,
                    "vanilla_interp_ef": vanilla_ef,
                    "vanilla_interp_latency_per_query_ms": vanilla_latency,
                    "iso_speedup_vs_vanilla": speedup,
                    "same_ef_qps_gain_pct": float(row.get("qps_gain_vs_vanilla_pct", np.nan)),
                    "same_ef_recall_loss_pp": float(row.get("recall_loss_vs_vanilla_pp", np.nan)),
                    "iso_status": status,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    summary["model_order"] = summary["dataset"].map(lambda value: order.get(str(value), 999))
    return summary.sort_values(["model_order", "ours_ef"]).drop(columns=["model_order"]).reset_index(drop=True)


def best_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in summary.groupby("dataset", sort=False):
        finite = sub[np.isfinite(sub["iso_speedup_vs_vanilla"].to_numpy(dtype=float))]
        rows.append((finite if not finite.empty else sub).sort_values("iso_speedup_vs_vanilla", ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def main() -> int:
    args = parse_args()
    if args.csv is None:
        if args.final_dir is None:
            raise SystemExit("pass --csv or --final-dir")
        source_csv = args.final_dir.expanduser().resolve() / "main_qps_latency_sweep.csv"
        out_dir = args.final_dir.expanduser().resolve()
    else:
        source_csv = args.csv.expanduser().resolve()
        out_dir = args.final_dir.expanduser().resolve() if args.final_dir else source_csv.parent
    df = pd.read_csv(source_csv)
    df = df[df["k"].astype(int).eq(10)].copy()
    summary = build_summary(df)
    best = best_rows(summary)
    summary_csv = out_dir / f"{args.output_prefix}.csv"
    best_csv = out_dir / f"{args.output_prefix}_best.csv"
    summary.to_csv(summary_csv, index=False)
    best.to_csv(best_csv, index=False)
    print(f"[DONE] {summary_csv}")
    print(f"[DONE] {best_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
