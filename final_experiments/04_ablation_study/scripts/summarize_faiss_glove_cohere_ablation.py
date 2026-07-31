#!/usr/bin/env python3
"""Summarize the FAISS GloVe/Cohere paper ablation run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "sage_ablation_faiss_glove_cohere_24t_m32_efc500_ef1024"
DATASET_ORDER = ("glove-100-angular", "cohere-768-angular")
STUDIES: dict[str, dict[str, Any]] = {
    "01_ncal": {
        "parameter": "Calibration set size (ncal)",
        "variants": [("ncal_100", "100"), ("ncal_500", "500"), ("ncal_1000", "1000")],
    },
    "02_classification_window": {
        "parameter": "CFR observation window",
        "variants": [("window_1_13", "[1,13]"), ("window_4_16", "[4,16]")],
    },
    "03_tiers": {
        "parameter": "Difficulty tier count (B)",
        "variants": [("b2", "2"), ("b4", "4"), ("b6", "6")],
    },
    "04_ema_alpha": {
        "parameter": "EMA decay weight (alpha)",
        "variants": [("alpha_0", "0.0"), ("alpha_0p4", "0.4"), ("alpha_0p8", "0.8")],
    },
    "05_pair_gap": {
        "parameter": "Safety margin (g)",
        "variants": [("gap_1x", "1"), ("gap_2x", "2"), ("gap_3x", "3"), ("gap_4x", "4")],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-prefix", default="paper_ablation_glove_cohere_faiss")
    return parser.parse_args()


def dataset_stem(value: object) -> str:
    return Path(str(value)).stem


def sort_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    order = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    out = df.copy()
    out["_dataset_order"] = out["dataset"].map(lambda value: order.get(dataset_stem(value), 999))
    return out.sort_values(["_dataset_order", "dataset"]).drop(columns=["_dataset_order"]).reset_index(drop=True)


def load_variant(path: Path, *, ef: int, k: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" in df.columns:
        df = df[df["method"].astype(str).str.lower().eq("ours")].copy()
    if "ef" in df.columns:
        df = df[df["ef"].astype(int).eq(int(ef))].copy()
    if "k" in df.columns:
        df = df[df["k"].astype(int).eq(int(k))].copy()
    return sort_dataset_frame(df.reset_index(drop=True))


def speedup(row: pd.Series) -> float:
    return 1.0 + float(row.get("qps_gain_vs_vanilla_pct", np.nan)) / 100.0


def loss_fraction(row: pd.Series) -> float:
    return float(row.get("recall_loss_vs_vanilla_pp", np.nan)) / 100.0


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = "" if pd.isna(value) else f"{float(value):.4g}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for study_dir, spec in STUDIES.items():
        for variant_order, (variant_dir, value) in enumerate(spec["variants"]):
            csv_path = output_root / study_dir / variant_dir / "final" / "main_qps_latency_sweep.csv"
            df = load_variant(csv_path, ef=int(args.ef), k=int(args.k))
            if df.empty:
                missing.append({"study": study_dir, "variant": variant_dir, "csv": str(csv_path)})
                continue
            for _, row in df.iterrows():
                rows.append(
                    {
                        "study": study_dir,
                        "parameter": spec["parameter"],
                        "value": value,
                        "variant_order": int(variant_order),
                        "dataset": dataset_stem(row.get("dataset", row.get("dataset_file", ""))),
                        "k": int(row.get("k", args.k)),
                        "ef": int(row.get("ef", args.ef)),
                        "recall": float(row.get("recall", np.nan)),
                        "recall_loss_fraction": loss_fraction(row),
                        "recall_loss_pp": float(row.get("recall_loss_vs_vanilla_pp", np.nan)),
                        "speedup_vs_vanilla": speedup(row),
                        "offline_calibration_wall_s": float(row.get("offline_calibration_wall_s", np.nan)),
                        "route_signature": row.get("route_signature", ""),
                        "bucket_gamma_signature": row.get("bucket_gamma_signature", ""),
                    }
                )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["dataset_order"] = summary["dataset"].map(lambda value: DATASET_ORDER.index(value) if value in DATASET_ORDER else 999)
        summary["study_order"] = summary["study"].map(lambda value: list(STUDIES).index(value))
        summary = summary.sort_values(["study_order", "variant_order", "dataset_order"]).drop(columns=["study_order", "dataset_order"]).reset_index(drop=True)
    final_dir = output_root / "summary"
    final_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = final_dir / f"{args.output_prefix}.csv"
    summary.to_csv(summary_csv, index=False)

    paper_rows: list[dict[str, Any]] = []
    if not summary.empty:
        for study_dir, spec in STUDIES.items():
            for _variant_dir, value in spec["variants"]:
                sub = summary[(summary["study"] == study_dir) & (summary["value"] == value)]
                row: dict[str, Any] = {"parameter": spec["parameter"], "value": value}
                for dataset in DATASET_ORDER:
                    dsub = sub[sub["dataset"].eq(dataset)]
                    if dsub.empty:
                        row[f"{dataset}_loss"] = np.nan
                        row[f"{dataset}_speedup"] = np.nan
                    else:
                        first = dsub.iloc[0]
                        row[f"{dataset}_loss"] = float(first["recall_loss_fraction"])
                        row[f"{dataset}_speedup"] = float(first["speedup_vs_vanilla"])
                paper_rows.append(row)
    paper_df = pd.DataFrame(paper_rows)
    paper_csv = final_dir / f"{args.output_prefix}_paper_table.csv"
    paper_df.to_csv(paper_csv, index=False)

    md_path = final_dir / f"{args.output_prefix}.md"
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# FAISS GloVe/Cohere Ablation Summary\n\n")
        handle.write(f"- source root: `{output_root}`\n")
        handle.write(f"- k: `{int(args.k)}`\n")
        handle.write(f"- efSearch: `{int(args.ef)}`\n")
        handle.write("- loss columns use paper-style fraction units; raw runner also writes percent-point units.\n\n")
        handle.write("## Paper Table Shape\n\n")
        handle.write(md_table(paper_df))
        handle.write("\n\n## Missing Inputs\n\n")
        handle.write(md_table(pd.DataFrame(missing)))
        handle.write("\n")
    print(f"[DONE] {summary_csv}")
    print(f"[DONE] {paper_csv}")
    print(f"[DONE] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
