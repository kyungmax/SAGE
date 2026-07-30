#!/usr/bin/env python3
"""Analyze large false-easy recall drops in main8 drill-down replay output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
DRILLDOWN_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_QUERYWISE = (
    DRILLDOWN_ROOT
    / "drilldown_faiss_SIMD_on_main8_24t"
    / "hard_loss_querywise_exactgt_24t"
    / "hard_loss_querywise.csv"
)
DEFAULT_QUERY_GROUPS = (
    DRILLDOWN_ROOT
    / "drilldown_faiss_SIMD_on_main8_24t"
    / "difficulty_exactgt_24t"
    / "query_groups.csv"
)
DEFAULT_OUTPUT_DIR = DRILLDOWN_ROOT / "drilldown_faiss_SIMD_on_main8_24t" / "large_false_easy_drop_analysis"
DEFAULT_DATASETS = (
    "glove-100-angular",
    "nytimes-256-angular",
    "msmarco-v1-openai-ada2-full-ip",
    "msspacev-100M-i8-euclidean",
    "cohere-768-angular",
    "youtube-15M-angular",
    "agnews-mxbai-1024-euclidean",
    "landmark-nomic-768-angular",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--querywise", type=Path, default=DEFAULT_QUERYWISE)
    parser.add_argument("--query-groups", type=Path, default=DEFAULT_QUERY_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-prefix", default="main8_SIMD_on")
    parser.add_argument("--min-drop", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def parse_numeric_list(value: object, cast=float) -> list:
    if value is None or pd.isna(value):
        return []
    return [cast(part) for part in str(value).split("/") if str(part).strip()]


def finite_quantile(values: pd.Series, q: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def pct(mask: pd.Series) -> float:
    if len(mask) == 0:
        return float("nan")
    return float(mask.astype(bool).mean() * 100.0)


def add_route_boundary_fields(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        routes = parse_numeric_list(getattr(row, "route_efs"), int)
        gammas = parse_numeric_list(getattr(row, "bucket_gammas"), float)
        routed = int(getattr(row, "routed_ef"))
        ef = int(getattr(row, "ef"))
        ratio = float(getattr(row, "chr")) / max(float(getattr(row, "tau")), 1e-12)
        route_idx = routes.index(routed) if routed in routes else -1
        boundary = float("nan")
        margin = float("nan")
        next_route = ef
        if route_idx >= 0 and route_idx < len(gammas):
            boundary = float(gammas[route_idx])
            margin = boundary - ratio
            if route_idx + 1 < len(routes):
                next_route = int(routes[route_idx + 1])
        rows.append(
            {
                "route_bucket": f"{ef}->{routed}",
                "chr_ratio": ratio,
                "route_index": route_idx,
                "next_harder_route": next_route,
                "harder_route_boundary_gamma": boundary,
                "margin_to_harder_route": margin,
                "near_boundary_0p005": bool(np.isfinite(margin) and margin <= 0.005),
                "near_boundary_0p01": bool(np.isfinite(margin) and margin <= 0.01),
                "near_boundary_0p02": bool(np.isfinite(margin) and margin <= 0.02),
                "near_boundary_0p05": bool(np.isfinite(margin) and margin <= 0.05),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def ordered_groups(df: pd.DataFrame, keys: Iterable[str]):
    return df.groupby(list(keys), sort=True, dropna=False)


def summarize_dataset_ef(hard: pd.DataFrame, fe: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, ef), part in ordered_groups(hard, ["dataset", "ef"]):
        fe_part = fe[(fe["dataset"].eq(dataset)) & (fe["ef"].eq(ef))]
        large_part = large[(large["dataset"].eq(dataset)) & (large["ef"].eq(ef))]
        fe_loss_sum = float(fe_part["hard_positive_loss"].sum())
        large_loss_sum = float(large_part["hard_positive_loss"].sum())
        rows.append(
            {
                "dataset": dataset,
                "ef": int(ef),
                "hard_q": int(len(part)),
                "false_easy_q": int(len(fe_part)),
                "large_false_easy_q": int(len(large_part)),
                "large_share_of_false_easy_q_pct": len(large_part) / len(fe_part) * 100.0 if len(fe_part) else np.nan,
                "large_share_of_false_easy_loss_pct": large_loss_sum / fe_loss_sum * 100.0 if fe_loss_sum > 0 else np.nan,
                "false_easy_drop_sum": fe_loss_sum,
                "large_drop_sum": large_loss_sum,
                "large_drop_mean": float(large_part["hard_positive_loss"].mean()) if len(large_part) else np.nan,
                "large_drop_p50": finite_quantile(large_part["hard_positive_loss"], 0.50),
                "large_drop_max": float(large_part["hard_positive_loss"].max()) if len(large_part) else np.nan,
                "large_hit_drop_sum": int(large_part["hit_drop_at_k"].sum()) if len(large_part) else 0,
                "large_vanilla_recall_mean": float(large_part["vanilla_recall"].mean()) if len(large_part) else np.nan,
                "large_route_recall_mean": float(large_part["route_recall"].mean()) if len(large_part) else np.nan,
                "large_exact_ours_recall_mean": float(large_part["exact_ours_recall"].mean()) if len(large_part) else np.nan,
                "large_route_only_loss_mean": float(large_part["route_only_loss"].mean()) if len(large_part) else np.nan,
                "large_stop_extra_loss_mean": float(large_part["stop_extra_loss"].mean()) if len(large_part) else np.nan,
                "large_group_def_vanilla_recall_mean": float(large_part["group_def_vanilla_recall"].mean()) if len(large_part) else np.nan,
                "large_group_def_first_final_step_p50": finite_quantile(large_part["group_def_first_final_recall_step"], 0.50),
                "large_easiness_percentile_mean": float(large_part["easiness_percentile"].mean()) if len(large_part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_routes(large: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, ef, route_bucket), part in ordered_groups(large, ["dataset", "ef", "route_bucket"]):
        rows.append(
            {
                "dataset": dataset,
                "ef": int(ef),
                "route_bucket": route_bucket,
                "q": int(len(part)),
                "drop_sum": float(part["hard_positive_loss"].sum()),
                "drop_mean": float(part["hard_positive_loss"].mean()),
                "drop_p50": finite_quantile(part["hard_positive_loss"], 0.50),
                "vanilla_recall_mean": float(part["vanilla_recall"].mean()),
                "route_recall_mean": float(part["route_recall"].mean()),
                "exact_ours_recall_mean": float(part["exact_ours_recall"].mean()),
                "route_only_loss_mean": float(part["route_only_loss"].mean()),
                "stop_extra_loss_mean": float(part["stop_extra_loss"].mean()),
                "chr_ratio_mean": float(part["chr_ratio"].mean()),
                "margin_to_harder_route_p50": finite_quantile(part["margin_to_harder_route"], 0.50),
                "near_boundary_0p01_pct": pct(part["near_boundary_0p01"]),
                "near_boundary_0p05_pct": pct(part["near_boundary_0p05"]),
                "group_first_final_step_p50": finite_quantile(part["group_def_first_final_recall_step"], 0.50),
            }
        )
    return pd.DataFrame(rows)


def summarize_drop_buckets(fe: pd.DataFrame, min_drop: float) -> pd.DataFrame:
    scoped = fe[fe["hard_positive_loss"].ge(min_drop - 1e-12)].copy()
    scoped["drop_bucket"] = scoped["hard_positive_loss"].round(1)
    out = (
        scoped.groupby(["dataset", "ef", "drop_bucket"], sort=True)
        .size()
        .reset_index(name="q")
        .sort_values(["dataset", "ef", "drop_bucket"])
    )
    return out


def summarize_overlap(large: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, part in large.groupby("dataset", sort=True):
        ef_sets = {int(ef): set(group["qid"].astype(int)) for ef, group in part.groupby("ef")}
        ef512 = ef_sets.get(512, set())
        ef1024 = ef_sets.get(1024, set())
        rows.append(
            {
                "dataset": dataset,
                "large_qid_ef512": len(ef512),
                "large_qid_ef1024": len(ef1024),
                "large_qid_both": len(ef512 & ef1024),
                "large_qid_512_only": len(ef512 - ef1024),
                "large_qid_1024_only": len(ef1024 - ef512),
                "large_qid_union": len(ef512 | ef1024),
            }
        )
    return pd.DataFrame(rows)


def feature_summary(fe: pd.DataFrame, min_drop: float) -> pd.DataFrame:
    scoped = fe.copy()
    scoped["tail_cohort"] = np.where(scoped["hard_positive_loss"].ge(min_drop - 1e-12), "large_drop_ge_threshold", "small_drop_lt_threshold")
    metrics = [
        "hard_positive_loss",
        "vanilla_recall",
        "route_recall",
        "exact_ours_recall",
        "route_only_loss",
        "stop_extra_loss",
        "chr_ratio",
        "margin_to_harder_route",
        "group_def_vanilla_recall",
        "group_def_first_final_recall_step",
        "easiness_percentile",
    ]
    rows: list[dict[str, object]] = []
    for (dataset, ef, cohort), part in scoped.groupby(["dataset", "ef", "tail_cohort"], sort=True):
        row: dict[str, object] = {"dataset": dataset, "ef": int(ef), "tail_cohort": cohort, "q": int(len(part))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean()) if len(part) else np.nan
            row[f"{metric}_p50"] = finite_quantile(part[metric], 0.50)
            row[f"{metric}_p90"] = finite_quantile(part[metric], 0.90)
        row["near_boundary_0p01_pct"] = pct(part["near_boundary_0p01"])
        row["near_boundary_0p05_pct"] = pct(part["near_boundary_0p05"])
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, *, floatfmt: str | None = None) -> str:
    if df.empty:
        return "_none_"
    local = df.copy()
    if floatfmt is not None:
        for col in local.columns:
            if pd.api.types.is_float_dtype(local[col]):
                local[col] = local[col].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
    cols = [str(col) for col in local.columns]

    def cell(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in local.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in local.columns) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    *,
    querywise: Path,
    query_groups: Path,
    min_drop: float,
    summary: pd.DataFrame,
    route_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    top: pd.DataFrame,
) -> None:
    lines = [
        "# Large False-Easy Recall-Drop Analysis",
        "",
        f"Source querywise: `{querywise}`",
        f"Source query groups: `{query_groups}`",
        "",
        "Definition used here:",
        "",
        "- Scope is the final exact-GT drilldown hard group.",
        "- `false_easy_loss = hard_positive_loss > 0 and routed_ef < efSearch`.",
        "- `hard_positive_loss = max(Vanilla Recall@10 - exact Ours Recall@10, 0)`.",
        f"- Large tail means `hard_positive_loss >= {min_drop:.1f}`; at `k=10`, this is at least {int(round(min_drop * 10))} missed GT neighbors.",
        "",
        "## Dataset / ef Summary",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        "## Large-Tail Route Buckets",
        "",
        markdown_table(summary, floatfmt=".4f"),
        "",
        "## Cross-ef Query Overlap",
        "",
        markdown_table(overlap),
        "",
        "## Top Large-Drop Rows",
        "",
        markdown_table(top, floatfmt=".4f"),
        "",
        "Reading:",
        "",
        "- `route_only_loss` compares vanilla at the selected efSearch against vanilla at the CHR-selected lower route.",
        "- `stop_extra_loss` is any additional loss of exact Ours versus the full lower-route search; if this is near zero, the problem is route assignment, not an extra early-stop artifact.",
        "- `margin_to_harder_route` is the CHR-ratio distance to the next harder route boundary. Small values mean boundary cases; large values mean the query looked confidently easy under CHR.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = tuple(part.strip() for part in str(args.datasets).split(",") if part.strip())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    querywise = pd.read_csv(args.querywise)
    qgroups = pd.read_csv(args.query_groups)
    querywise = querywise[querywise["dataset"].astype(str).isin(datasets)].copy()
    qgroups = qgroups[qgroups["dataset"].astype(str).isin(datasets)].copy()

    group_cols = [
        "dataset",
        "qid",
        "group_def_vanilla_recall",
        "group_def_target_hits",
        "group_def_first_final_recall_step",
        "group_def_first_step_reached_target",
        "group_def_first_step_achieved_hits",
        "easiness_group",
        "easiness_rank",
        "easiness_percentile",
    ]
    hard = querywise.merge(qgroups[group_cols], on=["dataset", "qid"], how="left", validate="many_to_one")
    hard["hard_positive_loss"] = pd.to_numeric(hard["hard_positive_loss"], errors="coerce").fillna(0.0)
    hard["route_only_loss"] = np.maximum(hard["vanilla_recall"].astype(float) - hard["route_recall"].astype(float), 0.0)
    hard["stop_extra_loss"] = np.maximum(hard["route_recall"].astype(float) - hard["exact_ours_recall"].astype(float), 0.0)
    hard["hit_drop_at_k"] = np.rint(hard["hard_positive_loss"].astype(float) * int(args.k)).astype(int)
    hard = add_route_boundary_fields(hard)

    fe = hard[hard["false_easy_loss"].astype(bool)].copy()
    large = fe[fe["hard_positive_loss"].ge(float(args.min_drop) - 1e-12)].copy()

    summary = summarize_dataset_ef(hard, fe, large)
    route_summary = summarize_routes(large)
    drop_buckets = summarize_drop_buckets(fe, float(args.min_drop))
    overlap = summarize_overlap(large)
    features = feature_summary(fe, float(args.min_drop))
    top_cols = [
        "dataset",
        "ef",
        "qid",
        "route_bucket",
        "hard_positive_loss",
        "hit_drop_at_k",
        "vanilla_recall",
        "route_recall",
        "exact_ours_recall",
        "route_only_loss",
        "stop_extra_loss",
        "chr_ratio",
        "margin_to_harder_route",
        "group_def_vanilla_recall",
        "group_def_first_final_recall_step",
        "easiness_rank",
        "easiness_percentile",
    ]
    top = large.sort_values(["hard_positive_loss", "dataset", "ef", "qid"], ascending=[False, True, True, True])[top_cols].head(40)

    prefix = str(args.output_prefix).strip() or "analysis"
    hard.to_csv(output_dir / f"{prefix}_hard_querywise_with_features.csv", index=False)
    fe.to_csv(output_dir / f"{prefix}_false_easy_cases.csv", index=False)
    large.to_csv(output_dir / f"{prefix}_large_false_easy_tail_cases.csv", index=False)
    summary.to_csv(output_dir / "large_false_easy_summary_by_dataset_ef.csv", index=False)
    route_summary.to_csv(output_dir / "large_false_easy_summary_by_route.csv", index=False)
    drop_buckets.to_csv(output_dir / "large_false_easy_drop_buckets.csv", index=False)
    overlap.to_csv(output_dir / "large_false_easy_cross_ef_overlap.csv", index=False)
    features.to_csv(output_dir / "false_easy_large_vs_small_feature_summary.csv", index=False)
    top.to_csv(output_dir / "top_large_false_easy_cases.csv", index=False)
    write_markdown(
        output_dir / "README.md",
        querywise=args.querywise,
        query_groups=args.query_groups,
        min_drop=float(args.min_drop),
        summary=summary,
        route_summary=route_summary,
        overlap=overlap,
        top=top,
    )
    print(f"[RESULT] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
