#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_HIT_DIR = (
    ROOT
    / "final_analysis/false_easy_analysis/first_pass_gt_spread_local_minima_20260622/gt_hit_burst_20260622"
)
DEFAULT_INPUT = DEFAULT_GT_HIT_DIR / "per_query_gt_hit_burst_metrics.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_GT_HIT_DIR / "tail_recovery_budget_20260623"
STANDARD_ROUTES = (256, 512, 768, 1024)
EXTRA_BUDGETS = (16, 32, 64, 128, 256, 384, 512, 768)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how much extra pop budget is needed to recover false-easy "
            "recall drops from precomputed GT hit-arrival traces."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def pipe_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(float(part)) for part in text.split("|") if part.strip()]


def finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def q(values: pd.Series | np.ndarray, quantile: float) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, quantile))


def mean(values: pd.Series | np.ndarray) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def pct(mask: pd.Series | np.ndarray) -> float:
    series = pd.Series(mask)
    if len(series) == 0:
        return float("nan")
    return float(series.fillna(False).mean() * 100.0)


def fmt_float(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_int(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.0f}"


def route_rank(route: int) -> int:
    try:
        return STANDARD_ROUTES.index(int(route))
    except ValueError:
        return -1


def smallest_recovery_route(step: float) -> float:
    if not np.isfinite(step):
        return float("nan")
    for route in STANDARD_ROUTES:
        if step <= route:
            return float(route)
    return float("nan")


def per_query_recovery(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for source in rows.itertuples(index=False):
        row = source._asdict()
        route = int(row["route"])
        drop = float(row["drop"])
        gt_k = int(row.get("gt_k", k) or k)
        hit_drop = max(0, int(round(drop * gt_k)))
        hit_steps = pipe_ints(row.get("gt_hit_path_steps", ""))
        before_or_at_route = [step for step in hit_steps if step <= route]
        post_route = [step for step in hit_steps if step > route]
        needed_steps = post_route[:hit_drop]
        trace_covers_drop = hit_drop == 0 or len(post_route) >= hit_drop
        recovery_step = float(needed_steps[-1]) if hit_drop > 0 and trace_covers_drop else float("nan")
        extra_budget = recovery_step - float(route) if np.isfinite(recovery_step) else float("nan")
        recovery_route = smallest_recovery_route(recovery_step)
        current_rank = route_rank(route)
        recovery_rank = route_rank(int(recovery_route)) if np.isfinite(recovery_route) else -1
        bucket_bumps = recovery_rank - current_rank if current_rank >= 0 and recovery_rank >= 0 else float("nan")
        recovered_hit_fraction = (
            min(len(post_route), hit_drop) / hit_drop if hit_drop > 0 else 1.0
        )

        result: dict[str, Any] = {
            "dataset": row["dataset"],
            "qid": int(row["qid"]),
            "cohort": row["cohort"],
            "route": route,
            "drop": drop,
            "hit_drop_at_k": hit_drop,
            "classify_chr_mean": float(row["classify_chr_mean"]),
            "classify_chr_ratio": float(row["classify_chr_ratio"]),
            "feature_first_final_step": float(row["feature_first_final_step"]),
            "gt_hit_unique_count": int(row["gt_hit_unique_count"]),
            "hits_before_or_at_route": len(before_or_at_route),
            "post_route_gt_hits": len(post_route),
            "trace_tail_hit_surplus_vs_drop": len(post_route) - hit_drop,
            "trace_covers_recall_drop": bool(trace_covers_drop),
            "trace_recovered_hit_fraction": recovered_hit_fraction,
            "first_post_route_gt_step": float(post_route[0]) if post_route else float("nan"),
            "first_post_route_extra_budget": float(post_route[0] - route) if post_route else float("nan"),
            "recovery_step_for_drop_hits": recovery_step,
            "extra_budget_to_recover_drop_hits": extra_budget,
            "recovery_step_over_route": recovery_step / route if np.isfinite(recovery_step) else float("nan"),
            "standard_route_recovering_drop": recovery_route,
            "standard_bucket_bumps_needed": bucket_bumps,
            "post_route_steps_covering_drop": "|".join(str(step) for step in needed_steps),
            "gt_hit_path_steps": row.get("gt_hit_path_steps", ""),
        }
        for budget in EXTRA_BUDGETS:
            result[f"recovered_by_route_plus_{budget}"] = (
                bool(np.isfinite(extra_budget) and extra_budget <= float(budget))
                if hit_drop > 0
                else True
            )
            result[f"post_route_gt_hits_within_{budget}"] = sum(
                1 for step in post_route if step <= route + budget
            )
        for candidate in STANDARD_ROUTES:
            result[f"recovered_by_standard_route_{candidate}"] = (
                bool(np.isfinite(recovery_step) and recovery_step <= float(candidate))
                if hit_drop > 0
                else route <= candidate
            )
        out_rows.append(result)
    return pd.DataFrame(out_rows)


def summarize_group(group: pd.DataFrame, prefix: dict[str, Any]) -> dict[str, Any]:
    covered = group[group["trace_covers_recall_drop"]].copy()
    out: dict[str, Any] = dict(prefix)
    out["n"] = int(len(group))
    out["mean_drop"] = mean(group["drop"])
    out["mean_hit_drop_at_k"] = mean(group["hit_drop_at_k"])
    out["total_hit_drop_at_k"] = int(group["hit_drop_at_k"].sum())
    out["pct_hit_drop_1"] = pct(group["hit_drop_at_k"].eq(1))
    out["pct_hit_drop_2"] = pct(group["hit_drop_at_k"].eq(2))
    out["pct_hit_drop_ge3"] = pct(group["hit_drop_at_k"].ge(3))
    out["pct_trace_covers_drop"] = pct(group["trace_covers_recall_drop"])
    out["trace_recovered_hit_fraction_mean"] = mean(group["trace_recovered_hit_fraction"])
    out["gt_hit_unique_count_p50"] = q(group["gt_hit_unique_count"], 0.50)
    out["post_route_gt_hits_p50"] = q(group["post_route_gt_hits"], 0.50)
    out["post_route_gt_hits_p90"] = q(group["post_route_gt_hits"], 0.90)
    out["first_post_route_extra_p50"] = q(group["first_post_route_extra_budget"], 0.50)
    out["first_post_route_extra_p90"] = q(group["first_post_route_extra_budget"], 0.90)
    out["extra_to_recover_p25"] = q(covered["extra_budget_to_recover_drop_hits"], 0.25)
    out["extra_to_recover_p50"] = q(covered["extra_budget_to_recover_drop_hits"], 0.50)
    out["extra_to_recover_p75"] = q(covered["extra_budget_to_recover_drop_hits"], 0.75)
    out["extra_to_recover_p90"] = q(covered["extra_budget_to_recover_drop_hits"], 0.90)
    out["extra_to_recover_mean"] = mean(covered["extra_budget_to_recover_drop_hits"])
    for budget in EXTRA_BUDGETS:
        out[f"pct_recovered_by_route_plus_{budget}"] = pct(
            group[f"recovered_by_route_plus_{budget}"]
        )
    out["pct_one_bucket_bump_enough"] = pct(group["standard_bucket_bumps_needed"].eq(1))
    out["pct_two_or_more_bucket_bumps_needed"] = pct(group["standard_bucket_bumps_needed"].ge(2))
    out["pct_recovered_by_1024"] = pct(group["recovered_by_standard_route_1024"])
    out["pct_not_recovered_by_trace_1024"] = pct(~group["trace_covers_recall_drop"])
    return out


def make_summaries(per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict[str, Any]] = []
    for dataset, group in per_query.groupby("dataset", sort=True):
        dataset_rows.append(summarize_group(group, {"dataset": dataset}))

    route_rows: list[dict[str, Any]] = []
    for (dataset, route), group in per_query.groupby(["dataset", "route"], sort=True):
        route_rows.append(summarize_group(group, {"dataset": dataset, "route": int(route)}))

    loss_size_rows: list[dict[str, Any]] = []
    for (dataset, hit_drop), group in per_query.groupby(["dataset", "hit_drop_at_k"], sort=True):
        loss_size_rows.append(
            summarize_group(group, {"dataset": dataset, "hit_drop_at_k": int(hit_drop)})
        )
    return pd.DataFrame(dataset_rows), pd.DataFrame(route_rows), pd.DataFrame(loss_size_rows)


def route_mix_text(group: pd.DataFrame) -> str:
    counts = group["route"].value_counts().sort_index()
    return ", ".join(f"{int(route)}:{int(count)}" for route, count in counts.items())


def write_markdown(
    output_dir: Path,
    input_csv: Path,
    per_query: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    route_summary: pd.DataFrame,
) -> None:
    lines = [
        "# False-Easy Tail Recovery Budget Analysis",
        "",
        f"Source: `{input_csv}`",
        "",
        "Definition:",
        "",
        "- `hit_drop_at_k = round(recall_drop * k)`, with `k=10`.",
        "- A query is trace-covered when the layer-0 GT hit trace has at least that many GT hits after the assigned route cutoff.",
        "- `extra_to_recover` is the pop-step distance from the assigned route to the post-route GT hit that covers the observed recall drop.",
        "- This is still a pop-trace proxy, not an exact intermediate top-k replay.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | n | routes | hit drop=1 | trace covers | extra p50 | extra p90 | <=+64 | <=+128 | <=+256 | one bucket | by 1024 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    route_mix = {dataset: route_mix_text(group) for dataset, group in per_query.groupby("dataset", sort=True)}
    for row in dataset_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.n} | {route_mix.get(row.dataset, '')} | "
            f"{row.pct_hit_drop_1:.1f}% | {row.pct_trace_covers_drop:.1f}% | "
            f"{fmt_int(row.extra_to_recover_p50)} | {fmt_int(row.extra_to_recover_p90)} | "
            f"{row.pct_recovered_by_route_plus_64:.1f}% | "
            f"{row.pct_recovered_by_route_plus_128:.1f}% | "
            f"{row.pct_recovered_by_route_plus_256:.1f}% | "
            f"{row.pct_one_bucket_bump_enough:.1f}% | "
            f"{row.pct_recovered_by_1024:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Route-Cut Summary",
            "",
            "| dataset | route | n | hit drop=1 | trace covers | first late p50 | recover p50 | recover p90 | <=+128 | <=+256 | one bucket |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in route_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.route} | {row.n} | {row.pct_hit_drop_1:.1f}% | "
            f"{row.pct_trace_covers_drop:.1f}% | {fmt_int(row.first_post_route_extra_p50)} | "
            f"{fmt_int(row.extra_to_recover_p50)} | {fmt_int(row.extra_to_recover_p90)} | "
            f"{row.pct_recovered_by_route_plus_128:.1f}% | "
            f"{row.pct_recovered_by_route_plus_256:.1f}% | "
            f"{row.pct_one_bucket_bump_enough:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The dominant failure is still a small-hit tail miss: most rows lose one GT neighbor, and the trace usually contains enough post-route GT hits to cover the observed drop.",
            "",
            "The recovery budget is not usually tiny. In the larger cohorts, the median extra budget is often hundreds of pops after the assigned route, so a small patience window after a low route would only recover a minority of false-easy losses.",
            "",
            "One standard route bump is a much more direct mitigation for the trace-covered cases. It is almost exactly the pair-gap story in route space: many 256-route false-easy losses recover by 512, while the residual tail needs 768 or 1024.",
            "",
            "## Files",
            "",
            "- `per_query_tail_recovery_budget.csv`: one row per false-easy loss query.",
            "- `tail_recovery_dataset_summary.csv`: dataset-level recovery budget summary.",
            "- `tail_recovery_by_route_summary.csv`: recovery budget by assigned route.",
            "- `tail_recovery_by_hit_drop_summary.csv`: recovery budget by observed hit loss size.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.input_csv)
    per_query = per_query_recovery(rows, int(args.k))
    dataset_summary, route_summary, loss_size_summary = make_summaries(per_query)

    per_query.to_csv(output_dir / "per_query_tail_recovery_budget.csv", index=False)
    dataset_summary.to_csv(output_dir / "tail_recovery_dataset_summary.csv", index=False)
    route_summary.to_csv(output_dir / "tail_recovery_by_route_summary.csv", index=False)
    loss_size_summary.to_csv(output_dir / "tail_recovery_by_hit_drop_summary.csv", index=False)
    write_markdown(output_dir, args.input_csv, per_query, dataset_summary, route_summary)

    print(f"[DONE] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
