#!/usr/bin/env python3
"""Build the experiment-script combined recall-vs-latency figure.

This script only consumes existing result artifacts:

- hnswlib experiments_scripts main sweep CSV
- Faiss experiments_scripts main sweep CSV
- precomputed Ada-EF/DARTH JSON outputs

Ada-EF and DARTH are not rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_SCRIPT_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if parent.name == "experiments_scripts"
)
DETECTED_PROJECT_ROOT = EXPERIMENTS_SCRIPT_ROOT.parent


def _find_default_project_root() -> Path:
    env_root = os.environ.get("SAGE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (DETECTED_PROJECT_ROOT, *DETECTED_PROJECT_ROOT.parents):
        if (candidate / ".git").exists() or (
            (candidate / "experiments_scripts").is_dir()
            and (candidate / "final_experiments").is_dir()
        ):
            return candidate
    return DETECTED_PROJECT_ROOT


PROJECT_ROOT = _find_default_project_root()
EXPERIMENT_ROOT = PROJECT_ROOT / "final_experiments" / "combined_recall_latency_six_m32_efc500"
DEFAULT_HNSW_CSV = (
    EXPERIMENT_ROOT
    / "hnswlib_main_qps_latency_total6_m32_efc500_ncal100_offline24_online1"
    / "final"
    / "main_qps_latency_sweep.csv"
)
DEFAULT_HNSW_RECOMMENDED_CSV = (
    EXPERIMENT_ROOT
    / "hnswlib_main_qps_latency_total6_m32_efc500_ncal100_offline24_online1"
    / "final"
    / "offline_recommended_efsearch.csv"
)
DEFAULT_FAISS_CSV = (
    EXPERIMENT_ROOT
    / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1"
    / "final"
    / "main_qps_latency_sweep.csv"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs_experiment_scripts_ncal100"
DEFAULT_OUTPUT_STEM = "combined_recall_latency_six_m32_efc500_experiment_scripts_ncal100"
DEFAULT_BASELINE_ROOT = (
    EXPERIMENT_ROOT / "baseline_results_m32_efc500_target095_efs1000_20260603"
)

DEFAULT_TARGET_RECALL = 0.95
DEFAULT_RECOMMENDED_CUMULATIVE_GAIN_EPS = 0.001
RECALL_K = 10
HIGH_RECALL_ZOOM_FLOOR_OFFSET = 0.06
DATASET_Y_ZOOM_FLOOR = {
    "msmarco-v1-openai-ada2-full-ip": 0.975,
    "sift-100M-euclidean": 0.93,
    "deep-100M": 0.95,
}

DATASETS = (
    ("nytimes-256-angular", "NYTimes"),
    ("glove-100-angular", "GloVe-100"),
    ("cohere-768-angular", "Cohere-768"),
    ("msmarco-v1-openai-ada2-full-ip", "MSMARCO"),
    ("sift-100M-euclidean", "SIFT-100M"),
    ("deep-100M", "DEEP-100M"),
)
DATASET_LABEL = dict(DATASETS)

SHORT_TO_DATASET = {
    "nytimes": "nytimes-256-angular",
    "glove-100": "glove-100-angular",
    "cohere": "cohere-768-angular",
    "msmarco": "msmarco-v1-openai-ada2-full-ip",
    "sift-100M": "sift-100M-euclidean",
    "deep-100M": "deep-100M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hnsw-csv", type=Path, default=DEFAULT_HNSW_CSV)
    parser.add_argument("--hnsw-recommended-csv", type=Path, default=DEFAULT_HNSW_RECOMMENDED_CSV)
    parser.add_argument("--faiss-csv", type=Path, default=DEFAULT_FAISS_CSV)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument(
        "--recommended-cumulative-gain-eps",
        type=float,
        default=DEFAULT_RECOMMENDED_CUMULATIVE_GAIN_EPS,
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def load_curve(path: Path, backend: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "k" in df.columns:
        df = df.loc[pd.to_numeric(df["k"], errors="coerce") == RECALL_K].copy()
    df = df.loc[df["dataset"].astype(str).isin(DATASET_LABEL)].copy()
    df["backend"] = backend
    df["method_label"] = df["method"].astype(str).map(
        {"Vanilla": f"{backend} Vanilla", "Ours": f"{backend} Ours"}
    )
    df["latency_ms"] = pd.to_numeric(df["latency_per_query_mean_ms"], errors="coerce")
    missing_latency = df["latency_ms"].isna()
    if missing_latency.any():
        df.loc[missing_latency, "latency_ms"] = 1000.0 / pd.to_numeric(
            df.loc[missing_latency, "qps"], errors="coerce"
        )
    keep = [
        "dataset",
        "backend",
        "method",
        "method_label",
        "ef",
        "recall",
        "latency_ms",
        "qps",
    ]
    out = df[keep].copy()
    out["ef"] = out["ef"].astype(int)
    out["recall"] = out["recall"].astype(float)
    out["latency_ms"] = out["latency_ms"].astype(float)
    out["qps"] = out["qps"].astype(float)
    out["target_recall"] = ""
    out["status"] = "ok"
    out["note"] = ""
    out["source_csv"] = str(path)
    return out


def parse_cmd_arg(cmd: Iterable[object], flag: str) -> str | None:
    items = [str(item) for item in cmd]
    try:
        idx = items.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(items):
        return None
    return items[idx + 1]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_adaef_points(baseline_root: Path, target_recall: float) -> pd.DataFrame:
    rows: list[dict] = []
    for short, dataset in SHORT_TO_DATASET.items():
        online = baseline_root / "adaef/results" / short / "online.wrapper.json"
        skip = baseline_root / "adaef/results" / short / "skip.json"
        if skip.exists():
            payload = load_json(skip)
            rows.append(
                {
                    "dataset": dataset,
                    "backend": "Ada-EF",
                    "method": "Ada-EF",
                    "method_label": f"Ada-EF target {target_recall:.2f}",
                    "ef": "",
                    "recall": math.nan,
                    "latency_ms": math.nan,
                    "qps": math.nan,
                    "target_recall": target_recall,
                    "status": payload.get("status", "skipped"),
                    "note": payload.get("reason", "skipped"),
                    "source_csv": str(skip),
                }
            )
            continue
        if not online.exists():
            continue
        payload = load_json(online)
        metrics = payload.get("backend_result", {}).get("metrics", {})
        achieved_recall = float(metrics["achieved_recall"])
        rows.append(
            {
                "dataset": dataset,
                "backend": "Ada-EF",
                "method": "Ada-EF",
                "method_label": f"Ada-EF target {target_recall:.2f}",
                "ef": metrics.get("weighted_average_ef", ""),
                "recall": achieved_recall,
                "latency_ms": float(metrics["mean_query_latency_ms"]),
                "qps": float(metrics["qps"]),
                "target_recall": target_recall,
                "status": payload.get("status", "ok"),
                "note": f"target={target_recall:.2f}, actual={achieved_recall:.4f}",
                "source_csv": str(online),
            }
        )
    return pd.DataFrame(rows)


def load_darth_points(baseline_root: Path, target_recall: float) -> pd.DataFrame:
    rows: list[dict] = []
    for short, dataset in SHORT_TO_DATASET.items():
        online = baseline_root / "darth/results" / short / "online.wrapper.json"
        if not online.exists():
            continue
        payload = load_json(online)
        metrics = payload.get("metrics", {})
        query_num_raw = parse_cmd_arg(payload.get("cmd", []), "--query-num")
        query_num = float(query_num_raw) if query_num_raw else math.nan
        search_time_s = float(metrics["search_time_s"])
        latency_ms = (search_time_s / query_num) * 1000.0
        achieved_recall = float(metrics["achieved_recall"])
        rows.append(
            {
                "dataset": dataset,
                "backend": "DARTH",
                "method": "DARTH",
                "method_label": f"DARTH target {target_recall:.2f}",
                "ef": int(metrics.get("ef_search", 1000)),
                "recall": achieved_recall,
                "latency_ms": latency_ms,
                "qps": query_num / search_time_s,
                "target_recall": target_recall,
                "status": payload.get("status", "ok"),
                "note": f"target={target_recall:.2f}, actual={achieved_recall:.4f}",
                "source_csv": str(online),
            }
        )
    return pd.DataFrame(rows)


def interpolate_log_latency(points: list[tuple[float, float]], recall: float) -> float | None:
    pts = sorted(points)
    if not pts or recall < pts[0][0] or recall > pts[-1][0]:
        return None
    for (r0, l0), (r1, l1) in zip(pts, pts[1:]):
        if r0 <= recall <= r1:
            if r1 == r0:
                return l0
            t = (recall - r0) / (r1 - r0)
            return math.exp(math.log(l0) + t * (math.log(l1) - math.log(l0)))
    return pts[-1][1]


def load_offline_recommended(hnsw: pd.DataFrame, path: Path) -> pd.DataFrame | None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None

    offline = pd.read_csv(resolved)
    if "k" in offline.columns:
        offline = offline.loc[pd.to_numeric(offline["k"], errors="coerce") == RECALL_K].copy()
    if offline.empty or "recommended_ef" not in offline.columns:
        return None

    rows: list[dict] = []
    for source_row in offline.itertuples(index=False):
        dataset = str(getattr(source_row, "dataset"))
        recommended_ef = int(getattr(source_row, "recommended_ef"))
        cur = hnsw.loc[hnsw["dataset"].astype(str).eq(dataset)]
        ours = cur.loc[
            (cur["method"].astype(str) == "Ours")
            & (cur["ef"].astype(int) == int(recommended_ef))
        ]
        if ours.empty:
            continue
        ours_row = ours.iloc[0]
        vanilla = cur.loc[cur["method"].astype(str) == "Vanilla"].sort_values("ef")
        vanilla_points = [
            (float(row.recall), float(row.latency_ms)) for row in vanilla.itertuples()
        ]
        recall = float(ours_row["recall"])
        vanilla_latency = interpolate_log_latency(vanilla_points, recall)
        row = {
            "dataset": dataset,
            "dataset_label": DATASET_LABEL.get(dataset, dataset),
            "recommended_ef": recommended_ef,
            "recall": recall,
            "latency_ms": float(ours_row["latency_ms"]),
            "vanilla_iso_latency_ms": math.nan if vanilla_latency is None else vanilla_latency,
            "iso_latency_speedup": (
                math.nan if vanilla_latency is None else vanilla_latency / float(ours_row["latency_ms"])
            ),
            "recommendation_source": "offline_calibration_proxy",
            "source_csv": str(resolved),
        }
        for column in [
            "offline_predicted_recall",
            "offline_cumulative_recall",
            "max_cumulative_recall",
            "remaining_cumulative_recall_gain",
            "previous_step_cumulative_gain",
            "recommendation_eps",
            "selection_rule",
            "calibration_query_count",
            "calibration_lid_pool_count",
            "curve_csv",
            "cache_path",
        ]:
            if hasattr(source_row, column):
                row[column] = getattr(source_row, column)
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_recommended(hnsw: pd.DataFrame, cumulative_gain_eps: float) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, _label in DATASETS:
        cur = hnsw.loc[hnsw["dataset"] == dataset]
        vanilla = cur.loc[cur["method"] == "Vanilla"].sort_values("ef")
        ours = cur.loc[cur["method"] == "Ours"].sort_values("ef")
        if ours.empty:
            continue

        vanilla_points = [
            (float(row.recall), float(row.latency_ms)) for row in vanilla.itertuples()
        ]
        best_observed_recall = float(ours["recall"].max())
        previous_recall: float | None = None
        selected: dict | None = None

        for row in ours.itertuples():
            recall = float(row.recall)
            remaining_gain = best_observed_recall - recall
            previous_step_gain = (
                math.nan if previous_recall is None else recall - previous_recall
            )
            previous_recall = recall
            if remaining_gain > float(cumulative_gain_eps):
                continue

            vanilla_latency = interpolate_log_latency(vanilla_points, recall)
            selected = {
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "recommended_ef": int(row.ef),
                "recall": recall,
                "latency_ms": float(row.latency_ms),
                "vanilla_iso_latency_ms": (
                    math.nan if vanilla_latency is None else vanilla_latency
                ),
                "iso_latency_speedup": (
                    math.nan
                    if vanilla_latency is None
                    else vanilla_latency / float(row.latency_ms)
                ),
                "remaining_cumulative_recall_gain": remaining_gain,
                "previous_step_recall_gain": previous_step_gain,
                "selection_rule": (
                    "first ef where best observed hnswlib Ours Recall@10 minus "
                    f"current Recall@10 <= {float(cumulative_gain_eps):g}"
                ),
            }
            break

        if selected is None:
            row = ours.iloc[-1]
            recall = float(row["recall"])
            vanilla_latency = interpolate_log_latency(vanilla_points, recall)
            selected = {
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "recommended_ef": int(row["ef"]),
                "recall": recall,
                "latency_ms": float(row["latency_ms"]),
                "vanilla_iso_latency_ms": (
                    math.nan if vanilla_latency is None else vanilla_latency
                ),
                "iso_latency_speedup": (
                    math.nan
                    if vanilla_latency is None
                    else vanilla_latency / float(row["latency_ms"])
                ),
                "remaining_cumulative_recall_gain": best_observed_recall - recall,
                "previous_step_recall_gain": math.nan,
                "selection_rule": "fallback last ef; no saturation point found",
            }
        rows.append(selected)
    return pd.DataFrame(rows)


def save_combined_csv(
    *,
    parts: list[pd.DataFrame],
    recommended: pd.DataFrame,
    combined_csv: Path,
    recommended_csv: Path,
) -> None:
    combined = pd.concat(parts, ignore_index=True)
    combined["is_hnswlib_recommended"] = False
    for row in recommended.itertuples():
        mask = (
            (combined["dataset"] == row.dataset)
            & (combined["backend"] == "hnswlib")
            & (combined["method"] == "Ours")
            & (combined["ef"].astype(str) == str(row.recommended_ef))
        )
        combined.loc[mask, "is_hnswlib_recommended"] = True
    combined.to_csv(combined_csv, index=False)
    recommended.to_csv(recommended_csv, index=False)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    color: str,
    linestyle: str,
    marker: str,
    markerfacecolor: str | None = None,
) -> None:
    if df.empty:
        return
    df = df.sort_values("ef")
    ax.plot(
        df["latency_ms"],
        df["recall"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.4,
        markerfacecolor=color if markerfacecolor is None else markerfacecolor,
        markeredgecolor=color,
        markeredgewidth=0.8,
        linewidth=1.6,
        alpha=0.96,
    )


def annotate_actual(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7.2,
        color=color,
        ha="left",
        va="bottom",
    )


def render(
    *,
    hnsw: pd.DataFrame,
    faiss: pd.DataFrame,
    adaef: pd.DataFrame,
    darth: pd.DataFrame,
    recommended: pd.DataFrame,
    target_recall: float,
    out_png: Path,
    out_pdf: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharey=False)
    axes_flat = axes.ravel()

    style = {
        ("hnswlib", "Vanilla"): {
            "label": "hnswlib Vanilla",
            "color": "#6baed6",
            "linestyle": ":",
            "marker": "o",
            "markerfacecolor": "white",
        },
        ("hnswlib", "Ours"): {
            "label": "hnswlib Ours",
            "color": "#08519c",
            "linestyle": "-",
            "marker": "o",
            "markerfacecolor": "#08519c",
        },
        ("FAISS", "Vanilla"): {
            "label": "FAISS Vanilla",
            "color": "#fdae6b",
            "linestyle": ":",
            "marker": "^",
            "markerfacecolor": "white",
        },
        ("FAISS", "Ours"): {
            "label": "FAISS Ours",
            "color": "#d94801",
            "linestyle": "--",
            "marker": "^",
            "markerfacecolor": "#d94801",
        },
    }

    def panel_ylim(dataset: str, panel_points: pd.DataFrame) -> tuple[float, float]:
        floor = DATASET_Y_ZOOM_FLOOR.get(
            dataset,
            float(target_recall) - HIGH_RECALL_ZOOM_FLOOR_OFFSET,
        )
        zoom_recalls = panel_points.loc[
            panel_points["recall"] >= floor, "recall"
        ].astype(float).tolist()
        baseline_recalls = pd.concat(
            [
                adaef.loc[adaef["dataset"] == dataset, ["recall"]],
                darth.loc[darth["dataset"] == dataset, ["recall"]],
            ],
            ignore_index=True,
        )["recall"].dropna().astype(float).tolist()
        zoom_recalls.extend(baseline_recalls)
        if floor <= float(target_recall):
            zoom_recalls.append(float(target_recall))
        if not zoom_recalls:
            zoom_recalls = panel_points["recall"].dropna().astype(float).tolist()
        ymin = min(zoom_recalls)
        ymax = max(zoom_recalls)
        yspan = max(0.01, ymax - ymin)
        ypad = max(0.0025, yspan * 0.08)
        return max(0.0, ymin - ypad), min(1.002, ymax + ypad)

    for ax, (dataset, title) in zip(axes_flat, DATASETS):
        panel_points = pd.concat(
            [
                hnsw.loc[hnsw["dataset"] == dataset, ["latency_ms", "recall"]],
                faiss.loc[faiss["dataset"] == dataset, ["latency_ms", "recall"]],
                adaef.loc[adaef["dataset"] == dataset, ["latency_ms", "recall"]],
                darth.loc[darth["dataset"] == dataset, ["latency_ms", "recall"]],
            ],
            ignore_index=True,
        ).dropna()
        if panel_points.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, "missing data", ha="center", va="center")
            continue

        ymin, ymax = panel_ylim(dataset, panel_points)
        if ymin <= float(target_recall) <= ymax:
            ax.axhline(
                float(target_recall),
                color="#999999",
                linestyle=":",
                linewidth=1.0,
                zorder=0,
                label=(
                    f"target recall {float(target_recall):.2f}"
                    if dataset == DATASETS[0][0]
                    else "_nolegend_"
                ),
            )

        for source, backend in ((hnsw, "hnswlib"), (faiss, "FAISS")):
            cur = source.loc[source["dataset"] == dataset]
            for method in ("Vanilla", "Ours"):
                spec = style[(backend, method)]
                plot_curve(
                    ax,
                    cur.loc[cur["method"] == method],
                    spec["label"],
                    spec["color"],
                    spec["linestyle"],
                    spec["marker"],
                    spec["markerfacecolor"],
                )

        rec_row = recommended.loc[recommended["dataset"] == dataset]
        if not rec_row.empty:
            rr = rec_row.iloc[0]
            ax.scatter(
                [float(rr["latency_ms"])],
                [float(rr["recall"])],
                marker="*",
                s=72,
                facecolor="#ffd54f",
                edgecolor="#111111",
                linewidth=0.8,
                zorder=6,
                label="recommended hnswlib Ours",
            )
            ax.annotate(
                f"ef={int(rr['recommended_ef'])}",
                xy=(float(rr["latency_ms"]), float(rr["recall"])),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=6.8,
                color="#111111",
                ha="left",
                va="top",
            )

        ada = adaef.loc[(adaef["dataset"] == dataset) & adaef["recall"].notna()]
        if not ada.empty:
            row = ada.iloc[0]
            x = float(row["latency_ms"])
            y = float(row["recall"])
            ax.scatter(
                [x],
                [y],
                marker="D",
                s=42,
                facecolor="#d62728",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
                label=f"Ada-EF target {float(target_recall):.2f}",
            )
            annotate_actual(ax, x, y, f"Ada {y:.3f}", "#b01d23")
        elif dataset == "sift-100M-euclidean":
            ax.text(
                0.03,
                0.05,
                "Ada-EF skipped (L2)",
                transform=ax.transAxes,
                fontsize=7.0,
                color="#b01d23",
                ha="left",
                va="bottom",
            )

        dar = darth.loc[(darth["dataset"] == dataset) & darth["recall"].notna()]
        if not dar.empty:
            row = dar.iloc[0]
            x = float(row["latency_ms"])
            y = float(row["recall"])
            ax.scatter(
                [x],
                [y],
                marker="s",
                s=46,
                facecolor="#2ca02c",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
                label=f"DARTH target {float(target_recall):.2f}",
            )
            annotate_actual(ax, x, y, f"DARTH {y:.3f}", "#1b7f1b")

        xmin = float(panel_points["latency_ms"].min())
        xmax = float(panel_points["latency_ms"].max())
        ax.set_xscale("log")
        ax.set_xlim(max(0.01, xmin * 0.7), xmax * 1.45)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.grid(True, which="major", color="#e2e2e2", linewidth=0.7)
        ax.grid(True, which="minor", axis="x", color="#efefef", linewidth=0.45)
        ax.tick_params(axis="both", length=3)

    for ax in axes[:, 0]:
        ax.set_ylabel("Recall@10")
    for ax in axes[-1, :]:
        ax.set_xlabel("Search latency (ms/query, log scale)")

    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Recall-latency tradeoff, M=32 efConstruction=500 k=10",
        y=1.055,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def write_readme(
    *,
    output_dir: Path,
    hnsw_csv: Path,
    hnsw_recommended_csv: Path,
    faiss_csv: Path,
    baseline_root: Path,
    combined_csv: Path,
    recommended_csv: Path,
    out_png: Path,
    out_pdf: Path,
) -> None:
    lines = [
        "# Six-Dataset Recall-Latency Benchmark Plot",
        "",
        "Source policy:",
        "- hnswlib and Faiss curves are read from experiment-local experiments_scripts sweep CSVs.",
        "- Ada-EF and DARTH are not rerun; precomputed JSON outputs are reused.",
        "",
        "Inputs:",
        f"- hnswlib CSV: `{hnsw_csv}`",
        f"- hnswlib offline recommended efSearch CSV: `{hnsw_recommended_csv}` if present; otherwise measured-curve fallback",
        f"- Faiss CSV: `{faiss_csv}`",
        f"- Ada-EF/DARTH root: `{baseline_root}`",
        "",
        "Outputs:",
        f"- `{combined_csv.name}`",
        f"- `{recommended_csv.name}`",
        f"- `{out_png.name}`",
        f"- `{out_pdf.name}`",
        "",
        "Regenerate:",
        "```bash",
        f"python3 {SCRIPT_PATH} \\",
        f"  --hnsw-csv {hnsw_csv} \\",
        f"  --faiss-csv {faiss_csv} \\",
        f"  --baseline-root {baseline_root} \\",
        f"  --output-dir {output_dir}",
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    hnsw_csv = require_file(args.hnsw_csv, "hnswlib experiments_scripts CSV")
    hnsw_recommended_csv = Path(args.hnsw_recommended_csv).expanduser().resolve()
    faiss_csv = require_file(args.faiss_csv, "Faiss experiments_scripts CSV")
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    if not baseline_root.exists():
        raise FileNotFoundError(f"Missing Ada-EF/DARTH baseline root: {baseline_root}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(args.output_stem)
    out_png = output_dir / f"{output_stem}.png"
    out_pdf = output_dir / f"{output_stem}.pdf"
    combined_csv = output_dir / f"{output_stem}.csv"
    recommended_csv = output_dir / "recommended_hnswlib_ours_efsearch.csv"

    hnsw = load_curve(hnsw_csv, "hnswlib")
    faiss = load_curve(faiss_csv, "FAISS")
    adaef = load_adaef_points(baseline_root, target_recall=float(args.target_recall))
    darth = load_darth_points(baseline_root, target_recall=float(args.target_recall))
    recommended = load_offline_recommended(hnsw, hnsw_recommended_csv)
    if recommended is None:
        recommended = compute_recommended(
            hnsw,
            cumulative_gain_eps=float(args.recommended_cumulative_gain_eps),
        )
    save_combined_csv(
        parts=[hnsw, faiss, adaef, darth],
        recommended=recommended,
        combined_csv=combined_csv,
        recommended_csv=recommended_csv,
    )
    render(
        hnsw=hnsw,
        faiss=faiss,
        adaef=adaef,
        darth=darth,
        recommended=recommended,
        target_recall=float(args.target_recall),
        out_png=out_png,
        out_pdf=out_pdf,
    )
    write_readme(
        output_dir=output_dir,
        hnsw_csv=hnsw_csv,
        hnsw_recommended_csv=hnsw_recommended_csv,
        faiss_csv=faiss_csv,
        baseline_root=baseline_root,
        combined_csv=combined_csv,
        recommended_csv=recommended_csv,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    print(f"[WRITE] {out_png}")
    print(f"[WRITE] {out_pdf}")
    print(f"[WRITE] {combined_csv}")
    print(f"[WRITE] {recommended_csv}")
    print(f"[WRITE] {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
