#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OURS_ROOT = HERE.parent

EXPERIMENT_ROOT = (
    OURS_ROOT
    / "experiments"
    / "sage_plan_20260601"
    / "offline_probe_representativeness_n100_20260608"
)
RECALL_CSV = EXPERIMENT_ROOT / "hide_node_recall_curve" / "recall_curves.csv"
RECALL_ERROR_CSV = EXPERIMENT_ROOT / "hide_node_recall_curve" / "curve_error_vs_online.csv"
CFR_VALUES_CSV = EXPERIMENT_ROOT / "probe_cfr_vs_online_ef1024" / "probe_cfr_values_ef1024.csv"
CFR_ERROR_CSV = EXPERIMENT_ROOT / "probe_cfr_vs_online_ef1024" / "probe_cfr_vs_online_summary_ef1024.csv"
ONLINE_CFR_CSV = (
    OURS_ROOT
    / "analysis"
    / "query_cfr_distribution_latest6_20260604"
    / "latest6_query_cfr_values_ef1024.csv"
)
OUTPUT_DIR = HERE / "generated_figures"

SOURCE_LID = "random_index_hide_node_lid_stratified_1k_pool10k"
SOURCE_RANDOM = "random_index_hide_node_uniform_1k"

DATASETS = [
    ("glove-100-angular", "GloVe-100"),
    ("cohere-768-angular", "Cohere-768"),
]

LABELS = {
    "online": "Online queries",
    SOURCE_LID: "LID-stratified 100",
    SOURCE_RANDOM: "Uniform random 100",
}

COLORS = {
    "online": "#222222",
    SOURCE_LID: "#1f77b4",
    SOURCE_RANDOM: "#ff7f0e",
}

LINESTYLES = {
    "online": "-",
    SOURCE_LID: "-",
    SOURCE_RANDOM: "--",
}

MARKERS = {
    "online": "o",
    SOURCE_LID: "s",
    SOURCE_RANDOM: "^",
}


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    if arr.size == 0:
        return arr, arr
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, y


def metric_value(df: pd.DataFrame, dataset: str, source: str, column: str) -> float | None:
    rows = df[(df["Dataset_Stem"].eq(dataset)) & (df["Query_Source"].eq(source))]
    if rows.empty:
        return None
    return float(rows.iloc[0][column])


def add_metric_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.03,
        0.05,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )


def plot_recall(
    ax: plt.Axes,
    recall: pd.DataFrame,
    recall_error: pd.DataFrame,
    dataset: str,
    dataset_label: str,
) -> None:
    dataset_recall = recall[recall["Dataset_Stem"].eq(dataset)].copy()
    for source in ["online", SOURCE_LID, SOURCE_RANDOM]:
        source_recall = dataset_recall[dataset_recall["Query_Source"].eq(source)].sort_values("ef")
        if source_recall.empty:
            continue
        ax.plot(
            source_recall["ef"],
            source_recall["mean_recall"],
            label=LABELS[source],
            color=COLORS[source],
            linestyle=LINESTYLES[source],
            marker=MARKERS[source],
            linewidth=2.0,
            markersize=4.5,
        )

    lid_mae = metric_value(recall_error, dataset, SOURCE_LID, "MAE_vs_Online")
    rand_mae = metric_value(recall_error, dataset, SOURCE_RANDOM, "MAE_vs_Online")
    if lid_mae is not None and rand_mae is not None:
        add_metric_box(ax, f"MAE vs online\nLID {lid_mae:.4f} | rand {rand_mae:.4f}")

    ax.set_title(f"{dataset_label}: recall curve", fontsize=12, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels(["64", "128", "256", "512", "1024", "2048"])
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Mean recall@10")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.5)

    yvals = dataset_recall["mean_recall"].dropna()
    ymin = max(0.0, float(yvals.min()) - 0.015)
    ymax = min(1.005, float(yvals.max()) + 0.008)
    ax.set_ylim(ymin, ymax)


def plot_cfr(
    ax: plt.Axes,
    online_cfr: pd.DataFrame,
    probe_cfr: pd.DataFrame,
    cfr_error: pd.DataFrame,
    dataset: str,
    dataset_label: str,
) -> None:
    plot_values: list[pd.Series] = []

    online_values = online_cfr[online_cfr["dataset_stem"].eq(dataset)]["cfr_mean"]
    x, y = ecdf(online_values)
    if x.size:
        plot_values.append(pd.Series(x))
        ax.step(
            x,
            y,
            where="post",
            label=LABELS["online"],
            color=COLORS["online"],
            linestyle=LINESTYLES["online"],
            linewidth=2.1,
        )

    dataset_probe = probe_cfr[probe_cfr["Dataset_Stem"].eq(dataset)].copy()
    for source in [SOURCE_LID, SOURCE_RANDOM]:
        source_values = dataset_probe[dataset_probe["Query_Source"].eq(source)]["cfr_mean"]
        x, y = ecdf(source_values)
        if not x.size:
            continue
        plot_values.append(pd.Series(x))
        ax.step(
            x,
            y,
            where="post",
            label=LABELS[source],
            color=COLORS[source],
            linestyle=LINESTYLES[source],
            linewidth=2.0,
        )

    lid_ks = metric_value(cfr_error, dataset, SOURCE_LID, "cfr_ks")
    rand_ks = metric_value(cfr_error, dataset, SOURCE_RANDOM, "cfr_ks")
    if lid_ks is not None and rand_ks is not None:
        add_metric_box(ax, f"KS vs online\nLID {lid_ks:.3f} | rand {rand_ks:.3f}")

    ax.set_title(f"{dataset_label}: CFR ECDF", fontsize=12, fontweight="bold")
    ax.set_xlabel("CFR mean at efSearch=1024")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)

    if plot_values:
        combined = pd.concat(plot_values, ignore_index=True)
        lo = float(combined.min())
        hi = float(combined.max())
        pad = max((hi - lo) * 0.06, 0.01)
        ax.set_xlim(lo - pad, hi + pad)


def main() -> None:
    recall = pd.read_csv(RECALL_CSV)
    recall_error = pd.read_csv(RECALL_ERROR_CSV)
    probe_cfr = pd.read_csv(CFR_VALUES_CSV)
    cfr_error = pd.read_csv(CFR_ERROR_CSV)
    online_cfr = pd.read_csv(ONLINE_CFR_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), constrained_layout=False)
    for row, (dataset, dataset_label) in enumerate(DATASETS):
        plot_recall(axes[row, 0], recall, recall_error, dataset, dataset_label)
        plot_cfr(axes[row, 1], online_cfr, probe_cfr, cfr_error, dataset, dataset_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Online Query Approximation from 100 Probes",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.045,
        "Recall uses hide-node probe evaluation; CFR uses the non-hide-node path metric comparison.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.075, 0.995, 0.95), h_pad=2.0, w_pad=2.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "glove_cohere_recall_cfr_n100_panel.png"
    pdf_path = OUTPUT_DIR / "glove_cohere_recall_cfr_n100_panel.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
