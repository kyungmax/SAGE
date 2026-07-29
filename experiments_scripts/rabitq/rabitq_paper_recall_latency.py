#!/usr/bin/env python3
"""Plot RaBitQ vs RaBitQ+SAGE recall/QPS and recall/latency curves."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[2]
BASE_METHOD = "RaBitQ"
ADAPTIVE_METHOD = "RaBitQ+SAGE"
METHODS = [BASE_METHOD, ADAPTIVE_METHOD]

ALIASES = {
    "agnews": "agnews-mxbai-1024-euclidean",
    "cohere": "cohere-768-angular",
    "msspacev": "msspacev-100M-i8-euclidean",
    "youtube": "youtube-15M-angular",
}
COLORS = {BASE_METHOD: "#2563eb", ADAPTIVE_METHOD: "#dc2626"}
MARKERS = {BASE_METHOD: "o", ADAPTIVE_METHOD: "s"}
LINESTYLES = {BASE_METHOD: "-", ADAPTIVE_METHOD: "--"}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
})


def slug(name: str) -> str:
    return name.replace("-", "_").replace("+", "plus")


def canonical_dataset(name: str) -> str:
    return ALIASES.get(name, name)


def default_plot_dir() -> str:
    return str(Path(os.environ.get("RABITQ_PLOT_DIR", REPO / "artifacts" / "plots")).expanduser())


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            method = raw["method"]
            if method not in METHODS:
                raise ValueError(f"Unsupported method {method!r}; expected one of {METHODS}")
            qps = float(raw["qps"])
            latency = raw.get("latency_per_query_ms")
            threads = raw.get("num_threads") or raw.get("query_threads") or raw.get("threads") or 0
            rows.append({
                "dataset": raw["dataset"],
                "method": method,
                "ef": int(raw["ef"]),
                "recall": float(raw["recall"]),
                "qps": qps,
                "latency_per_query_ms": float(latency) if latency else 1000.0 / qps,
                "num_threads": int(threads),
            })
    return rows


def frontier(rows: list[dict]) -> list[dict]:
    best_by_recall: dict[float, dict] = {}
    for row in rows:
        recall = round(float(row["recall"]), 10)
        if recall not in best_by_recall or float(row["qps"]) > float(best_by_recall[recall]["qps"]):
            best_by_recall[recall] = row

    ordered = [best_by_recall[key] for key in sorted(best_by_recall)]
    out: list[dict] = []
    best_qps = -1.0
    for row in reversed(ordered):
        if float(row["qps"]) > best_qps:
            out.append(row)
            best_qps = float(row["qps"])
    return list(reversed(out))


def interpolate_qps(rows: list[dict], targets: np.ndarray) -> np.ndarray:
    fr = frontier(rows)
    if len(fr) < 2:
        return np.full_like(targets, np.nan, dtype=np.float64)
    x = np.asarray([row["recall"] for row in fr], dtype=np.float64)
    y = np.asarray([row["qps"] for row in fr], dtype=np.float64)
    order = np.argsort(x)
    return np.interp(targets, x[order], y[order])


def method_rows(rows: list[dict], dataset: str, method: str) -> list[dict]:
    return [row for row in rows if row["dataset"] == dataset and row["method"] == method]


def iso_targets(base: list[dict], adaptive: list[dict], count: int) -> np.ndarray:
    base_frontier = frontier(base)
    adaptive_frontier = frontier(adaptive)
    if len(base_frontier) < 2 or len(adaptive_frontier) < 2:
        return np.asarray([], dtype=np.float64)
    lo = max(min(row["recall"] for row in base_frontier), min(row["recall"] for row in adaptive_frontier))
    hi = min(max(row["recall"] for row in base_frontier), max(row["recall"] for row in adaptive_frontier))
    if hi <= lo:
        return np.asarray([], dtype=np.float64)
    return np.linspace(lo, hi, int(count))


def save_figure(fig, paths: list[Path]) -> list[Path]:
    for output in paths:
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_recall_qps(rows: list[dict], dataset: str, out_dir: Path, suffix: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    for method in METHODS:
        raw = sorted(method_rows(rows, dataset, method), key=lambda row: row["ef"])
        if not raw:
            continue
        pareto = frontier(raw)
        ax.scatter(
            [row["recall"] for row in raw],
            [row["qps"] for row in raw],
            color=COLORS[method],
            marker=MARKERS[method],
            alpha=0.25,
            s=22,
            label=f"{method} raw",
        )
        ax.plot(
            [row["recall"] for row in pareto],
            [row["qps"] for row in pareto],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            linewidth=2.2,
            markersize=4.5,
            label=f"{method} Pareto",
        )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_title(f"{dataset}: recall-QPS Pareto frontier")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#d1d5db", ncol=2)
    stem = f"{slug(dataset)}_recall_qps_pareto_{suffix}"
    return save_figure(fig, [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"])


def plot_recall_latency(rows: list[dict], dataset: str, out_dir: Path, suffix: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    for method in METHODS:
        raw = sorted(method_rows(rows, dataset, method), key=lambda row: row["ef"])
        if not raw:
            continue
        pareto = frontier(raw)
        ax.scatter(
            [row["recall"] for row in raw],
            [row["latency_per_query_ms"] for row in raw],
            color=COLORS[method],
            marker=MARKERS[method],
            alpha=0.25,
            s=22,
            label=f"{method} raw",
        )
        ax.plot(
            [row["recall"] for row in pareto],
            [row["latency_per_query_ms"] for row in pareto],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            linewidth=2.2,
            markersize=4.5,
            label=f"{method} Pareto",
        )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Search time per query (ms)")
    ax.set_title(f"{dataset}: recall-search time Pareto frontier")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#d1d5db", ncol=2)
    stem = f"{slug(dataset)}_recall_latency_pareto_{suffix}"
    return save_figure(fig, [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"])


def plot_iso_qps(rows: list[dict], dataset: str, out_dir: Path, suffix: str, points: int) -> list[Path]:
    base = method_rows(rows, dataset, BASE_METHOD)
    adaptive = method_rows(rows, dataset, ADAPTIVE_METHOD)
    targets = iso_targets(base, adaptive, points)
    if targets.size == 0:
        return []
    base_qps = interpolate_qps(base, targets)
    adaptive_qps = interpolate_qps(adaptive, targets)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(targets, base_qps, label=BASE_METHOD, color=COLORS[BASE_METHOD], marker=MARKERS[BASE_METHOD], linestyle=LINESTYLES[BASE_METHOD], linewidth=2.1, markersize=3.6)
    ax.plot(targets, adaptive_qps, label=ADAPTIVE_METHOD, color=COLORS[ADAPTIVE_METHOD], marker=MARKERS[ADAPTIVE_METHOD], linestyle=LINESTYLES[ADAPTIVE_METHOD], linewidth=2.1, markersize=3.6)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_title(f"{dataset}: iso-recall QPS Pareto frontier")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d1d5db")
    stem = f"{slug(dataset)}_iso_recall_qps_pareto_{suffix}"
    return save_figure(fig, [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"])


def plot_iso_latency(rows: list[dict], dataset: str, out_dir: Path, suffix: str, points: int) -> list[Path]:
    base = method_rows(rows, dataset, BASE_METHOD)
    adaptive = method_rows(rows, dataset, ADAPTIVE_METHOD)
    targets = iso_targets(base, adaptive, points)
    if targets.size == 0:
        return []
    base_latency = 1000.0 / interpolate_qps(base, targets)
    adaptive_latency = 1000.0 / interpolate_qps(adaptive, targets)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(targets, base_latency, label=BASE_METHOD, color=COLORS[BASE_METHOD], marker=MARKERS[BASE_METHOD], linestyle=LINESTYLES[BASE_METHOD], linewidth=2.1, markersize=3.6)
    ax.plot(targets, adaptive_latency, label=ADAPTIVE_METHOD, color=COLORS[ADAPTIVE_METHOD], marker=MARKERS[ADAPTIVE_METHOD], linestyle=LINESTYLES[ADAPTIVE_METHOD], linewidth=2.1, markersize=3.6)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Search time per query (ms)")
    ax.set_title(f"{dataset}: iso-recall search time Pareto frontier")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#d1d5db")
    stem = f"{slug(dataset)}_iso_recall_latency_pareto_{suffix}"
    return save_figure(fig, [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"])


def plot_search_time_speedup(rows: list[dict], dataset: str, out_dir: Path, suffix: str, points: int) -> list[Path]:
    base = method_rows(rows, dataset, BASE_METHOD)
    adaptive = method_rows(rows, dataset, ADAPTIVE_METHOD)
    targets = iso_targets(base, adaptive, points)
    if targets.size == 0:
        return []
    base_qps = interpolate_qps(base, targets)
    adaptive_qps = interpolate_qps(adaptive, targets)
    speedup = adaptive_qps / base_qps

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.axhline(1.0, color="#6b7280", linewidth=1.0, linestyle=":")
    ax.plot(targets, speedup, color="#059669", marker="o", markersize=3.5, linewidth=2.0)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Search-time speedup over RaBitQ")
    ax.set_title(f"{dataset}: iso-recall search-time speedup")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    stem = f"{slug(dataset)}_iso_recall_search_time_speedup_{suffix}"
    return save_figure(fig, [out_dir / f"{stem}.png", out_dir / f"{stem}.svg"])


def write_summary(rows: list[dict], out_path: Path, points: int) -> None:
    datasets: list[str] = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    fieldnames = [
        "dataset",
        "iso_recall",
        "rabitq_qps",
        "sage_qps",
        "rabitq_latency_per_query_ms",
        "sage_latency_per_query_ms",
        "search_time_speedup",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            base = method_rows(rows, dataset, BASE_METHOD)
            adaptive = method_rows(rows, dataset, ADAPTIVE_METHOD)
            targets = iso_targets(base, adaptive, points)
            if targets.size == 0:
                continue
            base_qps = interpolate_qps(base, targets)
            adaptive_qps = interpolate_qps(adaptive, targets)
            for recall, bq, aq in zip(targets, base_qps, adaptive_qps):
                writer.writerow({
                    "dataset": dataset,
                    "iso_recall": float(recall),
                    "rabitq_qps": float(bq),
                    "sage_qps": float(aq),
                    "rabitq_latency_per_query_ms": float(1000.0 / bq),
                    "sage_latency_per_query_ms": float(1000.0 / aq),
                    "search_time_speedup": float(aq / bq),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Pareto recall/QPS and recall/search-time curves from a RaBitQ paper sweep CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default=default_plot_dir())
    parser.add_argument("--suffix", default="paper")
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--iso-points", type=int, default=25)
    parser.add_argument("--plot-kind", choices=("all", "qps", "latency"), default="all")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    if args.datasets:
        wanted = {canonical_dataset(name) for name in args.datasets}
        rows = [row for row in rows if row["dataset"] in wanted]
    if not rows:
        raise ValueError("No matching rows found in the input CSV.")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[str] = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])

    created: list[Path] = []
    for dataset in datasets:
        if args.plot_kind in ("all", "qps"):
            created += plot_recall_qps(rows, dataset, out_dir, args.suffix)
            created += plot_iso_qps(rows, dataset, out_dir, args.suffix, int(args.iso_points))
        if args.plot_kind in ("all", "latency"):
            created += plot_recall_latency(rows, dataset, out_dir, args.suffix)
            created += plot_iso_latency(rows, dataset, out_dir, args.suffix, int(args.iso_points))
            created += plot_search_time_speedup(rows, dataset, out_dir, args.suffix, int(args.iso_points))

    summary = out_dir / f"iso_recall_search_time_summary_{args.suffix}.csv"
    write_summary(rows, summary, int(args.iso_points))
    created.append(summary)
    for output in created:
        print(output)


if __name__ == "__main__":
    main()
