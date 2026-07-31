#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
BENCHMARKING_DARTH_ROOT = THIS_DIR.parent.parent


FEATURE_COLUMNS = [
    "step",
    "dists",
    "inserts",
    "first_nn_dist",
    "nn_dist",
    "furthest_dist",
    "avg_dist",
    "variance",
    "percentile_25",
    "percentile_50",
    "percentile_75",
]

TARGET_COLUMN = "r"
INTERVAL_RE = re.compile(
    r"^(?P<dataset>.+)_k(?P<k>\d+)_efC(?P<efc>\d+)_efS(?P<efs>\d+)_rt(?P<rt>[0-9.]+)_qs(?P<qs>\d+)\.json$"
)
QS_RE = re.compile(r"_qs(?P<qs>\d+)_li(?P<li>\d+)\.(csv|txt)$")


@dataclass(frozen=True)
class IntervalSpec:
    dataset: str
    k: int
    efc: int
    efs: int
    qs: int
    interval_path: Path | None


def _default_darth_index_root() -> Path:
    if os.environ.get("DARTH_INDEX_ROOT"):
        return Path(os.environ["DARTH_INDEX_ROOT"]).expanduser().resolve()
    if os.environ.get("INDEX_ROOT"):
        return (Path(os.environ["INDEX_ROOT"]).expanduser().resolve() / "DARTH").resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "faiss").exists() and (parent / "experiments_scripts").exists():
            return Path(os.environ.get("DARTH_INDEX_ROOT", str(parent / "index/DARTH"))).resolve()
    return Path(os.environ["DARTH_INDEX_ROOT"]).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DARTH LightGBM predictor models for all datasets that already have "
            "interval JSON files and/or existing training logs."
        )
    )
    parser.add_argument(
        "--interval-root",
        default=str(_default_darth_index_root() / "intervals"),
        help="Directory containing *_k*_efC*_efS*_rt*_qs*.json interval files.",
    )
    parser.add_argument(
        "--training-root",
        default=str(_default_darth_index_root() / "et_training_data"),
        help="Directory containing shared per-dataset training CSVs.",
    )
    parser.add_argument(
        "--legacy-training-root",
        default="",
        help=(
            "Optional legacy training root (txt/csv). Defaults to "
            "$DARTH_ROOT/et_training_data/early-stop-training when DARTH_ROOT is set."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Model output directory. Defaults to $DARTH_ROOT/predictor_models/darth "
            "if DARTH_ROOT is set, else ./predictor_models/darth"
        ),
    )
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional subset of dataset names to train.")
    parser.add_argument("--exclude-datasets", nargs="*", default=[], help="Optional dataset names to skip.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=2000)
    parser.add_argument("--li", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional explicit summary output path. Defaults to <output-root>/training_summary_from_intervals.json",
    )
    return parser.parse_args()


def resolve_output_root(path_value: str) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    darth_root = os.environ.get("DARTH_ROOT", "")
    if darth_root:
        return (Path(darth_root).expanduser().resolve() / "predictor_models" / "darth").resolve()
    return (BENCHMARKING_DARTH_ROOT / "predictor_models" / "darth").resolve()


def resolve_legacy_training_root(path_value: str) -> Path | None:
    if path_value:
        return Path(path_value).expanduser().resolve()
    darth_root = os.environ.get("DARTH_ROOT", "")
    if not darth_root:
        return None
    candidate = Path(darth_root).expanduser().resolve() / "et_training_data" / "early-stop-training"
    return candidate if candidate.exists() else None


def collect_interval_specs(
    interval_root: Path,
    *,
    keep_datasets: set[str],
    exclude_datasets: set[str],
    target_k: int,
    target_efc: int,
    target_efs: int,
) -> list[IntervalSpec]:
    specs: list[IntervalSpec] = []
    for path in sorted(interval_root.glob("*.json")):
        match = INTERVAL_RE.match(path.name)
        if not match:
            continue
        dataset = str(match.group("dataset"))
        k = int(match.group("k"))
        efc = int(match.group("efc"))
        efs = int(match.group("efs"))
        qs = int(match.group("qs"))
        if k != target_k or efc != target_efc or efs != target_efs:
            continue
        if keep_datasets and dataset not in keep_datasets:
            continue
        if dataset in exclude_datasets:
            continue
        specs.append(
            IntervalSpec(
                dataset=dataset,
                k=k,
                efc=efc,
                efs=efs,
                qs=qs,
                interval_path=path.resolve(),
            )
        )
    return specs


def parse_qs_from_training_filename(path: Path, li: int) -> int | None:
    match = QS_RE.search(path.name)
    if not match:
        return None
    if int(match.group("li")) != int(li):
        return None
    return int(match.group("qs"))


def collect_training_only_specs(
    *,
    training_root: Path,
    legacy_training_root: Path | None,
    existing_datasets: set[str],
    keep_datasets: set[str],
    exclude_datasets: set[str],
    target_k: int,
    target_efc: int,
    target_efs: int,
    li: int,
    m: int,
) -> list[IntervalSpec]:
    grouped: dict[str, tuple[int, Path]] = {}
    patterns: list[Path] = [
        training_root / "*" / f"k{target_k}" / f"M{m}_efC{target_efc}_efS{target_efs}_qs*_li{li}.csv",
        training_root / "*" / f"k{target_k}" / f"M{m}_efC{target_efc}_efS{target_efs}_qs*_li{li}.txt",
    ]
    if legacy_training_root is not None:
        patterns.extend(
            [
                legacy_training_root / "*" / f"k{target_k}" / f"M{m}_efC{target_efc}_efS{target_efs}_qs*_li{li}.csv",
                legacy_training_root / "*" / f"k{target_k}" / f"M{m}_efC{target_efc}_efS{target_efs}_qs*_li{li}.txt",
            ]
        )
    for pattern in patterns:
        for path_text in sorted(glob.glob(str(pattern))):
            path = Path(path_text)
            dataset = path.parent.parent.name
            if dataset in existing_datasets:
                continue
            if keep_datasets and dataset not in keep_datasets:
                continue
            if dataset in exclude_datasets:
                continue
            qs = parse_qs_from_training_filename(path, li)
            if qs is None:
                continue
            previous = grouped.get(dataset)
            if previous is None or qs > previous[0]:
                grouped[dataset] = (qs, path.resolve())
    specs: list[IntervalSpec] = []
    for dataset, (qs, _path) in sorted(grouped.items()):
        specs.append(
            IntervalSpec(
                dataset=dataset,
                k=target_k,
                efc=target_efc,
                efs=target_efs,
                qs=qs,
                interval_path=None,
            )
        )
    return specs


def resolve_training_file(
    *,
    training_root: Path,
    legacy_training_root: Path | None,
    spec: IntervalSpec,
    m: int,
    li: int,
) -> Path:
    roots = [training_root]
    if legacy_training_root is not None:
        roots.append(legacy_training_root)

    exact_candidates: list[Path] = []
    for root in roots:
        exact_candidates.extend(
            [
                root / spec.dataset / f"k{spec.k}" / f"M{m}_efC{spec.efc}_efS{spec.efs}_qs{spec.qs}_li{li}.csv",
                root / spec.dataset / f"k{spec.k}" / f"M{m}_efC{spec.efc}_efS{spec.efs}_qs{spec.qs}_li{li}.txt",
            ]
        )
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate.resolve()

    fuzzy_matches: list[Path] = []
    for root in roots:
        pattern_csv = root / spec.dataset / f"k{spec.k}" / f"M{m}_efC{spec.efc}_efS{spec.efs}_qs*_li{li}.csv"
        pattern_txt = root / spec.dataset / f"k{spec.k}" / f"M{m}_efC{spec.efc}_efS{spec.efs}_qs*_li{li}.txt"
        fuzzy_matches.extend(sorted(Path(path_text) for path_text in glob.glob(str(pattern_csv))))
        fuzzy_matches.extend(sorted(Path(path_text) for path_text in glob.glob(str(pattern_txt))))
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0].resolve()
    if not fuzzy_matches:
        raise FileNotFoundError(
            f"Training log not found for dataset={spec.dataset}, k={spec.k}, efC={spec.efc}, efS={spec.efs}"
        )
    raise ValueError(f"Multiple training logs for dataset={spec.dataset}: {[str(path) for path in fuzzy_matches]}")


def model_filename(dataset: str, m: int, efc: int, efs: int, qs: int, k: int, n_estimators: int, li: int) -> str:
    return f"{dataset}_M{m}_efC{efc}_efS{efs}_s{qs}_k{k}_nestim{n_estimators}_li{li}_all_feats.txt"


def ensure_columns(df: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def train_one(
    *,
    spec: IntervalSpec,
    training_csv: Path,
    output_root: Path,
    m: int,
    li: int,
    n_estimators: int,
    learning_rate: float,
    num_threads: int,
    seed: int,
) -> dict:
    start_time = time.time()
    df = pd.read_csv(training_csv, usecols=FEATURE_COLUMNS + [TARGET_COLUMN])
    ensure_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN], training_csv)
    df = df.dropna(axis=0)
    if df.empty:
        raise ValueError(f"No usable rows after dropna for {training_csv}")
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=int(seed),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        n_jobs=int(num_threads),
        verbose=-1,
    )
    fit_start = time.time()
    model.fit(x, y)
    fit_seconds = time.time() - fit_start

    output_root.mkdir(parents=True, exist_ok=True)
    model_path = output_root / model_filename(
        dataset=spec.dataset,
        m=int(m),
        efc=int(spec.efc),
        efs=int(spec.efs),
        qs=int(spec.qs),
        k=int(spec.k),
        n_estimators=int(n_estimators),
        li=int(li),
    )
    model.booster_.save_model(str(model_path))
    total_seconds = time.time() - start_time

    return {
        "dataset": spec.dataset,
        "k": spec.k,
        "ef_construction": spec.efc,
        "ef_search": spec.efs,
        "query_sample_size": spec.qs,
        "interval_json": str(spec.interval_path) if spec.interval_path is not None else "",
        "training_csv": str(training_csv),
        "rows_used": int(len(df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(total_seconds),
        "model_path": str(model_path.resolve()),
        "model_size_bytes": int(model_path.stat().st_size),
    }


def main() -> int:
    args = parse_args()
    interval_root = Path(args.interval_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    legacy_training_root = resolve_legacy_training_root(args.legacy_training_root)
    output_root = resolve_output_root(args.output_root)
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else (output_root / "training_summary_from_intervals.json").resolve()
    )

    keep_datasets = {str(name) for name in args.datasets if str(name)}
    exclude_datasets = {str(name) for name in args.exclude_datasets if str(name)}
    specs = collect_interval_specs(
        interval_root=interval_root,
        keep_datasets=keep_datasets,
        exclude_datasets=exclude_datasets,
        target_k=int(args.k),
        target_efc=int(args.ef_construction),
        target_efs=int(args.ef_search),
    )
    extra_specs = collect_training_only_specs(
        training_root=training_root,
        legacy_training_root=legacy_training_root,
        existing_datasets={spec.dataset for spec in specs},
        keep_datasets=keep_datasets,
        exclude_datasets=exclude_datasets,
        target_k=int(args.k),
        target_efc=int(args.ef_construction),
        target_efs=int(args.ef_search),
        li=int(args.li),
        m=int(args.m),
    )
    specs.extend(extra_specs)
    if not specs:
        raise ValueError("No target datasets found from intervals or training logs.")

    results: list[dict] = []
    errors: list[dict] = []
    for spec in specs:
        try:
            training_csv = resolve_training_file(
                training_root=training_root,
                legacy_training_root=legacy_training_root,
                spec=spec,
                m=int(args.m),
                li=int(args.li),
            )
            result = train_one(
                spec=spec,
                training_csv=training_csv,
                output_root=output_root,
                m=int(args.m),
                li=int(args.li),
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.learning_rate),
                num_threads=int(args.num_threads),
                seed=int(args.seed),
            )
            results.append(result)
            print(
                f"[OK] {spec.dataset} -> {result['model_path']} "
                f"(rows={result['rows_used']}, fit={result['fit_seconds']:.2f}s)"
            )
        except Exception as exc:  # pylint: disable=broad-except
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"dataset": spec.dataset, "error": message})
            print(f"[ERR] {spec.dataset}: {message}")

    payload = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "params": {
            "k": int(args.k),
            "m": int(args.m),
            "ef_construction": int(args.ef_construction),
            "ef_search": int(args.ef_search),
            "li": int(args.li),
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_threads": int(args.num_threads),
            "seed": int(args.seed),
        },
        "interval_root": str(interval_root),
        "training_root": str(training_root),
        "legacy_training_root": str(legacy_training_root) if legacy_training_root is not None else "",
        "output_root": str(output_root),
        "num_success": len(results),
        "num_failed": len(errors),
        "results": results,
        "errors": errors,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[SUMMARY] {summary_json}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
