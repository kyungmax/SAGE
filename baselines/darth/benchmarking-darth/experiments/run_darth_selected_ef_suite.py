#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from darth_shared_config import SHARED_DATASET_ROOT, find_dataset_spec, shared_index_path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTION_CSV = Path(
    os.environ.get(
        "DARTH_SELECTION_CSV",
        str((THIS_DIR.parent.parent / "configs" / "RESULT_target_recall_ef.csv").resolve()),
    )
).expanduser().resolve()
DEFAULT_BINARY = (THIS_DIR.parent / "build" / "hnsw-test" / "hnsw_test").resolve()

SUMMARY_RE = re.compile(
    r"Index\[M=(?P<m>\d+), efC=(?P<efc>\d+), efS=(?P<efs>\d+)\]"
    r"IndexTime:\s*(?P<index_time>[0-9.]+)s,\s*"
    r"SearchTime:\s*(?P<search_time>[0-9.]+)s,\s*"
    r"TotalTime:\s*(?P<total_time>[0-9.]+)s,\s*"
    r"Avg_Recall@\d+:\s*(?P<avg_recall>[0-9.]+),\s*"
    r"P1_Recall@\d+:\s*(?P<p1_recall>[0-9.]+),\s*"
    r"P5_Recall@\d+:\s*(?P<p5_recall>[0-9.]+)"
)

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


@dataclass(frozen=True)
class SelectionRow:
    dataset: str
    dataset_file: str
    target_recall: float
    selected_ef: int
    verified_recall: float
    verified_qps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DARTH training/evaluation using per-dataset efSearch values from "
            "darth/configs/RESULT_target_recall_ef.csv."
        )
    )
    parser.add_argument("--selection-csv", default=str(DEFAULT_SELECTION_CSV))
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--exclude-datasets", nargs="*", default=[])
    parser.add_argument("--output-root", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--training-query-num", type=int, default=10000)
    parser.add_argument("--logging-interval", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--offline-threads", type=int, default=24)
    parser.add_argument("--online-threads", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-training-csv", action="store_true")
    return parser.parse_args()


def timestamp_for_path() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_output_root(path_value: str) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return (THIS_DIR / "results" / f"darth_selected_ef_suite_{timestamp_for_path()}").resolve()


def load_selection_rows(
    *,
    selection_csv: Path,
    keep_datasets: set[str],
    exclude_datasets: set[str],
) -> list[SelectionRow]:
    rows: list[SelectionRow] = []
    with selection_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            dataset = str(raw["dataset"]).strip()
            if keep_datasets and dataset not in keep_datasets:
                continue
            if dataset in exclude_datasets:
                continue
            selected_ef_text = str(raw.get("selected_ef", "")).strip()
            if not selected_ef_text:
                continue
            rows.append(
                SelectionRow(
                    dataset=dataset,
                    dataset_file=str(raw["dataset_file"]).strip(),
                    target_recall=float(raw["target_recall"]),
                    selected_ef=int(selected_ef_text),
                    verified_recall=float(raw["verified_recall"]),
                    verified_qps=float(raw["verified_qps"]),
                )
            )
    if not rows:
        raise ValueError(f"No dataset rows selected from {selection_csv}")
    return rows


def build_env(num_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[key] = str(int(num_threads))
    return env


def run_command(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return combined


def parse_hnsw_summary(text: str) -> dict[str, float]:
    for line in reversed(text.splitlines()):
        match = SUMMARY_RE.search(line)
        if match:
            payload = match.groupdict()
            return {
                "index_time": float(payload["index_time"]),
                "search_time": float(payload["search_time"]),
                "total_time": float(payload["total_time"]),
                "avg_recall": float(payload["avg_recall"]),
                "p1_recall": float(payload["p1_recall"]),
                "p5_recall": float(payload["p5_recall"]),
            }
    raise ValueError("Could not parse hnsw_test summary line.")


def compute_first_reaching_dists(training_csv: Path, *, target_recall: float, chunksize: int) -> dict[int, float]:
    best_by_qid: dict[int, float] = {}
    for chunk in pd.read_csv(training_csv, usecols=["qid", "dists", "r"], chunksize=chunksize):
        filtered = chunk.loc[chunk["r"] >= target_recall, ["qid", "dists"]]
        if filtered.empty:
            continue
        chunk_best = filtered.groupby("qid", sort=False)["dists"].min()
        for qid, dists in chunk_best.items():
            qid_int = int(qid)
            dists_float = float(dists)
            current = best_by_qid.get(qid_int)
            if current is None or dists_float < current:
                best_by_qid[qid_int] = dists_float
    return best_by_qid


def round_interval(value: float) -> int:
    return int(max(1, round(value)))


def predictor_model_filename(
    *,
    dataset: str,
    m: int,
    efc: int,
    efs: int,
    qs: int,
    k: int,
    n_estimators: int,
    li: int,
) -> str:
    return f"{dataset}_M{m}_efC{efc}_efS{efs}_s{qs}_k{k}_nestim{n_estimators}_li{li}_all_feats.txt"


def train_predictor(
    *,
    training_csv: Path,
    output_model: Path,
    n_estimators: int,
    learning_rate: float,
    num_threads: int,
    seed: int,
) -> dict[str, Any]:
    started = time.time()
    df = pd.read_csv(training_csv, usecols=FEATURE_COLUMNS + [TARGET_COLUMN]).dropna(axis=0)
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
    fit_started = time.time()
    model.fit(x, y)
    fit_seconds = time.time() - fit_started

    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_model))

    return {
        "rows_used": int(len(df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(time.time() - started),
        "model_path": str(output_model.resolve()),
        "model_size_bytes": int(output_model.stat().st_size),
    }


def run_eval_once(
    *,
    binary: Path,
    dataset: str,
    query_num: int,
    k: int,
    m: int,
    efc: int,
    efs: int,
    dataset_root: Path,
    index_path: Path,
    target_recall: float,
    ipi: int,
    mpi: int,
    predictor_model_path: Path,
    out_csv: Path,
    log_path: Path,
    env: dict[str, str],
) -> dict[str, float]:
    cmd = [
        str(binary),
        "--dataset",
        dataset,
        "--M",
        str(m),
        "--efConstruction",
        str(efc),
        "--efSearch",
        str(efs),
        "--query-num",
        str(query_num),
        "--k",
        str(k),
        "--mode",
        "early-stop-testing",
        "--index-filepath",
        str(index_path),
        "--dataset-dir-prefix",
        str(dataset_root) + "/",
        "--target-recall",
        f"{float(target_recall):.6f}",
        "--initial-prediction-interval",
        str(int(ipi)),
        "--min-prediction-interval",
        str(int(mpi)),
        "--query-type",
        "testing",
        "--predictor-model-path",
        str(predictor_model_path),
        "--output",
        str(out_csv),
    ]
    text = run_command(cmd, log_path=log_path, env=env)
    summary = parse_hnsw_summary(text)
    query_num_float = float(query_num)
    qps = query_num_float / summary["search_time"] if summary["search_time"] > 0.0 else 0.0
    return {
        "search_time_s": summary["search_time"],
        "avg_recall": summary["avg_recall"],
        "p1_recall": summary["p1_recall"],
        "p5_recall": summary["p5_recall"],
        "qps": qps,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cleanup_training_csv(training_csv: Path) -> None:
    if not training_csv.exists():
        return
    training_csv.unlink()
    current = training_csv.parent
    while current.name and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    selection_csv: Path,
    output_root: Path,
) -> None:
    def as_float(row: dict[str, Any], key: str) -> float:
        return float(row[key])

    def as_int(row: dict[str, Any], key: str) -> int:
        return int(float(row[key]))

    lines = [
        "# DARTH Selected ef Suite",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Config",
        "",
        f"- Selection CSV: `{selection_csv}`",
        f"- Output root: `{output_root}`",
        "",
        "## Results",
        "",
        "| Dataset | Runtime Target | efS | IPI | MPI | Offline Total (s) | Avg Recall | Gap vs Runtime Target | QPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {runtime_target_recall:.2f} | {selected_ef} | {ipi} | {mpi} | "
            "{offline_total_seconds:.6f} | {avg_recall_mean:.6f} | {recall_gap_vs_runtime_target:+.6f} | "
            "{qps_mean:.6f} |".format(
                dataset=str(row["dataset"]),
                runtime_target_recall=as_float(row, "runtime_target_recall"),
                selected_ef=as_int(row, "selected_ef"),
                ipi=as_int(row, "ipi"),
                mpi=as_int(row, "mpi"),
                offline_total_seconds=as_float(row, "offline_total_seconds"),
                avg_recall_mean=as_float(row, "avg_recall_mean"),
                recall_gap_vs_runtime_target=as_float(row, "recall_gap_vs_runtime_target"),
                qps_mean=as_float(row, "qps_mean"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    selection_csv = Path(args.selection_csv).expanduser().resolve()
    binary = Path(args.binary).expanduser().resolve()
    output_root = resolve_output_root(args.output_root)
    keep_datasets = {str(name) for name in args.datasets if str(name)}
    exclude_datasets = {str(name) for name in args.exclude_datasets if str(name)}

    rows = load_selection_rows(
        selection_csv=selection_csv,
        keep_datasets=keep_datasets,
        exclude_datasets=exclude_datasets,
    )

    logs_root = output_root / "logs"
    training_root = output_root / "training_root"
    interval_root = output_root / "intervals"
    models_root = output_root / "models"
    eval_root = output_root / "eval"
    for path in [logs_root, training_root, interval_root, models_root, eval_root]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "selection_csv": str(selection_csv),
        "binary": str(binary),
        "dataset_root": str(SHARED_DATASET_ROOT),
        "k": int(args.k),
        "m": int(args.m),
        "ef_construction": int(args.ef_construction),
        "training_query_num": int(args.training_query_num),
        "logging_interval": int(args.logging_interval),
        "repeats": int(args.repeats),
        "offline_threads": int(args.offline_threads),
        "online_threads": int(args.online_threads),
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "datasets": [row.dataset for row in rows],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    offline_env = build_env(args.offline_threads)
    online_env = build_env(args.online_threads)
    summary_csv_path = output_root / "summary.csv"
    summary_md_path = output_root / "summary.md"
    summary_rows: list[dict[str, Any]] = load_existing_summary_rows(summary_csv_path) if args.resume else []
    completed_datasets = {str(row.get("dataset", "")).strip() for row in summary_rows}

    for row in rows:
        if args.resume and row.dataset in completed_datasets:
            print(f"[SKIP] {row.dataset} already present in {summary_csv_path}")
            continue
        dataset_spec = find_dataset_spec(row.dataset)
        training_csv = (
            training_root
            / row.dataset
            / f"k{int(args.k)}"
            / f"M{int(args.m)}_efC{int(args.ef_construction)}_efS{int(row.selected_ef)}_qs{int(args.training_query_num)}_li{int(args.logging_interval)}.csv"
        )
        training_log = logs_root / "training_data" / f"{row.dataset}.log"
        interval_json = interval_root / (
            f"{row.dataset}_k{int(args.k)}_efC{int(args.ef_construction)}_efS{int(row.selected_ef)}"
            f"_rt{float(row.target_recall):.2f}_qs{int(args.training_query_num)}.json"
        )
        predictor_summary_json = models_root / f"{row.dataset}.training_summary.json"
        predictor_model_path = models_root / predictor_model_filename(
            dataset=row.dataset,
            m=int(args.m),
            efc=int(args.ef_construction),
            efs=int(row.selected_ef),
            qs=int(args.training_query_num),
            k=int(args.k),
            n_estimators=int(args.n_estimators),
            li=int(args.logging_interval),
        )

        summary_row: dict[str, Any] = {
            "dataset": row.dataset,
            "dataset_file": row.dataset_file,
            "runtime_target_recall": float(row.target_recall),
            "selected_ef": int(row.selected_ef),
            "vanilla_verified_recall": float(row.verified_recall),
            "vanilla_verified_qps": float(row.verified_qps),
            "k": int(args.k),
            "M": int(args.m),
            "efC": int(args.ef_construction),
            "efS": int(row.selected_ef),
            "training_query_num": int(args.training_query_num),
            "eval_query_num": int(dataset_spec.eval_query_num),
            "training_csv": str(training_csv),
            "interval_json": str(interval_json),
            "predictor_model_path": str(predictor_model_path),
        }

        if args.skip_existing and training_csv.exists():
            training_text = training_log.read_text(encoding="utf-8") if training_log.exists() else ""
            training_summary = parse_hnsw_summary(training_text)
        else:
            training_csv.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(binary),
                "--dataset",
                row.dataset,
                "--M",
                str(args.m),
                "--efConstruction",
                str(args.ef_construction),
                "--efSearch",
                str(row.selected_ef),
                "--query-num",
                str(args.training_query_num),
                "--k",
                str(args.k),
                "--mode",
                "early-stop-training",
                "--logging-interval",
                str(args.logging_interval),
                "--index-filepath",
                str(shared_index_path(row.dataset, m=int(args.m), efc=int(args.ef_construction))),
                "--dataset-dir-prefix",
                str(SHARED_DATASET_ROOT) + "/",
                "--query-type",
                "training",
                "--output",
                str(training_csv),
            ]
            training_text = run_command(cmd, log_path=training_log, env=offline_env)
            training_summary = parse_hnsw_summary(training_text)
        summary_row["training_data_index_load_seconds"] = float(training_summary["index_time"])
        summary_row["training_data_search_seconds"] = float(training_summary["search_time"])
        summary_row["training_data_total_seconds"] = float(training_summary["total_time"])
        summary_row["training_data_log"] = str(training_log)

        interval_started = time.time()
        best_by_qid = compute_first_reaching_dists(
            training_csv,
            target_recall=float(row.target_recall),
            chunksize=int(args.chunksize),
        )
        if not best_by_qid:
            raise ValueError(
                f"No queries reached target recall {float(row.target_recall):.2f} in {training_csv}"
            )
        dists = list(best_by_qid.values())
        avg_dists_rt = sum(dists) / len(dists)
        ipi = round_interval(avg_dists_rt / 2.0)
        mpi = round_interval(avg_dists_rt / 10.0)
        if mpi > ipi:
            mpi = ipi
        interval_payload = {
            "avg_dists_rt": float(avg_dists_rt),
            "initial_prediction_interval": int(ipi),
            "min_prediction_interval": int(mpi),
            "queries_reaching_target": int(len(best_by_qid)),
            "target_recall": float(row.target_recall),
            "compute_seconds": float(time.time() - interval_started),
            "training_csv": str(training_csv),
        }
        interval_json.write_text(json.dumps(interval_payload, indent=2) + "\n", encoding="utf-8")
        summary_row["avg_dists_rt"] = float(avg_dists_rt)
        summary_row["queries_reaching_target"] = int(len(best_by_qid))
        summary_row["interval_compute_seconds"] = float(interval_payload["compute_seconds"])
        summary_row["ipi"] = int(ipi)
        summary_row["mpi"] = int(mpi)

        predictor_result = train_predictor(
            training_csv=training_csv,
            output_model=predictor_model_path,
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_threads=int(args.offline_threads),
            seed=int(args.seed),
        )
        predictor_summary = {
            "dataset": row.dataset,
            "runtime_target_recall": float(row.target_recall),
            "ef_search": int(row.selected_ef),
            **predictor_result,
        }
        predictor_summary_json.write_text(json.dumps(predictor_summary, indent=2) + "\n", encoding="utf-8")
        summary_row["predictor_rows_used"] = int(predictor_result["rows_used"])
        summary_row["predictor_fit_seconds"] = float(predictor_result["fit_seconds"])
        summary_row["predictor_total_seconds"] = float(predictor_result["total_seconds"])
        summary_row["predictor_model_size_bytes"] = int(predictor_result["model_size_bytes"])
        summary_row["predictor_training_summary_json"] = str(predictor_summary_json)

        eval_runs: list[dict[str, float]] = []
        log_paths: list[str] = []
        csv_paths: list[str] = []
        for repeat in range(1, int(args.repeats) + 1):
            eval_csv = eval_root / f"{row.dataset}.r{repeat}.csv"
            eval_log = logs_root / "eval" / f"{row.dataset}.r{repeat}.log"
            eval_result = run_eval_once(
                binary=binary,
                dataset=row.dataset,
                query_num=int(dataset_spec.eval_query_num),
                k=int(args.k),
                m=int(args.m),
                efc=int(args.ef_construction),
                efs=int(row.selected_ef),
                dataset_root=SHARED_DATASET_ROOT,
                index_path=shared_index_path(row.dataset, m=int(args.m), efc=int(args.ef_construction)),
                target_recall=float(row.target_recall),
                ipi=int(ipi),
                mpi=int(mpi),
                predictor_model_path=predictor_model_path,
                out_csv=eval_csv,
                log_path=eval_log,
                env=online_env,
            )
            eval_runs.append(eval_result)
            log_paths.append(str(eval_log))
            csv_paths.append(str(eval_csv))

        count = float(len(eval_runs))
        avg_recall_mean = sum(run["avg_recall"] for run in eval_runs) / count
        p1_recall_mean = sum(run["p1_recall"] for run in eval_runs) / count
        p5_recall_mean = sum(run["p5_recall"] for run in eval_runs) / count
        search_time_s_mean = sum(run["search_time_s"] for run in eval_runs) / count
        qps_mean = sum(run["qps"] for run in eval_runs) / count

        summary_row["repeats"] = int(args.repeats)
        summary_row["avg_recall_mean"] = float(avg_recall_mean)
        summary_row["p1_recall_mean"] = float(p1_recall_mean)
        summary_row["p5_recall_mean"] = float(p5_recall_mean)
        summary_row["search_time_s_mean"] = float(search_time_s_mean)
        summary_row["qps_mean"] = float(qps_mean)
        summary_row["recall_gap_vs_runtime_target"] = float(avg_recall_mean - float(row.target_recall))
        summary_row["offline_total_seconds"] = float(
            summary_row["training_data_total_seconds"]
            + summary_row["interval_compute_seconds"]
            + summary_row["predictor_total_seconds"]
        )
        summary_row["eval_log_paths"] = ";".join(log_paths)
        summary_row["eval_csv_paths"] = ";".join(csv_paths)

        summary_rows.append(summary_row)
        write_csv(summary_csv_path, summary_rows)
        write_markdown(
            summary_md_path,
            rows=summary_rows,
            selection_csv=selection_csv,
            output_root=output_root,
        )
        print(
            f"[DONE] {row.dataset} efS={row.selected_ef} rt={row.target_recall:.2f} "
            f"offline={summary_row['offline_total_seconds']:.3f}s "
            f"recall={avg_recall_mean:.6f} qps={qps_mean:.6f}"
        )
        if not args.keep_training_csv:
            cleanup_training_csv(training_csv)
            print(f"[CLEANUP] removed training CSV for {row.dataset}")

    print(f"[SUMMARY] {summary_csv_path}")
    print(f"[SUMMARY] {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
