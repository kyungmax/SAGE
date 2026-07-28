#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from darth_shared_config import (
    get_dataset_specs,
    predictor_model_path,
    shared_index_path,
    shared_interval_json_path,
)


SUMMARY_RE = re.compile(
    r"SearchTime:\s*([0-9.]+)s.*Avg_Recall@\d+:\s*([0-9.]+).*P1_Recall@\d+:\s*([0-9.]+).*P5_Recall@\d+:\s*([0-9.]+)"
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    query_num: int
    achieved_target: float
    ipi: int
    mpi: int
    index_path: Path
    model_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune DARTH configured target-recall per dataset so achieved recall is close to 0.95, "
            "then measure 3-run mean QPS with fixed ipi/mpi."
        )
    )
    parser.add_argument("--binary", required=True, help="Path to hnsw_test binary")
    parser.add_argument("--dataset-root", required=True, help="Processed DARTH dataset root (prefix)")
    parser.add_argument("--out-root", required=True, help="Output directory root")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Achieved recall target to match")
    parser.add_argument(
        "--dataset-targets-csv",
        default="",
        help="Optional CSV with per-dataset target recall. Columns: dataset,target_recall",
    )
    parser.add_argument("--min-config-target", type=float, default=0.80, help="Min configured --target-recall")
    parser.add_argument("--max-config-target", type=float, default=0.99, help="Max configured --target-recall")
    parser.add_argument("--calibration-iters", type=int, default=6, help="Binary-search iterations")
    parser.add_argument("--repeats", type=int, default=3, help="Final benchmark repeats")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--efc", type=int, default=200)
    parser.add_argument("--efs", type=int, default=2000)
    parser.add_argument("--training-query-num", type=int, default=10000)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "dbpedia-openai-1000k-angular",
            "glove-100-angular",
            "glove-200-angular",
            "gist-960-euclidean",
            "nytimes-256-angular",
            "deep-image-96-angular",
            "agnews-mxbai-1024-euclidean",
            "msmarco-v1-openai-ada2-1M-ip",
        ],
    )
    return parser.parse_args()


def load_dataset_targets(args: argparse.Namespace) -> dict[str, float]:
    targets: dict[str, float] = {}
    if not args.dataset_targets_csv:
        return targets
    csv_path = Path(args.dataset_targets_csv).expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            if not dataset:
                continue
            value = row.get("target_recall", row.get("target", ""))
            if value is None or str(value).strip() == "":
                raise ValueError(f"Missing target_recall for dataset={dataset} in {csv_path}")
            targets[dataset] = float(value)
    return targets


def build_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    shared_specs = {spec.dataset: spec for spec in get_dataset_specs(args.datasets)}
    target_map = load_dataset_targets(args)
    specs: list[DatasetSpec] = []
    for ds in [spec.dataset for spec in get_dataset_specs(args.datasets)]:
        achieved_target = float(target_map.get(ds, args.target_recall))
        interval_path = shared_interval_json_path(
            ds,
            k=int(args.k),
            efc=int(args.efc),
            efs=int(args.efs),
            target_recall=achieved_target,
            qs=int(args.training_query_num),
        )
        if not interval_path.exists():
            raise FileNotFoundError(f"Interval JSON not found: {interval_path}")
        interval_payload = json.loads(interval_path.read_text(encoding="utf-8"))
        ipi = int(interval_payload["initial_prediction_interval"])
        mpi = int(interval_payload["min_prediction_interval"])
        qn = int(shared_specs[ds].eval_query_num)
        specs.append(
            DatasetSpec(
                dataset=ds,
                query_num=qn,
                achieved_target=achieved_target,
                ipi=ipi,
                mpi=mpi,
                index_path=shared_index_path(ds, m=int(args.m), efc=int(args.efc)),
                model_path=predictor_model_path(
                    ds,
                    k=int(args.k),
                    m=int(args.m),
                    efc=int(args.efc),
                    efs=int(args.efs),
                    qs=int(args.training_query_num),
                    n_estimators=100,
                    li=2,
                ),
            )
        )
    return specs


def run_once(
    *,
    binary: Path,
    dataset_root: str,
    spec: DatasetSpec,
    configured_target: float,
    k: int,
    m: int,
    efc: int,
    efs: int,
    log_path: Path,
    out_csv: Path,
    env: dict,
) -> dict:
    cmd = [
        str(binary),
        "--dataset",
        spec.dataset,
        "--M",
        str(m),
        "--efConstruction",
        str(efc),
        "--efSearch",
        str(efs),
        "--query-num",
        str(spec.query_num),
        "--k",
        str(k),
        "--mode",
        "early-stop-testing",
        "--index-filepath",
        str(spec.index_path),
        "--dataset-dir-prefix",
        dataset_root,
        "--target-recall",
        f"{configured_target:.6f}",
        "--initial-prediction-interval",
        str(spec.ipi),
        "--min-prediction-interval",
        str(spec.mpi),
        "--query-type",
        "testing",
        "--predictor-model-path",
        str(spec.model_path),
        "--output",
        str(out_csv),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {spec.dataset} at target={configured_target:.6f}, rc={proc.returncode}")

    summary_line = ""
    for line in (proc.stdout or "").splitlines():
        if "SearchTime:" in line:
            summary_line = line
    match = SUMMARY_RE.search(summary_line)
    if not match:
        raise RuntimeError(f"Could not parse summary line for {spec.dataset} at target={configured_target:.6f}")

    search_time_s, avg_recall, p1_recall, p5_recall = map(float, match.groups())
    qps = spec.query_num / search_time_s if search_time_s > 0 else 0.0
    return {
        "configured_target_recall": configured_target,
        "avg_recall": avg_recall,
        "p1_recall": p1_recall,
        "p5_recall": p5_recall,
        "search_time_s": search_time_s,
        "qps": qps,
        "summary_line": summary_line,
    }


def choose_best(candidates: list[dict], achieved_target: float) -> dict:
    # Min absolute recall gap first, then higher qps.
    return sorted(
        candidates,
        key=lambda x: (abs(x["avg_recall"] - achieved_target), -x["qps"]),
    )[0]


def main() -> int:
    args = parse_args()
    binary = Path(args.binary).resolve()
    dataset_root = str(Path(args.dataset_root).resolve()) + "/"
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cal_log_root = out_root / "calibration_logs"
    cal_csv_root = out_root / "calibration_per_query"
    eval_log_root = out_root / "eval_logs"
    eval_csv_root = out_root / "eval_per_query"
    for p in [cal_log_root, cal_csv_root, eval_log_root, eval_csv_root]:
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[var] = "1"

    specs = build_specs(args)
    dataset_tag = f"{len(specs)}datasets"

    final_rows: list[dict] = []
    trace_rows: list[dict] = []

    for spec in specs:
        cache: dict[float, dict] = {}

        def eval_target(t: float, label: str) -> dict:
            t = round(float(t), 6)
            if t in cache:
                return cache[t]
            result = run_once(
                binary=binary,
                dataset_root=dataset_root,
                spec=spec,
                configured_target=t,
                k=args.k,
                m=args.m,
                efc=args.efc,
                efs=args.efs,
                log_path=cal_log_root / f"{spec.dataset}.{label}.log",
                out_csv=cal_csv_root / f"{spec.dataset}.{label}.csv",
                env=env,
            )
            cache[t] = result
            return result

        # Always evaluate configured target 0.95 as anchor
        anchor = eval_target(spec.achieved_target, "anchor")
        low = float(args.min_config_target)
        high = float(args.max_config_target)

        # Decide search direction around anchor
        tested = [anchor]
        if anchor["avg_recall"] > spec.achieved_target:
            # Need lower configured target
            hi = float(spec.achieved_target)
            lo = low
            for i in range(args.calibration_iters):
                mid = round((lo + hi) / 2.0, 6)
                res = eval_target(mid, f"bs_down_{i+1}")
                tested.append(res)
                if res["avg_recall"] >= spec.achieved_target:
                    hi = mid
                else:
                    lo = mid
        elif anchor["avg_recall"] < spec.achieved_target:
            # Need higher configured target
            lo = float(spec.achieved_target)
            hi = high
            for i in range(args.calibration_iters):
                mid = round((lo + hi) / 2.0, 6)
                res = eval_target(mid, f"bs_up_{i+1}")
                tested.append(res)
                if res["avg_recall"] >= spec.achieved_target:
                    hi = mid
                else:
                    lo = mid

        chosen = choose_best(tested, spec.achieved_target)
        configured_target = float(chosen["configured_target_recall"])

        trace_rows.append(
            {
                "dataset": spec.dataset,
                "target_achieved": spec.achieved_target,
                "selected_configured_target_recall": configured_target,
                "selected_avg_recall_single_run": chosen["avg_recall"],
                "tested": tested,
            }
        )

        # Final repeated evaluation at selected configured target
        rep_results = []
        for r in range(1, args.repeats + 1):
            rep = run_once(
                binary=binary,
                dataset_root=dataset_root,
                spec=spec,
                configured_target=configured_target,
                k=args.k,
                m=args.m,
                efc=args.efc,
                efs=args.efs,
                log_path=eval_log_root / f"{spec.dataset}.r{r}.log",
                out_csv=eval_csv_root / f"{spec.dataset}.r{r}.csv",
                env=env,
            )
            rep_results.append(rep)

        avg_recall_mean = statistics.mean(x["avg_recall"] for x in rep_results)
        p1_recall_mean = statistics.mean(x["p1_recall"] for x in rep_results)
        p5_recall_mean = statistics.mean(x["p5_recall"] for x in rep_results)
        search_time_s_mean = statistics.mean(x["search_time_s"] for x in rep_results)
        qps_mean = statistics.mean(x["qps"] for x in rep_results)

        final_row = {
            "dataset": spec.dataset,
            "query_num": spec.query_num,
            "k": args.k,
            "M": args.m,
            "efC": args.efc,
            "efS": args.efs,
            "ipi": spec.ipi,
            "mpi": spec.mpi,
            "repeats": args.repeats,
            "achieved_recall_target": spec.achieved_target,
            "configured_target_recall": configured_target,
            "avg_recall_mean": avg_recall_mean,
            "recall_gap_vs_target": avg_recall_mean - spec.achieved_target,
            "p1_recall_mean": p1_recall_mean,
            "p5_recall_mean": p5_recall_mean,
            "search_time_s_mean": search_time_s_mean,
            "qps_mean": qps_mean,
        }
        final_rows.append(final_row)
        print(
            f"[DONE] {spec.dataset}: configured_target={configured_target:.6f}, "
            f"avg_recall_mean={avg_recall_mean:.6f}, qps_mean={qps_mean:.6f}"
        )

    summary_csv = out_root / f"summary_darth_{dataset_tag}_recall_matched_qps{args.repeats}runs.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    trace_json = out_root / "calibration_trace.json"
    trace_json.write_text(json.dumps(trace_rows, indent=2))

    print(f"[SUMMARY] {summary_csv}")
    print(f"[TRACE] {trace_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
