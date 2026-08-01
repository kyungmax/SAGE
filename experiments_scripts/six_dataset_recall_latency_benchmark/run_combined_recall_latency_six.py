#!/usr/bin/env python3
"""Run experiment-script sweeps and build the combined six-dataset plot.

Default behavior is conservative:

- run hnswlib only if its final CSV is missing
- run Faiss only if its final CSV is missing
- always rebuild the combined CSV/PNG/PDF from local Ada-EF/DARTH result copies

Use `--mode all --force-recompute` to intentionally rerun completed sweep cells.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENTS_SCRIPT_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if parent.name == "experiments_scripts"
)
SAGE_ROOT = EXPERIMENTS_SCRIPT_ROOT.parent
EXPERIMENT_ROOT = SAGE_ROOT / "final_experiments" / "combined_recall_latency_six_m32_efc500"

HNSW_RUNNER = EXPERIMENTS_SCRIPT_ROOT / "hnswlib" / "run_main_qps_latency_sweep.py"
FAISS_RUNNER = EXPERIMENTS_SCRIPT_ROOT / "faiss" / "run_faiss_vanilla_ours_ef_sweep.py"
COMBINE_SCRIPT = SCRIPT_PATH.parent / "build_combined_recall_latency_six.py"

RUN_DIR = EXPERIMENT_ROOT / "run"
LOG_DIR = RUN_DIR / "logs"
STATUS_TSV = RUN_DIR / "status.tsv"

HNSW_ROOT = EXPERIMENT_ROOT / "hnswlib_main_qps_latency_total6_m32_efc500_ncal100_offline24_online1"
HNSW_RUN_ROOT = HNSW_ROOT / "run"
HNSW_FINAL_DIR = HNSW_ROOT / "final"
HNSW_CSV = HNSW_FINAL_DIR / "main_qps_latency_sweep.csv"
HNSW_RECOMMENDED_CSV = HNSW_FINAL_DIR / "offline_recommended_efsearch.csv"

FAISS_ROOT = EXPERIMENT_ROOT / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1"
FAISS_RUN_ROOT = FAISS_ROOT / "run"
FAISS_FINAL_DIR = FAISS_ROOT / "final"
FAISS_CSV = FAISS_FINAL_DIR / "main_qps_latency_sweep.csv"

BASELINE_ROOT = EXPERIMENT_ROOT / "baseline_results_m32_efc500_target095_efs1000_20260603"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs_experiment_scripts_ncal100"
HNSW_DRILLDOWN_SUBDIR = "hnswlib_easy_medium_hard_drilldown_pseudogt4096_groupdef1024"

def _find_default_project_root() -> Path:
    env_root = os.environ.get("HNSW_PLAYGROUND_ROOT") or os.environ.get("SAGE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (SAGE_ROOT, *SAGE_ROOT.parents):
        if (candidate / "datasets").exists():
            return candidate
    return SAGE_ROOT


PROJECT_ROOT = _find_default_project_root()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get(
        "FAISS_PYTHON_PATH",
        str(PROJECT_ROOT / "faiss/build_sage_avx512/faiss/python"),
    )
)
DEFAULT_FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_INDEX_DIR",
        str(PROJECT_ROOT / "index/m32_efc500_target095_adaef_darth_efs1000_20260603/index"),
    )
)
DEFAULT_PYTHON = Path(os.environ.get("SAGE_PYTHON", sys.executable))

LATEST6_DATASETS = (
    "nytimes-256-angular.hdf5",
    "glove-100-angular.hdf5",
    "cohere-768-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "sift-100M-euclidean.hdf5",
    "deep-100M.hdf5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("missing", "all", "combine-only"),
        default="missing",
        help="Which steps to run. 'missing' skips sweeps whose final CSV already exists.",
    )
    parser.add_argument("--datasets", default=",".join(LATEST6_DATASETS))
    parser.add_argument("--ef-sweep", default="")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--offline-num-threads", type=int, default=24)
    parser.add_argument("--online-num-threads", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--mixed-threshold-mode", default="paper_floor_half")
    parser.add_argument("--mixed-bucket-count", type=int, default=4)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--faiss-index-root", type=Path, default=DEFAULT_FAISS_INDEX_ROOT)
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python executable used to run the hnswlib/Faiss/build scripts.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--enable-hnsw-drilldown",
        action="store_true",
        help="Run the hnswlib-only easy/medium/hard drilldown piggyback sweep.",
    )
    parser.add_argument(
        "--hnsw-drilldown-output-dir",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/hnswlib_easy_medium_hard_drilldown_pseudogt4096_groupdef1024.",
    )
    parser.add_argument("--hnsw-drilldown-pseudo-gt-ef", type=int, default=4096)
    parser.add_argument("--hnsw-drilldown-group-def-ef", type=int, default=1024)
    parser.add_argument(
        "--hnsw-drilldown-ef-sweep",
        default="",
        help="Optional drilldown efSearch sweep. Defaults to the hnswlib main ef sweep.",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--skip-faiss-sweep",
        action="store_true",
        help="Do not run the Faiss sweep step. The combined plot still expects an existing Faiss CSV.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Do not rebuild the combined recall-latency plot. Useful for hnswlib-only drilldown backfills.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if int(args.num_calibration_queries) != 100:
        raise ValueError("This final experiment is fixed to calibration n=100.")
    if int(args.offline_num_threads) < 1 or int(args.online_num_threads) < 1:
        raise ValueError("thread counts must be positive.")
    if int(args.measured_runs) < 1 or int(args.warmup_runs) < 0:
        raise ValueError("invalid warmup/measured run counts.")
    if int(args.hnsw_drilldown_pseudo_gt_ef) < 1:
        raise ValueError("--hnsw-drilldown-pseudo-gt-ef must be positive.")
    if int(args.hnsw_drilldown_group_def_ef) < 1:
        raise ValueError("--hnsw-drilldown-group-def-ef must be positive.")
    if int(args.hnsw_drilldown_pseudo_gt_ef) < int(args.hnsw_drilldown_group_def_ef):
        raise ValueError("--hnsw-drilldown-pseudo-gt-ef must be >= --hnsw-drilldown-group-def-ef.")
    if args.hnsw_drilldown_output_dir is None:
        args.hnsw_drilldown_output_dir = Path(args.output_dir).expanduser() / HNSW_DRILLDOWN_SUBDIR
    return args


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def append_status(step: str, status: str, detail: str = "") -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not STATUS_TSV.exists():
        STATUS_TSV.write_text("timestamp\tstep\tstatus\tdetail\n", encoding="utf-8")
    with STATUS_TSV.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{now_label()}\t{step}\t{status}\t{detail}\n")


def shell_join(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_cmd(step: str, cmd: Sequence[object], *, dry_run: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step}.log"
    command_line = shell_join(cmd)
    print(f"[RUN] {step}: {command_line}", flush=True)
    if dry_run:
        return
    append_status(step, "start", command_line)

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n# {now_label()} {step}\n")
        log.write(command_line + "\n")
        log.flush()
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(SAGE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        rc = process.wait()
        if rc != 0:
            append_status(step, "failed", f"exit={rc} log={log_path}")
            raise subprocess.CalledProcessError(rc, [str(part) for part in cmd])
    append_status(step, "done", f"log={log_path}")


def maybe_add_common_sweep_args(cmd: list[object], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--datasets",
            args.datasets,
            "--k-values",
            args.k_values,
            "--offline-num-threads",
            str(int(args.offline_num_threads)),
            "--online-num-threads",
            str(int(args.online_num_threads)),
            "--warmup-runs",
            str(int(args.warmup_runs)),
            "--measured-runs",
            str(int(args.measured_runs)),
            "--param-m",
            str(int(args.param_m)),
            "--ef-construction",
            str(int(args.ef_construction)),
            "--num-calibration-queries",
            str(int(args.num_calibration_queries)),
            "--mixed-threshold-mode",
            str(args.mixed_threshold_mode),
            "--mixed-bucket-count",
            str(int(args.mixed_bucket_count)),
        ]
    )
    if args.ef_sweep:
        cmd.extend(["--ef-sweep", str(args.ef_sweep)])
    if args.force_recompute:
        cmd.append("--no-skip-existing")


def build_hnsw_cmd(args: argparse.Namespace) -> list[object]:
    cmd: list[object] = [
        Path(args.python).expanduser(),
        HNSW_RUNNER,
        "--run-root",
        HNSW_RUN_ROOT,
        "--final-dir",
        HNSW_FINAL_DIR,
    ]
    maybe_add_common_sweep_args(cmd, args)
    if args.enable_hnsw_drilldown:
        cmd.extend(
            [
                "--enable-drilldown",
                "--drilldown-output-dir",
                Path(args.hnsw_drilldown_output_dir).expanduser(),
                "--drilldown-pseudo-gt-ef",
                str(int(args.hnsw_drilldown_pseudo_gt_ef)),
                "--drilldown-group-def-ef",
                str(int(args.hnsw_drilldown_group_def_ef)),
            ]
        )
        if args.hnsw_drilldown_ef_sweep:
            cmd.extend(["--drilldown-ef-sweep", str(args.hnsw_drilldown_ef_sweep)])
    return cmd


def build_faiss_cmd(args: argparse.Namespace) -> list[object]:
    cmd: list[object] = [
        Path(args.python).expanduser(),
        FAISS_RUNNER,
        "--faiss-python-path",
        Path(args.faiss_python_path).expanduser(),
        "--faiss-index-root",
        Path(args.faiss_index_root).expanduser(),
        "--dataset-preset",
        "latest6",
        "--run-root",
        FAISS_RUN_ROOT,
        "--final-dir",
        FAISS_FINAL_DIR,
    ]
    maybe_add_common_sweep_args(cmd, args)
    return cmd


def build_combine_cmd(args: argparse.Namespace) -> list[object]:
    return [
        Path(args.python).expanduser(),
        COMBINE_SCRIPT,
        "--hnsw-csv",
        HNSW_CSV,
        "--hnsw-recommended-csv",
        HNSW_RECOMMENDED_CSV,
        "--faiss-csv",
        FAISS_CSV,
        "--baseline-root",
        BASELINE_ROOT,
        "--output-dir",
        Path(args.output_dir).expanduser(),
    ]


def should_run_sweep(mode: str, csv_path: Path) -> bool:
    if mode == "combine-only":
        return False
    if mode == "all":
        return True
    return not csv_path.exists()


def hnsw_drilldown_csv(args: argparse.Namespace) -> Path:
    return Path(args.hnsw_drilldown_output_dir).expanduser() / "group_ef_sweep.csv"


def should_run_hnsw(args: argparse.Namespace) -> bool:
    if should_run_sweep(str(args.mode), HNSW_CSV):
        return True
    if str(args.mode) != "combine-only" and not HNSW_RECOMMENDED_CSV.exists():
        return True
    if str(args.mode) == "combine-only" or not args.enable_hnsw_drilldown:
        return False
    return not hnsw_drilldown_csv(args).exists()


def append_drilldown_pointer(args: argparse.Namespace) -> None:
    if not args.enable_hnsw_drilldown or args.dry_run:
        return
    readme_path = Path(args.output_dir).expanduser() / "README.md"
    if not readme_path.exists():
        return
    drilldown_dir = Path(args.hnsw_drilldown_output_dir).expanduser()
    lines = [
        "",
        "## HNSW Easy/Medium/Hard Drilldown",
        "",
        "This is produced by the same final runner invocation as a piggyback hnswlib-only drilldown.",
        f"- output dir: `{drilldown_dir}`",
        "- pseudo ground truth: Vanilla HNSW at efSearch=4096",
        "- group definition: Vanilla HNSW at efSearch=1024",
        "- fixed groups: top 30% easy, middle 40% medium, bottom 30% hard",
        "- primary files: `query_groups.csv`, `group_ef_sweep.csv`, `group_pair_metrics.csv`",
        "",
    ]
    with readme_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n" + "\n".join(lines))


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        append_status("run", "configured", f"mode={args.mode}")

    if should_run_hnsw(args):
        run_cmd("hnswlib_sweep", build_hnsw_cmd(args), dry_run=bool(args.dry_run))
    else:
        if not args.dry_run:
            append_status("hnswlib_sweep", "skipped", f"existing_csv={HNSW_CSV} existing_recommended={HNSW_RECOMMENDED_CSV}")
        print(f"[SKIP] hnswlib_sweep existing_csv={HNSW_CSV} existing_recommended={HNSW_RECOMMENDED_CSV}", flush=True)

    if args.skip_faiss_sweep:
        if not args.dry_run:
            append_status("faiss_sweep", "skipped", "requested_by=--skip-faiss-sweep")
        print("[SKIP] faiss_sweep requested_by=--skip-faiss-sweep", flush=True)
    elif should_run_sweep(str(args.mode), FAISS_CSV):
        run_cmd("faiss_sweep", build_faiss_cmd(args), dry_run=bool(args.dry_run))
    else:
        if not args.dry_run:
            append_status("faiss_sweep", "skipped", f"existing_csv={FAISS_CSV}")
        print(f"[SKIP] faiss_sweep existing_csv={FAISS_CSV}", flush=True)

    if args.skip_combine:
        if not args.dry_run:
            append_status("combined_plot", "skipped", "requested_by=--skip-combine")
        print("[SKIP] combined_plot requested_by=--skip-combine", flush=True)
    else:
        run_cmd("combined_plot", build_combine_cmd(args), dry_run=bool(args.dry_run))
        append_drilldown_pointer(args)
    if not args.dry_run:
        append_status("run", "complete", f"output_dir={Path(args.output_dir).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
