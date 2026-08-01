#!/usr/bin/env python3
"""Run the paper ablation cells on FAISS for GloVe100 and CohereWiki."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
CELL_RUNNER = SCRIPT_PATH.parent / "run_faiss_ablation_cell_with_build_index.py"
SUMMARY_SCRIPT = SCRIPT_PATH.parent / "summarize_faiss_glove_cohere_ablation.py"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HNSW_PLAYGROUND_ROOT", os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT)))
).expanduser()
DEFAULT_DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get(
            "FAISS_INDEX_ROOT",
            str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8/index"),
        ),
    )
).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get(
        "FAISS_PYTHON_PATH",
        str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"),
    )
).expanduser()
DEFAULT_CELL_PYTHON = os.environ.get("SAGE_PYTHON", str(Path(sys.executable)))
DEFAULT_DATASETS = (
    "glove-100-angular.hdf5",
    "cohere-768-angular.hdf5",
)
DEFAULT_OUTPUT_ROOT = ROOT / "sage_ablation_faiss_glove_cohere_24t_m32_efc500_ef1024"


@dataclass(frozen=True)
class Cell:
    study_key: str
    study_dir: str
    variant_dir: str
    ablation_name: str
    ablation_value: str
    ncal: int = 100
    classify_start: int = 4
    classify_end: int = 16
    alpha: float = 0.8
    pair_gap: int = 2
    bucket_count: int = 4


CELLS: tuple[Cell, ...] = (
    Cell("ncal", "01_ncal", "ncal_100", "ncal", "100", ncal=100),
    Cell("ncal", "01_ncal", "ncal_500", "ncal", "500", ncal=500),
    Cell("ncal", "01_ncal", "ncal_1000", "ncal", "1000", ncal=1000),
    Cell("window", "02_classification_window", "window_1_13", "classification_window", "1_13", classify_start=1, classify_end=13),
    Cell("window", "02_classification_window", "window_4_16", "classification_window", "4_16", classify_start=4, classify_end=16),
    Cell("tiers", "03_tiers", "b2", "tiers", "2", bucket_count=2),
    Cell("tiers", "03_tiers", "b4", "tiers", "4", bucket_count=4),
    Cell("tiers", "03_tiers", "b6", "tiers", "6", bucket_count=6),
    Cell("ema", "04_ema_alpha", "alpha_0", "ema_alpha", "0", alpha=0.0),
    Cell("ema", "04_ema_alpha", "alpha_0p4", "ema_alpha", "0p4", alpha=0.4),
    Cell("ema", "04_ema_alpha", "alpha_0p8", "ema_alpha", "0p8", alpha=0.8),
    Cell("gap", "05_pair_gap", "gap_1x", "pair_gap", "1", pair_gap=1),
    Cell("gap", "05_pair_gap", "gap_2x", "pair_gap", "2", pair_gap=2),
    Cell("gap", "05_pair_gap", "gap_3x", "pair_gap", "3", pair_gap=3),
    Cell("gap", "05_pair_gap", "gap_4x", "pair_gap", "4", pair_gap=4),
)

STUDY_ALIASES = {
    "all": "all",
    "1": "ncal",
    "01": "ncal",
    "01_ncal": "ncal",
    "ncal": "ncal",
    "calibration": "ncal",
    "2": "window",
    "02": "window",
    "02_classification_window": "window",
    "window": "window",
    "classification_window": "window",
    "cfr_window": "window",
    "3": "tiers",
    "03": "tiers",
    "03_tiers": "tiers",
    "tiers": "tiers",
    "bucket": "tiers",
    "b": "tiers",
    "4": "ema",
    "04": "ema",
    "04_ema_alpha": "ema",
    "ema": "ema",
    "ema_alpha": "ema",
    "alpha": "ema",
    "5": "gap",
    "05": "gap",
    "05_pair_gap": "gap",
    "gap": "gap",
    "pair_gap": "gap",
    "g": "gap",
}


def parse_dataset_csv(value: str) -> tuple[str, ...]:
    datasets = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not datasets:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")
    return datasets


def selected_studies(value: str) -> set[str]:
    raw = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not raw or "all" in raw:
        return {"ncal", "window", "tiers", "ema", "gap"}
    out: set[str] = set()
    for part in raw:
        if part not in STUDY_ALIASES:
            raise argparse.ArgumentTypeError(f"unknown study selector: {part}")
        mapped = STUDY_ALIASES[part]
        if mapped != "all":
            out.add(mapped)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_CELL_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", type=parse_dataset_csv, default=DEFAULT_DATASETS)
    parser.add_argument("--studies", default="all", help="Comma list: all,ncal,window,tiers,ema,gap or 1..5")
    parser.add_argument("--ef-sweep", default="1024")
    parser.add_argument("--k-values", default="10")
    parser.add_argument("--offline-num-threads", type=int, default=24)
    parser.add_argument("--online-num-threads", type=int, default=24)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--tmin-pops", type=int, default=25)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--build-batch-size", type=int, default=int(os.environ.get("SAGE_FAISS_BUILD_BATCH_SIZE", "32768")))
    parser.add_argument("--index-build-threads", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute completed sweep cells.")
    parser.add_argument("--split-datasets", action="store_true", default=True)
    parser.add_argument("--no-split-datasets", action="store_false", dest="split_datasets")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args(argv)
    if int(args.offline_num_threads) < 1 or int(args.online_num_threads) < 1:
        raise ValueError("thread counts must be positive")
    if int(args.build_batch_size) < 1 or int(args.index_build_threads) < 1:
        raise ValueError("index build settings must be positive")
    return args


def common_sweep_args(args: argparse.Namespace, datasets: Sequence[str], cell_root: Path, cell: Cell) -> list[str]:
    cmd = [
        "--build-batch-size", str(int(args.build_batch_size)),
        "--index-build-threads", str(int(args.index_build_threads)),
        "--datasets", ",".join(datasets),
        "--base-path", str(Path(args.base_path).expanduser()),
        "--index-dir", str(Path(args.index_dir).expanduser()),
        "--faiss-python-path", str(Path(args.faiss_python_path).expanduser()),
        "--run-root", str(cell_root / "run"),
        "--final-dir", str(cell_root / "final"),
        "--k-values", str(args.k_values),
        "--ef-sweep", str(args.ef_sweep),
        "--offline-num-threads", str(int(args.offline_num_threads)),
        "--online-num-threads", str(int(args.online_num_threads)),
        "--warmup-runs", str(int(args.warmup_runs)),
        "--measured-runs", str(int(args.measured_runs)),
        "--param-m", str(int(args.param_m)),
        "--ef-construction", str(int(args.ef_construction)),
        "--query-method", "adaptive-light",
        "--num-calibration-queries", str(int(cell.ncal)),
        "--mixed-threshold-mode", "paper_floor_half",
        "--mixed-bucket-count", str(int(cell.bucket_count)),
        "--classify-start", str(int(cell.classify_start)),
        "--classify-end", str(int(cell.classify_end)),
        "--cfr-ema-decay", f"{float(cell.alpha):g}",
        "--pair-gap", str(int(cell.pair_gap)),
        "--tmin-pops", str(int(args.tmin_pops)),
        "--ablation-name", cell.ablation_name,
        "--ablation-value", cell.ablation_value,
        "--allow-system-faiss",
        "--no-conda-reexec",
    ]
    if bool(args.force):
        cmd.append("--no-skip-existing")
    return cmd


def append_manifest(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_command(cmd: Sequence[str]) -> None:
    print("[CMD] " + " ".join(shlex.quote(str(part)) for part in cmd), flush=True)


def run_summary(args: argparse.Namespace, output_root: Path) -> int:
    cmd = [
        str(args.python),
        str(SUMMARY_SCRIPT),
        "--output-root", str(output_root),
        "--ef", "1024",
    ]
    print_command(cmd)
    if bool(args.dry_run):
        return 0
    return int(subprocess.run(cmd, cwd=ROOT).returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    studies = selected_studies(str(args.studies))
    cells = [cell for cell in CELLS if cell.study_key in studies]
    if int(args.max_cells) > 0:
        cells = cells[: int(args.max_cells)]
    datasets = tuple(args.datasets)
    jobs: list[tuple[Cell, tuple[str, ...]]] = []
    if bool(args.split_datasets):
        jobs = [(cell, (dataset,)) for cell in cells for dataset in datasets]
    else:
        jobs = [(cell, datasets) for cell in cells]

    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(args.online_num_threads))

    manifest = output_root / "ablation_run_manifest.tsv"
    print(f"[FAISS-ABLATION] output_root={output_root}", flush=True)
    print(f"[FAISS-ABLATION] datasets={','.join(datasets)}", flush=True)
    print(f"[FAISS-ABLATION] studies={','.join(sorted(studies))} cells={len(cells)} jobs={len(jobs)}", flush=True)
    print(f"[FAISS-ABLATION] ef_sweep={args.ef_sweep} threads={int(args.online_num_threads)}", flush=True)

    for idx, (cell, job_datasets) in enumerate(jobs, start=1):
        cell_root = output_root / cell.study_dir / cell.variant_dir
        cmd = [str(args.python), str(CELL_RUNNER), *common_sweep_args(args, job_datasets, cell_root, cell)]
        started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[CELL {idx}/{len(jobs)}] {cell.study_dir}/{cell.variant_dir} datasets={','.join(job_datasets)}", flush=True)
        print_command(cmd)
        status = "dry_run"
        returncode = 0
        if not bool(args.dry_run):
            returncode = int(subprocess.run(cmd, cwd=ROOT).returncode)
            status = "done" if returncode == 0 else "failed"
        if not bool(args.dry_run):
            append_manifest(
                manifest,
                {
                    "started_at": started,
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "datasets": ",".join(job_datasets),
                    "study": cell.study_dir,
                    "variant": cell.variant_dir,
                    "ablation_name": cell.ablation_name,
                    "ablation_value": cell.ablation_value,
                    "status": status,
                    "returncode": returncode,
                    "command": " ".join(shlex.quote(str(part)) for part in cmd),
                },
            )
        if returncode != 0:
            return returncode

    if not bool(args.skip_summary):
        return run_summary(args, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
