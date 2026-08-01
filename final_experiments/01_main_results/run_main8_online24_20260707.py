#!/usr/bin/env python3
"""Run main eight-dataset SAGE sweeps with 24 online/offline threads.

This is the script-only artifact version of the original final-experiment
launcher. It dispatches to the local ``experiments_scripts/{hnswlib,faiss}``
runners and does not vendor generated results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
HNSW_RUNNER_ROOT = EXPERIMENTS_ROOT / "hnswlib"
FAISS_RUNNER_ROOT = EXPERIMENTS_ROOT / "faiss"
OUT_ROOT = ROOT / "main8_online24"


def _default_project_root() -> Path:
    for key in ("HNSW_PLAYGROUND_ROOT", "SAGE_PROJECT_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return REPO_ROOT


PROJECT_ROOT = _default_project_root()
DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(PROJECT_ROOT / "datasets"))).expanduser()
INDEX_DIR = Path(os.environ.get("SAGE_INDEX_DIR", str(PROJECT_ROOT / "index"))).expanduser()
FAISS_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get(
            "FAISS_INDEX_ROOT",
            str(INDEX_DIR / "faiss_m32_efc500_main8_20260707/index"),
        ),
    )
).expanduser()
FAISS_PYTHON_PATH = Path(
    os.environ.get("FAISS_PYTHON_PATH", str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"))
).expanduser()

CELL_PYTHON = os.environ.get("SAGE_PYTHON", str(Path(sys.executable)))

DATASETS = (
    "glove-100-angular.hdf5",
    "nytimes-256-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
    "cohere-768-angular.hdf5",
    "youtube-15M-angular.hdf5",
    "agnews-mxbai-1024-euclidean.hdf5",
    "landmark-nomic-768-angular.hdf5",
)
EF_SWEEP = "64,80,96,128,160,192,256,320,384,512,640,768,896,1024"
OFFLINE_THREADS = 24
SEARCH_THREADS = 24
DEFAULT_FAISS_OUTPUT_NAME = "faiss"


def _prepend_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def parse_dataset_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DATASETS
    datasets = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not datasets:
        raise ValueError("--datasets must not be empty.")
    return datasets


def resolve_out_root(value: str | None) -> Path:
    if not value:
        return OUT_ROOT
    return Path(value).expanduser().resolve()


def common_sweep_args(*, run_root: Path, final_dir: Path, datasets: Sequence[str]) -> list[str]:
    return [
        "--datasets",
        ",".join(datasets),
        "--base-path",
        str(DATASET_DIR),
        "--run-root",
        str(run_root),
        "--final-dir",
        str(final_dir),
        "--k-values",
        "10",
        "--ef-sweep",
        EF_SWEEP,
        "--offline-num-threads",
        str(OFFLINE_THREADS),
        "--online-num-threads",
        str(SEARCH_THREADS),
        "--warmup-runs",
        "1",
        "--measured-runs",
        "3",
        "--param-m",
        "32",
        "--ef-construction",
        "500",
        "--query-method",
        "adaptive-light",
        "--num-calibration-queries",
        "100",
        "--tmin-pops",
        "25",
        "--mixed-threshold-mode",
        "paper_floor_half",
        "--mixed-bucket-count",
        "4",
        "--no-skip-existing",
    ]


def run_hnswlib_cell(*, datasets: Sequence[str], out_root: Path) -> int:
    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(PROJECT_ROOT))
    _prepend_path(HNSW_RUNNER_ROOT)
    _prepend_path(EXPERIMENTS_ROOT)

    import run_main_qps_latency_sweep as sweep

    sys.argv = [
        str(HNSW_RUNNER_ROOT / "run_main_qps_latency_sweep.py"),
        "--index-dir",
        str(INDEX_DIR),
        *common_sweep_args(
            run_root=out_root / "hnswlib" / "run",
            final_dir=out_root / "hnswlib" / "final",
            datasets=datasets,
        ),
    ]
    return int(sweep.main())


def run_faiss_cell(*, datasets: Sequence[str], out_root: Path, output_name: str = DEFAULT_FAISS_OUTPUT_NAME) -> int:
    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(PROJECT_ROOT))
    os.environ["FAISS_INDEX_ROOT"] = str(FAISS_INDEX_ROOT)
    os.environ["FAISS_PYTHON_PATH"] = str(FAISS_PYTHON_PATH)
    os.environ["FAISS_OPT_LEVEL"] = "AVX512"

    _prepend_path(FAISS_RUNNER_ROOT)
    _prepend_path(EXPERIMENTS_ROOT)

    import run_main_qps_latency_sweep as sweep

    sys.argv = [
        str(FAISS_RUNNER_ROOT / "run_main_qps_latency_sweep.py"),
        "--no-conda-reexec",
        "--faiss-python-path",
        str(FAISS_PYTHON_PATH),
        "--index-dir",
        str(FAISS_INDEX_ROOT),
        *common_sweep_args(
            run_root=out_root / output_name / "run",
            final_dir=out_root / output_name / "final",
            datasets=datasets,
        ),
    ]
    return int(sweep.main())


def run_cell(args: argparse.Namespace) -> int:
    datasets = parse_dataset_csv(args.datasets)
    out_root = resolve_out_root(args.out_root)
    if args.cell == "hnswlib":
        return run_hnswlib_cell(datasets=datasets, out_root=out_root)
    if args.cell == "faiss":
        return run_faiss_cell(
            datasets=datasets,
            out_root=out_root,
            output_name=str(args.faiss_output_name),
        )
    raise ValueError(f"unknown cell: {args.cell!r}")


def run_all(args: argparse.Namespace) -> int:
    cells = ("hnswlib", "faiss") if args.cells == "all" else (args.cells,)
    out_root = resolve_out_root(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for idx, cell in enumerate(cells, start=1):
        cmd = [
            CELL_PYTHON,
            str(Path(__file__).resolve()),
            "run-cell",
            "--cell",
            cell,
        ]
        if args.datasets:
            cmd.extend(["--datasets", args.datasets])
        if args.out_root:
            cmd.extend(["--out-root", args.out_root])
        if cell == "faiss" and args.faiss_output_name != DEFAULT_FAISS_OUTPUT_NAME:
            cmd.extend(["--faiss-output-name", args.faiss_output_name])
        print(f"[RUN {idx}/{len(cells)}] cell={cell}", flush=True)
        print(" ".join(cmd), flush=True)
        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.perf_counter() - start
        print(
            f"[DONE {idx}/{len(cells)}] cell={cell} "
            f"returncode={completed.returncode} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_cell_parser = sub.add_parser("run-cell")
    run_cell_parser.add_argument("--cell", choices=("hnswlib", "faiss"), required=True)
    run_cell_parser.add_argument("--datasets", default=None)
    run_cell_parser.add_argument("--out-root", default=None)
    run_cell_parser.add_argument("--faiss-output-name", default=DEFAULT_FAISS_OUTPUT_NAME)

    run_all_parser = sub.add_parser("run-all")
    run_all_parser.add_argument("--cells", choices=("all", "hnswlib", "faiss"), default="all")
    run_all_parser.add_argument("--datasets", default=None)
    run_all_parser.add_argument("--out-root", default=None)
    run_all_parser.add_argument("--faiss-output-name", default=DEFAULT_FAISS_OUTPUT_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(SEARCH_THREADS))
    args = parse_args(argv)
    if args.command == "run-cell":
        return run_cell(args)
    if args.command == "run-all":
        return run_all(args)
    raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
