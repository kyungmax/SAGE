#!/usr/bin/env python3
"""Run vanilla vs Ours efSearch sweep on Faiss HNSW indexes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
EXP_ROOT = SCRIPT_PATH.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
EXPERIMENTS_SCRIPT_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if parent.name == "experiments_scripts"
)
if str(EXPERIMENTS_SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_SCRIPT_ROOT))

from final_index_utils import (  # noqa: E402
    DEFAULT_FAISS_INDEX_ROOT,
    DEFAULT_FAISS_PYTHON_PATH,
)


def _find_default_project_root() -> Path:
    env_root = os.environ.get("HNSW_PLAYGROUND_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (EXP_ROOT, *EXP_ROOT.parents):
        if (candidate / "datasets").exists():
            return candidate
    return EXP_ROOT


PROJECT_ROOT = _find_default_project_root()
DEFAULT_RUN_ROOT = EXP_ROOT / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1/run"
DEFAULT_FINAL_DIR = EXP_ROOT / "faiss_vanilla_ours_efsweep_total6_m32_efc500_ncal100_online1/final"
LATEST6_DATASETS = (
    "nytimes-256-angular.hdf5",
    "glove-100-angular.hdf5",
    "cohere-768-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "msspacev-100M-i8-euclidean.hdf5",
    "youtube-15M-angular.hdf5",
)
PAPER6_DATASETS = (
    "glove-100-angular.hdf5",
    "nytimes-256-angular.hdf5",
    "deep-image-96-angular.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "sift-100M-euclidean.hdf5",
    "cohere-768-angular.hdf5",
)


def has_forwarded_arg(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def parse_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--faiss-python-path",
        default=DEFAULT_FAISS_PYTHON_PATH,
        help=(
            "Optional path containing a built faiss Python package. When omitted, "
            "the faiss module installed in the active Python environment is used."
        ),
    )
    parser.add_argument(
        "--faiss-index-root",
        default=str(DEFAULT_FAISS_INDEX_ROOT),
        help="Root containing DARTH/Faiss HNSW index subdirectories.",
    )
    parser.add_argument(
        "--dataset-preset",
        choices=("latest6", "paper6"),
        default="latest6",
        help="Default dataset list to forward when --datasets is omitted.",
    )
    parser.add_argument(
        "--allow-system-faiss",
        action="store_true",
        help=(
            "When --faiss-python-path is provided, allow importing an already "
            "installed faiss module instead of requiring that exact path."
        ),
    )
    parser.add_argument(
        "--no-conda-reexec",
        action="store_true",
        help="Do not re-exec the direct Faiss runner under its configured conda env.",
    )
    parser.add_argument(
        "--print-effective-argv",
        action="store_true",
        help="Print the argv forwarded to run_main_qps_latency_sweep.py.",
    )
    return parser.parse_known_args(list(argv))


def default_datasets_for_preset(preset: str) -> tuple[str, ...]:
    if preset == "paper6":
        return PAPER6_DATASETS
    return LATEST6_DATASETS


def build_forwarded_argv(wrapper_args: argparse.Namespace, remaining: list[str]) -> list[str]:
    forwarded = list(remaining)
    defaults: list[str] = []
    if not has_forwarded_arg(forwarded, "--datasets"):
        defaults += ["--datasets", ",".join(default_datasets_for_preset(wrapper_args.dataset_preset))]
    if not has_forwarded_arg(forwarded, "--base-path"):
        defaults += ["--base-path", str(PROJECT_ROOT / "datasets")]
    if not has_forwarded_arg(forwarded, "--index-dir"):
        defaults += ["--index-dir", str(Path(wrapper_args.faiss_index_root).expanduser())]
    if not has_forwarded_arg(forwarded, "--faiss-python-path") and wrapper_args.faiss_python_path:
        defaults += ["--faiss-python-path", str(Path(wrapper_args.faiss_python_path).expanduser())]
    if wrapper_args.allow_system_faiss and not has_forwarded_arg(forwarded, "--allow-system-faiss"):
        defaults += ["--allow-system-faiss"]
    if wrapper_args.no_conda_reexec and not has_forwarded_arg(forwarded, "--no-conda-reexec"):
        defaults += ["--no-conda-reexec"]
    if not has_forwarded_arg(forwarded, "--run-root"):
        defaults += ["--run-root", str(DEFAULT_RUN_ROOT)]
    if not has_forwarded_arg(forwarded, "--final-dir"):
        defaults += ["--final-dir", str(DEFAULT_FINAL_DIR)]
    if not has_forwarded_arg(forwarded, "--param-m"):
        defaults += ["--param-m", "32"]
    if not has_forwarded_arg(forwarded, "--ef-construction"):
        defaults += ["--ef-construction", "500"]
    if not has_forwarded_arg(forwarded, "--offline-num-threads"):
        defaults += ["--offline-num-threads", "24"]
    if not has_forwarded_arg(forwarded, "--online-num-threads"):
        defaults += ["--online-num-threads", "24"]
    return [str(EXP_ROOT / "run_main_qps_latency_sweep.py")] + defaults + forwarded


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    forwarded_argv = build_forwarded_argv(wrapper_args, remaining)
    if wrapper_args.print_effective_argv:
        print("[FAISS] forwarded argv:")
        print(" ".join(forwarded_argv))

    sys.argv = forwarded_argv
    import run_main_qps_latency_sweep as sweep

    return int(sweep.main())


if __name__ == "__main__":
    raise SystemExit(main())
