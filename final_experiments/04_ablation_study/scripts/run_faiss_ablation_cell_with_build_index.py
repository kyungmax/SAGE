#!/usr/bin/env python3
"""Run one FAISS ablation cell through the local sweep runner.

This wrapper keeps the current ``experiments_scripts/faiss`` runner as the
source of truth, adding only build-on-miss FAISS index loading for artifact
reproduction.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments_scripts"
FAISS_IMPL_ROOT = EXPERIMENTS_ROOT / "faiss"
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("HNSW_PLAYGROUND_ROOT", os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT)))
).expanduser()
DEFAULT_DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_FAISS_INDEX_ROOT",
        os.environ.get(
            "FAISS_INDEX_ROOT",
            str(DEFAULT_PROJECT_ROOT / "index/faiss_m32_efc500_main8_20260707/index"),
        ),
    )
).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get(
        "FAISS_PYTHON_PATH",
        str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"),
    )
).expanduser()


def _prepend_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _arg_present(argv: Sequence[str], name: str) -> bool:
    prefix = f"{name}="
    return any(item == name or item.startswith(prefix) for item in argv)


def parse_wrapper_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-batch-size", type=int, default=int(os.environ.get("SAGE_FAISS_BUILD_BATCH_SIZE", "32768")))
    parser.add_argument("--index-build-threads", type=int, default=int(os.environ.get("SAGE_FAISS_INDEX_BUILD_THREADS", "24")))
    args, rest = parser.parse_known_args(argv)
    if int(args.build_batch_size) < 1:
        raise ValueError("--build-batch-size must be positive")
    if int(args.index_build_threads) < 1:
        raise ValueError("--index-build-threads must be positive")
    return args, rest


def install_build_on_miss_index_loader(final_index_utils, sweep_module, *, build_batch_size: int, index_build_threads: int) -> None:
    def build_or_load_faiss_index(
        *,
        train,
        dataset_name: str,
        index_dir: str,
        param_m: int,
        ef_construction: int,
        num_threads: int,
    ):
        spec = final_index_utils.resolve_dataset_spec(dataset_name)
        index_root = Path(index_dir).expanduser().resolve()
        index_path = index_root / spec.darth_name / f"{spec.darth_name}.M{int(param_m)}.efC{int(ef_construction)}.index"
        faiss_index_class = final_index_utils.import_faiss_index_class()
        dim = int(train.shape[1])
        index = faiss_index_class(space=spec.space, dim=dim)

        if index_path.exists() and index_path.stat().st_size > 0:
            print(f"[FAISS] loading index dataset={dataset_name} space={spec.space} path={index_path}", flush=True)
            index.load_index(str(index_path), max_elements=int(len(train)))
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[FAISS] building index dataset={dataset_name} rows={len(train)} dim={dim} "
                f"M={int(param_m)} efC={int(ef_construction)} threads={int(index_build_threads)} path={index_path}",
                flush=True,
            )
            build_start = time.perf_counter()
            index.set_num_threads(int(index_build_threads))
            index.init_index(max_elements=int(len(train)), M=int(param_m), ef_construction=int(ef_construction))
            for start in range(0, int(len(train)), int(build_batch_size)):
                stop = min(start + int(build_batch_size), int(len(train)))
                index.add_items(train[start:stop], num_threads=int(index_build_threads))
                batch_no = start // int(build_batch_size)
                if start == 0 or stop == int(len(train)) or batch_no % 10 == 0:
                    elapsed = time.perf_counter() - build_start
                    print(
                        f"[FAISS] add progress {stop}/{len(train)} "
                        f"({100.0 * stop / float(len(train)):.2f}%) elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
            tmp_path = index_path.with_name(index_path.name + f".tmp.{os.getpid()}")
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"[FAISS] saving index tmp={tmp_path}", flush=True)
            index.save_index(str(tmp_path))
            os.replace(tmp_path, index_path)
            print(f"[FAISS] built index path={index_path} wall_s={time.perf_counter() - build_start:.1f}", flush=True)

        index.set_num_threads(int(num_threads))
        count = int(index.get_current_count())
        if count != int(len(train)):
            raise ValueError(f"Index count mismatch for {dataset_name}: index ntotal={count}, train rows={len(train)}")
        return index, spec.space, spec.darth_name

    sweep_module.build_original_index = build_or_load_faiss_index


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_args, pass_through = parse_wrapper_args(sys.argv[1:] if argv is None else argv)

    os.environ.setdefault("HNSW_PLAYGROUND_ROOT", str(DEFAULT_PROJECT_ROOT))
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    os.environ.setdefault("FAISS_INDEX_ROOT", str(DEFAULT_INDEX_ROOT))
    os.environ.setdefault("FAISS_PYTHON_PATH", str(DEFAULT_FAISS_PYTHON_PATH))

    if not _arg_present(pass_through, "--base-path"):
        pass_through.extend(["--base-path", str(DEFAULT_DATASET_DIR)])
    if not _arg_present(pass_through, "--index-dir"):
        pass_through.extend(["--index-dir", str(DEFAULT_INDEX_ROOT)])
    if not _arg_present(pass_through, "--faiss-python-path"):
        pass_through.extend(["--faiss-python-path", str(DEFAULT_FAISS_PYTHON_PATH)])
    if not _arg_present(pass_through, "--allow-system-faiss"):
        pass_through.append("--allow-system-faiss")
    if not _arg_present(pass_through, "--no-conda-reexec"):
        pass_through.append("--no-conda-reexec")

    _prepend_path(FAISS_IMPL_ROOT)
    _prepend_path(EXPERIMENTS_ROOT)
    import final_index_utils
    import run_main_qps_latency_sweep as sweep

    install_build_on_miss_index_loader(
        final_index_utils,
        sweep,
        build_batch_size=int(wrapper_args.build_batch_size),
        index_build_threads=int(wrapper_args.index_build_threads),
    )

    original_argv = sys.argv[:]
    try:
        sys.argv = [str(FAISS_IMPL_ROOT / "run_main_qps_latency_sweep.py"), *pass_through]
        return int(sweep.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
