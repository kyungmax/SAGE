#!/usr/bin/env python3
"""Run main eight-dataset SAGE sweeps with one online search thread."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

import run_main8_online24_20260707 as base


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "main8_online1"
OFFLINE_THREADS = 24
SEARCH_THREADS = 1


def run_all(args) -> int:
    cells = ("hnswlib", "faiss") if args.cells == "all" else (args.cells,)
    out_root = base.resolve_out_root(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve()
    for idx, cell in enumerate(cells, start=1):
        cmd = [
            base.CELL_PYTHON,
            str(wrapper),
            "run-cell",
            "--cell",
            cell,
        ]
        if args.datasets:
            cmd.extend(["--datasets", args.datasets])
        if args.out_root:
            cmd.extend(["--out-root", args.out_root])
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


def configure() -> None:
    base.__doc__ = __doc__
    base.OUT_ROOT = OUT_ROOT
    base.OFFLINE_THREADS = OFFLINE_THREADS
    base.SEARCH_THREADS = SEARCH_THREADS
    base.run_all = run_all
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(SEARCH_THREADS)


def main(argv: Sequence[str] | None = None) -> int:
    configure()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
