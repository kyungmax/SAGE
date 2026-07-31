#!/usr/bin/env python3
"""Run the MSMARCO five-embedding SAGE sweep on the FAISS backend.

The five default HDF5 datasets are built from the same MSMARCO passage/query
texts with mean GloVe, mean fastText, OpenAI ada-002, BGE-M3, and
EmbeddingGemma-300M embeddings. This launcher keeps the shared FAISS sweep
runner intact and only adds dataset mappings plus build-on-miss FAISS indexes.
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
    os.environ.get("SAGE_PROJECT_ROOT", str(REPO_ROOT))
).expanduser()
DEFAULT_DATASET_DIR = Path(os.environ.get("SAGE_DATA_DIR", str(DEFAULT_PROJECT_ROOT / "datasets"))).expanduser()
DEFAULT_INDEX_ROOT = Path(
    os.environ.get(
        "SAGE_MSMARCO_EMBEDDING_FAISS_INDEX_ROOT",
        str(DEFAULT_PROJECT_ROOT / "index/msmarco_embedding_models_faiss_m32_efc500/darth/index"),
    )
).expanduser()
DEFAULT_FAISS_PYTHON_PATH = Path(
    os.environ.get("FAISS_PYTHON_PATH", str(REPO_ROOT / "faiss/build_sage_avx512/faiss/python"))
).expanduser()

DATASETS = (
    "msmarco-v1-glove6b300d-full-ip.hdf5",
    "msmarco-v1-fasttext-cc300d-full-ip.hdf5",
    "msmarco-v1-openai-ada2-full-ip.hdf5",
    "marco_embeddings/msmarco-v1-bge-m3-fp32-dev6980-ip.hdf5",
    "marco_embeddings/msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip.hdf5",
)
EF_SWEEP = "64,80,96,128,160,192,256,320,384,512,640,768,896,1024"
BUILD_BATCH_SIZE = 32768


def parse_dataset_list(value: str) -> str:
    datasets = [part.strip() for part in str(value).split(",") if part.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")
    return ",".join(datasets)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=parse_dataset_list, default=",".join(DATASETS))
    parser.add_argument("--base-path", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--faiss-python-path", type=Path, default=DEFAULT_FAISS_PYTHON_PATH)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--ef-sweep", default=EF_SWEEP)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--param-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=500)
    parser.add_argument("--num-calibration-queries", type=int, default=100)
    parser.add_argument("--build-batch-size", type=int, default=BUILD_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Recompute completed cells.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if int(args.threads) < 1:
        raise ValueError("--threads must be positive")
    if int(args.build_batch_size) < 1:
        raise ValueError("--build-batch-size must be positive")
    return args


def install_dataset_specs(final_index_utils) -> None:
    dataset_spec = final_index_utils.DatasetSpec
    specs = {
        "msmarco-v1-glove6b300d-full-ip": ("msmarco-v1-glove6b300d-full-ip", "ip"),
        "msmarco-v1-glove6b300d-tfidf-full-ip": ("msmarco-v1-glove6b300d-tfidf-full-ip", "ip"),
        "msmarco-v1-fasttext-cc300d-full-ip": ("msmarco-v1-fasttext-cc300d-full-ip", "ip"),
        "msmarco-v1-fasttext-cc300d-tfidf-full-ip": ("msmarco-v1-fasttext-cc300d-tfidf-full-ip", "ip"),
        "msmarco-v1-openai-ada2-full-ip": ("msmarco-v1-openai-ada2-full-ip", "ip"),
        "msmarco-v1-bge-m3-fp32-dev6980-ip": ("msmarco-v1-bge-m3-fp32-dev6980-ip", "ip"),
        "msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip": (
            "msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip",
            "ip",
        ),
    }
    for key, (darth_name, space) in specs.items():
        final_index_utils.DATASET_SPECS[key] = dataset_spec(darth_name, space)
        final_index_utils.DATASET_SPECS[f"{key}.hdf5"] = dataset_spec(darth_name, space)


def install_build_on_miss_index_loader(final_index_utils, sweep_module, *, build_batch_size: int) -> None:
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
        index.set_num_threads(int(num_threads))

        if index_path.exists():
            print(f"[FAISS] loading index dataset={dataset_name} space={spec.space} path={index_path}", flush=True)
            index.load_index(str(index_path), max_elements=int(len(train)))
        else:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[FAISS] building index dataset={dataset_name} rows={len(train)} dim={dim} "
                f"M={int(param_m)} efC={int(ef_construction)} threads={int(num_threads)} path={index_path}",
                flush=True,
            )
            build_start = time.perf_counter()
            index.init_index(max_elements=int(len(train)), M=int(param_m), ef_construction=int(ef_construction))
            for start in range(0, int(len(train)), int(build_batch_size)):
                stop = min(start + int(build_batch_size), int(len(train)))
                index.add_items(train[start:stop], num_threads=int(num_threads))
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


def install_faiss_legacy_signature_compat(faiss_sage_index) -> None:
    original_native_cfr_summary = faiss_sage_index.Index._native_cfr_summary

    def _uses_legacy_optional_kwarg(exc: TypeError) -> bool:
        message = str(exc)
        return "unexpected keyword argument" in message and (
            "classify_start" in message or "classify_end" in message or "cfr_ema_decay" in message
        )

    def _native_cfr_summary_compat(
        self,
        data,
        *,
        k: int,
        ef: int,
        hide_labels=None,
        num_threads: int = -1,
        classify_start: int = 4,
        classify_end: int = 16,
        cfr_ema_decay: float = faiss_sage_index.CFR_EMA_DECAY,
    ):
        try:
            return original_native_cfr_summary(
                self,
                data,
                k=int(k),
                ef=int(ef),
                hide_labels=hide_labels,
                num_threads=int(num_threads),
                classify_start=int(classify_start),
                classify_end=int(classify_end),
                cfr_ema_decay=float(cfr_ema_decay),
            )
        except TypeError as exc:
            if not _uses_legacy_optional_kwarg(exc):
                raise
            if int(classify_start) != 4 or int(classify_end) != 16 or abs(float(cfr_ema_decay) - 0.8) > 1e-12:
                raise RuntimeError(
                    "Installed FAISS CFR summary binding lacks classify-window kwargs; fallback is valid only "
                    "for classify_start=4, classify_end=16, cfr_ema_decay=0.8."
                ) from exc
            native_method = getattr(self._require_index(), "search_layer0_cfr_summary", None)
            if native_method is None:
                raise
            if not getattr(self, "_legacy_cfr_summary_signature_warned", False):
                print("[FAISS] using legacy default-window search_layer0_cfr_summary signature", flush=True)
                self._legacy_cfr_summary_signature_warned = True
            vectors = self._prepare_vectors(data)
            return native_method(
                vectors,
                k=int(k),
                ef=int(ef),
                hide_labels=None if hide_labels is None else faiss_sage_index.np.asarray(hide_labels, dtype=faiss_sage_index.np.int64),
                num_threads=self._num_threads if int(num_threads) <= 0 else int(num_threads),
            )

    faiss_sage_index.Index._native_cfr_summary = _native_cfr_summary_compat


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_root = args.out_root.expanduser() if args.out_root is not None else ROOT / "msmarco_embedding_models_faiss_SIMD_on_24t"
    out_root = out_root.resolve()
    index_root = Path(args.index_root).expanduser().resolve()
    base_path = Path(args.base_path).expanduser().resolve()
    datasets = str(args.datasets)

    runner_argv = [
        "--datasets", datasets,
        "--base-path", str(base_path),
        "--index-dir", str(index_root),
        "--faiss-python-path", str(Path(args.faiss_python_path).expanduser().resolve()),
        "--run-root", str(out_root / "sage_results/run"),
        "--final-dir", str(out_root / "sage_results/final"),
        "--ef-sweep", str(args.ef_sweep),
        "--offline-num-threads", str(int(args.threads)),
        "--online-num-threads", str(int(args.threads)),
        "--warmup-runs", str(int(args.warmup_runs)),
        "--measured-runs", str(int(args.measured_runs)),
        "--param-m", str(int(args.param_m)),
        "--ef-construction", str(int(args.ef_construction)),
        "--num-calibration-queries", str(int(args.num_calibration_queries)),
        "--allow-system-faiss",
        "--no-conda-reexec",
    ]
    if bool(args.force):
        runner_argv.append("--no-skip-existing")

    print(f"[OUT_ROOT] {out_root}")
    print(f"[INDEX_ROOT] {index_root}")
    print(f"[BASE_PATH] {base_path}")
    print(f"[DATASETS] {datasets}")
    print(f"[EF_SWEEP] {args.ef_sweep}")
    print(f"[THREADS] {int(args.threads)}")
    print(f"[PYTHON] {sys.executable}")
    if args.dry_run:
        print("[DRY-RUN] runner argv:")
        print(" ".join(runner_argv))
        return 0

    os.environ.setdefault("SAGE_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
    os.environ.setdefault("FAISS_OPT_LEVEL", "AVX512")
    os.environ["FAISS_PYTHON_PATH"] = str(Path(args.faiss_python_path).expanduser().resolve())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(int(args.threads))

    faiss_python_path = Path(args.faiss_python_path).expanduser().resolve()
    sys.path.insert(0, str(faiss_python_path))
    sys.path.insert(0, str(FAISS_IMPL_ROOT))
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
    import faiss_sage_index
    import final_index_utils
    import run_main_qps_latency_sweep as sweep

    install_faiss_legacy_signature_compat(faiss_sage_index)
    install_dataset_specs(final_index_utils)
    final_index_utils.configure_faiss_loader(
        python_path=Path(args.faiss_python_path),
        index_root=index_root,
        allow_system_faiss=True,
    )
    install_build_on_miss_index_loader(final_index_utils, sweep, build_batch_size=int(args.build_batch_size))

    original_argv = sys.argv[:]
    try:
        sys.argv = [str(FAISS_IMPL_ROOT / "run_main_qps_latency_sweep.py"), *runner_argv]
        return int(sweep.main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
