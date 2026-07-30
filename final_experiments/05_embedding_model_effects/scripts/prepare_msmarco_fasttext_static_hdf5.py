#!/usr/bin/env python3
"""Build MSMARCO passage static fastText HDF5 datasets.

The output follows the same local ANN-Benchmarks-style layout used by
prepare_msmarco_glove_static_hdf5.py:

  train      corpus passage embeddings from mean or TF-IDF-weighted fastText pooling
  test       query embeddings
  neighbors  exact top-k neighbors in this embedding space
  distances  1 - inner_product for the same neighbors

This builder targets fastText .bin models so OOV tokens are embedded through
fastText subword vectors instead of being dropped as in GloVe.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

import prepare_msmarco_glove_static_hdf5 as base


DATASET_ROOT = base.DATASET_ROOT
WORK_SUBDIR = "msmarco_passage_fasttext_static"
COLLECTION_URL = base.COLLECTION_URL
QUERIES_URL = base.QUERIES_URL
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
DEFAULT_ADA_HDF5 = base.DEFAULT_ADA_HDF5
DEFAULT_ADA_QUERY_IDS = base.DEFAULT_ADA_QUERY_IDS
TOKEN_RE = base.TOKEN_RE


def log(message: str) -> None:
    base.log(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--work-subdir", default=WORK_SUBDIR)
    parser.add_argument("--collection-tsv", type=Path, default=None)
    parser.add_argument("--queries-tsv", type=Path, default=None)
    parser.add_argument("--fasttext-bin", type=Path, default=None)
    parser.add_argument("--fasttext-label", default="fasttext-cc300d")
    parser.add_argument("--fasttext-name", default="cc.en.300.bin")
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--collection-url", default=COLLECTION_URL)
    parser.add_argument("--queries-url", default=QUERIES_URL)
    parser.add_argument("--fasttext-url", default=FASTTEXT_URL)
    parser.add_argument("--fasttext-dim", type=base.positive_int, default=300)
    parser.add_argument("--pooling", choices=("mean", "tfidf"), default="mean")
    parser.add_argument(
        "--idf-cache-npz",
        type=Path,
        default=None,
        help="Optional cache for token IDF values used by --pooling tfidf.",
    )
    parser.add_argument("--recompute-idf", action="store_true")
    parser.add_argument(
        "--tfidf-sublinear-tf",
        action="store_true",
        help="Use 1 + log(tf) instead of raw term frequency for --pooling tfidf.",
    )
    parser.add_argument(
        "--token-vector-cache-size",
        type=int,
        default=200_000,
        help="LRU cache size for fastText token vectors; 0 disables caching.",
    )
    parser.add_argument("--sample-size", type=int, default=100_000, help="0 means full corpus.")
    parser.add_argument("--sample-mode", choices=("first", "random"), default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--selected-rows-csv",
        type=Path,
        default=None,
        help="Optional CSV with a source_row_id column. Overrides --sample-size/--sample-mode.",
    )
    parser.add_argument("--query-ids-csv", type=Path, default=DEFAULT_ADA_QUERY_IDS)
    parser.add_argument("--query-limit", type=int, default=0, help="0 means all queries from --query-ids-csv.")
    parser.add_argument("--output-hdf5", type=Path, default=None)
    parser.add_argument("--neighbors-k", type=base.positive_int, default=10)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--write-batch", type=base.positive_int, default=8192)
    parser.add_argument("--exact-query-batch", type=base.positive_int, default=32)
    parser.add_argument("--exact-doc-block", type=base.positive_int, default=32768)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-ada-subset", action="store_true")
    parser.add_argument("--ada-source-hdf5", type=Path, default=DEFAULT_ADA_HDF5)
    parser.add_argument("--ada-output-hdf5", type=Path, default=None)
    args = parser.parse_args()
    if int(args.sample_size) < 0:
        raise ValueError("--sample-size must be >= 0")
    if int(args.query_limit) < 0:
        raise ValueError("--query-limit must be >= 0")
    if int(args.progress_every) < 0:
        raise ValueError("--progress-every must be >= 0")
    if int(args.token_vector_cache_size) < 0:
        raise ValueError("--token-vector-cache-size must be >= 0")
    return args


def resolved_asset_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.dataset_root).expanduser().resolve() / str(args.work_subdir)
    raw = root / "raw"
    return {
        "root": root,
        "raw": raw,
        "collection_tar": raw / "collection.tar.gz",
        "queries_tar": raw / "queries.tar.gz",
        "fasttext_gz": raw / Path(str(args.fasttext_url)).name,
        "collection_tsv": Path(args.collection_tsv).expanduser().resolve()
        if args.collection_tsv
        else raw / "collection.tsv",
        "queries_tsv": Path(args.queries_tsv).expanduser().resolve()
        if args.queries_tsv
        else raw / "queries.dev.tsv",
        "fasttext_bin": Path(args.fasttext_bin).expanduser().resolve()
        if args.fasttext_bin
        else raw / str(args.fasttext_name),
    }


def extract_gzip_file(gzip_path: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[EXTRACT] exists {dest}")
        return
    base.ensure_parent(dest)
    log(f"[EXTRACT] {gzip_path} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with gzip.open(gzip_path, "rb") as source, tmp.open("wb") as out:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def ensure_assets(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    paths["raw"].mkdir(parents=True, exist_ok=True)
    if args.download_assets:
        if not paths["collection_tsv"].exists():
            base.download_file(str(args.collection_url), paths["collection_tar"])
        if not paths["queries_tsv"].exists():
            base.download_file(str(args.queries_url), paths["queries_tar"])
        if not paths["fasttext_bin"].exists():
            base.download_file(str(args.fasttext_url), paths["fasttext_gz"])
    if paths["collection_tar"].exists():
        base.extract_tar_member(paths["collection_tar"], "collection.tsv", paths["collection_tsv"])
    if paths["queries_tar"].exists():
        query_member = Path(paths["queries_tsv"]).name
        try:
            base.extract_tar_member(paths["queries_tar"], query_member, paths["queries_tsv"])
        except FileNotFoundError:
            if query_member == "queries.dev.tsv":
                raise
            base.extract_tar_member(paths["queries_tar"], "queries.dev.tsv", paths["queries_tsv"])
    if paths["fasttext_gz"].exists():
        extract_gzip_file(paths["fasttext_gz"], paths["fasttext_bin"])

    missing = [name for name in ("collection_tsv", "queries_tsv", "fasttext_bin") if not paths[name].exists()]
    if missing:
        hint = "pass explicit paths or rerun with --download-assets"
        raise FileNotFoundError(f"Missing required asset(s): {missing}; {hint}")


def default_idf_cache_path(paths: dict[str, Path], fasttext_label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in str(fasttext_label))
    return paths["root"] / f"{safe_label}_msmarco_collection_token_idf.npz"


def load_idf_cache(path: Path) -> tuple[dict[str, float], float, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        tokens = np.asarray(data["tokens"]).astype(str)
        idf_values = np.asarray(data["idf"], dtype=np.float32)
        if tokens.shape != idf_values.shape:
            raise ValueError(f"IDF cache {path} has tokens shape={tokens.shape}, idf shape={idf_values.shape}")
        idf_by_token = {str(token): float(idf) for token, idf in zip(tokens, idf_values)}
        default_idf = float(np.asarray(data["default_idf"], dtype=np.float32).reshape(-1)[0])
        metadata = {
            "idf_cache_npz": str(path),
            "idf_doc_count": int(np.asarray(data["doc_count"]).reshape(-1)[0]),
            "idf_docs_with_tokens": int(np.asarray(data["docs_with_tokens"]).reshape(-1)[0]),
            "idf_unique_token_count": int(tokens.size),
            "idf_default_for_unseen_token": float(default_idf),
            "idf_formula": str(np.asarray(data["formula"]).reshape(-1)[0]),
        }
    log(
        f"[IDF] loaded {path} docs={metadata['idf_doc_count']:,} "
        f"tokens={metadata['idf_unique_token_count']:,}"
    )
    return idf_by_token, default_idf, metadata


def write_idf_cache(
    path: Path,
    *,
    idf_by_token: dict[str, float],
    df_by_token: dict[str, int],
    doc_count: int,
    docs_with_tokens: int,
    fasttext_dim: int,
) -> None:
    base.ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".part")
    formula = "smooth_idf: log((1 + n_docs) / (1 + df)) + 1"
    tokens = np.asarray(sorted(idf_by_token), dtype=str)
    idf = np.asarray([idf_by_token[str(token)] for token in tokens], dtype=np.float32)
    df = np.asarray([df_by_token[str(token)] for token in tokens], dtype=np.uint32)
    default_idf = np.asarray([math.log(1.0 + float(doc_count)) + 1.0], dtype=np.float32)
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            tokens=tokens,
            idf=idf,
            df=df,
            doc_count=np.asarray([int(doc_count)], dtype=np.int64),
            docs_with_tokens=np.asarray([int(docs_with_tokens)], dtype=np.int64),
            fasttext_dim=np.asarray([int(fasttext_dim)], dtype=np.int64),
            tokenizer=np.asarray([TOKEN_RE.pattern]),
            default_idf=default_idf,
            formula=np.asarray([formula]),
        )
    tmp.replace(path)
    log(f"[IDF] wrote {path}")


def compute_collection_idf(
    *,
    collection_tsv: Path,
    expected_docs: int,
    progress_every: int,
) -> tuple[dict[str, float], dict[str, int], float, dict[str, Any]]:
    df_by_token: dict[str, int] = {}
    doc_count = 0
    docs_with_tokens = 0
    log(f"[IDF] scanning collection for token document frequencies: {collection_tsv}")
    with collection_tsv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for source_row_id, raw_line in enumerate(handle):
            _doc_id, text = base.parse_collection_line(raw_line, source_row_id)
            seen = set(TOKEN_RE.findall(str(text).lower()))
            if seen:
                docs_with_tokens += 1
                for token in seen:
                    df_by_token[token] = df_by_token.get(token, 0) + 1
            doc_count += 1
            if int(progress_every) and doc_count % int(progress_every) == 0:
                log(f"[IDF] scanned {doc_count:,}/{int(expected_docs):,} unique_tokens={len(df_by_token):,}")
    if doc_count != int(expected_docs):
        raise RuntimeError(f"IDF scan saw {doc_count:,} docs, expected {int(expected_docs):,}")

    idf_by_token = {
        token: float(math.log((1.0 + float(doc_count)) / (1.0 + float(df))) + 1.0)
        for token, df in df_by_token.items()
    }
    default_idf = float(math.log(1.0 + float(doc_count)) + 1.0)
    metadata = {
        "idf_doc_count": int(doc_count),
        "idf_docs_with_tokens": int(docs_with_tokens),
        "idf_unique_token_count": int(len(idf_by_token)),
        "idf_default_for_unseen_token": float(default_idf),
        "idf_formula": "smooth_idf: log((1 + n_docs) / (1 + df)) + 1",
    }
    log(
        f"[IDF] computed docs={doc_count:,} docs_with_tokens={docs_with_tokens:,} "
        f"unique_tokens={len(idf_by_token):,}"
    )
    return idf_by_token, df_by_token, default_idf, metadata


def load_or_compute_idf(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    collection_tsv: Path,
    total_docs: int,
) -> tuple[dict[str, float] | None, float, dict[str, Any]]:
    if str(args.pooling) != "tfidf":
        return None, 1.0, {}
    cache_path = (
        Path(args.idf_cache_npz).expanduser().resolve()
        if args.idf_cache_npz
        else default_idf_cache_path(paths, str(args.fasttext_label)).resolve()
    )
    if cache_path.exists() and not bool(args.recompute_idf):
        idf_by_token, default_idf, metadata = load_idf_cache(cache_path)
        return idf_by_token, default_idf, metadata

    idf_by_token, df_by_token, default_idf, metadata = compute_collection_idf(
        collection_tsv=collection_tsv,
        expected_docs=int(total_docs),
        progress_every=int(args.progress_every),
    )
    write_idf_cache(
        cache_path,
        idf_by_token=idf_by_token,
        df_by_token=df_by_token,
        doc_count=int(metadata["idf_doc_count"]),
        docs_with_tokens=int(metadata["idf_docs_with_tokens"]),
        fasttext_dim=int(args.fasttext_dim),
    )
    metadata["idf_cache_npz"] = str(cache_path)
    return idf_by_token, default_idf, metadata


def load_fasttext_model(path: Path, expected_dim: int) -> tuple[Any, int]:
    try:
        import fasttext  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The fasttext Python package is required for .bin models. "
            "Install it in the hnsw environment, for example: "
            "python -m pip install fasttext-wheel"
        ) from exc

    log(f"[FASTTEXT] loading {path}")
    model = fasttext.load_model(str(path))
    dim = int(model.get_dimension())
    if dim != int(expected_dim):
        raise ValueError(f"fastText model dimension={dim}, expected {int(expected_dim)}")
    log(f"[FASTTEXT] loaded dim={dim}")
    return model, dim


def make_word_vector_fn(
    model: Any,
    *,
    dim: int,
    cache_size: int,
) -> Callable[[str], np.ndarray]:
    def uncached(token: str) -> np.ndarray:
        vector = np.asarray(model.get_word_vector(token), dtype=np.float32)
        if vector.shape != (int(dim),):
            raise ValueError(f"fastText vector for {token!r} has shape={vector.shape}, expected=({int(dim)},)")
        return vector

    if int(cache_size) == 0:
        return uncached

    @lru_cache(maxsize=int(cache_size))
    def cached(token: str) -> np.ndarray:
        return uncached(token)

    return cached


def embed_text(
    text: str,
    *,
    word_vector: Callable[[str], np.ndarray],
    dim: int,
    normalize: bool,
    pooling: str,
    idf_by_token: dict[str, float] | None,
    default_idf: float,
    tfidf_sublinear_tf: bool,
) -> tuple[np.ndarray, int, int, int, float]:
    tokens = TOKEN_RE.findall(str(text).lower())
    if not tokens:
        return np.zeros(int(dim), dtype=np.float32), 0, 0, 0, 0.0

    if str(pooling) == "mean":
        accum = np.zeros(int(dim), dtype=np.float32)
        for token in tokens:
            accum += word_vector(token)
        vector = (accum / float(len(tokens))).astype(np.float32, copy=False)
    elif str(pooling) == "tfidf":
        if idf_by_token is None:
            raise ValueError("idf_by_token must be provided when pooling='tfidf'")
        accum = np.zeros(int(dim), dtype=np.float32)
        weight_sum = 0.0
        for token, count in Counter(tokens).items():
            tf = float(count)
            if bool(tfidf_sublinear_tf):
                tf = 1.0 + math.log(tf)
            weight = tf * float(idf_by_token.get(token, default_idf))
            accum += np.float32(weight) * word_vector(token)
            weight_sum += weight
        if weight_sum <= 1e-12:
            vector = np.zeros(int(dim), dtype=np.float32)
        else:
            vector = (accum / np.float32(weight_sum)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported pooling={pooling!r}")

    raw_norm = float(np.linalg.norm(vector))
    if normalize:
        vector = base.normalize_row(vector)
    return vector, len(tokens), len(tokens), 0, raw_norm


def embed_queries(
    *,
    query_ids: list[str],
    queries_tsv: Path,
    word_vector: Callable[[str], np.ndarray],
    dim: int,
    normalize: bool,
    pooling: str,
    idf_by_token: dict[str, float] | None,
    default_idf: float,
    tfidf_sublinear_tf: bool,
    stats_csv: Path,
) -> tuple[np.ndarray, base.TextStats]:
    query_text_by_id = base.load_queries_tsv(queries_tsv)
    missing = [qid for qid in query_ids if qid not in query_text_by_id]
    if missing:
        raise KeyError(f"Missing {len(missing)} query texts; first missing IDs: {missing[:10]}")
    stats = base.TextStats()
    embeddings = np.empty((len(query_ids), int(dim)), dtype=np.float32)
    base.ensure_parent(stats_csv)
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "query_id", "token_count", "matched_count", "oov_count", "match_rate", "raw_norm"])
        for row_index, query_id in enumerate(query_ids):
            vector, token_count, matched_count, oov_count, raw_norm = embed_text(
                query_text_by_id[query_id],
                word_vector=word_vector,
                dim=int(dim),
                normalize=normalize,
                pooling=pooling,
                idf_by_token=idf_by_token,
                default_idf=float(default_idf),
                tfidf_sublinear_tf=bool(tfidf_sublinear_tf),
            )
            embeddings[row_index] = vector
            stats.add(token_count, matched_count)
            match_rate = float(matched_count / token_count) if token_count else 0.0
            writer.writerow([row_index, query_id, token_count, matched_count, oov_count, match_rate, raw_norm])
    return embeddings, stats


def derive_output_path(args: argparse.Namespace, root: Path) -> Path:
    if args.output_hdf5:
        return Path(args.output_hdf5).expanduser().resolve()
    size = "full" if int(args.sample_size) == 0 else f"{int(args.sample_size) // 1000}k"
    suffix = f"{size}-{str(args.sample_mode)}"
    if str(args.sample_mode) == "random":
        suffix += f"-seed{int(args.sample_seed)}"
    pooling_suffix = "" if str(args.pooling) == "mean" else f"-{str(args.pooling)}"
    label = str(args.fasttext_label).strip()
    name = f"msmarco-v1-{label}{pooling_suffix}-{suffix}-ip.hdf5"
    return root / name


def derive_ada_output_path(args: argparse.Namespace, static_output_hdf5: Path) -> Path:
    if args.ada_output_hdf5:
        return Path(args.ada_output_hdf5).expanduser().resolve()
    name = static_output_hdf5.name.replace(str(args.fasttext_label), "openai-ada2")
    if name == static_output_hdf5.name:
        name = static_output_hdf5.with_suffix("").name + "__ada_subset.hdf5"
    return static_output_hdf5.with_name(name)


def write_static_hdf5(
    *,
    args: argparse.Namespace,
    output_hdf5: Path,
    collection_tsv: Path,
    queries_tsv: Path,
    selected_rows: np.ndarray,
    total_docs: int,
    selection_source: str,
    query_ids: list[str],
    word_vector: Callable[[str], np.ndarray],
    dim: int,
    idf_by_token: dict[str, float] | None,
    default_idf: float,
    idf_metadata: dict[str, Any],
) -> dict[str, Any]:
    paths = base.sidecar_paths(output_hdf5)
    base.maybe_remove_existing([output_hdf5, *paths.values()], bool(args.overwrite))
    base.ensure_parent(output_hdf5)

    log(f"[QUERY] embedding {len(query_ids):,} queries")
    query_embeddings, query_stats = embed_queries(
        query_ids=query_ids,
        queries_tsv=queries_tsv,
        word_vector=word_vector,
        dim=int(dim),
        normalize=bool(args.normalize),
        pooling=str(args.pooling),
        idf_by_token=idf_by_token,
        default_idf=float(default_idf),
        tfidf_sublinear_tf=bool(args.tfidf_sublinear_tf),
        stats_csv=paths["query_stats"],
    )

    passage_stats = base.TextStats()
    doc_ids: list[str] = []
    train_count = int(selected_rows.size)
    log(f"[PASSAGE] embedding selected passages count={train_count:,}")
    with h5py.File(output_hdf5, "w") as h5f, paths["passage_stats"].open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stats_handle:
        train_ds = h5f.create_dataset(
            "train",
            shape=(train_count, int(dim)),
            dtype=np.float32,
            chunks=(min(int(args.write_batch), train_count), int(dim)),
        )
        h5f.create_dataset("test", data=query_embeddings, dtype=np.float32)
        stats_writer = csv.writer(stats_handle)
        stats_writer.writerow(
            [
                "row_index",
                "source_row_id",
                "doc_id",
                "token_count",
                "matched_count",
                "oov_count",
                "match_rate",
                "raw_norm",
            ]
        )
        pointer = 0
        next_row = int(selected_rows[pointer]) if train_count else -1
        with collection_tsv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for source_row_id, raw_line in enumerate(handle):
                if pointer >= train_count:
                    break
                if source_row_id != next_row:
                    continue
                doc_id, text = base.parse_collection_line(raw_line, source_row_id)
                vector, token_count, matched_count, oov_count, raw_norm = embed_text(
                    text,
                    word_vector=word_vector,
                    dim=int(dim),
                    normalize=bool(args.normalize),
                    pooling=str(args.pooling),
                    idf_by_token=idf_by_token,
                    default_idf=float(default_idf),
                    tfidf_sublinear_tf=bool(args.tfidf_sublinear_tf),
                )
                train_ds[pointer] = vector
                doc_ids.append(str(doc_id))
                passage_stats.add(token_count, matched_count)
                match_rate = float(matched_count / token_count) if token_count else 0.0
                stats_writer.writerow(
                    [pointer, int(source_row_id), doc_id, token_count, matched_count, oov_count, match_rate, raw_norm]
                )
                pointer += 1
                if int(args.progress_every) and (
                    pointer % int(args.progress_every) == 0 or pointer == train_count
                ):
                    log(f"[PASSAGE] embedded {pointer:,}/{train_count:,}")
                if pointer < train_count:
                    next_row = int(selected_rows[pointer])
        if pointer != train_count:
            raise RuntimeError(f"Embedded {pointer:,} passages, expected {train_count:,}")

        if args.skip_ground_truth:
            h5f.create_dataset(
                "neighbors",
                data=np.full((len(query_ids), int(args.neighbors_k)), -1, dtype=np.int32),
            )
            h5f.create_dataset(
                "distances",
                data=np.full((len(query_ids), int(args.neighbors_k)), np.nan, dtype=np.float32),
            )
        else:
            neighbors, distances = base.exact_topk_from_hdf5(
                train_ds=train_ds,
                queries=query_embeddings,
                k=int(args.neighbors_k),
                query_batch_size=int(args.exact_query_batch),
                doc_block_size=int(args.exact_doc_block),
            )
            h5f.create_dataset("neighbors", data=neighbors, dtype=np.int32)
            h5f.create_dataset("distances", data=distances, dtype=np.float32)

    base.write_train_row_csv(paths["train_rows"], selected_rows, doc_ids=doc_ids)
    base.write_query_id_csv(paths["query_ids"], query_ids)
    embedding_name = (
        f"{args.fasttext_name}_mean"
        if str(args.pooling) == "mean"
        else f"{args.fasttext_name}_tfidf_weighted_mean"
    )
    metric_note = (
        "L2-normalized mean fastText subword vectors with inner product equivalent to cosine for nonzero rows"
        if str(args.pooling) == "mean"
        else "L2-normalized TF-IDF weighted mean fastText subword vectors with inner product equivalent to cosine for nonzero rows"
    )
    metadata: dict[str, Any] = {
        "output_hdf5": str(output_hdf5),
        "created_at_unix": time.time(),
        "dataset": "msmarco-v1-passage",
        "embedding": embedding_name,
        "embedding_backend": "fasttext_bin",
        "pooling": str(args.pooling),
        "tfidf_sublinear_tf": bool(args.tfidf_sublinear_tf),
        "tokenizer": TOKEN_RE.pattern,
        "lowercase": True,
        "stopword_removal": False,
        "oov_policy": "fasttext_subword_vectors_for_all_tokens",
        "normalize": bool(args.normalize),
        "space": "ip",
        "metric_note": metric_note,
        "fasttext_dim": int(dim),
        "fasttext_bin": str(Path(args.fasttext_bin).expanduser().resolve()) if args.fasttext_bin else str(args.fasttext_name),
        "fasttext_label": str(args.fasttext_label),
        "token_vector_cache_size": int(args.token_vector_cache_size),
        "collection_tsv": str(collection_tsv),
        "queries_tsv": str(queries_tsv),
        "query_ids_csv": str(paths["query_ids"]),
        "train_rows_csv": str(paths["train_rows"]),
        "passage_stats_csv": str(paths["passage_stats"]),
        "query_stats_csv": str(paths["query_stats"]),
        "sample_size_requested": int(args.sample_size),
        "sample_mode": str(args.sample_mode),
        "sample_seed": int(args.sample_seed),
        "selection_source": str(selection_source),
        "total_docs_seen": int(total_docs),
        "train_count": int(train_count),
        "query_count": int(len(query_ids)),
        "embedding_dim": int(dim),
        "neighbors_k": int(args.neighbors_k),
        "ground_truth_mode": "skipped" if args.skip_ground_truth else "exact_block_inner_product",
        "exact_query_batch": int(args.exact_query_batch),
        "exact_doc_block": int(args.exact_doc_block),
        "passage_oov_summary": passage_stats.summary(),
        "query_oov_summary": query_stats.summary(),
    }
    metadata.update(idf_metadata)
    paths["meta"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[WRITE] {output_hdf5}")
    log(f"[WRITE] {paths['meta']}")
    return metadata


def main() -> int:
    args = parse_args()
    asset_paths = resolved_asset_paths(args)
    ensure_assets(args, asset_paths)
    output_hdf5 = derive_output_path(args, asset_paths["root"]).resolve()
    selected_rows, total_docs, selection_source = base.choose_selected_rows(args, asset_paths["collection_tsv"])
    query_ids = base.load_query_ids(Path(args.query_ids_csv).expanduser().resolve(), int(args.query_limit))

    log(
        f"[CONFIG] train_count={len(selected_rows):,} total_docs={total_docs:,} "
        f"queries={len(query_ids):,} pooling={args.pooling} output={output_hdf5}"
    )
    model, dim = load_fasttext_model(asset_paths["fasttext_bin"], int(args.fasttext_dim))
    word_vector = make_word_vector_fn(
        model,
        dim=int(dim),
        cache_size=int(args.token_vector_cache_size),
    )
    idf_by_token, default_idf, idf_metadata = load_or_compute_idf(
        args=args,
        paths=asset_paths,
        collection_tsv=asset_paths["collection_tsv"],
        total_docs=int(total_docs),
    )
    write_static_hdf5(
        args=args,
        output_hdf5=output_hdf5,
        collection_tsv=asset_paths["collection_tsv"],
        queries_tsv=asset_paths["queries_tsv"],
        selected_rows=selected_rows,
        total_docs=total_docs,
        selection_source=selection_source,
        query_ids=query_ids,
        word_vector=word_vector,
        dim=int(dim),
        idf_by_token=idf_by_token,
        default_idf=float(default_idf),
        idf_metadata=idf_metadata,
    )
    if args.write_ada_subset:
        base.write_ada_subset_hdf5(
            args=args,
            source_hdf5=Path(args.ada_source_hdf5).expanduser().resolve(),
            output_hdf5=derive_ada_output_path(args, output_hdf5),
            selected_rows=selected_rows,
            query_ids=query_ids,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
