#!/usr/bin/env python3
"""Build MSMARCO passage static GloVe HDF5 datasets.

The output follows the local ANN-Benchmarks-style layout:

  train      corpus passage embeddings from mean or TF-IDF-weighted GloVe pooling
  test       query embeddings
  neighbors  exact top-k neighbors in this embedding space
  distances  1 - inner_product for the same neighbors

The script also writes sidecar CSV/JSON files matching the conventions used by
the existing MSMARCO ada-002 HDF5 dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", "/home/kyungmin/vectordb/hnsw-playground/datasets")).expanduser()
WORK_SUBDIR = "msmarco_passage_glove_static"
COLLECTION_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
QUERIES_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
DEFAULT_ADA_HDF5 = DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip.hdf5"
DEFAULT_ADA_QUERY_IDS = DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip_query_ids.csv"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._'-][a-z0-9]+)*")


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--work-subdir", default=WORK_SUBDIR)
    parser.add_argument("--collection-tsv", type=Path, default=None)
    parser.add_argument("--queries-tsv", type=Path, default=None)
    parser.add_argument("--glove-txt", type=Path, default=None)
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--collection-url", default=COLLECTION_URL)
    parser.add_argument("--queries-url", default=QUERIES_URL)
    parser.add_argument("--glove-url", default=GLOVE_URL)
    parser.add_argument("--glove-dim", type=positive_int, default=300)
    parser.add_argument("--pooling", choices=("mean", "tfidf"), default="mean")
    parser.add_argument(
        "--idf-cache-npz",
        type=Path,
        default=None,
        help="Optional cache for collection IDF values used by --pooling tfidf.",
    )
    parser.add_argument("--recompute-idf", action="store_true")
    parser.add_argument(
        "--tfidf-sublinear-tf",
        action="store_true",
        help="Use 1 + log(tf) instead of raw term frequency for --pooling tfidf.",
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
    parser.add_argument("--neighbors-k", type=positive_int, default=10)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--write-batch", type=positive_int, default=8192)
    parser.add_argument("--exact-query-batch", type=positive_int, default=32)
    parser.add_argument("--exact-doc-block", type=positive_int, default=32768)
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
    return args


def resolved_asset_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.dataset_root).expanduser().resolve() / str(args.work_subdir)
    raw = root / "raw"
    return {
        "root": root,
        "raw": raw,
        "collection_tar": raw / "collection.tar.gz",
        "queries_tar": raw / "queries.tar.gz",
        "glove_zip": raw / "glove.6B.zip",
        "collection_tsv": Path(args.collection_tsv).expanduser().resolve()
        if args.collection_tsv
        else raw / "collection.tsv",
        "queries_tsv": Path(args.queries_tsv).expanduser().resolve()
        if args.queries_tsv
        else raw / "queries.dev.tsv",
        "glove_txt": Path(args.glove_txt).expanduser().resolve()
        if args.glove_txt
        else raw / f"glove.6B.{int(args.glove_dim)}d.txt",
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_remove_existing(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    if not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s): {joined}")
    for path in existing:
        path.unlink()


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[DOWNLOAD] exists {dest}")
        return

    ensure_parent(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_at = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    request = urllib.request.Request(url, headers=headers)

    log(f"[DOWNLOAD] {url} -> {dest}")
    with urllib.request.urlopen(request) as response:  # noqa: S310
        status = getattr(response, "status", None)
        mode = "ab" if resume_at and status == 206 else "wb"
        if mode == "wb":
            resume_at = 0
        total_header = response.headers.get("Content-Length")
        total = int(total_header) + resume_at if total_header and total_header.isdigit() else 0
        copied = resume_at
        next_report = copied + 100 * 1024 * 1024
        with tmp.open(mode + "") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                copied += len(chunk)
                if total and copied >= next_report:
                    log(f"[DOWNLOAD] {dest.name}: {copied / 1e9:.2f}/{total / 1e9:.2f} GB")
                    next_report += 100 * 1024 * 1024
    tmp.replace(dest)
    log(f"[DOWNLOAD] complete {dest}")


def extract_tar_member(tar_path: Path, member_basename: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[EXTRACT] exists {dest}")
        return
    ensure_parent(dest)
    log(f"[EXTRACT] {member_basename} from {tar_path}")
    with tarfile.open(tar_path, "r:gz") as archive:
        member = None
        for candidate in archive.getmembers():
            if Path(candidate.name).name == member_basename:
                member = candidate
                break
        if member is None:
            names = [Path(item.name).name for item in archive.getmembers()[:10]]
            raise FileNotFoundError(f"{member_basename!r} not found in {tar_path}; first members={names}")
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not extract {member.name} from {tar_path}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with source, tmp.open("wb") as out:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)


def extract_zip_member(zip_path: Path, member_basename: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[EXTRACT] exists {dest}")
        return
    ensure_parent(dest)
    log(f"[EXTRACT] {member_basename} from {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        member_name = None
        for name in archive.namelist():
            if Path(name).name == member_basename:
                member_name = name
                break
        if member_name is None:
            raise FileNotFoundError(f"{member_basename!r} not found in {zip_path}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with archive.open(member_name) as source, tmp.open("wb") as out:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)


def ensure_assets(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    paths["raw"].mkdir(parents=True, exist_ok=True)
    if args.download_assets:
        download_file(str(args.collection_url), paths["collection_tar"])
        download_file(str(args.queries_url), paths["queries_tar"])
        download_file(str(args.glove_url), paths["glove_zip"])
    if paths["collection_tar"].exists():
        extract_tar_member(paths["collection_tar"], "collection.tsv", paths["collection_tsv"])
    if paths["queries_tar"].exists():
        query_member = Path(paths["queries_tsv"]).name
        try:
            extract_tar_member(paths["queries_tar"], query_member, paths["queries_tsv"])
        except FileNotFoundError:
            if query_member == "queries.dev.tsv":
                raise
            extract_tar_member(paths["queries_tar"], "queries.dev.tsv", paths["queries_tsv"])
    if paths["glove_zip"].exists():
        extract_zip_member(paths["glove_zip"], f"glove.6B.{int(args.glove_dim)}d.txt", paths["glove_txt"])

    missing = [name for name in ("collection_tsv", "queries_tsv", "glove_txt") if not paths[name].exists()]
    if missing:
        hint = "pass explicit paths or rerun with --download-assets"
        raise FileNotFoundError(f"Missing required asset(s): {missing}; {hint}")


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def load_selected_rows_csv(path: Path) -> np.ndarray:
    rows: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        field_map = {field.lower(): field for field in reader.fieldnames}
        source_field = field_map.get("source_row_id") or field_map.get("row_id") or field_map.get("doc_id")
        if source_field is None:
            raise ValueError(f"{path} must contain source_row_id, row_id, or doc_id")
        for row in reader:
            rows.append(int(row[source_field]))
    if not rows:
        raise ValueError(f"No selected rows in {path}")
    selected = np.asarray(rows, dtype=np.int64)
    selected.sort()
    if np.unique(selected).size != selected.size:
        raise ValueError(f"{path} contains duplicate source rows")
    return selected


def choose_selected_rows(args: argparse.Namespace, collection_tsv: Path) -> tuple[np.ndarray, int, str]:
    if args.selected_rows_csv:
        total_docs = count_lines(collection_tsv)
        selected = load_selected_rows_csv(Path(args.selected_rows_csv).expanduser().resolve())
        if int(selected[-1]) >= int(total_docs):
            raise ValueError(f"Selected source row {int(selected[-1])} >= total docs {total_docs}")
        return selected, total_docs, f"selected_rows_csv:{args.selected_rows_csv}"

    sample_size = int(args.sample_size)
    total_docs = count_lines(collection_tsv)

    if sample_size == 0:
        selected = np.arange(total_docs, dtype=np.int64)
        return selected, total_docs, "full"
    if sample_size > total_docs:
        raise ValueError(f"--sample-size {sample_size:,} exceeds total docs {total_docs:,}")
    if str(args.sample_mode) == "first":
        selected = np.arange(sample_size, dtype=np.int64)
        return selected, total_docs, "first"

    rng = np.random.default_rng(int(args.sample_seed))
    selected = rng.choice(total_docs, size=sample_size, replace=False).astype(np.int64)
    selected.sort()
    return selected, total_docs, f"random_seed_{int(args.sample_seed)}"


def load_query_ids(path: Path, limit: int) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        field_map = {field.lower(): field for field in reader.fieldnames}
        id_field = field_map.get("query_id") or field_map.get("id") or field_map.get("_id")
        if id_field is None:
            raise ValueError(f"{path} must contain query_id")
        for row in reader:
            ids.append(str(row[id_field]))
            if limit and len(ids) >= int(limit):
                break
    if not ids:
        raise ValueError(f"No query IDs loaded from {path}")
    return ids


def load_queries_tsv(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            queries[str(parts[0])] = parts[1]
    if not queries:
        raise ValueError(f"No queries loaded from {path}")
    return queries


def write_query_id_csv(path: Path, query_ids: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "query_id"])
        for row_index, query_id in enumerate(query_ids):
            writer.writerow([row_index, query_id])


def write_train_row_csv(path: Path, selected_rows: np.ndarray, doc_ids: list[str] | None = None) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if doc_ids is None:
            writer.writerow(["row_index", "source_row_id"])
            for row_index, source_row_id in enumerate(np.asarray(selected_rows, dtype=np.int64)):
                writer.writerow([row_index, int(source_row_id)])
        else:
            writer.writerow(["row_index", "source_row_id", "doc_id"])
            for row_index, (source_row_id, doc_id) in enumerate(zip(selected_rows, doc_ids)):
                writer.writerow([row_index, int(source_row_id), str(doc_id)])


@dataclass
class TextStats:
    total_items: int = 0
    total_tokens: int = 0
    total_matched: int = 0
    zero_vector_items: int = 0
    token_counts: list[int] = field(default_factory=list)
    match_rates: list[float] = field(default_factory=list)

    def add(self, token_count: int, matched_count: int) -> None:
        self.total_items += 1
        self.total_tokens += int(token_count)
        self.total_matched += int(matched_count)
        if int(matched_count) == 0:
            self.zero_vector_items += 1
        if len(self.token_counts) < 1_000_000:
            self.token_counts.append(int(token_count))
            rate = float(matched_count) / float(token_count) if token_count else 0.0
            self.match_rates.append(rate)

    def summary(self) -> dict[str, Any]:
        token_values = np.asarray(self.token_counts, dtype=np.float64)
        rate_values = np.asarray(self.match_rates, dtype=np.float64)
        return {
            "total_items": int(self.total_items),
            "total_tokens": int(self.total_tokens),
            "total_matched_tokens": int(self.total_matched),
            "total_oov_tokens": int(self.total_tokens - self.total_matched),
            "overall_match_rate": (
                float(self.total_matched / self.total_tokens) if self.total_tokens else 0.0
            ),
            "zero_vector_items": int(self.zero_vector_items),
            "zero_vector_rate": (
                float(self.zero_vector_items / self.total_items) if self.total_items else 0.0
            ),
            "token_count_mean_sampled": float(np.mean(token_values)) if token_values.size else 0.0,
            "token_count_p50_sampled": float(np.percentile(token_values, 50)) if token_values.size else 0.0,
            "token_count_p95_sampled": float(np.percentile(token_values, 95)) if token_values.size else 0.0,
            "match_rate_mean_sampled": float(np.mean(rate_values)) if rate_values.size else 0.0,
            "match_rate_p05_sampled": float(np.percentile(rate_values, 5)) if rate_values.size else 0.0,
            "match_rate_p50_sampled": float(np.percentile(rate_values, 50)) if rate_values.size else 0.0,
        }


def load_glove(path: Path, expected_dim: int) -> tuple[dict[str, int], np.ndarray]:
    words: list[str] = []
    vectors: list[np.ndarray] = []
    seen: set[str] = set()
    log(f"[GLOVE] loading {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                word, rest = stripped.split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed GloVe line {line_index} in {path}") from exc
            vec = np.fromstring(rest, sep=" ", dtype=np.float32)
            if vec.size != int(expected_dim):
                raise ValueError(
                    f"GloVe line {line_index} has dim={vec.size}, expected {int(expected_dim)}"
                )
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
            vectors.append(vec)
            if line_index % 100_000 == 0:
                log(f"[GLOVE] loaded {line_index:,} lines")
    if not vectors:
        raise ValueError(f"No vectors loaded from {path}")
    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    vocab = {word: idx for idx, word in enumerate(words)}
    log(f"[GLOVE] loaded vocab={len(vocab):,} dim={matrix.shape[1]}")
    return vocab, matrix


def default_idf_cache_path(paths: dict[str, Path], glove_dim: int) -> Path:
    return paths["root"] / f"glove.6B.{int(glove_dim)}d_msmarco_collection_idf.npz"


def load_idf_cache(path: Path, expected_vocab_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        idf = np.asarray(data["idf"], dtype=np.float32)
        if idf.shape != (int(expected_vocab_size),):
            raise ValueError(
                f"IDF cache {path} has shape={idf.shape}, expected=({int(expected_vocab_size)},)"
            )
        metadata = {
            "idf_cache_npz": str(path),
            "idf_doc_count": int(np.asarray(data["doc_count"]).reshape(-1)[0]),
            "idf_docs_with_matched_tokens": int(
                np.asarray(data["docs_with_matched_tokens"]).reshape(-1)[0]
            ),
            "idf_formula": str(np.asarray(data["formula"]).reshape(-1)[0]),
        }
    log(
        f"[IDF] loaded {path} docs={metadata['idf_doc_count']:,} "
        f"matched_docs={metadata['idf_docs_with_matched_tokens']:,}"
    )
    return idf, metadata


def write_idf_cache(
    path: Path,
    *,
    idf: np.ndarray,
    df: np.ndarray,
    doc_count: int,
    docs_with_matched_tokens: int,
    glove_dim: int,
) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".part")
    formula = "smooth_idf: log((1 + n_docs) / (1 + df)) + 1"
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            idf=np.asarray(idf, dtype=np.float32),
            df=np.asarray(df, dtype=np.uint32),
            doc_count=np.asarray([int(doc_count)], dtype=np.int64),
            docs_with_matched_tokens=np.asarray([int(docs_with_matched_tokens)], dtype=np.int64),
            glove_dim=np.asarray([int(glove_dim)], dtype=np.int64),
            tokenizer=np.asarray([TOKEN_RE.pattern]),
            formula=np.asarray([formula]),
        )
    tmp.replace(path)
    log(f"[IDF] wrote {path}")


def compute_collection_idf(
    *,
    collection_tsv: Path,
    vocab: dict[str, int],
    expected_docs: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vocab_size = int(len(vocab))
    df = np.zeros(vocab_size, dtype=np.uint32)
    doc_count = 0
    docs_with_matched_tokens = 0
    log(f"[IDF] scanning collection for document frequencies: {collection_tsv}")
    with collection_tsv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for source_row_id, raw_line in enumerate(handle):
            _doc_id, text = parse_collection_line(raw_line, source_row_id)
            ids: set[int] = set()
            for token in TOKEN_RE.findall(str(text).lower()):
                idx = vocab.get(token)
                if idx is not None:
                    ids.add(int(idx))
            if ids:
                docs_with_matched_tokens += 1
                df[np.fromiter(ids, dtype=np.int64)] += 1
            doc_count += 1
            if int(progress_every) and doc_count % int(progress_every) == 0:
                log(f"[IDF] scanned {doc_count:,}/{int(expected_docs):,}")
    if doc_count != int(expected_docs):
        raise RuntimeError(f"IDF scan saw {doc_count:,} docs, expected {int(expected_docs):,}")
    idf = (np.log((1.0 + float(doc_count)) / (1.0 + df.astype(np.float64))) + 1.0).astype(np.float32)
    metadata = {
        "idf_doc_count": int(doc_count),
        "idf_docs_with_matched_tokens": int(docs_with_matched_tokens),
        "idf_formula": "smooth_idf: log((1 + n_docs) / (1 + df)) + 1",
    }
    log(
        f"[IDF] computed vocab={vocab_size:,} docs={doc_count:,} "
        f"matched_docs={docs_with_matched_tokens:,}"
    )
    return idf, df, metadata


def load_or_compute_idf(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    collection_tsv: Path,
    vocab: dict[str, int],
    total_docs: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if str(args.pooling) != "tfidf":
        return None, {}
    cache_path = (
        Path(args.idf_cache_npz).expanduser().resolve()
        if args.idf_cache_npz
        else default_idf_cache_path(paths, int(args.glove_dim)).resolve()
    )
    if cache_path.exists() and not bool(args.recompute_idf):
        idf, metadata = load_idf_cache(cache_path, int(len(vocab)))
        return idf, metadata

    idf, df, metadata = compute_collection_idf(
        collection_tsv=collection_tsv,
        vocab=vocab,
        expected_docs=int(total_docs),
        progress_every=int(args.progress_every),
    )
    write_idf_cache(
        cache_path,
        idf=idf,
        df=df,
        doc_count=int(metadata["idf_doc_count"]),
        docs_with_matched_tokens=int(metadata["idf_docs_with_matched_tokens"]),
        glove_dim=int(args.glove_dim),
    )
    metadata["idf_cache_npz"] = str(cache_path)
    return idf, metadata


def normalize_row(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def embed_text(
    text: str,
    *,
    vocab: dict[str, int],
    matrix: np.ndarray,
    normalize: bool,
    pooling: str,
    idf: np.ndarray | None,
    tfidf_sublinear_tf: bool,
) -> tuple[np.ndarray, int, int, int, float]:
    tokens = TOKEN_RE.findall(str(text).lower())
    ids: list[int] = []
    for token in tokens:
        idx = vocab.get(token)
        if idx is not None:
            ids.append(int(idx))
    dim = int(matrix.shape[1])
    if ids:
        id_array = np.asarray(ids, dtype=np.int64)
        if str(pooling) == "mean":
            vector = np.mean(matrix[id_array], axis=0, dtype=np.float32)
        elif str(pooling) == "tfidf":
            if idf is None:
                raise ValueError("idf must be provided when pooling='tfidf'")
            unique_ids, counts = np.unique(id_array, return_counts=True)
            tf = counts.astype(np.float32)
            if bool(tfidf_sublinear_tf):
                tf = (1.0 + np.log(tf)).astype(np.float32)
            weights = tf * np.asarray(idf[unique_ids], dtype=np.float32)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                vector = np.zeros(dim, dtype=np.float32)
            else:
                vector = np.average(matrix[unique_ids], axis=0, weights=weights).astype(np.float32)
        else:
            raise ValueError(f"Unsupported pooling={pooling!r}")
        raw_norm = float(np.linalg.norm(vector))
        if normalize:
            vector = normalize_row(vector)
    else:
        vector = np.zeros(dim, dtype=np.float32)
        raw_norm = 0.0
    return vector, len(tokens), len(ids), len(tokens) - len(ids), raw_norm


def sidecar_paths(output_hdf5: Path) -> dict[str, Path]:
    stem = output_hdf5.with_suffix("")
    return {
        "train_rows": stem.with_name(stem.name + "_train_rows.csv"),
        "query_ids": stem.with_name(stem.name + "_query_ids.csv"),
        "meta": stem.with_name(stem.name + "_meta.json"),
        "passage_stats": stem.with_name(stem.name + "_passage_embedding_stats.csv"),
        "query_stats": stem.with_name(stem.name + "_query_embedding_stats.csv"),
    }


def derive_output_path(args: argparse.Namespace, root: Path) -> Path:
    if args.output_hdf5:
        return Path(args.output_hdf5).expanduser().resolve()
    size = "full" if int(args.sample_size) == 0 else f"{int(args.sample_size) // 1000}k"
    suffix = f"{size}-{str(args.sample_mode)}"
    if str(args.sample_mode) == "random":
        suffix += f"-seed{int(args.sample_seed)}"
    pooling_suffix = "" if str(args.pooling) == "mean" else f"-{str(args.pooling)}"
    name = f"msmarco-v1-glove6b{int(args.glove_dim)}d{pooling_suffix}-{suffix}-ip.hdf5"
    return root / name


def topk_largest(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[1] <= int(k):
        idx = np.argsort(-scores, axis=1)
    else:
        kth = scores.shape[1] - int(k)
        idx = np.argpartition(scores, kth=kth, axis=1)[:, -int(k) :]
        local_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-local_scores, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
    values = np.take_along_axis(scores, idx, axis=1)
    return idx.astype(np.int32, copy=False), values.astype(np.float32, copy=False)


def exact_topk_from_hdf5(
    *,
    train_ds: Any,
    queries: np.ndarray,
    k: int,
    query_batch_size: int,
    doc_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_queries = int(queries.shape[0])
    total_docs = int(train_ds.shape[0])
    if total_docs < int(k):
        raise ValueError(f"k={int(k)} exceeds train_count={total_docs}")
    neighbors = np.empty((total_queries, int(k)), dtype=np.int32)
    distances = np.empty((total_queries, int(k)), dtype=np.float32)
    log(f"[GT] exact top-{int(k)} queries={total_queries:,} docs={total_docs:,}")
    for q_start in range(0, total_queries, int(query_batch_size)):
        q_end = min(q_start + int(query_batch_size), total_queries)
        qblock = np.asarray(queries[q_start:q_end], dtype=np.float32, order="C")
        qsize = int(q_end - q_start)
        best_scores = np.full((qsize, int(k)), -np.inf, dtype=np.float32)
        best_ids = np.full((qsize, int(k)), -1, dtype=np.int32)
        for d_start in range(0, total_docs, int(doc_block_size)):
            d_end = min(d_start + int(doc_block_size), total_docs)
            dblock = np.asarray(train_ds[d_start:d_end], dtype=np.float32, order="C")
            scores = qblock @ dblock.T
            local_ids, local_scores = topk_largest(scores, int(k))
            local_ids = local_ids + int(d_start)
            merged_scores = np.concatenate([best_scores, local_scores], axis=1)
            merged_ids = np.concatenate([best_ids, local_ids], axis=1)
            keep, keep_scores = topk_largest(merged_scores, int(k))
            best_scores = keep_scores
            best_ids = np.take_along_axis(merged_ids, keep, axis=1).astype(np.int32, copy=False)
        neighbors[q_start:q_end] = best_ids
        distances[q_start:q_end] = (1.0 - best_scores).astype(np.float32, copy=False)
        log(f"[GT] finished query rows {q_start:,}..{q_end - 1:,}")
    return neighbors, distances


def embed_queries(
    *,
    query_ids: list[str],
    queries_tsv: Path,
    vocab: dict[str, int],
    matrix: np.ndarray,
    normalize: bool,
    pooling: str,
    idf: np.ndarray | None,
    tfidf_sublinear_tf: bool,
    stats_csv: Path,
) -> tuple[np.ndarray, TextStats]:
    query_text_by_id = load_queries_tsv(queries_tsv)
    missing = [qid for qid in query_ids if qid not in query_text_by_id]
    if missing:
        raise KeyError(f"Missing {len(missing)} query texts; first missing IDs: {missing[:10]}")
    stats = TextStats()
    embeddings = np.empty((len(query_ids), int(matrix.shape[1])), dtype=np.float32)
    ensure_parent(stats_csv)
    with stats_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "query_id", "token_count", "matched_count", "oov_count", "match_rate", "raw_norm"])
        for row_index, query_id in enumerate(query_ids):
            vector, token_count, matched_count, oov_count, raw_norm = embed_text(
                query_text_by_id[query_id],
                vocab=vocab,
                matrix=matrix,
                normalize=normalize,
                pooling=pooling,
                idf=idf,
                tfidf_sublinear_tf=bool(tfidf_sublinear_tf),
            )
            embeddings[row_index] = vector
            stats.add(token_count, matched_count)
            match_rate = float(matched_count / token_count) if token_count else 0.0
            writer.writerow([row_index, query_id, token_count, matched_count, oov_count, match_rate, raw_norm])
    return embeddings, stats


def parse_collection_line(raw_line: str, source_row_id: int) -> tuple[str, str]:
    line = raw_line.rstrip("\n")
    parts = line.split("\t", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(source_row_id), ""


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
    vocab: dict[str, int],
    glove_matrix: np.ndarray,
    idf: np.ndarray | None,
    idf_metadata: dict[str, Any],
) -> dict[str, Any]:
    paths = sidecar_paths(output_hdf5)
    maybe_remove_existing([output_hdf5, *paths.values()], bool(args.overwrite))
    ensure_parent(output_hdf5)

    log(f"[QUERY] embedding {len(query_ids):,} queries")
    query_embeddings, query_stats = embed_queries(
        query_ids=query_ids,
        queries_tsv=queries_tsv,
        vocab=vocab,
        matrix=glove_matrix,
        normalize=bool(args.normalize),
        pooling=str(args.pooling),
        idf=idf,
        tfidf_sublinear_tf=bool(args.tfidf_sublinear_tf),
        stats_csv=paths["query_stats"],
    )

    passage_stats = TextStats()
    doc_ids: list[str] = []
    train_count = int(selected_rows.size)
    dim = int(glove_matrix.shape[1])
    log(f"[PASSAGE] embedding selected passages count={train_count:,}")
    with h5py.File(output_hdf5, "w") as h5f, paths["passage_stats"].open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stats_handle:
        train_ds = h5f.create_dataset(
            "train",
            shape=(train_count, dim),
            dtype=np.float32,
            chunks=(min(int(args.write_batch), train_count), dim),
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
                doc_id, text = parse_collection_line(raw_line, source_row_id)
                vector, token_count, matched_count, oov_count, raw_norm = embed_text(
                    text,
                    vocab=vocab,
                    matrix=glove_matrix,
                    normalize=bool(args.normalize),
                    pooling=str(args.pooling),
                    idf=idf,
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
            h5f.create_dataset("neighbors", data=np.full((len(query_ids), int(args.neighbors_k)), -1, dtype=np.int32))
            h5f.create_dataset("distances", data=np.full((len(query_ids), int(args.neighbors_k)), np.nan, dtype=np.float32))
        else:
            neighbors, distances = exact_topk_from_hdf5(
                train_ds=train_ds,
                queries=query_embeddings,
                k=int(args.neighbors_k),
                query_batch_size=int(args.exact_query_batch),
                doc_block_size=int(args.exact_doc_block),
            )
            h5f.create_dataset("neighbors", data=neighbors, dtype=np.int32)
            h5f.create_dataset("distances", data=distances, dtype=np.float32)

    write_train_row_csv(paths["train_rows"], selected_rows, doc_ids=doc_ids)
    write_query_id_csv(paths["query_ids"], query_ids)
    embedding_name = (
        f"glove.6B.{int(args.glove_dim)}d_mean"
        if str(args.pooling) == "mean"
        else f"glove.6B.{int(args.glove_dim)}d_tfidf_weighted_mean"
    )
    metric_note = (
        "L2-normalized mean vectors with inner product equivalent to cosine for nonzero rows"
        if str(args.pooling) == "mean"
        else "L2-normalized TF-IDF weighted mean vectors with inner product equivalent to cosine for nonzero rows"
    )
    metadata: dict[str, Any] = {
        "output_hdf5": str(output_hdf5),
        "created_at_unix": time.time(),
        "dataset": "msmarco-v1-passage",
        "embedding": embedding_name,
        "pooling": str(args.pooling),
        "tfidf_sublinear_tf": bool(args.tfidf_sublinear_tf),
        "tokenizer": TOKEN_RE.pattern,
        "lowercase": True,
        "stopword_removal": False,
        "oov_policy": "skip_oov_tokens_zero_vector_if_all_oov",
        "normalize": bool(args.normalize),
        "space": "ip",
        "metric_note": metric_note,
        "glove_dim": int(args.glove_dim),
        "glove_vocab_size": int(len(vocab)),
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


def take_h5_rows(dataset: Any, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    unique_ids, inverse = np.unique(sorted_ids, return_inverse=True)
    values = np.asarray(dataset[unique_ids], dtype=np.float32)
    sorted_values = values[inverse]
    out = np.empty((ids.size,) + values.shape[1:], dtype=np.float32)
    out[order] = sorted_values
    return out


def derive_ada_output_path(args: argparse.Namespace, static_output_hdf5: Path) -> Path:
    if args.ada_output_hdf5:
        return Path(args.ada_output_hdf5).expanduser().resolve()
    name = static_output_hdf5.name.replace("glove6b300d", "openai-ada2")
    name = name.replace(f"glove6b{int(args.glove_dim)}d", "openai-ada2")
    if name == static_output_hdf5.name:
        name = static_output_hdf5.with_suffix("").name + "__ada_subset.hdf5"
    return static_output_hdf5.with_name(name)


def write_ada_subset_hdf5(
    *,
    args: argparse.Namespace,
    source_hdf5: Path,
    output_hdf5: Path,
    selected_rows: np.ndarray,
    query_ids: list[str],
) -> None:
    paths = sidecar_paths(output_hdf5)
    maybe_remove_existing([output_hdf5, paths["train_rows"], paths["query_ids"], paths["meta"]], bool(args.overwrite))
    ensure_parent(output_hdf5)
    train_count = int(selected_rows.size)
    log(f"[ADA] writing subset {output_hdf5} rows={train_count:,}")
    with h5py.File(source_hdf5, "r") as src, h5py.File(output_hdf5, "w") as out:
        if "train" not in src or "test" not in src:
            raise KeyError(f"{source_hdf5} must contain train and test datasets")
        source_train = src["train"]
        source_test = src["test"]
        dim = int(source_train.shape[1])
        train_ds = out.create_dataset(
            "train",
            shape=(train_count, dim),
            dtype=np.float32,
            chunks=(min(int(args.write_batch), train_count), dim),
        )
        for start in range(0, train_count, int(args.write_batch)):
            end = min(start + int(args.write_batch), train_count)
            train_ds[start:end] = take_h5_rows(source_train, selected_rows[start:end])
            if int(args.progress_every) and (end % int(args.progress_every) == 0 or end == train_count):
                log(f"[ADA] copied {end:,}/{train_count:,}")
        test = np.asarray(source_test[: len(query_ids)], dtype=np.float32)
        out.create_dataset("test", data=test, dtype=np.float32)
        if args.skip_ground_truth:
            out.create_dataset("neighbors", data=np.full((len(query_ids), int(args.neighbors_k)), -1, dtype=np.int32))
            out.create_dataset("distances", data=np.full((len(query_ids), int(args.neighbors_k)), np.nan, dtype=np.float32))
        else:
            neighbors, distances = exact_topk_from_hdf5(
                train_ds=train_ds,
                queries=test,
                k=int(args.neighbors_k),
                query_batch_size=int(args.exact_query_batch),
                doc_block_size=int(args.exact_doc_block),
            )
            out.create_dataset("neighbors", data=neighbors, dtype=np.int32)
            out.create_dataset("distances", data=distances, dtype=np.float32)

    write_train_row_csv(paths["train_rows"], selected_rows)
    write_query_id_csv(paths["query_ids"], query_ids)
    metadata = {
        "output_hdf5": str(output_hdf5),
        "source_hdf5": str(source_hdf5),
        "created_at_unix": time.time(),
        "dataset": "msmarco-v1-passage",
        "embedding": "openai-ada2",
        "subset_aligned_to_static_hdf5": True,
        "space": "ip",
        "train_rows_csv": str(paths["train_rows"]),
        "query_ids_csv": str(paths["query_ids"]),
        "train_count": int(train_count),
        "query_count": int(len(query_ids)),
        "neighbors_k": int(args.neighbors_k),
        "ground_truth_mode": "skipped" if args.skip_ground_truth else "exact_block_inner_product",
        "sample_size_requested": int(args.sample_size),
        "sample_mode": str(args.sample_mode),
        "sample_seed": int(args.sample_seed),
    }
    paths["meta"].write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[ADA] wrote {output_hdf5}")


def main() -> int:
    args = parse_args()
    asset_paths = resolved_asset_paths(args)
    ensure_assets(args, asset_paths)
    output_hdf5 = derive_output_path(args, asset_paths["root"]).resolve()
    selected_rows, total_docs, selection_source = choose_selected_rows(args, asset_paths["collection_tsv"])
    query_ids = load_query_ids(Path(args.query_ids_csv).expanduser().resolve(), int(args.query_limit))

    log(
        f"[CONFIG] train_count={len(selected_rows):,} total_docs={total_docs:,} "
        f"queries={len(query_ids):,} pooling={args.pooling} output={output_hdf5}"
    )
    vocab, glove_matrix = load_glove(asset_paths["glove_txt"], int(args.glove_dim))
    idf, idf_metadata = load_or_compute_idf(
        args=args,
        paths=asset_paths,
        collection_tsv=asset_paths["collection_tsv"],
        vocab=vocab,
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
        vocab=vocab,
        glove_matrix=glove_matrix,
        idf=idf,
        idf_metadata=idf_metadata,
    )
    if args.write_ada_subset:
        write_ada_subset_hdf5(
            args=args,
            source_hdf5=Path(args.ada_source_hdf5).expanduser().resolve(),
            output_hdf5=derive_ada_output_path(args, output_hdf5),
            selected_rows=selected_rows,
            query_ids=query_ids,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
