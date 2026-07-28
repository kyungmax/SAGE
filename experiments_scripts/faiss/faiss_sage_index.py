"""Faiss-backed SAGE index adapter.

The class exposes the backend API used by the shared SAGE pipeline. LID
sampling, hide-node probes, path traces, adaptive search dispatch, and
drilldown first-hit instrumentation all run through Faiss.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any, Iterator

import faiss
import numpy as np


def _metric_for_space(space: str) -> tuple[int, bool]:
    normalized = False
    if space == "l2":
        return faiss.METRIC_L2, normalized
    if space == "ip":
        return faiss.METRIC_INNER_PRODUCT, normalized
    if space == "cosine":
        normalized = True
        return faiss.METRIC_INNER_PRODUCT, normalized
    raise ValueError(f"Unsupported space: {space!r}")


@contextmanager
def _faiss_num_threads(num_threads: int | None) -> Iterator[None]:
    prev_threads = None
    if num_threads is not None and int(num_threads) > 0:
        prev_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(int(num_threads))
    try:
        yield
    finally:
        if prev_threads is not None:
            faiss.omp_set_num_threads(int(prev_threads))



CHR_EMA_DECAY = 0.8
CHR_EMA_UPDATE = 1.0 - CHR_EMA_DECAY





def _selector_from_filter(filter_arg):
    if filter_arg is None:
        return None
    if isinstance(filter_arg, faiss.IDSelector):
        return filter_arg
    raise NotImplementedError(
        "Faiss HNSW backend accepts filter only as a faiss.IDSelector; "
        "Python callables are not supported."
    )


class Index:
    """Faiss-backed wrapper exposing the shared SAGE backend API."""

    def __init__(self, space: str, dim: int):
        self.space = str(space)
        self.dim = int(dim)
        self._metric, self._normalize = _metric_for_space(self.space)
        self._index: faiss.IndexHNSWFlat | None = None
        self._max_elements = 0
        self._num_threads = faiss.omp_get_max_threads()
        self._cached_lids: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"<faiss_sage_index.Index(space={self.space!r}, dim={self.dim})>"

    def _require_index(self) -> faiss.IndexHNSWFlat:
        if self._index is None:
            raise RuntimeError("Index is not initialized.")
        return self._index

    def _prepare_vectors(self, data: np.ndarray) -> np.ndarray:
        vectors = np.asarray(data, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Wrong dimensionality: expected {self.dim}, got {vectors.shape[1]}"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._normalize:
            vectors = vectors.copy()
            faiss.normalize_L2(vectors)
        return vectors

    def set_num_threads(self, num_threads: int) -> None:
        self._num_threads = int(num_threads)

    def init_index(
        self,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 100,
        allow_replace_deleted: bool = False,
    ) -> None:
        del random_seed
        if allow_replace_deleted:
            raise NotImplementedError("Faiss HNSW backend does not support deleted-slot reuse.")
        if self._index is not None:
            raise RuntimeError("Index is already initialized.")

        index = faiss.IndexHNSWFlat(self.dim, int(M), self._metric)
        index.hnsw.efConstruction = int(ef_construction)
        self._index = index
        self._max_elements = int(max_elements)
        self._cached_lids = None

    def add_items(
        self,
        data,
        ids=None,
        num_threads: int = -1,
        replace_deleted: bool = False,
    ) -> None:
        if replace_deleted:
            raise NotImplementedError("Faiss HNSW backend does not support replace_deleted.")

        index = self._require_index()
        vectors = self._prepare_vectors(data)
        n = int(vectors.shape[0])
        if ids is not None:
            ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
            if ids_arr.shape[0] != n:
                raise ValueError("ids length does not match the number of vectors.")
            expected = np.arange(index.ntotal, index.ntotal + n, dtype=np.int64)
            if not np.array_equal(ids_arr, expected):
                raise NotImplementedError(
                    "Faiss SAGE backend currently supports only sequential ids."
                )
        with _faiss_num_threads(self._num_threads if num_threads <= 0 else num_threads):
            index.add(vectors)
        self._cached_lids = None

    def save_index(self, path_to_index: str) -> None:
        faiss.write_index(self._require_index(), str(Path(path_to_index)))

    def load_index(
        self,
        path_to_index: str,
        max_elements: int = 0,
        allow_replace_deleted: bool = False,
    ) -> None:
        if allow_replace_deleted:
            raise NotImplementedError("Faiss HNSW backend does not support deleted-slot reuse.")
        index = faiss.read_index(str(Path(path_to_index)))
        if not isinstance(index, faiss.IndexHNSWFlat):
            raise TypeError(f"Expected faiss.IndexHNSWFlat, got {type(index)!r}")
        if int(index.d) != self.dim:
            raise ValueError(f"Loaded index dim {int(index.d)} does not match expected {self.dim}.")
        self._index = index
        self._max_elements = max(int(max_elements), int(index.ntotal))
        self._cached_lids = None

    def set_ef(self, ef: int) -> None:
        self._require_index().hnsw.efSearch = int(ef)

    def get_max_elements(self) -> int:
        return int(self._max_elements)

    def get_current_count(self) -> int:
        return int(self._require_index().ntotal)

    @property
    def M(self) -> int:
        return int(self._require_index().hnsw.nb_neighbors(1))

    @property
    def ef_construction(self) -> int:
        return int(self._require_index().hnsw.efConstruction)

    @property
    def ef(self) -> int:
        return int(self._require_index().hnsw.efSearch)

    @ef.setter
    def ef(self, value: int) -> None:
        self.set_ef(int(value))

    def knn_query(self, data, k: int = 1, num_threads: int = -1, filter=None):
        selector = _selector_from_filter(filter)

        index = self._require_index()
        vectors = self._prepare_vectors(data)
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(index.hnsw.efSearch)
        params.sel = selector

        with _faiss_num_threads(self._num_threads if num_threads <= 0 else num_threads):
            distances, labels = index.search(vectors, int(k), params=params)

        if self._metric == faiss.METRIC_INNER_PRODUCT:
            distances = 1.0 - distances
            distances[labels < 0] = np.inf
        return labels, distances

    def knn_query_hide_node(self, data, k: int, hide_labels, num_threads: int = -1, filter=None):
        native_method = getattr(self._require_index(), "knn_query_hide_node", None)
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native knn_query_hide_node(). "
                "Rebuild/install Faiss with SAGE hide-node instrumentation."
            )
        vectors = self._prepare_vectors(data)
        hidden = np.asarray(hide_labels, dtype=np.int64)
        return native_method(
            vectors,
            hidden,
            k=int(k),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            filter=filter,
        )

    def knn_query_adaptive_light(
        self,
        data,
        k: int = 1,
        ef_init: int = 128,
        enable_stop: bool = True,
        num_threads: int = -1,
        filter=None,
        early_stop_ratio: float = 0.6,
        tmin_pops: int = 25,
        super_easy_gamma_ratio: float = float("nan"),
        mid_easy_upper_gamma_ratio: float = float("nan"),
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ):
        vectors = self._prepare_vectors(data)
        return self._require_index().knn_query_adaptive_light(
            vectors,
            k=int(k),
            ef_init=int(ef_init),
            enable_stop=bool(enable_stop),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            early_stop_ratio=float(early_stop_ratio),
            tmin_pops=int(tmin_pops),
            super_easy_gamma_ratio=float(super_easy_gamma_ratio),
            mid_easy_upper_gamma_ratio=float(mid_easy_upper_gamma_ratio),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
            filter=filter,
        )

    def _paper_bucket_query_method(self):
        index = self._require_index()
        method = getattr(index, "knn_query_sage", None)
        if method is not None:
            return method
        return index.knn_query_adaptive_light_paper_bucket

    def knn_query_adaptive_light_paper_bucket(
        self,
        data,
        k: int = 1,
        ef_init: int = 128,
        enable_stop: bool = True,
        num_threads: int = -1,
        filter=None,
        early_stop_ratio: float = 0.6,
        tmin_pops: int = 25,
        paper_bucket_count: int = 4,
        bucket_gamma_ratios=(),
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ):
        vectors = self._prepare_vectors(data)
        return self._paper_bucket_query_method()(
            vectors,
            k=int(k),
            ef_init=int(ef_init),
            enable_stop=bool(enable_stop),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            filter=filter,
            early_stop_ratio=float(early_stop_ratio),
            tmin_pops=int(tmin_pops),
            paper_bucket_count=int(paper_bucket_count),
            bucket_gamma_ratios=list(bucket_gamma_ratios),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def knn_query_adaptive_analysis_paper_bucket(
        self,
        data,
        k: int = 1,
        ef_init: int = 128,
        ef_max: int | None = None,
        enable_stop: bool = True,
        num_threads: int = -1,
        filter=None,
        early_stop_ratio: float = 0.6,
        tmin_pops: int = 25,
        paper_bucket_count: int = 4,
        bucket_gamma_ratios=(),
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ):
        native_method = getattr(
            self._require_index(),
            "knn_query_adaptive_analysis_paper_bucket",
            None,
        )
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native "
                "knn_query_adaptive_analysis_paper_bucket(). Rebuild Faiss with "
                "the SAGE analysis instrumentation."
            )

        vectors = self._prepare_vectors(data)
        return native_method(
            vectors,
            k=int(k),
            ef_init=int(ef_init),
            ef_max=None if ef_max is None else int(ef_max),
            enable_stop=bool(enable_stop),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            filter=filter,
            early_stop_ratio=float(early_stop_ratio),
            tmin_pops=int(tmin_pops),
            paper_bucket_count=int(paper_bucket_count),
            bucket_gamma_ratios=list(bucket_gamma_ratios),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def knn_query_sage(self, *args, **kwargs):
        return self.knn_query_adaptive_light_paper_bucket(*args, **kwargs)

    def _reconstruct_batch(self, ids: np.ndarray) -> np.ndarray:
        return self._require_index().reconstruct_batch(
            np.ascontiguousarray(ids, dtype=np.int64)
        )

    def _compute_lids_for_ids(
        self,
        ids: np.ndarray,
        *,
        k_lid: int,
        num_threads: int,
    ) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0,), dtype=np.float32)
        native_method = getattr(self._require_index(), "compute_internal_lids", None)
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native compute_internal_lids(). "
                "Rebuild/install Faiss with SAGE LID instrumentation."
            )
        return np.asarray(
            native_method(
                ids,
                k_lid=int(k_lid),
                num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            ),
            dtype=np.float32,
        )

    def calc_lids_internal(self, k_lid: int, num_threads: int = -1) -> None:
        total = self.get_current_count()
        lids = np.full(total, np.nan, dtype=np.float32)
        batch_size = 4096
        effective_threads = self._num_threads if num_threads <= 0 else int(num_threads)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch_ids = np.arange(start, stop, dtype=np.int64)
            lids[start:stop] = self._compute_lids_for_ids(
                batch_ids,
                k_lid=int(k_lid),
                num_threads=effective_threads,
            )
        self._cached_lids = lids

    def calc_lids_internal_sampled(
        self,
        k_lid: int,
        sample_fraction: float = 0.001,
        min_sample_size: int = 1000,
        random_seed: int = 42,
        num_threads: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = self.get_current_count()
        if total <= 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        sample_size = max(int(min_sample_size), int(math.ceil(total * float(sample_fraction))))
        sample_size = min(sample_size, total)
        rng = np.random.default_rng(int(random_seed))
        effective_threads = self._num_threads if num_threads <= 0 else int(num_threads)

        seen: set[int] = set()
        finite_query_ids: list[np.ndarray] = []
        finite_lids: list[np.ndarray] = []
        finite_count = 0

        # Some angular datasets contain zero vectors or many exact duplicates.
        # Their MLE-LID is undefined, so keep drawing unsampled ids until the
        # requested finite pool size is reached.
        draw_size = sample_size
        max_attempts = 64
        attempts = 0
        while finite_count < sample_size and len(seen) < total and attempts < max_attempts:
            attempts += 1
            draw_size = min(int(draw_size), total - len(seen))
            if draw_size <= 0:
                break

            raw_ids = rng.choice(total, size=draw_size, replace=False).astype(np.int64)
            if seen:
                raw_ids = np.asarray(
                    [int(query_id) for query_id in raw_ids if int(query_id) not in seen],
                    dtype=np.int64,
                )
            if raw_ids.size == 0:
                draw_size = min(max(draw_size * 2, 1), total - len(seen))
                continue

            seen.update(int(query_id) for query_id in raw_ids)
            query_ids = np.sort(raw_ids)
            lids = self._compute_lids_for_ids(
                query_ids,
                k_lid=int(k_lid),
                num_threads=effective_threads,
            )
            finite_mask = np.isfinite(lids)
            if np.any(finite_mask):
                finite_query_ids.append(query_ids[finite_mask])
                finite_lids.append(lids[finite_mask])
                finite_count += int(np.count_nonzero(finite_mask))

            remaining = sample_size - finite_count
            draw_size = max(remaining * 2, min(1024, max(remaining, 1)))

        if not finite_query_ids:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        query_ids = np.concatenate(finite_query_ids)[:sample_size].astype(np.int64, copy=False)
        lids = np.concatenate(finite_lids)[:sample_size].astype(np.float32, copy=False)
        order = np.argsort(query_ids)
        return query_ids[order], lids[order]

    def sample_internal_lids(
        self,
        k_lid: int,
        sample_fraction: float = 0.001,
        min_sample_size: int = 1000,
        random_seed: int = 42,
        num_threads: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.calc_lids_internal_sampled(
            k_lid=int(k_lid),
            sample_fraction=float(sample_fraction),
            min_sample_size=int(min_sample_size),
            random_seed=int(random_seed),
            num_threads=int(num_threads),
        )

    def get_lids(self) -> np.ndarray:
        if self._cached_lids is None:
            raise RuntimeError("No cached internal LIDs. Call calc_lids_internal() first.")
        return self._cached_lids.copy()

    def knn_query_beam_width_first_target_hit_step(
        self,
        data,
        target_labels,
        target_hits,
        k: int = 1,
        ef_before: int = 128,
        switch_pop: int = 0,
        switch_full_pop: int = 0,
        ef_after: int = 128,
        num_threads: int = -1,
        filter=None,
    ):
        native_method = getattr(
            self._require_index(),
            "knn_query_beam_width_first_target_hit_step",
            None,
        )
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native "
                "knn_query_beam_width_first_target_hit_step(). Rebuild Faiss with "
                "the SAGE first-hit instrumentation before running drilldown."
            )

        vectors = self._prepare_vectors(data)
        return native_method(
            vectors,
            np.asarray(target_labels, dtype=np.int64),
            np.asarray(target_hits, dtype=np.uint64),
            k=int(k),
            ef_before=int(ef_before),
            switch_pop=int(switch_pop),
            switch_full_pop=int(switch_full_pop),
            ef_after=int(ef_after),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            filter=filter,
        )

    @staticmethod
    def _native_trace_to_paths(trace: dict[str, np.ndarray]) -> tuple[list[list[dict[str, Any]]], int, list[float]]:
        step_counts = np.asarray(trace["step_counts"], dtype=np.int64)
        total_distance_count = int(np.asarray(trace["distance_counts"], dtype=np.uint64).sum())
        closest_dists = [float(value) for value in np.asarray(trace["closest_dists"], dtype=np.float32)]
        paths: list[list[dict[str, Any]]] = []
        empty_vec = np.empty((0,), dtype=np.float32)
        for row in range(step_counts.shape[0]):
            row_path: list[dict[str, Any]] = []
            for step in range(int(step_counts[row])):
                row_path.append(
                    {
                        "node_label": int(trace["node_labels"][row, step]),
                        "node_internal_lid": float("nan"),
                        "rs_size": int(trace["rs_sizes"][row, step]),
                        "rs_size_after": int(trace["rs_sizes_after"][row, step]),
                        "is_full_pop_after": bool(trace["is_full_pop_after"][row, step]),
                        "full_pop_count_after": int(trace["full_pop_counts_after"][row, step]),
                        "popped_degree": int(trace["popped_degrees"][row, step]),
                        "unvisited_neighbor_count": int(trace["unvisited_counts"][row, step]),
                        "accepted_neighbor_count": int(trace["accepted_counts"][row, step]),
                        "runtime_accepted_rate": float(trace["runtime_accepted_rates"][row, step]),
                        "runtime_chr": float(trace["runtime_cfrs"][row, step]),
                        "runtime_smoothed_chr": float(trace["runtime_smoothed_cfrs"][row, step]),
                        "runtime_classify_chr_mean": float("nan"),
                        "runtime_classification_evaluated": False,
                        "runtime_is_easy_query": False,
                        "runtime_is_super_easy_query": False,
                        "runtime_is_mid_easy_query": False,
                        "runtime_effective_ef": int(trace.get("ef", 0)),
                        "internal_dist": float(trace["internal_dists"][row, step]),
                        "popped_query_dist": float(trace["popped_query_dists"][row, step]),
                        "furthest_dist": float(trace["furthest_dists"][row, step]),
                        "best_dist": float(trace["best_dists"][row, step]),
                        "top_k_dist": float(trace["top_k_dists"][row, step]),
                        "ef_half_dist": float(trace["ef_half_dists"][row, step]),
                        "ef_quarter_dist": float(trace["ef_quarter_dists"][row, step]),
                        "sqrt_ef_dist": float(trace["sqrt_ef_dists"][row, step]),
                        "top_2k_dist": float(trace["top_2k_dists"][row, step]),
                        "top_3k_dist": float(trace["top_3k_dists"][row, step]),
                        "furthest_vec": empty_vec,
                    }
                )
            paths.append(row_path)
        return paths, total_distance_count, closest_dists

    def _native_layer0_trace(
        self,
        data,
        *,
        k: int,
        ef: int,
        hide_labels=None,
        num_threads: int = -1,
    ) -> tuple[list[list[dict[str, Any]]], int, list[float]]:
        native_method = getattr(self._require_index(), "search_layer0_trace", None)
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native search_layer0_trace(). "
                "Rebuild/install Faiss with SAGE trace instrumentation."
            )
        vectors = self._prepare_vectors(data)
        trace = native_method(
            vectors,
            k=int(k),
            ef=int(ef),
            hide_labels=None if hide_labels is None else np.asarray(hide_labels, dtype=np.int64),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
        )
        trace["ef"] = int(ef)
        return self._native_trace_to_paths(trace)

    def _native_chr_summary(
        self,
        data,
        *,
        k: int,
        ef: int,
        hide_labels=None,
        num_threads: int = -1,
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ) -> dict[str, np.ndarray]:
        native_method = getattr(self._require_index(), "search_layer0_chr_summary", None)
        if native_method is None:
            raise RuntimeError(
                "Current Faiss build does not expose native search_layer0_chr_summary(). "
                "Rebuild/install Faiss with SAGE CHR summary instrumentation."
            )
        vectors = self._prepare_vectors(data)
        return native_method(
            vectors,
            k=int(k),
            ef=int(ef),
            hide_labels=None if hide_labels is None else np.asarray(hide_labels, dtype=np.int64),
            num_threads=self._num_threads if num_threads <= 0 else int(num_threads),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def search_layer0_chr_summary(
        self,
        data,
        *,
        k: int,
        ef: int,
        hide_labels=None,
        num_threads: int = -1,
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ) -> dict[str, np.ndarray]:
        return self._native_chr_summary(
            data,
            k=int(k),
            ef=int(ef),
            hide_labels=hide_labels,
            num_threads=int(num_threads),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def search_layer0_chr_summary_hide_node_batch(
        self,
        data,
        *args,
        k: int | None = None,
        ef: int | None = None,
        hide_labels=None,
        num_threads: int = -1,
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ) -> dict[str, np.ndarray]:
        if args:
            if len(args) > 1:
                raise TypeError(
                    "search_layer0_chr_summary_hide_node_batch accepts at most one "
                    "positional argument after data."
                )
            if ef is not None:
                raise TypeError("ef was provided both positionally and by keyword.")
            ef = int(args[0])
        if ef is None:
            raise TypeError("Missing required argument: ef")
        if hide_labels is None:
            raise TypeError("Missing required argument: hide_labels")
        return self._native_chr_summary(
            data,
            k=10 if k is None else int(k),
            ef=int(ef),
            hide_labels=hide_labels,
            num_threads=int(num_threads),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def search_layer0_chr_summary_batch(
        self,
        data,
        *args,
        k: int | None = None,
        ef: int | None = None,
        num_threads: int = -1,
        classify_start: int = 4,
        classify_end: int = 16,
        chr_ema_decay: float = CHR_EMA_DECAY,
    ) -> dict[str, np.ndarray]:
        if args:
            if len(args) > 1:
                raise TypeError(
                    "search_layer0_chr_summary_batch accepts at most one "
                    "positional argument after data."
                )
            if ef is not None:
                raise TypeError("ef was provided both positionally and by keyword.")
            ef = int(args[0])
        if ef is None:
            raise TypeError("Missing required argument: ef")
        return self._native_chr_summary(
            data,
            k=10 if k is None else int(k),
            ef=int(ef),
            hide_labels=None,
            num_threads=int(num_threads),
            classify_start=int(classify_start),
            classify_end=int(classify_end),
            chr_ema_decay=float(chr_ema_decay),
        )

    def search_layer0_path_batch(self, data, ef: int, num_threads: int = -1):
        paths, _, _ = self.search_layer0_path_with_dist_metrics_batch(
            data,
            ef=int(ef),
            k=10,
            num_threads=int(num_threads),
        )
        return paths

    def search_layer0_path(self, query, ef: int):
        vectors = self._prepare_vectors(query)
        if vectors.shape[0] != 1:
            raise ValueError("search_layer0_path expects a single query vector.")
        return self.search_layer0_path_batch(vectors, ef=int(ef), num_threads=1)[0]

    def search_layer0_path_with_dist_metrics_hide_node_batch(
        self,
        data,
        *args,
        k: int | None = None,
        ef: int | None = None,
        hide_labels=None,
        num_threads: int = -1,
    ) -> tuple[list[list[dict[str, Any]]], int, list[float]]:
        if args:
            if len(args) > 1:
                raise TypeError(
                    "search_layer0_path_with_dist_metrics_hide_node_batch accepts at most one "
                    "positional argument after data."
                )
            if ef is not None:
                raise TypeError("ef was provided both positionally and by keyword.")
            ef = int(args[0])
        if ef is None:
            raise TypeError("Missing required argument: ef")
        if hide_labels is None:
            raise TypeError("Missing required argument: hide_labels")
        return self._native_layer0_trace(
            data,
            k=10 if k is None else int(k),
            ef=int(ef),
            hide_labels=hide_labels,
            num_threads=int(num_threads),
        )

    def search_layer0_cfr_trace_hide_node_batch(self, *args, **kwargs):
        return self.search_layer0_path_with_dist_metrics_hide_node_batch(*args, **kwargs)

    def search_layer0_path_with_dist_metrics_batch(
        self,
        data,
        *args,
        k: int | None = None,
        ef: int | None = None,
        num_threads: int = -1,
    ) -> tuple[list[list[dict[str, Any]]], int, list[float]]:
        if args:
            if len(args) > 1:
                raise TypeError(
                    "search_layer0_path_with_dist_metrics_batch accepts at most one "
                    "positional argument after data."
                )
            if ef is not None:
                raise TypeError("ef was provided both positionally and by keyword.")
            ef = int(args[0])
        if ef is None:
            raise TypeError("Missing required argument: ef")
        return self._native_layer0_trace(
            data,
            k=10 if k is None else int(k),
            ef=int(ef),
            hide_labels=None,
            num_threads=int(num_threads),
        )
