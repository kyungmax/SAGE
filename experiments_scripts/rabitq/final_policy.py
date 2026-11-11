"""RaBitQ-Library adapter for SAGE's shared calibration and query policy.

Quantized indexes do not retain the original vectors. Supply a row reader
(`train_vectors(ids)`) or a float32 array for LID sampling and calibration.
Labels in RaBitQ-Library are zero-based input row numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..common.offline_calibration import DEFAULT_CALIBRATION_PROBE_COUNT
from ..common.final_policy import POLICY_VERSION, calibrate_policy as _calibrate, query as _query
from ..common.adaptive_runtime import DEFAULT_LID_SEARCH_EF


def load_library():
    import rabitqlib
    expected = Path(__file__).resolve().parents[2] / "rabitq/build_sage/python/rabitqlib"
    if Path(rabitqlib.__file__).resolve().parent != expected:
        raise RuntimeError("Use scripts/python.sh with this checkout's RaBitQ build.")
    if (getattr(rabitqlib, "SAGE_POLICY_API", 0) < 2 or
            getattr(rabitqlib, "SAGE_TOP1_BUCKET_RECOVERY", None) != 1):
        raise RuntimeError("Rebuild RaBitQ: bash scripts/build.sh rabitq")
    return rabitqlib


def _threads(value):
    return max(1, os.cpu_count() or 1) if value <= 0 else int(value)


class RaBitQAdapter:
    def __init__(self, index, train_vectors=None):
        self.native = index
        self.train_vectors = train_vectors
        self.ef = 64

    def get_current_count(self):
        return self.native.num_points

    def get_items(self, ids):
        if self.train_vectors is None:
            raise ValueError("Calibration requires original train vectors or a row reader.")
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.get_current_count()):
            raise ValueError("Probe label outside the index.")
        rows = self.train_vectors(ids) if callable(self.train_vectors) else self.train_vectors[ids]
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.shape != (len(ids), self.native.dim) or not np.isfinite(rows).all():
            raise ValueError("Train row reader returned invalid vectors.")
        return rows

    def set_ef(self, value):
        self.ef = int(value)

    def knn_query_hide_node(self, vectors, *, k, hide_labels, num_threads):
        return self.native.search_adaptive_light(
            vectors, k=k, ef_init=self.ef, enable_stop=False,
            hide_labels=hide_labels, num_threads=_threads(num_threads))

    def search_layer0_cfr_summary(self, vectors, *, k, ef, hide_labels, num_threads,
                                  classify_start, classify_end, shadow_observation=False,
                                  shadow_pre_expansion=True, include_shadow=False):
        if include_shadow and shadow_observation:
            raise ValueError("include_shadow requires the main summary mode")
        if include_shadow and ef < 128:
            raise ValueError("Shadow observation requires ef >= 128")
        _, _, stats = self.native.search_adaptive_light_with_stats(
            vectors, k=k, ef_init=ef, enable_stop=False, hide_labels=hide_labels,
            num_threads=_threads(num_threads), classify_start=classify_start,
            classify_end=classify_end, shadow_observation=shadow_observation or include_shadow,
            shadow_pre_expansion=shadow_pre_expansion)
        signal = stats["shadow_cfr" if shadow_observation else "classify_cfr_mean"]
        result = dict(mean_cfrs=signal, usable_flags=np.isfinite(signal))
        if shadow_observation:
            result["decision_steps"] = stats["shadow_decision_step"]
        if include_shadow:
            result["shadow"] = dict(mean_cfrs=stats["shadow_cfr"],
                                    usable_flags=np.isfinite(stats["shadow_cfr"]),
                                    decision_steps=stats["shadow_decision_step"])
        return result

    def calc_lids_internal_sampled(self, *, k_lid, sample_fraction, min_sample_size,
                                   random_seed, num_threads, lid_search_ef=DEFAULT_LID_SEARCH_EF):
        if int(lid_search_ef) <= 0:
            raise ValueError("lid_search_ef must be positive.")
        count = self.get_current_count()
        sample = min(count, max(min_sample_size, int(count * sample_fraction)))
        ids = np.random.default_rng(random_seed).choice(count, sample, replace=False)
        vectors = self.get_items(ids)
        _, distances = self.native.search_adaptive_light(
            vectors, k=k_lid, ef_init=max(int(lid_search_ef), k_lid + 1),
            enable_stop=False, hide_labels=ids,
            num_threads=_threads(num_threads))
        # RaBitQ IP estimates store -dot; use cosine distance for normalized data.
        distances = np.asarray(distances, dtype=np.float64)
        if self.native.metric == "ip":
            distances = 1.0 + distances
        distances = np.sqrt(np.maximum(distances, 1e-12))
        distances.sort(axis=1)
        denom = np.log(distances / distances[:, -1:]).mean(axis=1)
        # Match FAISS/hnswlib's undefined-LID sentinel: keep the sampled node
        # in the fixed pool, then let shared calibration exclude nonpositive
        # LIDs when choosing probes. Do not silently shrink or refill the pool.
        lids = np.zeros(sample, dtype=np.float64)
        valid = np.isfinite(denom) & (denom < -1e-12)
        lids[valid] = -1.0 / denom[valid]
        return ids, lids

    def knn_query_sage(self, vectors, *, filter=None, return_diagnostics=False, **kwargs):
        if filter is not None:
            raise ValueError("RaBitQ-Library does not support label filters.")
        kwargs["num_threads"] = _threads(kwargs["num_threads"])
        kwargs["paper_bucket_mode"] = True
        fn = (self.native.search_adaptive_light_with_stats if return_diagnostics
              else self.native.search_sage)
        return fn(vectors, **kwargs)


def calibrate_policy(index, *, train_vectors=None, **kwargs):
    adapter = index if isinstance(index, RaBitQAdapter) else RaBitQAdapter(index, train_vectors)
    if adapter.get_current_count() < DEFAULT_CALIBRATION_PROBE_COUNT or adapter.get_current_count() <= kwargs.get("k", 10):
        raise ValueError("Calibration requires 500 distinct probes and at least k+1 indexed rows.")
    return _calibrate(adapter, backend="rabitq", **kwargs)


def query(index, vectors, policy, **kwargs):
    if policy.get("backend") != "rabitq":
        raise ValueError("Use a policy calibrated on this RaBitQ graph.")
    adapter = index if isinstance(index, RaBitQAdapter) else RaBitQAdapter(index)
    return _query(adapter, vectors, policy, **kwargs)
