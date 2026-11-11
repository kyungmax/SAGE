"""Extend a frozen current policy without repeating recommendation or sampling.

The recommended policy and its three deployment costs remain unchanged. Every
extra curve point uses the frozen probe labels on the same graph. Work repeated
to reconstruct probe vectors/pseudo-GT is charged to curve preparation only.
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from .final_policy import POLICY_VERSION, _fit_policy_buckets, _grid_for_cap
from .offline_calibration import DEFAULT_CALIBRATION_PROBE_COUNT


def curve_grid(*, recommended_ef, offline_cap, k=10, minimum=64, initial_cap=2048):
    """Use the current (128-step above 1024) grid, including the frozen point."""
    cap = max(int(initial_cap), int(offline_cap), int(recommended_ef))
    return sorted({int(recommended_ef), *_grid_for_cap(cap, min_ef=minimum, k=k)})


def fit_curve_policies(index, policy, ef_values, *, num_threads=1):
    """Return per-ef policies and separately scoped curve-preparation costs.

    Callers must verify the graph/source fingerprint before invoking this
    function. It never samples nodes, recommends an ef, or reads test queries.
    """
    if policy['version'] != POLICY_VERSION:
        raise ValueError('Curve extension requires the current frozen policy version')
    ids = np.asarray(policy['probe_ids'], dtype=np.int64)
    if ids.shape != (DEFAULT_CALIBRATION_PROBE_COUNT,) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid frozen calibration probe labels')
    k, baseline = int(policy['k']), int(policy['ef_init'])
    points = sorted({max(k, int(ef)) for ef in ef_values} | {baseline})
    curves = deepcopy(policy['curve_policies'])
    if str(baseline) not in curves:
        raise ValueError('Frozen recommended-point policy is missing')
    missing = [ef for ef in points if str(ef) not in curves]
    started = time.perf_counter()
    if not missing:
        return {str(ef): curves[str(ef)] for ef in points}, dict(
            probe_reload_seconds=0., threshold_fit_seconds=0., curve_extra_seconds=0.,
            extra_policy_count=0, probe_count=len(ids), recommendation_reused=True)
    previous_ef = index.ef
    try:
        vectors = np.ascontiguousarray(index.get_items(ids), dtype=np.float32)
        index.set_ef(int(policy['pseudo_gt_ef']))
        truth, _ = index.knn_query_hide_node(vectors, k=k, hide_labels=ids,
                                            num_threads=num_threads)
        if np.any(truth == ids[:, None]):
            raise RuntimeError('Self-hidden pseudo ground truth contains a probe label')
        fixed = {int(ef): np.asarray(hits, dtype=np.int64)
                 for ef, hits in policy['probe_hits_by_ef'].items()}
        if any(hits.shape != (len(ids),) for hits in fixed.values()):
            raise ValueError('Saved probe hit count differs from frozen probe count')
        prepared = time.perf_counter()

        def get_hits(ef):
            ef = max(k, int(ef))
            if ef not in fixed:
                index.set_ef(ef)
                labels, _ = index.knn_query_hide_node(vectors, k=k, hide_labels=ids,
                                                      num_threads=num_threads)
                if np.any(labels == ids[:, None]):
                    raise RuntimeError('Probe search did not hide self')
                matches = ((labels[:, :, None] >= 0) & (truth[:, None, :] >= 0)
                           & (labels[:, :, None] == truth[:, None, :]))
                fixed[ef] = matches.any(axis=2).sum(axis=1)
            return fixed[ef]

        for ef in missing:
            main, shadow = _fit_policy_buckets(
                index, vectors, ids, initial_ef=ef,
                shadow_enabled=policy['shadow_enabled'], k=k,
                threads=num_threads, get_hits=get_hits)
            curves[str(ef)] = dict(version=POLICY_VERSION, backend=policy['backend'],
                k=k, ef_init=ef, calibration_sampling=deepcopy(policy['calibration_sampling']),
                cfr_signal='raw_cfr_window_mean', classify_start=1, classify_end=16,
                shadow_enabled=bool(policy['shadow_enabled']), main=main, shadow=shadow)
        finished = time.perf_counter()
        return {str(ef): curves[str(ef)] for ef in points}, dict(
            probe_reload_seconds=prepared - started,
            threshold_fit_seconds=finished - prepared,
            curve_extra_seconds=finished - started, extra_policy_count=len(missing),
            probe_count=len(ids), recommendation_reused=True)
    finally:
        index.set_ef(previous_ef)
