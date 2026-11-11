"""Shared final-policy calibration for FAISS, hnswlib and RaBitQ, with staged timing.

Calibration uses this backend's own graph, distances and self-hidden probes.
It temporarily changes index.ef, so do not calibrate concurrently with searches.
"""
from __future__ import annotations

import time

import numpy as np
from .adaptive_runtime import DEFAULT_INTERNAL_LID_K, DEFAULT_LID_SEARCH_EF

from .offline_calibration import (
    DEFAULT_CALIBRATION_PROBE_COUNT,
    calibration_sampling_metadata,
    compute_fixed_calibration_lid_pool,
)

from .projected_local_acceptable_runtime import (
    _build_mixed_threshold_config,
    _quantile_theta,
    _select_dummy_queries,
)

POLICY_VERSION = "sage-final-raw-min64-fixed10k-v7"
OFFLINE_MAX_EF = 4096
HIGH_EF_GRID_STEP = 128


def _grid_for_cap(cap, *, min_ef=64, k=10):
    """Quarter-doubling through 1024, then steps of 128 up to the cap."""
    grid = {min_ef, cap}
    low = min_ef
    while low < cap:
        step = min(low // 4, HIGH_EF_GRID_STEP)
        grid.update(e for e in range(low, 2 * low + 1, step) if e <= cap)
        low *= 2
    return sorted({max(k, ef) for ef in grid})


def _recommend_ef(measure_hits, *, total_hits, min_ef=64, k=10):
    """Extend by 1024 on cap selection, up to 4096; cache measured grid points."""
    cap, history = 1024, []
    measured = {}
    while True:
        for ef in _grid_for_cap(cap, min_ef=min_ef, k=k):
            if ef not in measured:
                measured[ef] = int(measure_hits(ef))
        hits = dict(measured)
        best = max(hits.values())
        eligible = [ef for ef, value in hits.items()
                    if best - value <= .001 * total_hits + 1e-12]
        selected = min(eligible)
        history.append(dict(cap=cap, selected_ef=selected, max_hits=best,
                            eligible_efs=eligible, hits_by_ef=hits,
                            saturated=selected < cap))
        if selected < cap or cap == OFFLINE_MAX_EF:
            return selected, history
        cap += 1024


def _fit_buckets(index, vectors, ids, *, rank, routes, margin, k, threads, get_hits,
                 shadow_observation=False, observation_ef=None, summary=None):
    observation_ef = rank if observation_ef is None else observation_ef
    if summary is None:
        observer_args = {"shadow_observation": True, "shadow_pre_expansion": True} if shadow_observation else {}
        summary = index.search_layer0_cfr_summary(
            vectors, k=k, ef=observation_ef, hide_labels=ids, num_threads=threads,
            classify_start=1, classify_end=16, **observer_args)
    signal = np.asarray(summary["mean_cfrs"], dtype=np.float32)
    mask = np.asarray(summary["usable_flags"], dtype=bool) & np.isfinite(signal)
    if not mask.any():
        raise ValueError(f"No complete CFR windows among probes at ef={rank}.")
    pairs = [max(1, route // margin) for route in routes]
    rates = [float(np.mean(get_hits(ef)[mask] == k)) for ef in pairs]
    raw = [max(_quantile_theta(signal[mask], p), 1e-6) for p in rates]
    thetas = np.maximum.accumulate(raw)
    return dict(rank=rank, margin=margin, route_efs=routes, pair_target_efs=pairs,
                observation_ef=int(observation_ef),
                cfr_observation_mode=("shadow_pre_expansion" if shadow_observation
                                      else "main_pre_expansion"),
                effective_pair_target_efs=[max(k, ef) for ef in pairs],
                acceptable_rates=rates, usable_count=int(mask.sum()),
                raw_thetas=raw, thetas=thetas.tolist(), tau=float(thetas[-1]),
                gammas=(thetas / thetas[-1]).tolist())


def _fit_policy_buckets(index, vectors, ids, *, initial_ef, shadow_enabled,
                        k, threads, get_hits):
    """Fit independent main/shadow thresholds from one uncut wide traversal."""
    use_shadow = shadow_enabled and initial_ef >= 256
    summary = index.search_layer0_cfr_summary(
        vectors, k=k, ef=initial_ef, hide_labels=ids, num_threads=threads,
        classify_start=1, classify_end=16, include_shadow=use_shadow,
        shadow_pre_expansion=True)
    config = _build_mixed_threshold_config(initial_ef, mixed_bucket_count=4)
    main = _fit_buckets(index, vectors, ids, rank=initial_ef,
                        routes=list(config.resolved_route_efs), margin=2,
                        k=k, threads=threads, get_hits=get_hits, summary=summary)
    shadow = None
    if use_shadow:
        # Each observer keeps its own window and usable-probe mask. An ef128
        # traversal or the main CFR distribution cannot substitute for shadow.
        shadow = _fit_buckets(index, vectors, ids, rank=128,
                              routes=[32, 64, 96], margin=4,
                              k=k, threads=threads, get_hits=get_hits,
                              shadow_observation=True, observation_ef=initial_ef,
                              summary=summary["shadow"])
    return main, shadow


def calibrate_policy(index, *, probe_ids=None, lid_pool=None, k=10, num_threads=1,
                     backend="hnswlib", min_ef=64, shadow_enabled=None, curve_efs=None):
    """Return a JSON-serializable policy using exactly 500 self-hidden probes.

    Supply frozen external probe labels to reuse a selection. Otherwise supply
    a validated 10,000-row DataFrame with query_id/lid, or sample exactly 10,000
    graph LIDs (seed 42, k_lid 15, ef 512, self ID excluded), independently
    of dataset size. Trim 1--99%
    and select the midpoint of each of 500 equal-count LID bins.
    Smaller automatic pools are an error.
    Pseudo ground truth is vanilla ef4096 on the same graph with self hidden.
    The epsilon recommendation is relative to the best recall inside each cap.
    """
    started = time.perf_counter()
    if backend not in ("faiss", "hnswlib", "rabitq"):
        raise ValueError("Unsupported backend")
    if shadow_enabled is None:
        shadow_enabled = k != 100
    if k == 100 and shadow_enabled:
        raise ValueError("k=100 requires shadow disabled")
    if not 1 <= k <= 128:
        raise ValueError("The final shadow128 policy requires 1 <= k <= 128.")
    if probe_ids is not None and lid_pool is not None:
        raise ValueError("Supply probe_ids or lid_pool, not both.")
    if probe_ids is None:
        supplied_pool = lid_pool is not None
        if lid_pool is None:
            lid_pool = compute_fixed_calibration_lid_pool(
                index, internal_lid_k=DEFAULT_INTERNAL_LID_K, num_nodes=index.get_current_count(),
                lid_sample_seed=42, num_threads=num_threads,
                lid_search_ef=DEFAULT_LID_SEARCH_EF)
        sampling = calibration_sampling_metadata(
            lid_pool, source="supplied_pool" if supplied_pool else "sampled_graph",
            internal_lid_k=None if supplied_pool else 15, seed=None if supplied_pool else 42)
        if not supplied_pool:
            sampling.update(label_space="external",
                            lid_estimator=f"{backend}-mle-self-id-excluded-v1")
        samp_end = time.perf_counter()
        pool = lid_pool.copy()
        pool = pool[np.isfinite(pool["lid"]) & (pool["lid"] > 0)].copy()
        if pool.empty:
            raise ValueError("No finite positive LID candidates.")
        pool["query_source"] = "train"
        selected = _select_dummy_queries(
            lid_df=pool, num_calibration_queries=DEFAULT_CALIBRATION_PROBE_COUNT,
            trim_low_percentile=1., trim_high_percentile=99.)
        probe_ids = selected["query_id"].to_numpy(np.int64)
        selection = f"lid_quantile{DEFAULT_CALIBRATION_PROBE_COUNT}_trim1_99"
    else:
        samp_end = time.perf_counter()
        selection = "supplied_frozen_labels"
        sampling = calibration_sampling_metadata(None)
    ids = np.asarray(probe_ids, dtype=np.int64)
    if ids.shape != (DEFAULT_CALIBRATION_PROBE_COUNT,) or len(np.unique(ids)) != DEFAULT_CALIBRATION_PROBE_COUNT:
        raise ValueError(f"Exactly {DEFAULT_CALIBRATION_PROBE_COUNT} distinct probe labels are required.")
    vectors = np.ascontiguousarray(index.get_items(ids), dtype=np.float32)
    previous_ef = index.ef
    try:
        index.set_ef(4096)
        truth, _ = index.knn_query_hide_node(
            vectors, k=k, hide_labels=ids, num_threads=num_threads)
        if np.any(truth == ids[:, None]):
            raise RuntimeError("Self-hidden pseudo ground truth contains a probe label.")
        select_end = time.perf_counter()
        fixed = {}

        def get_hits(ef):
            ef = max(k, int(ef))
            if ef not in fixed:
                index.set_ef(ef)
                labels, _ = index.knn_query_hide_node(
                    vectors, k=k, hide_labels=ids, num_threads=num_threads)
                if np.any(labels == ids[:, None]):
                    raise RuntimeError("Probe search did not hide self.")
                matches = ((labels[:, :, None] >= 0) & (truth[:, None, :] >= 0)
                           & (labels[:, :, None] == truth[:, None, :]))
                fixed[ef] = matches.any(axis=2).sum(axis=1)
            return fixed[ef]

        ef, history = _recommend_ef(lambda e: get_hits(e).sum(), total_hits=len(ids) * k,
                                    min_ef=min_ef, k=k)
        sweep_end = time.perf_counter()
        def fit(initial_ef):
            main, shadow = _fit_policy_buckets(
                index, vectors, ids, initial_ef=initial_ef,
                shadow_enabled=shadow_enabled, k=k, threads=num_threads,
                get_hits=get_hits)
            return dict(version=POLICY_VERSION, backend=backend, k=k, ef_init=initial_ef,
                        calibration_sampling=dict(sampling),
                        cfr_signal="raw_cfr_window_mean", classify_start=1, classify_end=16,
                        shadow_enabled=bool(shadow_enabled), main=main, shadow=shadow)

        policy = fit(ef)
        recommended_end = time.perf_counter()
        curve_policies = {str(ef): dict(policy)}
        points = curve_efs(ef, history[-1]['cap']) if callable(curve_efs) else (curve_efs or [])
        for point in sorted(set(points)):
            point = max(k, int(point))
            if str(point) not in curve_policies:
                curve_policies[str(point)] = fit(point)
        finished = time.perf_counter()
        # Three deployment stages; additional full-curve fitting is reported separately.
        timings = {"Samp.": samp_end - started, "Select.": select_end - samp_end,
                   "Eval.": recommended_end - select_end,
                   "eval_sweep_seconds": sweep_end - select_end,
                   "eval_policy_fit_seconds": recommended_end - sweep_end,
                   "curve_extra_seconds": finished - recommended_end,
                   "total_seconds": recommended_end - started,
                   "including_curve_seconds": finished - started}
        policy.update(probe_ids=ids.tolist(), probe_selection=selection,
                      pseudo_gt_ef=4096, epsilon=.001, offline_min_ef=min_ef,
                      offline_max_ef=OFFLINE_MAX_EF, high_ef_grid_step=HIGH_EF_GRID_STEP,
                      saturation_status="SATURATED" if history[-1]['saturated'] else "CAP_HIT",
                      cap_history=history, offline_timings=timings,
                      curve_policies=curve_policies,
                      probe_hits_by_ef={str(e): h.tolist() for e, h in fixed.items()})
        return policy
    finally:
        index.set_ef(previous_ef)


def query(index, vectors, policy, *, num_threads=-1, filter=None, return_diagnostics=False,
          return_routing=False):
    """Run a frozen policy; default return is (labels, distances).

    FAISS diagnostic calls can additionally request native final ef and main /
    shadow CFR arrays with return_routing=True (requires a rebuilt extension).
    """
    if policy["version"] != POLICY_VERSION or policy["backend"] not in ("hnswlib", "faiss", "rabitq"):
        raise ValueError("Expected the current SAGE final-policy calibration; recalibrate before querying.")
    if policy["k"] == 100 and policy["shadow"] is not None:
        raise ValueError("k=100 requires shadow disabled")
    main, shadow = policy["main"], policy["shadow"]
    if return_routing and (policy["backend"] != "faiss" or not return_diagnostics):
        raise ValueError("Routing diagnostics require FAISS and return_diagnostics=True")
    kwargs = dict(k=policy["k"], ef_init=policy["ef_init"], num_threads=num_threads,
                  filter=filter, early_stop_ratio=main["tau"], paper_bucket_count=4,
                  bucket_gamma_ratios=main["gammas"],
                  classify_start=1, classify_end=16, return_diagnostics=return_diagnostics,
                  shadow_stop_enabled=False,
                  shadow_pre_expansion=True)
    if return_routing:
        kwargs["return_routing"] = True
    if shadow is not None:
        kwargs.update(shadow_stop_enabled=True, shadow_ef=128, shadow_bucket_mode=True,
                      shadow_early_stop_ratio=shadow["tau"],
                      shadow_bucket_gamma_ratios=shadow["gammas"])
    return index.knn_query_sage(vectors, **kwargs)
