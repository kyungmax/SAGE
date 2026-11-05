"""W/B/g calibration at one fixed recommended ef, using the public FAISS path."""
from __future__ import annotations

import numpy as np

from experiments_scripts.common.projected_local_acceptable_runtime import (
    _build_mixed_threshold_config, _quantile_theta,
)


def variants():
    default = dict(W=16, B=4, g=2)
    result = {'default': default}
    for key, values in (('W', (8, 24)), ('B', (2, 6)), ('g', (1, 3, 4))):
        for value in values:
            result[f'{key}_{value}'] = dict(default, **{key: value})
    return result


class Calibration:
    def __init__(self, index, default_policy, threads):
        self.index, self.threads = index, threads
        self.ids = np.asarray(default_policy['probe_ids'], dtype=np.int64)
        self.vectors = np.ascontiguousarray(index.get_items(self.ids), dtype=np.float32)
        self.ef = default_policy['ef_init']
        self.truth = self.search(4096)
        self.hits = {}
        self.observations = {}

    def search(self, ef):
        previous = self.index.ef
        try:
            self.index.set_ef(max(10, int(ef)))
            labels, _ = self.index.knn_query_hide_node(self.vectors, k=10,
                hide_labels=self.ids, num_threads=self.threads)
        finally:
            self.index.set_ef(previous)
        if np.any(labels == self.ids[:, None]):
            raise RuntimeError('Calibration search did not hide self')
        return labels

    def get_hits(self, ef):
        ef = max(10, int(ef))
        if ef not in self.hits:
            labels = self.search(ef)
            self.hits[ef] = (labels[:, :, None] == self.truth[:, None, :]).any(2).sum(1)
        return self.hits[ef]

    def fit_branch(self, *, W, rank, routes, margin, shadow):
        key = (W, shadow)
        if key not in self.observations:
            self.observations[key] = self.index.search_layer0_cfr_summary(
                self.vectors, k=10, ef=self.ef, hide_labels=self.ids, num_threads=self.threads,
                classify_start=1, classify_end=W, shadow_observation=shadow,
                shadow_pre_expansion=True)
        observation = self.observations[key]
        signal = np.asarray(observation['mean_cfrs'], dtype=np.float32)
        mask = np.asarray(observation['usable_flags'], dtype=bool) & np.isfinite(signal)
        if not mask.any():
            raise ValueError(f'No complete CFR windows at W={W}, shadow={shadow}')
        pairs = [max(1, route // margin) for route in routes]
        rates = [float(np.mean(self.get_hits(ef)[mask] == 10)) for ef in pairs]
        raw = [max(_quantile_theta(signal[mask], rate), 1e-6) for rate in rates]
        thetas = np.maximum.accumulate(raw)
        return dict(rank=rank, margin=margin, route_efs=routes, pair_target_efs=pairs,
            observation_ef=self.ef,
            cfr_observation_mode='shadow_pre_expansion' if shadow else 'main_pre_expansion',
            effective_pair_target_efs=[max(10, ef) for ef in pairs],
            acceptable_rates=rates, usable_count=int(mask.sum()), raw_thetas=raw,
            thetas=thetas.tolist(), tau=float(thetas[-1]), gammas=(thetas / thetas[-1]).tolist())

    def fit(self, knobs):
        routes = list(_build_mixed_threshold_config(self.ef, mixed_bucket_count=knobs['B']).resolved_route_efs)
        main = self.fit_branch(W=knobs['W'], rank=self.ef, routes=routes, margin=knobs['g'], shadow=False)
        shadow = (self.fit_branch(W=knobs['W'], rank=128, routes=[32, 64, 96], margin=4, shadow=True)
                  if self.ef >= 256 else None)
        return dict(knobs=knobs, ef_init=self.ef, recommended_ef=self.ef, k=10,
                    classify_start=1, classify_end=knobs['W'], main=main, shadow=shadow)


def query(index, vectors, policy, threads):
    main, shadow, knobs = policy['main'], policy['shadow'], policy['knobs']
    kwargs = dict(k=10, ef_init=policy['ef_init'], num_threads=threads, filter=None,
        early_stop_ratio=main['tau'], paper_bucket_count=knobs['B'],
        bucket_gamma_ratios=main['gammas'], enable_stop=True,
        classify_start=1, classify_end=knobs['W'], return_diagnostics=False,
        shadow_stop_enabled=False, shadow_pre_expansion=True)
    if shadow is not None:
        kwargs.update(shadow_stop_enabled=True, shadow_ef=128, shadow_bucket_mode=True,
            shadow_early_stop_ratio=shadow['tau'], shadow_bucket_gamma_ratios=shadow['gammas'])
    return index.knn_query_sage(vectors, **kwargs)
