#!/usr/bin/env python3
"""Vary W, B, and g independently at the original recommended search budget."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
common = importlib.import_module('experiments_scripts.03_calibration.calibration_common')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_inputs(parser, output_name='04_parameter_sensitivity')
    args = parser.parse_args()
    common.validate_args(args)

    import numpy as np
    from experiments_scripts.common.final_policy import calibrate_policy, query as final_query
    from parameter_policy import Calibration, query, variants

    cfg, index = common.load_inputs(args)
    folder = args.output_root / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    default = calibrate_policy(index, k=10, backend='faiss', num_threads=args.threads,
                               min_ef=64, shadow_enabled=True)
    common.save(folder / 'default_policy.json', default)
    calibration = Calibration(index, default, args.threads)
    policies = {name: calibration.fit(knobs) for name, knobs in variants().items()}
    for name, policy in policies.items():
        common.save(folder / f'policy_{name}.json', policy)
    for branch in ('main', 'shadow'):
        if policies['default'][branch] != default[branch]:
            raise RuntimeError(f'Default sensitivity calibration differs from production: {branch}')
    np.savez_compressed(folder / 'calibration.npz', probe_ids=calibration.ids,
        pseudo_labels=calibration.truth, **{f'hits_ef{ef}': value for ef, value in calibration.hits.items()})
    common.save(folder / 'protocol.json', dict(dataset=cfg, recommended_ef=default['ef_init'],
        recommended_ef_fixed_across_variants=True, W_applies_to_both_streams=True,
        B_and_g_apply_to_main=True, shadow_buckets=4, shadow_margin=4,
        threads=args.threads, measured_runs=args.measured_runs, warmup_runs=args.warmup_runs,
        variants=variants()))
    queries, truth = common.load_online(cfg)
    # Catch any divergence in the experiment helper before collecting timing samples.
    public_labels, public_distances = final_query(index, queries, default, num_threads=args.threads)
    labels, distances = query(index, queries, policies['default'], args.threads)
    np.testing.assert_array_equal(labels, public_labels)
    np.testing.assert_array_equal(distances, public_distances)

    def invoke(name):
        if name == 'vanilla':
            index.set_ef(default['ef_init'])
            return index.knn_query(queries, k=10, num_threads=args.threads)
        return query(index, queries, policies[name], args.threads)

    measured = common.measure_jobs(['vanilla', *policies], invoke, truth, args, folder)
    vanilla = measured['vanilla']
    rows = [dict(dataset=args.dataset, setting=name, ef=default['ef_init'], **policy['knobs'],
        **measured[name], vanilla_recall=vanilla['recall'],
        recall_loss=vanilla['recall'] - measured[name]['recall'],
        same_ef_speedup=vanilla['batch_seconds_mean'] / measured[name]['batch_seconds_mean'])
        for name, policy in policies.items()]
    common.write_csv(folder / 'summary.csv', rows)
    common.save(folder / 'summary.json', dict(default_matches_public_api=True, vanilla=vanilla, settings=rows))
    print(json.dumps(dict(output=str(folder / 'summary.csv'))), flush=True)


if __name__ == '__main__':
    main()
