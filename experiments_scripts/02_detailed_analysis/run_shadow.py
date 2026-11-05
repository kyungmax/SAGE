#!/usr/bin/env python3
"""Time Vanilla and SAGE with/without shadow at the same recommended ef."""
import argparse
import copy
import statistics
import time

import numpy as np

from detail_common import add_inputs, hits, load, save_csv, save_json, vanilla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser, 'shadow')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--measured-runs', type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 1 or args.measured_runs < 1:
        parser.error('Require at least one warmup and one measured run')
    data = load(args)
    if data is None:
        return
    from experiments_scripts.common.final_policy import query
    index, queries, truth, policy = data
    plain = copy.deepcopy(policy)
    plain['shadow'] = None
    plain['shadow_enabled'] = False
    ef, k = policy['ef_init'], policy['k']
    def sage(p):
        labels, distances = query(index, queries, p, num_threads=args.threads)
        return dict(labels=labels, distances=distances)
    methods = {'vanilla': lambda: vanilla(index, queries, k, ef, args.threads),
               'without_shadow': lambda: sage(plain), 'with_shadow': lambda: sage(policy)}
    reference, samples, timing = {}, {name: [] for name in methods}, []
    for repeat in range(args.warmups + args.measured_runs):
        for name, function in methods.items():
            started = time.perf_counter()
            values = function()
            seconds = time.perf_counter() - started
            if name not in reference:
                reference[name] = values
            else:
                for key in ('labels', 'distances'):
                    np.testing.assert_array_equal(values[key], reference[name][key])
            measured = repeat >= args.warmups
            if measured:
                samples[name].append(seconds)
            timing.append(dict(method=name, run=repeat, measured=measured, seconds=seconds))
    means = {name: statistics.mean(values) for name, values in samples.items()}
    recalls = {name: float(hits(value['labels'], truth).mean() / k) for name, value in reference.items()}
    rows = [dict(method=name, ef=ef, recall=recalls[name], recall_loss=recalls['vanilla'] - recalls[name],
                 mean_seconds=means[name], qps=len(queries) / means[name],
                 speedup=means['vanilla'] / means[name]) for name in methods]
    save_csv(args.output / 'timings.csv', timing)
    save_csv(args.output / 'summary.csv', rows)
    save_json(args.output / 'protocol.json', dict(comparison='same_ef', ef=ef,
        warmups=args.warmups, measured_runs=args.measured_runs, total_runs=args.warmups + args.measured_runs,
        shadow_active=policy['shadow'] is not None, top1_second_chance=True,
        cfr_window=[1, 16], timing_scope='Whole query calls; index loading and parity checks excluded'))


if __name__ == '__main__':
    main()
