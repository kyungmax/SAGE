#!/usr/bin/env python3
"""Collect exact level-0 work and shared easiest/hardest 10% query cohorts."""
import argparse
import math
from pathlib import Path

import numpy as np

from detail_common import (add_inputs, array_hash, hits, load, sage_work, save_csv,
                           save_json, upper_hops, vanilla_work)


def summarize(methods, first_steps, k, ef, percent=10):
    n = len(first_steps)
    order = np.lexsort((np.arange(n), first_steps, -methods['Vanilla']['hits']))
    groups = {'Easy': order[:math.ceil(percent * n / 100)],
              'Hard': order[math.ceil((100 - percent) * n / 100):]}
    if any(not len(ids) for ids in groups.values()):
        raise ValueError('Too few queries for requested cohorts')
    rows = []
    for method, values in methods.items():
        for cohort, ids in groups.items():
            rows.append(dict(method=method, cohort=cohort, percent=percent,
                queries=len(ids), total_queries=n, reference_ef=ef,
                mean_steps=float(values['steps'][ids].mean()),
                mean_recall=float(values['hits'][ids].mean() / k),
                total_steps=int(values['steps'][ids].sum()), total_hits=int(values['hits'][ids].sum())))
    return rows, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser, 'drilldown')
    parser.add_argument('--baseline', action='append', default=[], metavar='NAME=NPZ',
        help='Counter output from collect_baseline.py or a checked external collector')
    args = parser.parse_args()
    data = load(args)
    if data is None:
        return
    index, queries, truth, policy = data
    k, ef = policy['k'], policy['ef_init']
    upper = upper_hops(index, queries)
    reference = vanilla_work(index, queries, k, ef, args.threads, upper)
    observed = sage_work(index, queries, policy, args.threads, upper)
    methods = {'Vanilla': reference, 'SAGE': observed}
    for values in methods.values():
        values['hits'] = hits(values['labels'], truth)
    first, reached, _, _ = index._require_index().knn_query_beam_width_first_target_hit_step(
        index._prepare_vectors(queries), truth, reference['hits'], k=k,
        ef_before=ef, ef_after=ef, num_threads=args.threads, bounded_queue=True)
    if not np.all(reached):
        raise ValueError('First-attainment replay failed to reach the Vanilla reference hits')
    for specification in args.baseline:
        name, path = specification.split('=', 1)
        if name in methods:
            raise ValueError(f'Duplicate method: {name}')
        with np.load(Path(path), allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        for field, expected in [('query_sha256', array_hash(queries)), ('truth_sha256', array_hash(truth))]:
            if str(values[field]) != expected:
                raise ValueError(f'{name}: mismatched {field}')
        if str(values['step_definition']) != 'completed_L0_expansions':
            raise ValueError(f'{name}: expected completed expansions, not total hops or started steps')
        np.testing.assert_array_equal(values['qids'], np.arange(len(queries)))
        np.testing.assert_array_equal(values['hits'], hits(values['labels'], truth))
        if (values['steps'].shape != (len(queries),)
                or not np.issubdtype(values['steps'].dtype, np.integer)
                or np.any(values['steps'] < 0)):
            raise ValueError(f'{name}: invalid work counters')
        methods[name] = values
    rows, groups = summarize(methods, first, k, ef)
    save_csv(args.output / 'mean_steps_recall.csv', rows)
    save_csv(args.output / 'cohort_query_ids.csv', [dict(cohort=name, query_id=int(q))
        for name, ids in groups.items() for q in ids])
    save_csv(args.output / 'per_query.csv', [dict(method=name, query_id=q,
        steps=int(values['steps'][q]), hits=int(values['hits'][q]),
        vanilla_first_attainment_step=int(first[q]))
        for name, values in methods.items() for q in range(len(queries))])
    for name in ('Vanilla', 'SAGE'):
        np.savez_compressed(args.output / f'{name.lower()}.npz', **methods[name],
            qids=np.arange(len(queries)), first_steps=first, upper_hops=upper)
    save_json(args.output / 'complete.json', dict(complete=True, methods=list(methods),
        difficulty='Vanilla final hits descending; first attainment L0 step ascending; query ID ascending',
        same_cohorts_across_methods=True, measured_latency=False,
        expected_paper_methods=['Vanilla', 'PiP', 'ABS', 'DARTH', 'Ada-EF', 'SAGE'],
        missing_paper_methods=sorted({'PiP', 'ABS', 'DARTH', 'Ada-EF'} - methods.keys())))


if __name__ == '__main__':
    main()
