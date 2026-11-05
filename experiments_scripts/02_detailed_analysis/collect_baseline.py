#!/usr/bin/env python3
"""Collect untimed PiP/ABS completed expansions at frozen main-result settings."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from detail_common import ROOT, add_inputs, array_hash, hits, preflight, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser, 'baseline')
    parser.add_argument('--method', required=True, choices=('pip', 'abs'))
    parser.add_argument('--settings', type=Path, required=True,
        help='01_main_results baseline tuning/<dataset>/selection.json')
    parser.add_argument('--reference', type=Path, required=True,
        help='Matching online/<dataset>/result.npz with labels/distances')
    args = parser.parse_args()
    policy, dim = preflight(args)
    selection = json.loads(args.settings.read_text())
    selected = selection['selected']
    settings = selected['config']
    receipt = json.loads((args.policy_receipt or args.policy.with_name('target.json')).read_text())
    if selected['method'] != args.method:
        raise ValueError('Selected baseline method differs from --method')
    if (selection['matched_target'] is not True
            or selection['target_hits'] != receipt['target_hits']
            or selection['recommended_ef'] != policy['ef_init']
            or selected['hits'] < receipt['target_hits']):
        raise ValueError('Baseline selection does not match the frozen main-run target')
    with np.load(args.reference, allow_pickle=False) as archive:
        expected = {key: archive[key] for key in ('labels', 'distances')}
    for key, value in expected.items():
        if (value.shape != (receipt['query_count'], policy['k'])
                or array_hash(value) != selected[f'{key}_sha256']):
            raise ValueError(f'Reference {key} differ from the selected main-run result')
    sys.path.insert(0, str(ROOT / f'baselines/{args.method}/scripts'))
    if args.method == 'pip':
        from pip_runtime import load_faiss, parameters, search_with_stats
    else:
        from abs_runtime import load_faiss
        from unit_or_zero_runtime import UnitOrZeroIPABSSearch
    faiss = load_faiss()
    if args.preflight:
        print(json.dumps(dict(passed=True, method=args.method, config=settings, search_calls=0)))
        return
    import h5py
    if args.cpus:
        os.sched_setaffinity(0, set(map(int, args.cpus.split(','))))
    faiss.omp_set_num_threads(args.threads)
    raw = faiss.read_index(str(args.index))
    if raw.d != dim:
        raise ValueError('Graph/query dimension mismatch')
    with h5py.File(args.dataset) as data:
        queries = np.asarray(data['test'], dtype=np.float32, order='C')
        truth = np.asarray(data['neighbors'][:, :policy['k']], dtype=np.int64, order='C')
        if raw.ntotal != data['train'].shape[0]:
            raise ValueError('Graph count differs from the dataset')
    if (receipt['query_sha256'], receipt['truth_sha256']) != (array_hash(queries), array_hash(truth)):
        raise ValueError('Baseline replay queries/truth differ from the frozen main run')
    if int(hits(expected['labels'], truth).sum()) != selected['hits']:
        raise ValueError('Reference label hits differ from the selected main-run result')
    expected_metric = faiss.METRIC_L2 if args.space == 'l2' else faiss.METRIC_INNER_PRODUCT
    if raw.metric_type != expected_metric:
        raise ValueError('Baseline graph metric differs from --space')
    prepared = queries.copy()
    if args.space == 'cosine':
        faiss.normalize_L2(prepared)
    k = policy['k']
    if args.method == 'pip':
        params = parameters(faiss, settings['ef_search'], pip=True,
            threshold=settings['threshold'], patience=settings['patience'], enabled=settings.get('enable', True))
        labels = np.empty((len(queries), k), np.int64)
        distances = np.empty((len(queries), k), np.float32)
        steps = np.empty(len(queries), np.int64)
        for q in range(len(queries)):
            d, ids, stats = search_with_stats(faiss, raw, prepared[q:q+1], k, params)
            labels[q], distances[q], steps[q] = ids[0], d[0], stats['nexpan']
    else:
        with UnitOrZeroIPABSSearch(raw, threads=args.threads,
                unit_normalized=args.space == 'cosine') as runtime:
            distances, labels, counters = runtime.search(prepared, k, gamma=settings['gamma'])
            steps = counters['nexpan'].astype(np.int64)
    np.testing.assert_array_equal(labels, expected['labels'], err_msg='Counter replay changed labels')
    np.testing.assert_array_equal(distances, expected['distances'], err_msg='Counter replay changed distances')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'counters.npz', labels=labels, distances=distances,
        hits=hits(labels, truth), steps=steps, qids=np.arange(len(queries)),
        query_sha256=array_hash(queries), truth_sha256=array_hash(truth),
        step_definition='completed_L0_expansions')
    save_json(args.output / 'complete.json', dict(complete=True, method=args.method, settings=settings,
        labels_distances_match_main=True, measured_latency=False, faiss_package=faiss.__file__,
        dataset=str(args.dataset.resolve()), index=str(args.index.resolve()),
        settings_source=str(args.settings.resolve()), reference=str(args.reference.resolve())))


if __name__ == '__main__':
    main()
