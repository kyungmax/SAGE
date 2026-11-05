#!/usr/bin/env python3
"""Measure a trained DARTH model with the corrected persistent-worker runtime.

Port of darth_persistent_models_main6_k10_20261030/campaign.py's measurement
stage. The model, training-derived intervals and vector preparation are inputs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from main6_common import SAGE_ROOT
from run_baselines_main6 import csvwrite, identity, save, sha


def read_vecs(path, dtype, rows):
    import numpy as np
    raw = np.fromfile(path, dtype='<i4')
    if not raw.size or raw[0] <= 0 or raw.size != rows * (int(raw[0]) + 1):
        raise ValueError(f'Invalid vector-file size: {path}')
    matrix = raw.reshape(rows, int(raw[0]) + 1)
    if not (matrix[:, 0] == matrix[0, 0]).all():
        raise ValueError(f'Inconsistent vector-file dimensions: {path}')
    return np.ascontiguousarray(matrix[:, 1:].view(dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'query-file', 'gt-file', 'gt-dist-file', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--binary', type=Path,
        default=SAGE_ROOT / 'baselines/darth_faiss1141/build/benchmark_new')
    for name in ('queries', 'ef', 'initial-interval', 'min-interval'):
        parser.add_argument('--' + name, type=int, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--target-hits', type=int)
    target.add_argument('--target-recall', type=float,
                        help='Frozen training recall, for a validation set of different size.')
    target.add_argument('--targets', type=Path, help='New FAISS targets.json; selects --dataset.')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--normalize', type=int, choices=[0, 1], default=1)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measured-runs', type=int, default=5)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if min(args.queries, args.ef, args.initial_interval, args.min_interval, args.k, args.threads) <= 0:
        parser.error('Counts, ef and prediction intervals must be positive.')
    if args.measured_runs < 2 or args.warmup_runs < 0:
        parser.error('Need at least two measured runs and nonnegative warmups.')
    inputs = {name: identity(getattr(args, name)) for name in
              ('index', 'query_file', 'gt_file', 'gt_dist_file', 'model', 'binary')}
    import hashlib
    import numpy as np
    queries = read_vecs(args.query_file, '<f4', args.queries)
    truth = read_vecs(args.gt_file, '<i4', args.queries).astype(np.int64)
    truth_distances = read_vecs(args.gt_dist_file, '<f4', args.queries)
    if truth.shape != (args.queries, args.k) or truth_distances.shape != truth.shape:
        parser.error('Prepared ground truth must contain exactly k columns.')
    if not np.isfinite(queries).all() or not np.isfinite(truth_distances).all():
        parser.error('Query and truth-distance vectors must be finite.')
    if (truth < 0).any() or (np.diff(np.sort(truth, axis=1), axis=1) <= 0).any():
        parser.error('Ground truth contains invalid or duplicate IDs.')
    query_sha = hashlib.sha256(queries.tobytes()).hexdigest()
    truth_sha = hashlib.sha256(truth.tobytes()).hexdigest()
    target_hits = args.target_hits
    if args.targets:
        document = json.loads(args.targets.read_text())
        if not document.get('complete'):
            parser.error('Incomplete FAISS target manifest.')
        row = next((r for r in document['datasets'] if r['dataset'] == args.dataset), None)
        if row is None or (row['k'], row['query_count']) != (args.k, args.queries):
            parser.error('Target dataset, k or query count differs.')
        if row['faiss_index'] != inputs['index']:
            parser.error('DARTH must use the same graph as the FAISS target.')
        if row['query_sha256'] != query_sha or row['truth_sha256'] != truth_sha:
            parser.error('Prepared queries/GT differ from the FAISS target inputs; supply raw query vectors.')
        target_hits = row['target_hits']
    target_recall = args.target_recall
    if target_recall is not None:
        if not math.isfinite(target_recall) or not 0 < target_recall <= 1:
            parser.error('Target recall must be in (0, 1].')
        target_hits = math.ceil(target_recall * (args.queries * args.k))
        # Multiplication can round across an integer boundary in either
        # direction; compare recalls to obtain the minimum satisfying hit count.
        if target_hits / (args.queries * args.k) < target_recall:
            target_hits += 1
        if target_hits > 1 and (target_hits - 1) / (args.queries * args.k) >= target_recall:
            target_hits -= 1
    if not 0 < target_hits <= args.queries * args.k:
        parser.error('Invalid exact integer target hit count.')
    if target_recall is None:
        target_recall = target_hits / (args.queries * args.k)
    args.output = args.output.resolve()
    argv = [str(args.binary.resolve())]
    values = {'index': args.index.resolve(), 'query-file': args.query_file.resolve(),
        'gt-file': args.gt_file.resolve(), 'gt-dist-file': args.gt_dist_file.resolve(),
        'model': args.model.resolve(), 'queries': args.queries, 'k': args.k,
        'threads': args.threads, 'ef': args.ef, 'initial-interval': args.initial_interval,
        'min-interval': args.min_interval, 'target': target_recall,
        'mode': 'darth', 'visited': 'auto', 'normalize': args.normalize,
        'model-lifetime': 'persistent', 'warmups': args.warmup_runs,
        'repeats': args.measured_runs, 'result-prefix': args.output / 'result',
        'audit-prefix': args.output / 'audit'}
    for key, value in values.items():
        argv += ['--' + key, str(value)]
    plan = dict(dataset=args.dataset, command=argv, inputs=inputs,
        model_sha256=sha(args.model), binary_sha256=sha(args.binary), target_hits=target_hits,
        target_recall=target_recall,
        query_sha256=query_sha, truth_sha256=truth_sha,
        warmups=args.warmup_runs, measured_runs=args.measured_runs,
        model_lifetime='persistent', no_ann_run=args.preflight)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'plan.json', plan)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND='FALSE', OPENBLAS_NUM_THREADS='1')
    env.pop('OMP_PLACES', None)
    with (args.output / 'native.log').open('x') as log:
        subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = []
    for line in (args.output / 'native.log').read_text().splitlines():
        if line.startswith('{'):
            records.append(json.loads(line))
    timed = [r for r in records if r.get('phase') == 'timed']
    if len(timed) != args.measured_runs:
        raise ValueError('Native runner did not complete all measured calls.')
    stable = {(r['hits'], r['ids_fnv1a64'], r['distances_fnv1a64']) for r in timed}
    if len(stable) != 1:
        raise ValueError('DARTH results changed between repeated calls.')
    seconds = [r['e2e_seconds'] for r in timed]
    hits = timed[0]['hits']
    csvwrite(args.output / 'summary.csv', [dict(dataset=args.dataset, method='darth',
        backend='faiss', k=args.k, query_count=args.queries, hits=hits,
        recall=hits / (args.queries * args.k), target_hits=target_hits,
        target_recall=target_recall,
        meets_target=hits >= target_hits, batch_mean_seconds=statistics.mean(seconds),
        batch_stdev_seconds=statistics.stdev(seconds), repeats=len(seconds), threads=args.threads)])
    save(args.output / 'timings.json', records)
    save(args.output / 'complete.json', dict(passed=True, meets_target=hits >= target_hits))


if __name__ == '__main__':
    main()
