#!/usr/bin/env python3
"""Convert current DARTH/Ada-EF main-run diagnostics to checked drill-down inputs."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from detail_common import array_hash, hits, save_json, validate_identity


def xvecs(path, dtype):
    words = np.fromfile(path, dtype='<i4')
    if not len(words) or words[0] < 1:
        raise ValueError('Invalid vector file')
    width = int(words[0])
    rows = words.reshape(-1, width + 1)
    if not np.all(rows[:, 0] == width):
        raise ValueError('Inconsistent vector dimensions')
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=('darth', 'adaef'))
    parser.add_argument('--dataset', required=True, type=Path, help='Same HDF5 test queries used in the main run')
    parser.add_argument('--run', required=True, type=Path, help='Output directory from 01/run_darth.py or run_adaef.py')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('--k must be positive')
    with h5py.File(args.dataset) as data:
        queries = np.asarray(data['test'], dtype=np.float32, order='C')
        truth = np.asarray(data['neighbors'][:, :args.k], dtype=np.int64, order='C')
    if args.method == 'darth':
        plan = json.loads((args.run / 'plan.json').read_text())
        for key in ('query_file', 'gt_file'):
            validate_identity(plan['inputs'][key]['path'], plan['inputs'][key])
        original_queries = xvecs(plan['inputs']['query_file']['path'], '<f4')
        original_truth = xvecs(plan['inputs']['gt_file']['path'], '<i4')
        np.testing.assert_array_equal(original_queries[:len(queries)], queries,
            err_msg='DARTH query file does not contain these HDF5 test queries in order')
        np.testing.assert_array_equal(original_truth[:len(truth), :args.k], truth,
            err_msg='DARTH ground truth differs from the supplied HDF5')
        records = json.loads((args.run / 'timings.json').read_text())
        if not any(r.get('phase') == 'untimed_audit' and r.get('matches_timed_outputs') is True for r in records):
            raise ValueError('DARTH native audit did not confirm output parity')
        label_file, dtype, counter_file = 'result.ids.i64', '<i8', 'audit.counters.csv'
    else:
        plan = json.loads((args.run / 'run_plan.json').read_text())
        validate_identity(args.dataset, plan['inputs']['data'])
        if plan['inputs']['k'] != args.k:
            raise ValueError('Ada-EF run used a different k')
        label_file, dtype, counter_file = 'labels.u64', '<u8', 'per_query.csv'
    labels = np.fromfile(args.run / label_file, dtype=dtype).astype(np.int64).reshape(truth.shape)
    with (args.run / counter_file).open(newline='') as stream:
        counters = list(csv.DictReader(stream))
    if len(counters) != len(queries):
        raise ValueError('Incomplete query counters')
    np.testing.assert_array_equal([int(row['qid']) for row in counters], np.arange(len(queries)))
    steps = np.array([int(row['completed_expansions']) for row in counters], np.int64)
    if np.any(steps < 0):
        raise ValueError('Negative completed expansion count')
    observed = hits(labels, truth)
    if args.method == 'adaef':
        online = json.loads((args.run / 'online.json').read_text())
        if int(observed.sum()) != online['metrics']['exact_hits']:
            raise ValueError('Ada-EF label hits differ from the measured run')
        if not all(row['instrumentation_labels_distances_equal'] == '1' for row in counters):
            raise ValueError('Ada-EF instrumented replay failed parity')
        np.testing.assert_allclose([float(row['recall']) for row in counters], observed / args.k, atol=1e-6)
    else:
        timed = [r for r in records if r.get('phase') == 'timed']
        if not timed or any(r['hits'] != int(observed.sum()) for r in timed):
            raise ValueError('DARTH label hits differ from the measured run')
        for row in counters:
            if int(row['started_expansions']) - int(row['completed_expansions']) != int(row['unfinished_expansions']):
                raise ValueError('Inconsistent DARTH completion counters')
            if int(row['unfinished_expansions']) not in (0, 1):
                raise ValueError('Invalid DARTH unfinished expansion count')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'counters.npz', labels=labels, hits=observed,
        steps=steps, qids=np.arange(len(queries)), query_sha256=array_hash(queries),
        truth_sha256=array_hash(truth), step_definition='completed_L0_expansions')
    save_json(args.output / 'complete.json', dict(complete=True, method=args.method,
        native_run=str(args.run.resolve()), query_count=len(queries), k=args.k,
        native_instrumentation_matches_measured_outputs=True, measured_latency=False))


if __name__ == '__main__':
    main()
