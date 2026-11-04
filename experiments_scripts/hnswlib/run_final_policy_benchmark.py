#!/usr/bin/env python3
"""Benchmark vanilla HNSW against the calibrated final shadow128 SAGE policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time

import h5py
import hnswlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments_scripts.hnswlib.final_policy import calibrate_policy, query


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def parse_efs(value):
    values = sorted({int(e.strip()) for e in value.split(',')})
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError('ef values must be positive integers.')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='HDF5 with train, test, neighbors.')
    parser.add_argument('--index', type=Path, required=True, help='Existing native hnswlib index; never rebuilt.')
    parser.add_argument('--space', choices=('l2', 'ip', 'cosine'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory.')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--offline-threads', type=int, default=24)
    parser.add_argument('--online-threads', type=int, default=24)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measured-runs', type=int, default=3)
    parser.add_argument('--query-limit', type=int, default=0, help='0: all queries; otherwise the first N.')
    parser.add_argument('--vanilla-efs', type=parse_efs,
                        help='Override the vanilla curve grid; the recommended ef is always included.')
    parser.add_argument('--expected-m', type=int, default=32)
    parser.add_argument('--expected-ef-construction', type=int, default=500)
    probe = parser.add_mutually_exclusive_group()
    probe.add_argument('--probe-ids', type=Path, help='NPY containing 500 distinct external labels.')
    probe.add_argument('--lid-pool', type=Path, help='CSV containing query_id and lid.')
    parser.add_argument('--timing-seed', type=int, default=42, help='Reproducible interleaved method order.')
    parser.add_argument('--preflight', action='store_true',
                        help='Inspect file headers/shapes only; no index load, calibration or search.')
    args = parser.parse_args(argv)
    if not 1 <= args.k <= 128:
        parser.error('--k must be in 1..128.')
    if min(args.offline_threads, args.online_threads, args.measured_runs,
           args.expected_m, args.expected_ef_construction) < 1:
        parser.error('Thread counts, measured runs and index parameters must be positive.')
    if min(args.warmup_runs, args.query_limit) < 0:
        parser.error('Warmup runs and query limit must be non-negative.')
    return args


def preflight(args):
    module_path = Path(hnswlib.__file__).resolve()
    if not module_path.is_relative_to(ROOT / 'hnswlib'):
        raise RuntimeError('Use this checkout\'s scripts/python.sh to select its rebuilt hnswlib.')
    files = {name: identity(getattr(args, name)) for name in ('dataset', 'index')}
    for name in ('probe_ids', 'lid_pool'):
        if getattr(args, name) is not None:
            files[name] = identity(getattr(args, name))
    # This checkout uses the native 64-bit little-endian hnswlib serialization.
    # Reject wrong dimensions/backend files before native code dereferences data.
    if struct.calcsize('P') != 8 or sys.byteorder != 'little':
        raise RuntimeError('Index header inspection currently requires a 64-bit little-endian host.')
    header_format = '<6QiI3QdQ'
    with args.index.open('rb') as stream:
        header = stream.read(struct.calcsize(header_format))
    values = struct.unpack(header_format, header)
    _, capacity, count, stride, label_offset, data_offset, _, _, max_m, max_m0, m, _, efc = values
    if not (0 < count <= capacity and 0 < data_offset < label_offset < stride
            and (label_offset - data_offset) % 4 == 0
            and len(header) + count * stride <= files['index']['bytes']):
        raise ValueError('Not a compatible native hnswlib float32 index.')
    dim = (label_offset - data_offset) // 4
    if (m, efc) != (args.expected_m, args.expected_ef_construction):
        raise ValueError(f'Index M/efConstruction is {(m, efc)}, expected '
                         f'{(args.expected_m, args.expected_ef_construction)}.')
    with h5py.File(args.dataset, 'r') as data:
        shapes = {name: tuple(data[name].shape) for name in ('train', 'test', 'neighbors')}
        distance_attribute = str(data.attrs.get('distance', 'unspecified'))
    if any(len(shape) != 2 for shape in shapes.values()):
        raise ValueError('train, test and neighbors must be 2D datasets.')
    if shapes['train'] != (count, dim) or shapes['test'][1] != dim:
        raise ValueError(f'Dataset dimensions/count do not match index: {shapes}, index={(count, dim)}.')
    if shapes['neighbors'][0] != shapes['test'][0] or shapes['neighbors'][1] < args.k:
        raise ValueError('Ground-truth rows/k do not cover the test queries.')
    query_count = min(args.query_limit or shapes['test'][0], shapes['test'][0])
    if query_count < 1:
        raise ValueError('No evaluation queries.')
    return dict(files=files, dataset_shapes=shapes, query_count=query_count,
                dataset_distance_attribute=distance_attribute, space=args.space,
                index_header=dict(count=count, dim=dim, M=m, maxM=max_m, maxM0=max_m0,
                                  ef_construction=efc,
                                  sha256=hashlib.sha256(header).hexdigest()),
                extension=identity(module_path), preflight_only=bool(args.preflight))


def hits_per_query(labels, truth):
    # Row-wise set intersection avoids an O(n*k*k) temporary for large batches.
    return np.asarray([np.intersect1d(a, b).size for a, b in zip(labels, truth)], dtype=np.int64)


def validate_index_vectors(index, args, inspected):
    count = inspected['index_header']['count']
    ids = np.unique(np.linspace(0, count - 1, min(count, 16), dtype=np.int64))
    with h5py.File(args.dataset, 'r') as data:
        expected = np.asarray(data['train'][ids], dtype=np.float32)
    if args.space == 'cosine':
        expected = expected / np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-30)
    actual = np.asarray(index.get_items(ids), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                               err_msg='Index vectors/row labels do not match the dataset and space.')
    return ids.tolist()


def summarize(rows, selected_ef):
    sage = next(row for row in rows if row['method'] == 'sage_shadow128')
    same = next(row for row in rows if row['method'] == 'vanilla' and row['ef_init'] == selected_ef)
    eligible = [row for row in rows if row['method'] == 'vanilla' and row['hits'] >= sage['hits']]
    matched = max(eligible, key=lambda row: row['qps']) if eligible else None
    return dict(selected_ef=selected_ef,
                same_initial_ef=dict(vanilla_recall=same['recall'], sage_recall=sage['recall'],
                                     recall_delta=sage['recall'] - same['recall'],
                                     sage_qps_over_vanilla=sage['qps'] / same['qps']),
                matched_recall_vanilla=None if matched is None else dict(
                    ef=matched['ef_init'], recall=matched['recall'], qps=matched['qps'],
                    sage_qps_over_vanilla=sage['qps'] / matched['qps']),
                matched_recall_rule='Fastest measured vanilla point with hits >= SAGE; '
                                    'retrospective comparison only, no interpolation or policy tuning.')


def run(args, inspected):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    status = output / 'status.json'

    def stage(name, **details):
        print(f'[{name}] {details}', flush=True)
        write_json(status, dict(stage=name, time=time.time(), **details))

    start = time.perf_counter()
    try:
        tracked = [Path(__file__), ROOT / 'experiments_scripts/hnswlib/final_policy.py',
                   ROOT / 'experiments_scripts/common/projected_local_acceptable_runtime.py',
                   ROOT / 'experiments_scripts/common/lid_selection.py',
                   ROOT / 'hnswlib/hnswlib/hnswalg.h', ROOT / 'hnswlib/python_bindings/bindings.cpp',
                   Path(hnswlib.__file__)]
        protocol = dict(inputs=inspected, arguments={k: str(v) if isinstance(v, Path) else v
                                                  for k, v in vars(args).items()},
                        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in tracked},
                        host=dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                                  cpu_affinity=sorted(os.sched_getaffinity(0)), load_average=os.getloadavg()),
                        methods=['vanilla', 'sage_shadow128'],
                        timing='Full query batch wall time; output allocation included; '
                               'calibration, evaluation and diagnostics excluded. QPS=N/mean(seconds).')
        write_json(output / 'protocol.json', protocol)
        stage('loading_index')
        begin = time.perf_counter()
        index = hnswlib.Index(space=args.space, dim=inspected['index_header']['dim'])
        index.load_index(str(args.index))
        index.set_num_threads(args.offline_threads)
        checked_labels = validate_index_vectors(index, args, inspected)
        index_load_seconds = time.perf_counter() - begin

        stage('calibrating_probe500')
        begin = time.perf_counter()
        probe_ids = None if args.probe_ids is None else np.load(args.probe_ids, allow_pickle=False)
        lid_pool = None if args.lid_pool is None else pd.read_csv(args.lid_pool)
        policy = calibrate_policy(index, probe_ids=probe_ids, lid_pool=lid_pool,
                                  k=args.k, num_threads=args.offline_threads)
        offline_seconds = time.perf_counter() - begin
        # Freeze the recommendation and thresholds before reading test vectors/GT.
        write_json(output / 'policy.json', policy)
        policy_sha = sha256(output / 'policy.json')
        cap = policy['cap_history'][-1]['cap']
        curve = [dict(cap=step['cap'], ef=ef, hits=hits, recall=hits / (len(policy['probe_ids']) * args.k),
                      selected_ef=step['selected_ef'])
                 for step in policy['cap_history'] for ef, hits in step['hits_by_ef'].items()]
        write_csv(output / 'offline_curve.csv', curve)
        write_json(output / 'calibration.json', dict(seconds=offline_seconds, policy_sha256=policy_sha,
                   index_load_seconds=index_load_seconds, checked_index_labels=checked_labels,
                   selected_ef=policy['ef_init'], stop_cap=cap, online_data_read=False))
        stage('policy_frozen', ef=policy['ef_init'], cap=cap, shadow_enabled=policy['shadow'] is not None)

        n = inspected['query_count']
        with h5py.File(args.dataset, 'r') as data:
            vectors = np.ascontiguousarray(data['test'][:n], dtype=np.float32)
            truth = np.asarray(data['neighbors'][:n, :args.k], dtype=np.int64)
        if not np.isfinite(vectors).all() or np.any(truth < 0) or np.any(truth >= index.element_count):
            raise ValueError('Invalid query vectors or ground-truth row labels.')
        if np.any(np.diff(np.sort(truth, axis=1), axis=1) == 0):
            raise ValueError('Duplicate labels in a ground-truth row.')
        np.savez_compressed(output / 'evaluation_inputs.npz', query_ids=np.arange(n), truth=truth)
        grid = args.vanilla_efs
        if grid is None:
            grid = list(policy['cap_history'][-1]['hits_by_ef']) + [16, 24, 32, 48, 64, 96]
        grid = sorted({max(args.k, int(e)) for e in grid} | {policy['ef_init']})
        jobs = [('vanilla', ef) for ef in grid] + [('sage_shadow128', policy['ef_init'])]
        write_json(output / 'online_plan.json', dict(policy_sha256=policy_sha, vanilla_efs=grid,
                   query_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),
                   truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
                   query_count=n, query_slice=[0, n], sage_initial_ef=policy['ef_init']))

        def call(job):
            method, _ = job
            if method == 'vanilla':
                return index.knn_query(vectors, k=args.k, num_threads=args.online_threads)
            return query(index, vectors, policy, num_threads=args.online_threads)

        def prepare(job):
            if job[0] == 'vanilla':
                index.set_ef(job[1])

        stage('warmup', settings=len(jobs), repeats=args.measured_runs)
        for job in jobs:
            prepare(job)
            for _ in range(args.warmup_runs):
                call(job)
        timings, snapshots = [], {}
        rng = np.random.default_rng(args.timing_seed)
        for repeat in range(args.measured_runs):
            stage('timing', repeat=repeat + 1)
            for position, job_id in enumerate(rng.permutation(len(jobs))):
                job = jobs[int(job_id)]
                prepare(job)
                begin = time.perf_counter_ns()
                result = call(job)
                seconds = (time.perf_counter_ns() - begin) / 1e9
                # Outside timing: verify identical outputs across repetitions.
                if job in snapshots:
                    for got, expected in zip(result, snapshots[job]):
                        np.testing.assert_array_equal(got, expected)
                snapshots[job] = result
                timings.append(dict(method=job[0], ef_init=job[1], repeat=repeat + 1,
                                    position=position, seconds=seconds, query_count=n))
            write_csv(output / 'timings.csv', timings)

        stage('diagnostics')
        rows = []
        for job in jobs:
            method, ef = job
            labels, distances = snapshots[job]
            hits = hits_per_query(labels, truth)
            if method == 'sage_shadow128':
                diagnostic = query(index, vectors, policy, num_threads=args.online_threads,
                                   return_diagnostics=True)
            else:
                diagnostic = index.knn_query_sage(
                    vectors, k=args.k, ef_init=ef, num_threads=args.online_threads,
                    enable_stop=False, bucket_gamma_ratios=[1., 1., 1.], return_diagnostics=True)
            np.testing.assert_array_equal(diagnostic['labels'], labels)
            np.testing.assert_array_equal(diagnostic['distances'], distances)
            if np.any(diagnostic['final_ef'] > ef):
                raise AssertionError('Search expanded its initial ef.')
            decided = diagnostic['shadow_decision_step'] > 0
            np.testing.assert_array_equal(diagnostic['shadow_updates'][decided],
                                          diagnostic['shadow_updates_at_decision'][decided])
            seconds = [row['seconds'] for row in timings if (row['method'], row['ef_init']) == job]
            mean_seconds = float(np.mean(seconds))
            rows.append(dict(dataset=args.dataset.stem, method=method, ef_init=ef, k=args.k,
                             query_count=n, hits=int(hits.sum()), recall=float(hits.mean() / args.k),
                             qps=n / mean_seconds, batch_seconds_mean=mean_seconds,
                             batch_seconds_std=float(np.std(seconds)),
                             amortized_query_ms=mean_seconds * 1000 / n,
                             online_threads=args.online_threads, measured_runs=args.measured_runs,
                             final_ef_mean=float(diagnostic['final_ef'].mean()),
                             hops_mean=float(diagnostic['hops'].mean()),
                             ndis_level0_mean=float(diagnostic['ndis_level0'].mean()),
                             shadow_route_fraction=float(diagnostic['shadow_routes'].mean())))
            np.savez_compressed(output / f'{method}_ef{ef}.npz', **diagnostic,
                                query_ids=np.arange(n), hits=hits, recall=hits / args.k)
            if method == 'sage_shadow128':
                buckets, counts = np.unique(diagnostic['final_ef'], return_counts=True)
                write_csv(output / 'route_distribution.csv', [dict(final_ef=int(b), queries=int(c),
                    fraction=float(c / n), recall=float(hits[diagnostic['final_ef'] == b].mean() / args.k))
                    for b, c in zip(buckets, counts)])
                pd.DataFrame({name: values for name, values in diagnostic.items() if values.ndim == 1}
                             | dict(query_id=np.arange(n), hits=hits, recall=hits / args.k)
                             ).to_csv(output / 'sage_per_query.csv', index=False)
        write_csv(output / 'summary.csv', rows)
        for name, expected in inspected['files'].items():
            if identity(getattr(args, name)) != expected:
                raise RuntimeError(f'Input changed during the run: {name}')
        comparison = summarize(rows, policy['ef_init'])
        write_json(output / 'summary.json', dict(comparison=comparison, rows=rows,
                   offline_seconds=offline_seconds, index_load_seconds=index_load_seconds,
                   policy_sha256=policy_sha, total_seconds=time.perf_counter() - start,
                   latency_note='amortized_query_ms is batch time / query count, not individual latency.',
                   diagnostic_note='Diagnostic timings excluded; ndis_level0 excludes upper layers.'))
        stage('complete', output=str(output), selected_ef=policy['ef_init'])
        return comparison
    except BaseException as exc:
        stage('failed', error=repr(exc))
        raise


def main(argv=None):
    args = parse_args(argv)
    inspected = preflight(args)
    if args.preflight:
        print(json.dumps(inspected, indent=2))
        return
    print(json.dumps(run(args, inspected), indent=2))


if __name__ == '__main__':
    main()
