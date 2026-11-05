"""Shared input validation and untimed FAISS counters for paper section 2."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty analysis')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def add_inputs(parser, experiment):
    parser.add_argument('--dataset', required=True, type=Path, help='HDF5 with train, test and neighbors arrays')
    parser.add_argument('--index', required=True, type=Path, help='FAISS graph used to calibrate this policy')
    parser.add_argument('--policy', required=True, type=Path, help='Frozen FAISS policy.json from 01_main_results')
    parser.add_argument('--policy-receipt', type=Path, help='Default: target.json beside policy.json')
    parser.add_argument('--M', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=500)
    parser.add_argument('--space', required=True, choices=('l2', 'cosine', 'ip'))
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/02_detailed_analysis' / experiment)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--cpus', help='Optional comma-separated CPU affinity list')
    parser.add_argument('--preflight', action='store_true', help='Check input schemas and native APIs, without loading the index/searching')


def preflight(args):
    import h5py
    from experiments_scripts.common.final_policy import POLICY_VERSION
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    for path in (args.dataset, args.index, args.policy):
        if not path.is_file():
            raise FileNotFoundError(path)
    policy = json.loads(args.policy.read_text())
    if policy.get('version') != POLICY_VERSION or policy.get('backend') != 'faiss':
        raise ValueError('Recalibrate with the current FAISS final policy in 01_main_results')
    if (policy.get('classify_start'), policy.get('classify_end')) != (1, 16):
        raise ValueError('Expected raw CFR window 1–16')
    receipt_path = args.policy_receipt or args.policy.with_name('target.json')
    receipt = json.loads(receipt_path.read_text())
    graph = receipt['faiss_index']
    validate_identity(args.index, graph)
    if (receipt['k'], receipt['recommended_ef']) != (policy['k'], policy['ef_init']):
        raise ValueError('Policy and reference receipt have different k/ef')
    proof = receipt if 'policy_sha256' in receipt else json.loads(args.policy.with_name('online_plan.json').read_text())
    if hashlib.sha256(args.policy.read_bytes()).hexdigest() != proof['policy_sha256']:
        raise ValueError('Policy differs from the frozen main-run receipt')
    with h5py.File(args.dataset) as data:
        n, dim = data['test'].shape
        if n < 1 or data['neighbors'].shape[0] != n or data['neighbors'].shape[1] < policy['k']:
            raise ValueError('Invalid test/ground-truth shape')
        if data['train'].shape[1] != dim or data['train'].shape[0] < policy['k'] or receipt['query_count'] != n:
            raise ValueError('Training/query shapes do not match the reference')
    return policy, dim


def validate_identity(path, expected):
    path = Path(path).resolve()
    stat = path.stat()
    if (path != Path(expected['path']).resolve() or stat.st_size != expected['bytes']
            or stat.st_mtime_ns != expected['mtime_ns']):
        raise ValueError(f'Input differs from the recorded main-run identity: {path}')


def load(args):
    import h5py
    import faiss
    from experiments_scripts.faiss.faiss_sage_index import Index
    policy, dim = preflight(args)
    for name in ('greedy_update_nearest', 'storage_distance_computer'):
        if not hasattr(faiss, name):
            raise RuntimeError(f'Missing {name}; build the current FAISS extension')
    if args.preflight:
        print(json.dumps(dict(passed=True, dataset=str(args.dataset), k=policy['k'], ef=policy['ef_init'], search_calls=0)))
        return None
    if args.cpus:
        requested = set(map(int, args.cpus.split(',')))
        if not requested <= os.sched_getaffinity(0):
            raise ValueError('Requested CPUs are outside the available affinity')
        os.sched_setaffinity(0, requested)
    faiss.omp_set_num_threads(args.threads)
    index = Index(space=args.space, dim=dim)
    index.load_index(str(args.index))
    index.set_num_threads(args.threads)
    with h5py.File(args.dataset) as data:
        queries = np.asarray(data['test'], dtype=np.float32, order='C')
        truth = np.asarray(data['neighbors'][:, :policy['k']], dtype=np.int64, order='C')
        count = data['train'].shape[0]
        if (index.get_current_count(), index.M, index.ef_construction) != (count, args.M, args.ef_construction):
            raise ValueError('Index count/M/efConstruction differs from the dataset/protocol')
        ids = np.unique(np.linspace(0, count - 1, min(33, count), dtype=np.int64))
        expected = np.asarray(data['train'][ids], dtype=np.float32)
    metric = faiss.METRIC_L2 if args.space == 'l2' else faiss.METRIC_INNER_PRODUCT
    if index._require_index().metric_type != metric:
        raise ValueError('Graph metric differs from --space')
    if args.space == 'cosine':
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-30)
    np.testing.assert_allclose(index.get_items(ids), expected, rtol=1e-4, atol=1e-5,
                               err_msg='Stored index vectors differ from HDF5 train rows')
    receipt = json.loads((args.policy_receipt or args.policy.with_name('target.json')).read_text())
    if (receipt['query_sha256'], receipt['truth_sha256']) != (array_hash(queries), array_hash(truth)):
        raise ValueError('Test query/ground-truth arrays differ from the main policy run')
    if not np.isfinite(queries).all():
        raise ValueError('Non-finite query vectors')
    args.output.mkdir(parents=True, exist_ok=False)
    stat = args.index.stat()
    save_json(args.output / 'inputs.json', dict(
        dataset=str(args.dataset.resolve()), index=str(args.index.resolve()),
        index_bytes=stat.st_size, index_mtime_ns=stat.st_mtime_ns,
        policy=str(args.policy.resolve()), policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        query_sha256=array_hash(queries), truth_sha256=array_hash(truth),
        k=policy['k'], ef=policy['ef_init'], threads=args.threads,
        cpus=sorted(os.sched_getaffinity(0)), faiss_package=faiss.__file__))
    return index, queries, truth, policy


def hits(labels, truth):
    if labels.shape != truth.shape:
        raise ValueError('Labels and ground truth have different shapes')
    return (labels[:, :, None] == truth[:, None, :]).any(axis=2).sum(axis=1).astype(np.int64)


def upper_hops(index, queries):
    """Run the same native greedy descent, using its metric-aware distance computer."""
    import faiss
    raw = index._require_index()
    vectors = index._prepare_vectors(queries)
    computer = faiss.storage_distance_computer(raw.storage)
    result = np.zeros(len(vectors), np.int64)
    for i, vector in enumerate(vectors):
        computer.set_query(faiss.swig_ptr(vector))
        nearest = np.array([raw.hnsw.entry_point], np.int32)
        distance = np.array([computer(int(nearest[0]))], np.float32)
        for level in range(raw.hnsw.max_level, 0, -1):
            stats = faiss.greedy_update_nearest(raw.hnsw, computer, level,
                        faiss.swig_ptr(nearest), faiss.swig_ptr(distance))
            result[i] += stats.nhops
    return result


def vanilla(index, queries, k, ef, threads):
    index.set_ef(ef)
    labels, distances = index.knn_query(queries, k=k, num_threads=threads)
    return dict(labels=labels, distances=distances)


def vanilla_work(index, queries, k, ef, threads, upper):
    expected = vanilla(index, queries, k, ef, threads)
    diagnostic = index.knn_query_sage(queries, k=k, ef_init=ef, enable_stop=False,
        num_threads=threads, return_diagnostics=True, bucket_gamma_ratios=[1., 1., 1.])
    for key in ('labels', 'distances'):
        np.testing.assert_array_equal(diagnostic[key], expected[key], err_msg='Vanilla counter replay changed outputs')
    steps = diagnostic['hops'].astype(np.int64) - upper
    if np.any(steps < 0):
        raise ValueError('Negative level-0 expansion count')
    return dict(**expected, steps=steps)


def sage_work(index, queries, policy, threads, upper):
    from experiments_scripts.common.final_policy import query
    labels, distances = query(index, queries, policy, num_threads=threads)
    observed = query(index, queries, policy, num_threads=threads, return_diagnostics=True, return_routing=True)
    np.testing.assert_array_equal(observed['labels'], labels)
    np.testing.assert_array_equal(observed['distances'], distances)
    observed['steps'] = observed['hops'].astype(np.int64) - upper
    if np.any(observed['steps'] < 0):
        raise ValueError('Negative SAGE level-0 expansion count')
    return observed


def candidate_tiers(policy):
    """Ported from false_easy_tier_oracle_20261029/aggregate.py."""
    k, ef = int(policy['k']), int(policy['ef_init'])
    tiers = {ef, *map(int, policy['main']['route_efs'])}
    if list(policy['main']['route_efs']) != [max(k, ef * i // 4) for i in (1, 2, 3)]:
        raise ValueError('Expected four main tiers')
    if policy.get('shadow') is not None:
        shadow = policy['shadow']
        if shadow['rank'] != 128 or list(shadow['route_efs']) != [max(k, e) for e in (32, 64, 96)]:
            raise ValueError('Expected shadow128 tiers')
        tiers.update(map(int, shadow['route_efs']))
        tiers.add(min(ef, max(k, 128)))
    return sorted(tiers)


def initial_budgets(policy, diagnostic):
    """Recover initial assignment from native CFRs; validate one-tier promotion.

    The native API exports final budgets and observed CFRs, not initial cuts.
    Float32 arithmetic mirrors the native threshold comparison.
    """
    ef, k = int(policy['ef_init']), int(policy['k'])
    final = diagnostic['final_efs'].astype(np.int64)
    initial = np.full(len(final), ef, np.int64)
    for q in range(len(final)):
        shadow = bool(diagnostic['shadow_routes'][q])
        config = policy['shadow'] if shadow else policy['main']
        value = diagnostic['shadow_cfrs' if shadow else 'main_cfrs'][q]
        base = 128 if shadow else ef
        if config is not None and np.isfinite(value):
            tau = np.float32(config['tau'])
            ratio = np.float32(value) / max(tau, np.float32(1e-6))
            if shadow or value <= tau:
                for i, gamma in enumerate(config['gammas']):
                    if ratio <= np.float32(gamma):
                        initial[q] = max(k, base * (i + 1) // 4)
                        break
        next_tiers = sorted({min(ef, max(k, base * i // 4)) for i in range(1, 5)} | {ef})
        larger = [value for value in next_tiers if value > initial[q]]
        allowed = {initial[q], larger[0] if larger else ef}
        if int(final[q]) not in allowed or (shadow and initial[q] == ef):
            raise ValueError(f'Cannot reconstruct native initial route at query {q}')
    return initial
