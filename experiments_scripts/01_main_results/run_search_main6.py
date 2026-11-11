#!/usr/bin/env python3
"""Fresh native SAGE/vanilla calibration and full curves; one process per graph."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from main6_common import (ROOT, RUN_ROOT, SAGE_ROOT, effective_config, resolve_datasets,
                         hits_per_query, validate_policy_config)
from run_baselines_main6 import (read, save, sha, array_sha, identity, csvwrite, require,
                                 environment, SAGE_BINARY)
from tune_baselines import ef_grid


def fingerprint(args, datasets):
    sources = [Path(__file__), ROOT / 'main6_common.py', ROOT / 'tune_baselines.py',
               ROOT / 'run_baselines_main6.py']
    sources += [SAGE_ROOT / ('experiments_scripts/' + name) for name in (
        'common/final_policy.py', 'common/offline_calibration.py', 'common/adaptive_runtime.py',
        'common/projected_local_acceptable_runtime.py',
        'common/lid_selection.py', 'faiss/faiss_sage_index.py')]
    binary = SAGE_BINARY if args.backend == 'faiss' else next((SAGE_ROOT / 'hnswlib').glob('hnswlib*.so'))
    return dict(k=args.k, backend=args.backend, config_sha256=sha(args.config),
                datasets_config_sha256=sha(args.datasets_config), binary_sha256=sha(binary),
                source_sha256={str(p.relative_to(SAGE_ROOT)): sha(p) for p in sources},
                inputs={d['dataset']: {'data': identity(d['dataset_path']),
                        'index': identity(d[args.backend + '_index']['path'])} for d in datasets})


def dataset_run(args, config, ds):
    import h5py
    import numpy as np
    sys.path.insert(0, str(SAGE_ROOT))
    cpus, threads, k = config['timing']['cpus'], config['online_threads'], config['k']
    os.sched_setaffinity(0, cpus)
    if args.backend == 'faiss':
        sys.path.insert(0, str(SAGE_BINARY.parents[1]))
        import faiss
        require(Path(faiss.__file__).resolve().parent == SAGE_BINARY.parent, 'Wrong FAISS package')
        from experiments_scripts.faiss.faiss_sage_index import Index
        faiss.InterruptCallback.clear_instance()
        faiss.omp_set_num_threads(threads)
    else:
        sys.path.insert(0, str(SAGE_ROOT / 'hnswlib'))
        import hnswlib
        require(Path(hnswlib.__file__).resolve().parent == SAGE_ROOT / 'hnswlib', 'Wrong hnswlib package')
        Index = hnswlib.Index
    from experiments_scripts.common.final_policy import calibrate_policy, query
    os.sched_setaffinity(0, cpus)
    folder = args.output / 'datasets' / ds['dataset']
    folder.mkdir(parents=True, exist_ok=False)

    def stage(name, **values):
        save(folder / 'status.json', dict(stage=name, time=time.time(), **values))
        print(f'[{ds["dataset"]}] {name} {values}', flush=True)

    stage('loading_index')
    started = time.perf_counter()
    index = Index(space=ds['space'], dim=ds['dimension'])
    index.load_index(ds[args.backend + '_index']['path'])
    index.set_num_threads(config['offline_threads'])
    require((index.get_current_count(), index.M, index.ef_construction) ==
            (ds['train_count'], ds['M'], ds['efConstruction']), 'Wrong graph parameters')
    if args.backend == 'faiss':
        require(index._require_index().metric_type ==
                (faiss.METRIC_L2 if ds['space'] == 'l2' else faiss.METRIC_INNER_PRODUCT), 'Wrong metric')
    ids = np.unique(np.linspace(0, ds['train_count'] - 1, 33, dtype=np.int64))
    with h5py.File(ds['dataset_path']) as f:
        expected = np.asarray(f['train'][ids], dtype=np.float32)
    if ds['space'] == 'cosine':
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-30)
    np.testing.assert_allclose(index.get_items(ids), expected, rtol=1e-4, atol=1e-5)
    load_seconds = time.perf_counter() - started

    def sage_grid(recommended, cap):
        return [ef for ef in ef_grid(config, dict(offline_cap=cap, recommended_ef=recommended))
                if ef >= max(k, config['calibration']['offline_min_ef'])]

    stage('calibrating')
    policy = calibrate_policy(index, k=k, backend=args.backend, num_threads=config['offline_threads'],
                min_ef=config['calibration']['offline_min_ef'], shadow_enabled=k != 100,
                curve_efs=sage_grid)
    save(folder / 'policy.json', policy)
    offline = dict(dataset=ds['dataset'], backend=args.backend, k=k,
                   recommended_ef=policy['ef_init'], offline_cap=policy['cap_history'][-1]['cap'],
                   **policy['offline_timings'], index_load_seconds=load_seconds)
    csvwrite(folder / 'offline_times.csv', [offline])
    csvwrite(folder / 'offline_curve.csv', [dict(cap=step['cap'], ef=int(ef), hits=hits,
        recall=hits / (len(policy['probe_ids']) * k), recommended_ef=step['selected_ef'])
        for step in policy['cap_history'] for ef, hits in step['hits_by_ef'].items()])
    stage('policy_frozen', recommended_ef=policy['ef_init'], k=k, shadow_enabled=k != 100)
    # Test queries and evaluation GT enter only after every policy is frozen.
    with h5py.File(ds['dataset_path']) as f:
        vectors = np.asarray(f['test'], dtype=np.float32, order='C')
        truth = np.asarray(f['neighbors'][:, :k], dtype=np.int64, order='C')
    require(vectors.shape == (ds['query_count'], ds['dimension']) and
            truth.shape == (ds['query_count'], k), 'Query/GT shape mismatch')
    require(np.isfinite(vectors).all() and ((truth >= 0) & (truth < ds['train_count'])).all(), 'Invalid inputs')
    require((np.diff(np.sort(truth, axis=1), axis=1) > 0).all(), 'Duplicate GT IDs')
    vanilla_efs = ef_grid(config, offline)
    method_name = 'sage_shadow128' if k != 100 else 'sage_main4'
    jobs = [('vanilla', e) for e in vanilla_efs] + [(method_name, int(e)) for e in policy['curve_policies']]
    n = len(vectors)
    save(folder / 'online_plan.json', dict(jobs=jobs, k=k, policy_sha256=sha(folder / 'policy.json'),
        query_sha256=array_sha(vectors), truth_sha256=array_sha(truth), timing=config['timing']))
    index.set_num_threads(threads)
    snapshots, timings, parameters = {}, [], {}
    if args.backend == 'faiss':
        for ef in vanilla_efs:
            p = faiss.SearchParametersHNSW()
            p.efSearch, p.bounded_queue = ef, True
            parameters[ef] = p

    def call(job):
        method, ef = job
        if method == 'vanilla' and args.backend == 'hnswlib':
            index.set_ef(ef)
        os.sched_setaffinity(0, [cpus[0]] if args.backend == 'faiss' else cpus)
        try:
            begin = time.perf_counter()
            if method != 'vanilla':
                result = query(index, vectors, policy['curve_policies'][str(ef)], num_threads=threads)
            elif args.backend == 'faiss':
                prepared = index._prepare_vectors(vectors)
                distances, labels = index._require_index().search(prepared, k, params=parameters[ef])
                result = labels, distances
            else:
                result = index.knn_query(vectors, k=k, num_threads=threads)
            seconds = time.perf_counter() - begin
        finally:
            os.sched_setaffinity(0, cpus)
        return result, seconds

    stage('warmup', settings=len(jobs))
    for job in jobs:
        for _ in range(config['warmups_per_setting']):
            call(job)
    rng = np.random.default_rng(42)
    for repeat in range(config['timed_repeats']):
        stage('timing', repeat=repeat + 1)
        for position, job_id in enumerate(rng.permutation(len(jobs))):
            job = jobs[int(job_id)]
            result, seconds = call(job)
            if job in snapshots:
                for actual, expected in zip(result, snapshots[job]):
                    np.testing.assert_array_equal(actual, expected)
            snapshots[job] = result
            timings.append(dict(dataset=ds['dataset'], backend=args.backend, k=k, method=job[0],
                ef_init=job[1], repeat=repeat + 1, position=position, seconds=seconds, query_count=n))
        csvwrite(folder / 'raw_timings.csv', timings)
    stage('diagnostics')
    rows, target = [], None
    for job in jobs:
        method, ef = job
        labels, distances = snapshots[job]
        labels = labels.astype(np.int64, copy=False)
        hits = hits_per_query(labels, truth)
        require(((labels >= 0) & (labels < ds['train_count'])).all(), 'Invalid result IDs')
        require((np.diff(np.sort(labels, axis=1), axis=1) > 0).all(), 'Duplicate result IDs')
        diagnostic = {}
        if method != 'vanilla':
            diagnostic = query(index, vectors, policy['curve_policies'][str(ef)],
                               num_threads=threads, return_diagnostics=True)
            np.testing.assert_array_equal(labels, diagnostic.pop('labels'))
            np.testing.assert_array_equal(distances, diagnostic.pop('distances'))
            if k == 100:
                require(not np.any(diagnostic['shadow_routes']), 'Unexpected k100 shadow route')
                if 'shadow_updates' in diagnostic:
                    require(not np.any(diagnostic['shadow_updates']), 'Unexpected k100 shadow work')
        seconds = [r['seconds'] for r in timings if (r['method'], r['ef_init']) == job]
        mean = statistics.mean(seconds)
        row = dict(dataset=ds['dataset'], backend=args.backend, k=k, method=method, ef_init=ef,
            recommended=ef == policy['ef_init'], recommended_ef=policy['ef_init'], query_count=n,
            hits=int(hits.sum()), recall=float(hits.sum() / (n * k)), batch_mean_seconds=mean,
            batch_stdev_seconds=statistics.stdev(seconds) if len(seconds) > 1 else 0,
            qps=n / mean, amortized_query_ms=1000 * mean / n, repeats=len(seconds), threads=threads,
            shadow_enabled=method != 'vanilla' and policy['curve_policies'][str(ef)]['shadow'] is not None,
            ndis_level0_mean=float(np.mean(diagnostic['ndis_level0'])) if diagnostic else '',
            shadow_route_fraction=float(np.mean(diagnostic['shadow_routes'])) if diagnostic else '')
        rows.append(row)
        result_file = folder / f'{method}_ef{ef}.npz'
        np.savez_compressed(result_file, labels=labels, distances=distances, hits=hits, **diagnostic)
        if method != 'vanilla' and ef == policy['ef_init']:
            target = dict(dataset=ds['dataset'], recommended_ef=ef, offline_cap=offline['offline_cap'],
                target_hits=int(hits.sum()), query_count=n, k=k,
                **{args.backend + '_index': identity(ds[args.backend + '_index']['path'])},
                query_sha256=array_sha(vectors),
                truth_sha256=array_sha(truth), labels_file=str(result_file.relative_to(args.output)),
                labels_sha256=array_sha(labels))
    csvwrite(folder / 'summary.csv', rows)
    save(folder / 'target.json', target)
    save(folder / 'complete.json', dict(passed=True, offline=offline, rows=len(rows)))
    stage('complete')


def run(args, config, datasets):
    frozen = fingerprint(args, datasets)
    if args.dataset:
        require(read(args.output / 'manifest.json')['signature'] == frozen, 'Inputs or source changed')
        ds = next(d for d in datasets if d['dataset'] == args.dataset)
        dataset_run(args, config, ds)
        require(fingerprint(args, datasets) == frozen, 'Inputs or source changed during measurement')
        return
    require(not args.output.exists(), f'Use an unused output directory: {args.output}')
    require(len(config['timing']['cpus']) == config['online_threads'] == config['offline_threads'],
            'This protocol uses the same cores for offline and online')
    for ds in datasets:
        expected = ds[args.backend + '_index']
        require(identity(expected['path']) == dict(expected, path=str(Path(expected['path']).resolve())),
                f'{ds["dataset"]}: graph identity changed')
    lock_path = RUN_ROOT / config['timing']['lock_file']
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another main experiment owns the timing cores') from error
        args.output.mkdir(parents=True)
        (args.output / 'logs').mkdir()
        save(args.output / 'manifest.json', dict(signature=frozen, config=config, result_reuse=False))
        summaries, offline, targets = [], [], []
        try:
            for ds in datasets:
                name = ds['dataset']
                print(f'[{args.backend} k={args.k}] {name}', flush=True)
                argv = [sys.executable, str(Path(__file__).resolve()), '--backend', args.backend,
                    '--k', str(args.k), '--config', str(args.config), '--datasets-config', str(args.datasets_config),
                    '--output', str(args.output), '--dataset', name]
                node = config['timing']['numa_node']
                if node is not None:
                    argv = ['numactl', '--physcpubind=' + ','.join(map(str, config['timing']['cpus'])),
                            f'--membind={node}', *argv]
                with (args.output / 'logs' / f'{name}.log').open('w') as log:
                    subprocess.run(argv, env=environment(config), stdout=log, stderr=subprocess.STDOUT, check=True)
                folder = args.output / 'datasets' / name
                for filename, accumulated in [('summary.csv', summaries), ('offline_times.csv', offline)]:
                    with (folder / filename).open() as stream:
                        accumulated.extend(csv.DictReader(stream))
                    csvwrite(args.output / filename, accumulated)
                targets.append(read(folder / 'target.json'))
            require(fingerprint(args, datasets) == frozen, 'Inputs or source changed during run')
            if args.backend == 'faiss':
                save(args.output / 'targets.json', dict(schema_version=1, complete=True, k=args.k,
                    config_sha256=sha(args.config), datasets_config_sha256=sha(args.datasets_config),
                    faiss_binary_sha256=sha(SAGE_BINARY), datasets=targets))
            save(args.output / 'complete.json', dict(passed=True, datasets=config['datasets'], k=args.k))
        except BaseException as error:
            save(args.output / 'failed.json', dict(error=repr(error), dataset=name))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['faiss', 'hnswlib'], required=True)
    for key in ('config', 'datasets-config', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--k', type=int, choices=[10, 100])
    parser.add_argument('--dataset', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ('config', 'datasets_config', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    config = effective_config(args.config, args.k)
    validate_policy_config(config)
    args.k = config['k']
    run(args, config, resolve_datasets(args.datasets_config, config))


if __name__ == '__main__':
    main()
