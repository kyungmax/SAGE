#!/usr/bin/env python3
"""One dataset per process: frozen FAISS calibration, or new full online curves.

The supervisor owns NUMA placement, timing locks, index builds and aggregation.
This worker never builds or deletes an index and never starts another stage.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import statistics
import sys
import time

from main6_common import SAGE_ROOT, hits_per_query
from run_baselines_main6 import (read, save, sha, array_sha, identity, csvwrite,
                                 require, SAGE_BINARY)


def signature(args, ds):
    sources = [Path(__file__), Path(__file__).with_name('main6_common.py'),
               Path(__file__).with_name('run_baselines_main6.py')]
    sources += sorted((SAGE_ROOT / 'experiments_scripts/common').glob('*.py'))
    sources.append(SAGE_ROOT / 'experiments_scripts/faiss/faiss_sage_index.py')
    binary = (SAGE_BINARY if args.backend == 'faiss' else
              next((SAGE_ROOT / 'hnswlib').glob('hnswlib*.so')))
    return dict(backend=args.backend, dataset=ds['dataset'],
        config_sha256=sha(args.config), binary_sha256=sha(binary),
        data=identity(ds['dataset_path']), index=identity(ds[args.backend + '_index']['path']),
        sources={str(p.relative_to(SAGE_ROOT)): sha(p) for p in sources})


def load_index(args, config, ds):
    import h5py
    import numpy as np
    cpus = config['timing']['cpus']
    os.sched_setaffinity(0, cpus)
    sys.path.insert(0, str(SAGE_ROOT))
    if args.backend == 'faiss':
        sys.path.insert(0, str(SAGE_BINARY.parents[1]))
        import faiss
        require(Path(faiss.__file__).resolve().parent == SAGE_BINARY.parent, 'Wrong FAISS package')
        from experiments_scripts.faiss.faiss_sage_index import Index
        faiss.InterruptCallback.clear_instance()
        faiss.omp_set_num_threads(config['offline_threads'])
    else:
        sys.path.insert(0, str(SAGE_ROOT / 'hnswlib'))
        import hnswlib
        require(Path(hnswlib.__file__).resolve().parent == SAGE_ROOT / 'hnswlib',
                'Wrong hnswlib package')
        Index = hnswlib.Index
    begin = time.perf_counter()
    index = Index(space=ds['space'], dim=ds['dimension'])
    index.load_index(ds[args.backend + '_index']['path'])
    index.set_num_threads(config['offline_threads'])
    require((index.get_current_count(), index.M, index.ef_construction) ==
            (ds['train_count'], ds['M'], ds['efConstruction']), 'Wrong graph parameters')
    if args.backend == 'faiss':
        require(index._require_index().metric_type ==
                (faiss.METRIC_L2 if ds['space'] == 'l2' else faiss.METRIC_INNER_PRODUCT),
                'Wrong graph metric')
    ids = np.unique(np.linspace(0, ds['train_count'] - 1, 33, dtype=np.int64))
    with h5py.File(ds['dataset_path']) as stream:
        expected = np.asarray(stream['train'][ids], dtype=np.float32)
    if ds['space'] == 'cosine':
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-30)
    np.testing.assert_allclose(index.get_items(ids), expected, rtol=1e-4, atol=1e-5)
    return index, time.perf_counter() - begin


def write_calibration(folder, ds, backend, policy, load_seconds):
    save(folder / 'policy.json', policy)
    common = backend == 'faiss'
    row = dict(dataset=ds['dataset'], backend=backend, k=policy['k'],
        recommended_ef=policy['ef_init'] if common else '',
        diagnostic_selected_ef='' if common else policy['ef_init'],
        recommendation_published=common, offline_cap=policy['cap_history'][-1]['cap'],
        saturation_status=policy['saturation_status'],
        calibration_probe_count=len(policy['probe_ids']),
        **policy['offline_timings'], index_load_seconds=load_seconds)
    csvwrite(folder / 'offline_times.csv', [row])
    csvwrite(folder / 'offline_curve.csv', [dict(cap=step['cap'], ef=int(ef), hits=hits,
        recall=hits / (len(policy['probe_ids']) * policy['k']), selected_ef=step['selected_ef'],
        selection_role='common_faiss_reference' if common else 'internal_grid_diagnostic')
        for step in policy['cap_history'] for ef, hits in step['hits_by_ef'].items()])
    return row


def run(args, config, ds):
    import numpy as np
    sys.path.insert(0, str(SAGE_ROOT))
    from experiments_scripts.common.final_policy import POLICY_VERSION, calibrate_policy, query
    from experiments_scripts.common.full_run_policy import curve_grid, fit_curve_policies
    from experiments_scripts.common.offline_calibration import DEFAULT_CALIBRATION_PROBE_COUNT
    require(DEFAULT_CALIBRATION_PROBE_COUNT == config['calibration']['probe_count'] == 500,
            'Approved run requires the current 500-probe calibration')
    require(config['k'] == 10 and config['timed_repeats'] >= 2 and
            config['warmups_per_setting'] >= 0, 'Expected k10, at least two measured runs and nonnegative warmups')
    require(len(config['timing']['cpus']) == config['online_threads'] == config['offline_threads'],
            'Same assigned cores required for offline and online work')
    initial_signature = signature(args, ds)
    frozen = None
    if args.backend == 'faiss' and args.stage == 'online':
        require(args.policy is not None, 'FAISS online requires --policy from the calibration stage')
        manifest = read(args.policy.parent / 'manifest.json')
        require(manifest['signature'] == initial_signature, 'Frozen calibration source/input changed')
        completion = read(args.policy.parent / 'complete.json')
        require(completion['passed'] and completion['policy_sha256'] == sha(args.policy),
                'Incomplete or changed frozen calibration')
        frozen = read(args.policy)
        require(frozen['version'] == POLICY_VERSION and frozen['backend'] == 'faiss',
                'Wrong frozen policy')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'manifest.json', dict(stage=args.stage, signature=initial_signature,
        dataset=ds, config=config, result_reuse=False,
        policy_input=str(args.policy) if args.policy else None,
        policy_input_sha256=sha(args.policy) if args.policy else None,
        recommended_timing_order=['vanilla', 'sage_shadow128'] if frozen else [],
        recommendation_scope='FAISS only; hnswlib curve only'))

    def stage(name, **details):
        save(args.output / 'status.json', dict(stage=name, time=time.time(), **details))
        print(f'[{args.backend} {ds["dataset"]}] {name} {details}', flush=True)

    try:
        stage('loading_index')
        index, load_seconds = load_index(args, config, ds)
        if args.stage == 'calibrate':
            require(args.backend == 'faiss', 'The recommendation stage is FAISS only')
            stage('calibrating')
            policy = calibrate_policy(index, k=10, backend='faiss',
                num_threads=config['offline_threads'], min_ef=64, shadow_enabled=True)
            offline = write_calibration(args.output, ds, 'faiss', policy, load_seconds)
            require(signature(args, ds) == initial_signature, 'Source/input changed during calibration')
            save(args.output / 'complete.json', dict(passed=True,
                policy_sha256=sha(args.output / 'policy.json'), offline=offline,
                test_queries_loaded=False, online_measurement_run=False))
            stage('complete', recommended_ef=policy['ef_init'])
            return

        def grid_for(ef, cap):
            return curve_grid(recommended_ef=ef, offline_cap=cap, k=10,
                              initial_cap=config['online'].get('initial_curve_cap', 2048))

        if frozen is None:
            stage('calibrating_curve_thresholds')
            policy = calibrate_policy(index, k=10, backend='hnswlib',
                num_threads=config['offline_threads'], min_ef=64, shadow_enabled=True,
                curve_efs=grid_for)
            offline = write_calibration(args.output, ds, 'hnswlib', policy, load_seconds)
            curves = policy['curve_policies']
            curve_setup = dict(curve_extra_seconds=policy['offline_timings']['curve_extra_seconds'],
                               recommendation_published=False)
        else:
            policy = frozen
            offline = write_calibration(args.output, ds, 'faiss', policy, load_seconds)
            curves = deepcopy(policy['curve_policies'])
            curve_setup = None
        ef_reference = int(policy['ef_init'])
        curve_efs = grid_for(ef_reference, policy['cap_history'][-1]['cap'])
        vanilla_efs = sorted({10, *config['online'].get('low_ef_extra', [16, 24, 32, 48]), *curve_efs})
        method_name = 'sage_shadow128'
        jobs = [('vanilla', ef) for ef in vanilla_efs] + [(method_name, ef) for ef in curve_efs]
        stage('loading_evaluation_queries')
        import h5py
        with h5py.File(ds['dataset_path']) as stream:
            vectors = np.asarray(stream['test'], dtype=np.float32, order='C')
            truth = np.asarray(stream['neighbors'][:, :10], dtype=np.int64, order='C')
        n, threads, cpus = len(vectors), config['online_threads'], config['timing']['cpus']
        require(vectors.shape == (ds['query_count'], ds['dimension']) and truth.shape == (n, 10),
                'Query or ground-truth shape changed')
        require(np.isfinite(vectors).all() and ((truth >= 0) & (truth < ds['train_count'])).all(),
                'Invalid evaluation input')
        require((np.diff(np.sort(truth, axis=1), axis=1) > 0).all(), 'Duplicate ground-truth IDs')
        save(args.output / 'online_plan.json', dict(jobs=jobs, k=10,
            query_sha256=array_sha(vectors), truth_sha256=array_sha(truth), timing=config['timing'],
            recommended_first=bool(frozen), recommended_method_order=['vanilla', method_name] if frozen else [],
            hnswlib_recommendation_published=False))
        index.set_num_threads(threads)
        parameters, snapshots, timings = {}, {}, []
        if args.backend == 'faiss':
            import faiss
            for ef in vanilla_efs:
                parameter = faiss.SearchParametersHNSW()
                parameter.efSearch, parameter.bounded_queue = ef, True
                parameters[ef] = parameter

        def call(job):
            method, ef = job
            if method == 'vanilla' and args.backend == 'hnswlib':
                index.set_ef(ef)
            os.sched_setaffinity(0, [cpus[0]] if args.backend == 'faiss' else cpus)
            try:
                begin = time.perf_counter()
                if method != 'vanilla':
                    result = query(index, vectors, curves[str(ef)], num_threads=threads)
                elif args.backend == 'faiss':
                    prepared = index._prepare_vectors(vectors)
                    distances, labels = index._require_index().search(prepared, 10, params=parameters[ef])
                    result = labels, distances
                else:
                    result = index.knn_query(vectors, k=10, num_threads=threads)
                seconds = time.perf_counter() - begin
            finally:
                os.sched_setaffinity(0, cpus)
            return result, seconds

        def timed_call(job, repeat, position, phase):
            result, seconds = call(job)
            if job in snapshots:
                for actual, expected in zip(result, snapshots[job]):
                    np.testing.assert_array_equal(actual, expected)
            snapshots[job] = result
            timings.append(dict(dataset=ds['dataset'], backend=args.backend, k=10,
                method=job[0], ef_init=job[1], repeat=repeat, position=position,
                seconds=seconds, query_count=n, measurement_phase=phase))
            csvwrite(args.output / 'raw_timings.csv', timings)

        recommended_jobs = [('vanilla', ef_reference), (method_name, ef_reference)] if frozen else []
        # No full-curve warmup or threshold fitting precedes these headline timers.
        for position, job in enumerate(recommended_jobs):
            stage('timing_recommended', method=job[0], ef=job[1])
            for _ in range(config['warmups_per_setting']):
                call(job)
            for repeat in range(1, config['timed_repeats'] + 1):
                timed_call(job, repeat, position, 'recommended_first')
        if frozen:
            stage('fitting_frozen_probe_curve')
            curves, curve_setup = fit_curve_policies(index, policy, curve_efs,
                                                     num_threads=config['offline_threads'])
            require(curves[str(ef_reference)] == frozen['curve_policies'][str(ef_reference)],
                    'Frozen recommended-point thresholds changed')
        save(args.output / 'curve_policies.json', curves)
        save(args.output / 'curve_preparation.json', curve_setup)
        remaining = [job for job in jobs if job not in recommended_jobs]
        stage('warming_curve', settings=len(remaining))
        for job in remaining:
            for _ in range(config['warmups_per_setting']):
                call(job)
        rng = np.random.default_rng(42)
        for repeat in range(1, config['timed_repeats'] + 1):
            stage('timing_curve', repeat=repeat, settings=len(remaining))
            for position, job_id in enumerate(rng.permutation(len(remaining))):
                timed_call(remaining[int(job_id)], repeat, position, 'curve')
        stage('diagnostics')
        rows, target = [], None
        for job in jobs:
            method, ef = job
            labels, distances = snapshots[job]
            labels = labels.astype(np.int64, copy=False)
            require(((labels >= 0) & (labels < ds['train_count'])).all(), 'Invalid result IDs')
            require((np.diff(np.sort(labels, axis=1), axis=1) > 0).all(), 'Duplicate result IDs')
            hits = hits_per_query(labels, truth)
            diagnostic = {}
            if method != 'vanilla':
                diagnostic = query(index, vectors, curves[str(ef)], num_threads=threads,
                                   return_diagnostics=True)
                np.testing.assert_array_equal(labels, diagnostic.pop('labels'))
                np.testing.assert_array_equal(distances, diagnostic.pop('distances'))
            seconds = [r['seconds'] for r in timings if (r['method'], r['ef_init']) == job]
            mean = statistics.mean(seconds)
            recommended = bool(frozen) and ef == ef_reference
            rows.append(dict(dataset=ds['dataset'], backend=args.backend, k=10, method=method,
                ef_init=ef, recommended=recommended, recommended_ef=ef_reference if frozen else '',
                query_count=n, hits=int(hits.sum()), recall=float(hits.sum() / (n * 10)),
                batch_mean_seconds=mean, batch_stdev_seconds=statistics.stdev(seconds),
                qps=n / mean, amortized_query_ms=1000 * mean / n, repeats=len(seconds), threads=threads,
                shadow_enabled=method != 'vanilla' and curves[str(ef)]['shadow'] is not None,
                ndis_level0_mean=float(np.mean(diagnostic['ndis_level0'])) if diagnostic else '',
                shadow_route_fraction=float(np.mean(diagnostic['shadow_routes'])) if diagnostic else '',
                measurement_phase='recommended_first' if job in recommended_jobs else 'curve'))
            result_file = args.output / f'{method}_ef{ef}.npz'
            np.savez_compressed(result_file, labels=labels, distances=distances, hits=hits, **diagnostic)
            if method != 'vanilla' and recommended:
                target = dict(dataset=ds['dataset'], recommended_ef=ef,
                    offline_cap=policy['cap_history'][-1]['cap'], target_hits=int(hits.sum()),
                    query_count=n, k=10, faiss_index=identity(ds['faiss_index']['path']),
                    query_sha256=array_sha(vectors), truth_sha256=array_sha(truth),
                    labels_file=result_file.name, labels_sha256=array_sha(labels))
        csvwrite(args.output / 'summary.csv', rows)
        if target is not None:
            save(args.output / 'target.json', target)
        require(signature(args, ds) == initial_signature, 'Source/input changed during measurement')
        save(args.output / 'complete.json', dict(passed=True, rows=len(rows), offline=offline,
            recommendations_published=args.backend == 'faiss',
            target_written=target is not None, curve_preparation=curve_setup))
        stage('complete')
    except BaseException as exc:
        stage('failed', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('calibrate', 'online'), required=True)
    parser.add_argument('--backend', choices=('faiss', 'hnswlib'), required=True)
    for option in ('config', 'dataset-config', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--policy', type=Path)
    args = parser.parse_args()
    for field in ('config', 'dataset_config', 'output', 'policy'):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    config, ds = read(args.config), read(args.dataset_config)
    require(args.stage != 'calibrate' or args.backend == 'faiss', 'Calibration-only stage is FAISS only')
    require(args.backend == 'faiss' or args.policy is None, 'hnswlib uses only its own curve calibration')
    run(args, config, ds)


if __name__ == '__main__':
    main()
