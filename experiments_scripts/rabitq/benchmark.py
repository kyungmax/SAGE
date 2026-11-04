#!/usr/bin/env python3
"""RaBitQ benchmark driver with train-only calibration and paired curves."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'experiments_scripts/main'))
from main6_common import ROOT, RUN_ROOT, SAGE_ROOT, hits_per_query, validate_policy_config
from run_baselines_main6 import read, save, sha, array_sha, csvwrite, require, environment
from launch_main6_online24 import siblings, cpu_ticks, utilization
from tune_baselines import ef_grid

sys.path.insert(0, str(SAGE_ROOT))


def rabitq_environment(plan):
    env = environment(plan)
    # RaBitQ uses std::thread. OpenMP's `close` binding pins the importing
    # thread to its first place; every native worker then inherits that CPU.
    # numactl owns the 24-core mask for this backend; BLAS stays single-threaded.
    env['OMP_PROC_BIND'] = 'FALSE'
    env.pop('OMP_PLACES', None)
    return env


def require_search_affinity(cpus):
    actual = sorted(os.sched_getaffinity(0))
    require(actual == sorted(cpus), f'RaBitQ caller affinity narrowed: {actual}; expected {sorted(cpus)}')


def dataset(entry):
    from experiments_scripts.rabitq.runner import Dataset
    return Dataset(entry['data_path'], raw=entry['raw'], metric=entry['metric'],
                   normalize=entry['normalize'])


def fingerprint(plan_path, plan):
    from experiments_scripts.rabitq.runner import file_identity
    sources = [Path(__file__), ROOT / 'main6_common.py', ROOT / 'run_baselines_main6.py',
               ROOT / 'launch_main6_online24.py', ROOT / 'tune_baselines.py',
               SAGE_ROOT / 'scripts/python.sh', SAGE_ROOT / 'sage_env.sh']
    for folder in ('common', 'rabitq'):
        sources.extend(sorted((SAGE_ROOT / 'experiments_scripts' / folder).glob('*.py')))
    sources.extend(sorted((SAGE_ROOT / 'rabitq/build_sage/python/rabitqlib').glob('*.py')))
    binaries = sorted((SAGE_ROOT / 'rabitq/build_sage/python/rabitqlib').glob('*.so'))
    require(len(binaries) == 1, 'Expected exactly one local RaBitQ extension')
    return dict(plan_sha256=sha(plan_path), binary_sha256=sha(binaries[0]),
                source_sha256={str(p): sha(p) for p in sources},
                inputs={e['dataset']: dict(data=dataset(e).identity(),
                        index=file_identity(e['index_path'])) for e in plan['entries']})


def preflight(plan):
    """Only file identities, runtime import and HDF5 metadata; no graph loads/search."""
    import h5py
    from experiments_scripts.rabitq.final_policy import load_library
    from experiments_scripts.common.offline_calibration import FIXED_CALIBRATION_LID_POOL_SIZE
    from experiments_scripts.rabitq.runner import file_identity
    validate_policy_config(plan)
    threads = plan['online_threads']
    require(threads == plan['offline_threads'] == len(set(plan['timing']['cpus'])),
            'Offline/online must use the same physical cores')
    require(plan['warmups_per_setting'] >= 1 and plan['timed_repeats'] >= 2,
            'Need warmup and repeated timings')
    require(plan['timing']['include_query_preprocessing'], 'Preprocessing must be timed')
    require(set(plan['timing']['cpus']) <= os.sched_getaffinity(0), 'Timing CPUs unavailable')
    library = load_library()
    entries = {e['dataset']: e for e in plan['entries']}
    require(len(entries) == len(plan['entries']), 'Duplicate dataset entries')
    rows = []
    for entry in entries.values():
        data = dataset(entry)
        require(data.identity() == entry['dataset_identity'], f"Changed dataset: {entry['dataset']}")
        require(file_identity(entry['index_path']) == entry['index_identity'],
                f"Changed index: {entry['dataset']}")
        require(entry['saved_reloaded_exact_match'], 'Index needs save/reload verification')
        with h5py.File(data.path) as handle:
            require(handle['train'].shape == data.shape == tuple(entry['shape']), 'Train shape mismatch')
            require(handle['test'].shape == (entry['query_count'], data.shape[1]), 'Query shape mismatch')
            require(handle['neighbors'].shape[0] == entry['query_count'], 'GT row mismatch')
            width = handle['neighbors'].shape[1]
        selected = [int(k) for k, names in plan['datasets_by_k'].items() if entry['dataset'] in names]
        require(selected and max(selected) <= width, 'Ground truth has insufficient neighbors')
        require(data.shape[0] >= FIXED_CALIBRATION_LID_POOL_SIZE,
                'Production calibration requires at least 10000 indexed vectors')
        rows.append(dict(dataset=entry['dataset'], k=selected, queries=entry['query_count'],
                         ground_truth_width=width, index_bytes=entry['index_identity']['size'],
                         lid_pool_size=FIXED_CALIBRATION_LID_POOL_SIZE))
    for k, names in plan['datasets_by_k'].items():
        require(int(k) in (10, 100) and len(names) == len(set(names)) and
                set(names) <= entries.keys(), 'Invalid dataset/k scope')
    return dict(passed=True, real_search_started=False, runtime=library.__file__,
                policy_api=library.SAGE_POLICY_API, cells=sum(len(v) for v in plan['datasets_by_k'].values()),
                datasets=rows)


def cell_run(plan, entry, k, folder):
    import h5py
    import numpy as np
    from experiments_scripts.rabitq.final_policy import load_library, calibrate_policy, query
    from experiments_scripts.rabitq.runner import validate_index
    config = dict(plan, k=k)
    threads = plan['online_threads']
    folder.mkdir(parents=True, exist_ok=False)

    def stage(name, **details):
        save(folder / 'status.json', dict(stage=name, time=time.time(), pid=os.getpid(), **details))
        print(f"[{entry['dataset']} k={k}] {name} {details}", flush=True)

    os.sched_setaffinity(0, plan['timing']['cpus'])
    data = dataset(entry)
    stage('loading_index')
    started = time.perf_counter()
    index = load_library().HnswIndex.load(entry['index_path'])
    validate_index(index, data, SimpleNamespace(**entry))
    require_search_affinity(plan['timing']['cpus'])
    save(folder / 'runtime_affinity.json', dict(pid=os.getpid(), cpus=sorted(os.sched_getaffinity(0)),
         native_threading='std::thread', omp_proc_bind=os.environ.get('OMP_PROC_BIND'),
         omp_places=os.environ.get('OMP_PLACES'), threads=threads))
    load_seconds = time.perf_counter() - started

    def sage_grid(recommended, cap):
        return [ef for ef in ef_grid(config, dict(offline_cap=cap, recommended_ef=recommended))
                if ef >= max(k, config['calibration']['offline_min_ef'])]

    stage('calibrating')
    policy = calibrate_policy(index, train_vectors=data.read, k=k, num_threads=threads,
                              min_ef=64, shadow_enabled=k != 100, curve_efs=sage_grid)
    require_search_affinity(plan['timing']['cpus'])
    save(folder / 'policy.json', policy)
    offline = dict(dataset=entry['dataset'], backend='rabitq', k=k,
                   recommended_ef=policy['ef_init'], offline_cap=policy['cap_history'][-1]['cap'],
                   index_load_seconds=load_seconds, **policy['offline_timings'])
    csvwrite(folder / 'offline_times.csv', [offline])
    csvwrite(folder / 'offline_curve.csv', [dict(cap=step['cap'], ef=int(ef), hits=hits,
        recall=hits / (len(policy['probe_ids']) * k), recommended_ef=step['selected_ef'])
        for step in policy['cap_history'] for ef, hits in step['hits_by_ef'].items()])
    stage('policy_frozen', recommended_ef=policy['ef_init'])
    # Evaluation values are first read after all policies have been frozen.
    with h5py.File(data.path) as handle:
        vectors = np.asarray(handle['test'], dtype=np.float32, order='C')
        truth = np.asarray(handle['neighbors'][:, :k], dtype=np.int64)
    n = len(vectors)
    require(vectors.shape == (entry['query_count'], index.dim) and truth.shape == (n, k), 'Invalid query/GT shapes')
    require(np.isfinite(vectors).all() and ((truth >= 0) & (truth < index.num_points)).all(), 'Invalid query/GT')
    require((np.diff(np.sort(truth, axis=1), axis=1) > 0).all(), 'Duplicate GT labels')
    jobs = [('vanilla', ef) for ef in ef_grid(config, offline)]
    jobs += [('sage', int(ef)) for ef in policy['curve_policies']]
    save(folder / 'online_plan.json', dict(jobs=jobs, policy_sha256=sha(folder / 'policy.json'),
         query_sha256=array_sha(vectors), truth_sha256=array_sha(truth), timing=plan['timing']))

    def call(job, diagnostic=False):
        method, ef = job
        require_search_affinity(plan['timing']['cpus'])
        begin = time.perf_counter()
        prepared = data.prepare(vectors)
        if method == 'vanilla':
            result = index.search(prepared, k=k, ef=ef, num_threads=threads)
        else:
            result = query(index, prepared, policy['curve_policies'][str(ef)],
                           num_threads=threads, return_diagnostics=diagnostic)
        return result, time.perf_counter() - begin

    stage('warmup', settings=len(jobs))
    for job in jobs:
        for _ in range(plan['warmups_per_setting']):
            call(job)
    snapshots, timings = {}, []
    rng = np.random.default_rng(42)
    for repeat in range(plan['timed_repeats']):
        stage('timing', repeat=repeat + 1)
        for position, job_id in enumerate(rng.permutation(len(jobs))):
            job = jobs[int(job_id)]
            result, seconds = call(job)
            if job in snapshots:
                for actual, expected in zip(result, snapshots[job]):
                    np.testing.assert_array_equal(actual, expected)
            snapshots[job] = result
            timings.append(dict(dataset=entry['dataset'], backend='rabitq', k=k, method=job[0],
                ef_init=job[1], repeat=repeat + 1, position=position, seconds=seconds, query_count=n))
        csvwrite(folder / 'raw_timings.csv', timings)
    stage('diagnostics')
    rows = []
    for job in jobs:
        method, ef = job
        labels, distances = snapshots[job]
        require(((labels >= 0) & (labels < index.num_points)).all(), 'Invalid result labels')
        require((np.diff(np.sort(labels, axis=1), axis=1) > 0).all(), 'Duplicate result labels')
        hits = hits_per_query(labels, truth)
        diagnostic = {}
        if method == 'sage':
            result, _ = call(job, diagnostic=True)
            np.testing.assert_array_equal(labels, result[0])
            np.testing.assert_array_equal(distances, result[1])
            diagnostic = result[2]
            if k == 100:
                require(not np.any(diagnostic['shadow_routes']) and not np.any(diagnostic['shadow_updates']),
                        'k100 unexpectedly used shadow')
        seconds = [r['seconds'] for r in timings if (r['method'], r['ef_init']) == job]
        mean = statistics.mean(seconds)
        rows.append(dict(dataset=entry['dataset'], backend='rabitq', k=k, method=method, ef_init=ef,
            recommended=ef == policy['ef_init'], recommended_ef=policy['ef_init'], query_count=n,
            hits=int(hits.sum()), recall=float(hits.sum() / (n * k)), batch_mean_seconds=mean,
            batch_stdev_seconds=statistics.stdev(seconds), qps=n / mean, amortized_query_ms=1000 * mean / n,
            repeats=len(seconds), threads=threads, shadow_enabled=method == 'sage' and
            policy['curve_policies'][str(ef)]['shadow'] is not None,
            shadow_route_fraction=float(np.mean(diagnostic['shadow_routes'])) if diagnostic else ''))
        np.savez_compressed(folder / f'{method}_ef{ef}.npz', labels=labels, distances=distances,
                            hits=hits, **diagnostic)
    csvwrite(folder / 'summary.csv', rows)
    save(folder / 'complete.json', dict(passed=True, rows=len(rows), offline=offline))
    stage('complete')


def wait_quiet(plan, output):
    cpus = plan['timing']['cpus']
    watched = siblings(cpus)
    quiet = 0
    while quiet < 30:
        before = cpu_ticks()
        time.sleep(10)
        load = utilization(before, cpu_ticks(), watched, len(cpus))
        idle = load['physical_capacity_busy_percent'] < 5 and load['busiest_logical_cpu_percent'] < 80
        quiet = quiet + 10 if idle else 0
        save(output / 'status.json', dict(stage='waiting_for_quiet_cpus', stable_seconds=quiet, cpu_load=load))
        print(f'Waiting for idle CPUs: stable={quiet}s, busy={load["physical_capacity_busy_percent"]:.1f}%', flush=True)


def run(args, plan):
    require(plan.get('execution_authorized') is True, 'User must approve this RaBitQ plan before execution')
    output = Path(plan['output'])
    frozen = fingerprint(args.plan, plan)
    if args.cell:
        require(read(output / 'manifest.json')['signature'] == frozen, 'Sources or inputs changed')
        k_text, name = args.cell.split('/', 1)
        k = int(k_text.removeprefix('k'))
        require(name in plan['datasets_by_k'][str(k)], 'Cell outside approved scope')
        entry = next(e for e in plan['entries'] if e['dataset'] == name)
        cell_run(plan, entry, k, output / args.cell)
        require(fingerprint(args.plan, plan) == frozen, 'Sources or inputs changed during measurement')
        return
    report = preflight(plan)
    require(not output.exists(), f'Use a new output directory: {output}')
    lock_path = RUN_ROOT / plan['timing']['lock_file']
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    save(output / 'manifest.json', dict(signature=frozen, plan=plan, preflight=report, result_reuse=False))
    completed = []
    cell = None
    try:
        with lock_path.open('a') as lock:
            save(output / 'status.json', dict(stage='waiting_for_timing_lock'))
            fcntl.flock(lock, fcntl.LOCK_EX)
            for k, names in plan['datasets_by_k'].items():
                for name in names:
                    cell = f'k{k}/{name}'
                    wait_quiet(plan, output)
                    require(fingerprint(args.plan, plan) == frozen, 'Sources or inputs changed while queued')
                    save(output / 'status.json', dict(stage='running', cell=cell, completed=completed))
                    argv = ['numactl', '--physcpubind=' + ','.join(map(str, plan['timing']['cpus'])),
                            '--membind=' + str(plan['timing']['numa_node']), 'bash',
                            str(SAGE_ROOT / 'scripts/python.sh'), str(Path(__file__).resolve()),
                            '--plan', str(args.plan), '--cell', cell]
                    with (output / 'logs' / f'k{k}_{name}.log').open('x') as log:
                        subprocess.run(argv, env=rabitq_environment(plan), stdout=log, stderr=subprocess.STDOUT,
                                       check=True, pass_fds=(lock.fileno(),))
                    completed.append(cell)
                    for filename in ('summary.csv', 'offline_times.csv'):
                        rows = []
                        for done in completed:
                            with (output / done / filename).open() as stream:
                                rows.extend(csv.DictReader(stream))
                        csvwrite(output / filename, rows)
            require(fingerprint(args.plan, plan) == frozen, 'Sources or inputs changed during run')
            save(output / 'complete.json', dict(passed=True, completed=completed))
            save(output / 'status.json', dict(stage='complete', completed=completed))
    except BaseException as error:
        save(output / 'status.json', dict(stage='failed', cell=cell, completed=completed, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--cell', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    plan = read(args.plan)
    if args.preflight:
        report = preflight(plan)
        report['signature'] = fingerprint(args.plan, plan)
        save(args.plan.parent / 'preflight.json', report)
        print(f"Preflight passed: {report['cells']} cells; no real calibration or measurement started")
    else:
        run(args, plan)


if __name__ == '__main__':
    main()
