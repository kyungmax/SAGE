"""RaBitQ cell worker used by the numbered robustness experiment."""
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'experiments_scripts/01_main_results'))
from main6_common import SAGE_ROOT, hits_per_query
from run_baselines_main6 import save, sha, array_sha, csvwrite, require
from tune_baselines import ef_grid

sys.path.insert(0, str(SAGE_ROOT))


def require_search_affinity(cpus):
    actual = sorted(os.sched_getaffinity(0))
    require(actual == sorted(cpus), f'RaBitQ caller affinity narrowed: {actual}; expected {sorted(cpus)}')


def dataset(entry):
    from experiments_scripts.rabitq.runner import Dataset
    return Dataset(entry['data_path'], raw=entry['raw'], metric=entry['metric'],
                   normalize=entry['normalize'])


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
