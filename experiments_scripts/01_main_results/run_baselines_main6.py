#!/usr/bin/env python3
"""Run fresh FAISS PiP/ABS tuning and separate final measurements for main6."""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tune_baselines import ef_grid, same_result, tune_abs, tune_pip
from main6_common import (RUN_ROOT, effective_config, resolve_datasets, hits_per_query,
                         validate_timing_config, with_numa_binding)

ROOT = Path(__file__).resolve().parent
SAGE_ROOT = ROOT.parents[1]
PACKAGES = {
    'pip': SAGE_ROOT / 'baselines/pip/build/paper095/faiss/python',
    'abs': SAGE_ROOT / 'baselines/abs/build/faiss/avx512/faiss/python',
}
SAGE_BINARY = SAGE_ROOT / 'faiss/build_sage_avx512/faiss/python/faiss/_swigfaiss_avx512.so'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def csvwrite(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_targets(args, config, datasets):
    """Reject unrelated/stale targets before creating any baseline output."""
    require(args.targets.resolve() == (args.output.parent / 'faiss/targets.json').resolve(),
            'Targets must come from faiss/targets.json in this same experiment directory')
    require(args.targets.is_file(), 'Run the new FAISS experiment first: missing faiss/targets.json')
    document = read(args.targets)
    require(document.get('schema_version') == 1 and document.get('complete') is True,
            'Expected a complete version-1 FAISS target manifest')
    require(document.get('config_sha256') == sha(args.config), 'Targets use a different experiment config')
    require(document.get('datasets_config_sha256') == sha(args.datasets_config),
            'Targets use a different dataset/index config')
    require(document.get('faiss_binary_sha256') == sha(SAGE_BINARY), 'Targets use a different SAGE binary')
    rows = document['datasets']
    require([r['dataset'] for r in rows] == config['datasets'], 'Missing or reordered FAISS targets')
    for ds, target in zip(datasets, rows):
        name = ds['dataset']
        require((target['query_count'], target['k']) == (ds['query_count'], config['k']),
                f'{name}: target query count/k differs')
        require(type(target['target_hits']) is int and 0 <= target['target_hits'] <= ds['query_count'] * config['k'],
                f'{name}: target must contain an exact integer hit count')
        require(type(target['recommended_ef']) is int and target['recommended_ef'] >= config['calibration']['offline_min_ef'],
                f'{name}: invalid recommended ef')
        require(type(target['offline_cap']) is int and target['offline_cap'] >= target['recommended_ef'],
                f'{name}: invalid offline cap')
        expected = dict(ds['faiss_index'], path=str(Path(ds['faiss_index']['path']).resolve()))
        require(target['faiss_index'] == expected == identity(expected['path']), f'{name}: graph changed')
        labels = (args.targets.parent / target['labels_file']).resolve()
        require(labels.is_relative_to(args.targets.parent.resolve()) and labels.is_file(),
                f'{name}: missing result labels inside the new FAISS output directory')
        for field in ('query_sha256', 'truth_sha256', 'labels_sha256'):
            require(isinstance(target[field], str) and len(target[field]) == 64, f'{name}: missing {field}')
    return {row['dataset']: row for row in rows}


def signature(args, datasets):
    binary = PACKAGES[args.method] / 'faiss/_swigfaiss_avx512.so'
    sources = [Path(__file__), ROOT / 'tune_baselines.py',
               SAGE_ROOT / f'baselines/{args.method}/scripts/{args.method}_runtime.py']
    if args.method == 'abs':
        sources += [SAGE_ROOT / 'baselines/abs/scripts/unit_or_zero_runtime.py',
                    SAGE_ROOT / 'baselines/abs/scripts/native_abs.cpp', SAGE_ROOT / 'baselines/abs/build/libpaper_abs.so']
    sources.append(ROOT / 'main6_common.py')
    return dict(k=args.k, config_sha256=sha(args.config), datasets_config_sha256=sha(args.datasets_config),
                targets_sha256=sha(args.targets), binary_sha256=sha(binary),
                sources_sha256={str(p): sha(p) for p in sources},
                data_files={d['dataset']: identity(d['dataset_path']) for d in datasets})


def environment(config):
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'KMP_AFFINITY', 'GOMP_CPU_AFFINITY', 'KMP_HW_SUBSET',
                'ABS_BUILD_DIR', 'ABS_FAISS_PYTHON_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS=str(config['online_threads']), OMP_DYNAMIC='FALSE',
               OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', FAISS_OPT_LEVEL='AVX512')
    cpus = config['timing'].get('cpus')
    env['OMP_PROC_BIND'] = 'close' if cpus else 'FALSE'
    if cpus:
        env['OMP_PLACES'] = ','.join('{%d}' % cpu for cpu in cpus)
    else:
        env.pop('OMP_PLACES', None)
    return env


class Experiment:
    def __init__(self, args, config, ds, target):
        import h5py
        import numpy as np
        self.np, self.config, self.ds, self.target = np, config, ds, target
        self.method, self.k = args.method, config['k']
        self.cpus = config['timing'].get('cpus')
        if self.cpus:
            os.sched_setaffinity(0, self.cpus)
        sys.path.insert(0, str(PACKAGES[self.method]))
        import faiss
        require(Path(faiss.__file__).resolve().parent == PACKAGES[self.method] / 'faiss', 'Wrong FAISS package')
        self.faiss = faiss
        # When binding, initialize both OpenMP runtimes against the full CPU mask.
        if self.method == 'abs':
            if self.cpus:
                os.sched_setaffinity(0, self.cpus)
            ctypes.CDLL(str(SAGE_ROOT / 'baselines/abs/build/libpaper_abs.so'))
        if self.cpus:
            for library in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                   if any(token in line for token in ('libgomp', 'libiomp', 'libomp'))}):
                os.sched_setaffinity(0, self.cpus)
                runtime = ctypes.CDLL(library)
                require(runtime.omp_get_num_places() == len(self.cpus), f'Wrong OpenMP placement: {library}')
            os.sched_setaffinity(0, self.cpus)
        faiss.omp_set_num_threads(config['online_threads'])
        faiss.InterruptCallback.clear_instance()
        self.index = faiss.read_index(ds['faiss_index']['path'])
        require((self.index.ntotal, self.index.d, self.index.hnsw.nb_neighbors(1), self.index.hnsw.efConstruction)
                == (ds['train_count'], ds['dimension'], ds['M'], ds['efConstruction']), 'Wrong graph structure')
        metric = faiss.METRIC_L2 if ds['space'] == 'l2' else faiss.METRIC_INNER_PRODUCT
        require(ds['space'] in ('cosine', 'l2') and self.index.metric_type == metric, 'Unsupported/mismatched metric')
        with h5py.File(ds['dataset_path']) as file:
            self.raw_queries = np.asarray(file['test'], dtype=np.float32, order='C')
            self.truth = np.asarray(file['neighbors'][:, :self.k], dtype=np.int64, order='C')
            ids = np.unique(np.linspace(0, ds['train_count'] - 1, 33, dtype=np.int64))
            vectors = np.asarray(file['train'][ids], dtype=np.float32, order='C')
        if ds['space'] == 'cosine':
            faiss.normalize_L2(vectors)
        np.testing.assert_allclose(self.index.reconstruct_batch(ids), vectors, rtol=1e-5, atol=1e-6)
        require(self.raw_queries.shape == (ds['query_count'], ds['dimension']), 'Query shape differs')
        require(self.truth.shape == (ds['query_count'], self.k), 'Ground truth shape differs')
        require(((self.truth >= 0) & (self.truth < self.index.ntotal)).all(), 'Invalid ground truth IDs')
        require((np.diff(np.sort(self.truth, axis=1), axis=1) > 0).all(), 'Duplicate ground truth IDs')
        require(np.isfinite(self.raw_queries).all(), 'Non-finite query vectors')
        require(array_sha(self.raw_queries) == target['query_sha256'], 'Raw float32 query hash differs')
        require(array_sha(self.truth) == target['truth_sha256'], 'Ground truth hash differs')
        with np.load(args.targets.parent / target['labels_file'], allow_pickle=False) as data:
            labels = data['labels']
            require(labels.dtype == np.int64 and array_sha(labels) == target['labels_sha256'], 'SAGE labels hash differs')
            require(self.hits(labels) == target['target_hits'], 'SAGE target hits do not match saved labels')
        self.abs = None
        if self.method == 'abs':
            sys.path.insert(0, str(SAGE_ROOT / 'baselines/abs/scripts'))
            from unit_or_zero_runtime import UnitOrZeroIPABSSearch
            if self.cpus:
                os.sched_setaffinity(0, self.cpus)
            self.abs = UnitOrZeroIPABSSearch(self.index, threads=config['online_threads'],
                                           unit_normalized=ds['space'] == 'cosine')
        self.parameters = {}

    def hits(self, labels):
        np = self.np
        require(labels.shape == self.truth.shape, 'Result shape differs')
        require(((labels >= -1) & (labels < self.index.ntotal)).all(), 'Invalid result IDs')
        ordered = np.sort(labels, axis=1)
        require(((np.diff(ordered, axis=1) > 0) | (ordered[:, 1:] == -1)).all(), 'Duplicate result IDs')
        return int(hits_per_query(labels, self.truth).sum())

    def call(self, method, settings):
        np, faiss = self.np, self.faiss
        key = (method, json.dumps(settings, sort_keys=True))
        if method != 'abs' and key not in self.parameters:
            p = faiss.SearchParametersHNSWPiP() if method == 'pip' else faiss.SearchParametersHNSW()
            p.efSearch, p.bounded_queue = settings['ef_search'], True
            if method == 'pip':
                p.enable_pip = settings.get('enable', True)
                p.patience = settings['patience']
                p.saturation_threshold = settings['threshold']
            self.parameters[key] = p
        faiss.cvar.hnsw_stats.reset()
        if self.cpus:
            os.sched_setaffinity(0, {self.cpus[0]})
        try:
            # Whole batch API scope: query preparation and output allocation included.
            start = time.perf_counter()
            queries = np.ascontiguousarray(self.raw_queries, dtype=np.float32)
            if self.ds['space'] == 'cosine':
                queries = queries.copy()
                faiss.normalize_L2(queries)
            if method == 'abs':
                distances, labels, counters = self.abs.search(queries, self.k, gamma=settings['gamma'])
            else:
                distances, labels = self.index.search(queries, self.k, params=self.parameters[key])
            seconds = time.perf_counter() - start
        finally:
            if self.cpus:
                os.sched_setaffinity(0, self.cpus)
        self.labels, self.distances = labels, distances
        ndis = int(counters['ndis'].sum()) - len(queries) if method == 'abs' else int(faiss.cvar.hnsw_stats.ndis)
        return dict(method=method, config=settings, seconds=seconds, hits=self.hits(labels), ndis=ndis,
                    labels_sha256=array_sha(labels), distances_sha256=array_sha(distances))

    def placement(self):
        masks = [sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()]
        if self.cpus:
            require(all(set(mask).issubset(self.cpus) for mask in masks), 'Threads escaped requested CPU mask')
            require(set(self.cpus[1:]).issubset({m[0] for m in masks if len(m) == 1}), 'Workers not pinned separately')
        return dict(thread_masks=masks, numa_maps=Path('/proc/self/numa_maps').read_text())

    def close(self):
        if self.abs is not None:
            self.abs.close()


def phase(args, config, ds, target):
    folder = args.output / ('tuning' if args.phase == 'tune' else 'online') / ds['dataset']
    folder.mkdir(parents=True, exist_ok=False)
    exp = Experiment(args, config, ds, target)
    try:
        if args.phase == 'tune':
            tuning_started = time.perf_counter()
            def record(row):
                with (folder / 'evaluations.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            if args.method == 'abs':
                result = tune_abs(exp.call, target['target_hits'], config['baseline_tuning']['abs'], record)
            else:
                vanilla = exp.call('vanilla', dict(ef_search=target['recommended_ef']))
                disabled = exp.call('pip', dict(ef_search=target['recommended_ef'], patience=0,
                                                threshold=config['baseline_tuning']['pip']['threshold'], enable=False))
                same_result(disabled, vanilla)
                save(folder / 'disabled_parity.json', dict(passed=True, vanilla=vanilla, disabled=disabled))
                result = tune_pip(exp.call, target['target_hits'], ef_grid(config, target), exp.index.ntotal,
                                  config['baseline_tuning']['pip'], record,
                                  config['warmups_per_setting'], config['timed_repeats'])
            selected = result['selected']
            require(selected['hits'] >= target['target_hits'], 'Selection failed to match target')
            same_result(exp.call(args.method, selected['config']), selected)
            np = exp.np
            np.savez_compressed(folder / 'selected.npz', labels=exp.labels, distances=exp.distances)
            save(folder / 'selection.json', dict(**result, target_hits=target['target_hits'],
                 recommended_ef=target['recommended_ef'], matched_target=True,
                 tuning_seconds=time.perf_counter() - tuning_started,
                 retrospective_tuning=True, global_optimum_claimed=False, pid=os.getpid()))
        else:
            selection_path = args.output / 'tuning' / ds['dataset'] / 'selection.json'
            selection = read(selection_path)
            require(selection['pid'] != os.getpid(), 'Final timing requires a fresh process')
            selected, rows = selection['selected'], []
            warmups, repeats = config['warmups_per_setting'], config['timed_repeats']
            for repeat in range(warmups + repeats):
                result = exp.call(args.method, selected['config'])
                same_result(result, selected)
                rows.append(dict(dataset=ds['dataset'], method=args.method, repeat=repeat,
                    phase='warmup' if repeat < warmups else 'timed',
                    seconds=result['seconds'], batch_ms=result['seconds'] * 1000,
                    hits=result['hits'], recall=result['hits'] / exp.truth.size,
                    target_hits=target['target_hits'], target_delta_hits=result['hits'] - target['target_hits'],
                    recommended_ef=target['recommended_ef'], query_count=len(exp.raw_queries), k=exp.k,
                    threads=config['online_threads'], config=json.dumps(selected['config'], sort_keys=True),
                    labels_sha256=result['labels_sha256'], distances_sha256=result['distances_sha256'], ndis=result['ndis']))
                csvwrite(folder / 'raw_timings.csv', rows)
            samples = [r['seconds'] for r in rows if r['phase'] == 'timed']
            summary = {k: v for k, v in rows[-1].items() if k not in ('repeat', 'phase', 'seconds', 'batch_ms')}
            summary.update(batch_mean_seconds=statistics.mean(samples),
                           batch_stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0,
                           repeats=repeats, **selected['config'])
            csvwrite(folder / 'summary.csv', [summary])
            exp.np.savez_compressed(folder / 'result.npz', labels=exp.labels, distances=exp.distances)
            save(folder / 'measurement.json', dict(selection_sha256=sha(selection_path),
                                                  fresh_process=True, pid=os.getpid(), timing=config['timing']))
        save(folder / 'complete.json', dict(passed=True, placement=exp.placement()))
    finally:
        exp.close()


def run(args, config, datasets, targets):
    validate_timing_config(config)
    require(config['timing']['include_query_preprocessing'] and config['timing']['include_output_allocation'],
            'This runner measures whole batch API time including preprocessing and allocation')
    frozen = signature(args, datasets)
    if args.phase:
        require(read(args.output / 'manifest.json')['signature'] == frozen, 'Inputs/code changed during this run')
        ds = next(d for d in datasets if d['dataset'] == args.dataset)
        phase(args, config, ds, targets[ds['dataset']])
        return
    require(not args.output.exists(), f'Use an unused output directory: {args.output}')
    lock_path = RUN_ROOT / config['timing']['lock_file']
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another main experiment owns the timing cores; retry after it finishes') from error
        args.output.mkdir(parents=True)
        save(args.output / 'manifest.json', dict(signature=frozen, method=args.method,
             config=config, targets=read(args.targets), result_reuse=False, started_at=time.time()))
        (args.output / 'logs').mkdir()
        summaries, timings = [], []
        try:
            for ds in datasets:
                for stage in ('tune', 'measure'):
                    name = ds['dataset']
                    print(f'[{args.method}] {name}: {stage}', flush=True)
                    argv = [sys.executable, str(Path(__file__).resolve()), '--method', args.method,
                            '--config', str(args.config), '--datasets-config', str(args.datasets_config),
                            '--output', str(args.output), '--targets', str(args.targets),
                            '--phase', stage, '--dataset', name, '--k', str(config['k'])]
                    argv = with_numa_binding(argv, config)
                    with (args.output / 'logs' / f'{name}_{stage}.log').open('w') as log:
                        subprocess.run(argv, env=environment(config), stdout=log, stderr=subprocess.STDOUT, check=True)
                for filename, rows in [('summary.csv', summaries), ('raw_timings.csv', timings)]:
                    with (args.output / 'online' / name / filename).open() as stream:
                        rows.extend(csv.DictReader(stream))
                    csvwrite(args.output / filename, rows)
            save(args.output / 'complete.json', dict(passed=True, datasets=list(targets), finished_at=time.time()))
        except Exception as error:
            save(args.output / 'failed.json', dict(error=str(error), dataset=name, phase=stage))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['abs', 'pip'], required=True)
    for flag in ('config', 'datasets-config', 'output', 'targets'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--phase', choices=['tune', 'measure'], help=argparse.SUPPRESS)
    parser.add_argument('--dataset', help=argparse.SUPPRESS)
    parser.add_argument('--k', type=int, choices=[10, 100])
    args = parser.parse_args()
    for key in ('config', 'datasets_config', 'output', 'targets'):
        setattr(args, key, getattr(args, key).resolve())
    require(bool(args.phase) == bool(args.dataset), 'Internal phases require --dataset')
    config = effective_config(args.config, args.k)
    if args.method == 'pip':
        # Preserve the original config/target SHA and all SAGE settings. Only
        # this PiP process uses the selected paper formula and positive Delta.
        config['baseline_tuning']['pip'].update(threshold=0.95, minimum_patience=1)
        config['pip_implementation'] = 'paper top-k set intersection; streak >= positive Delta'
    args.k = config['k']
    datasets = resolve_datasets(args.datasets_config, config)
    require([d['dataset'] for d in datasets] == config['datasets'], 'Dataset configuration differs')
    targets = validate_targets(args, config, datasets)
    run(args, config, datasets, targets)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f'Main baseline run stopped: {error}', file=sys.stderr)
        raise SystemExit(2)
