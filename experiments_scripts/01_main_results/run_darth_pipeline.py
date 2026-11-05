#!/usr/bin/env python3
"""Prepare, train, validate and measure DARTH against a fresh FAISS SAGE target."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from main6_common import ROOT, SAGE_ROOT, resolve_datasets
from run_baselines_main6 import identity, save, sha

STAGES = ('prepare', 'train', 'validation', 'test')
STAGE_DIRS = dict(prepare='data', train='training', validation='evaluation/validation',
                  test='evaluation/test')


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    return dict(**identity(path), sha256=sha(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_inputs(plan):
    for record in plan['inputs'].values():
        require(identity(record['path']) == record, f'Input changed: {record["path"]}')
    for record in plan['sources'].values():
        require(fingerprint(record['path']) == record, f'Source changed: {record["path"]}')


def make_plan(args):
    import h5py
    import numpy as np

    require(min(args.threads, args.learn_queries, args.validation_queries, args.batch_rows) > 0,
            'Counts and threads must be positive.')
    require(math.isfinite(args.trace_timeout_seconds) and args.trace_timeout_seconds > 0,
            'Trace timeout must be positive and finite.')
    require(args.seed >= 0, 'Seed must be nonnegative.')
    require(args.threads <= len(os.sched_getaffinity(0)), 'Threads exceed available CPU affinity.')
    row = resolve_datasets(args.catalog, dict(datasets=[args.dataset]))[0]
    require(row['space'] in ('l2', 'cosine'), 'DARTH pipeline supports l2 and cosine only.')
    targets = read(args.targets)
    require(targets.get('complete') is True and targets.get('schema_version') == 1,
            'Run the FAISS main cell first: a complete version-1 target manifest is required.')
    matches = [r for r in targets['datasets'] if r['dataset'] == args.dataset]
    require(len(matches) == 1, 'Target manifest must contain this dataset exactly once.')
    target = matches[0]
    require(target['k'] == 10 and targets['k'] == 10, 'The DARTH training protocol is k10 only.')
    full_index = identity(row['faiss_index']['path'])
    require(full_index == target['faiss_index'], 'Catalog and FAISS target must use the same full graph.')
    require(isinstance(target['recommended_ef'], int) and target['recommended_ef'] >= 10,
            'Invalid FAISS recommended ef.')
    require(isinstance(target['target_hits'], int)
            and 0 < target['target_hits'] <= target['query_count'] * 10, 'Invalid FAISS target hits.')
    inputs = dict(dataset=identity(row['dataset_path']), full_index=full_index)
    with h5py.File(row['dataset_path'], 'r') as data:
        base, queries, truth = (data[name] for name in ('train', 'test', 'neighbors'))
        require(base.ndim == queries.ndim == truth.ndim == 2, 'Expected matrix-shaped HDF5 inputs.')
        require(base.shape[1] == queries.shape[1] and queries.shape[0] == truth.shape[0]
                and truth.shape[1] >= 10, 'Dataset train/test/ground-truth shapes disagree.')
        require(queries.shape[0] == target['query_count'], 'FAISS target query count differs.')
        require(base.shape[0] > args.learn_queries + args.validation_queries + 10,
                'Too few vectors for the requested holdouts and k.')
        for key, actual in (('dimension', base.shape[1]), ('train_count', base.shape[0]),
                            ('query_count', queries.shape[0])):
            require(row.get(key, actual) == actual, f'Catalog {key} differs from the HDF5 input.')
        query_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
        for begin in range(0, len(queries), 1024):
            vectors = np.asarray(queries[begin:begin + 1024], dtype=np.float32, order='C')
            ids = np.asarray(truth[begin:begin + 1024, :10], dtype=np.int64, order='C')
            require(np.isfinite(vectors).all(), 'Nonfinite online queries.')
            require((ids >= 0).all() and (ids < len(base)).all()
                    and (np.diff(np.sort(ids, axis=1), axis=1) > 0).all(), 'Invalid online GT IDs.')
            query_digest.update(vectors.tobytes())
            truth_digest.update(ids.tobytes())
        require(query_digest.hexdigest() == target['query_sha256']
                and truth_digest.hexdigest() == target['truth_sha256'],
                'Online queries or GT differ from the FAISS target.')
        for name in ('train', 'test', 'neighbors'):
            source = data[name]
            inputs[f'{name}_hdf5'] = identity(source.file.filename)
            for number, (filename, _, _) in enumerate(source.external or []):
                path = Path(filename)
                if not path.is_absolute():
                    path = Path(source.file.filename).parent / path
                inputs[f'{name}_external_{number}'] = identity(path)
        shape = dict(train_count=base.shape[0], query_count=queries.shape[0], dimension=base.shape[1])
    for key in ('M', 'efConstruction'):
        require(isinstance(row[key], int) and row[key] > 0, f'Invalid graph parameter: {key}')
    for binary in (args.binary, args.trace_binary):
        require(binary.is_file() and os.access(binary, os.X_OK), f'Build DARTH first: {binary}')
    packages = {}
    for module, package in (('numpy', 'numpy'), ('pandas', 'pandas'), ('h5py', 'h5py'),
                            ('lightgbm', 'lightgbm'), ('sklearn', 'scikit-learn')):
        require(importlib.util.find_spec(module) is not None, f'Missing Python dependency: {package}')
        packages[package] = importlib.metadata.version(package)
    faiss_spec = importlib.util.find_spec('faiss')
    require(faiss_spec is not None and faiss_spec.origin
            and Path(faiss_spec.origin).resolve().is_relative_to(SAGE_ROOT / 'faiss'),
            'Use scripts/python.sh with the locally built SAGE FAISS extension.')
    source_paths = dict(catalog=args.catalog, targets=args.targets, binary=args.binary,
        trace_binary=args.trace_binary, runner=Path(__file__), measurement=ROOT / 'run_darth.py',
        preparation=ROOT / 'darth_preparation.py', training=ROOT / 'darth_training.py',
        common=ROOT / 'main6_common.py', io_helpers=ROOT / 'run_baselines_main6.py')
    for name, binary in (('online', args.binary), ('trace', args.trace_binary)):
        linked = subprocess.run(['ldd', str(binary.resolve())], check=True, capture_output=True, text=True).stdout
        require('not found' not in linked, f'Unresolved DARTH runtime library: {binary}')
        for line in linked.splitlines():
            if '=>' in line and ('libfaiss' in line or 'lib_lightgbm' in line):
                library = Path(line.split('=>', 1)[1].strip().split(' (', 1)[0])
                source_paths[f'{name}_{library.name}'] = library
    faiss_extensions = sorted(Path(faiss_spec.origin).parent.glob('_swigfaiss*.so'))
    require(faiss_extensions, 'Local FAISS native extension is missing.')
    for path in faiss_extensions:
        source_paths[path.name] = path
    output = (args.output or SAGE_ROOT / 'runs/01_main_results/darth' / args.dataset).resolve()
    return dict(schema_version=1, dataset=args.dataset, output=str(output),
        paper_holdout_counts=(args.learn_queries, args.validation_queries, args.seed) == (10000, 1000, 12345),
        inputs=inputs, sources={key: fingerprint(path) for key, path in source_paths.items()},
        target=target, shape=shape, python=dict(executable=sys.executable, version=sys.version, packages=packages),
        protocol=dict(k=10, ef=target['recommended_ef'], target_recall=target['target_hits'] / (target['query_count'] * 10),
            space=row['space'], M=row['M'], ef_construction=row['efConstruction'],
            learn_count=args.learn_queries, validation_count=args.validation_queries, seed=args.seed,
            batch_rows=args.batch_rows, threads=args.threads, logging_interval=2,
            warmups=1, measured_runs=5, trace_timeout_seconds=args.trace_timeout_seconds,
            intervals='Derived from learning trace only; frozen before validation and test'),
        affinity=dict(cpus=sorted(os.sched_getaffinity(0)),
                      memory_policy='Inherited; use the main experiment NUMA placement'))


def worker(manifest, stage):
    plan = read(manifest)
    check_inputs(plan)
    output, config = Path(plan['output']), plan['protocol']
    if stage == 'prepare':
        from darth_preparation import prepare_inputs
        prepared = prepare_inputs(plan['inputs']['dataset']['path'], plan['inputs']['full_index']['path'],
            output / 'data', **{key: config[key] for key in ('k', 'ef', 'space', 'M', 'ef_construction',
                'learn_count', 'validation_count', 'seed', 'threads', 'batch_rows')})
        for key in ('query_sha256', 'truth_sha256'):
            require(prepared[key] == plan['target'][key], 'Prepared online inputs differ from the FAISS target.')
    elif stage == 'train':
        from darth_training import train
        prepared = read(output / 'data/prepared.json')
        train(dataset=plan['dataset'], trace_binary=plan['sources']['trace_binary']['path'],
            index=prepared['auxiliary_index'], data_dir=output / 'data', output=output / 'training',
            ef=config['ef'], target=config['target_recall'], queries=config['learn_count'],
            threads=config['threads'], logging_interval=config['logging_interval'],
            trace_timeout_seconds=config['trace_timeout_seconds'])
    else:
        raise ValueError(f'Unknown worker stage: {stage}')


def evaluation_command(plan, stage):
    output, config = Path(plan['output']), plan['protocol']
    data, model = read(output / 'data/prepared.json'), read(output / 'training/training.json')
    validation = stage == 'validation'
    split = data['validation' if validation else 'evaluation']
    values = dict(dataset=plan['dataset'], binary=plan['sources']['binary']['path'],
        index=data['auxiliary_index'] if validation else data['full_index_path'],
        query_file=split['query_file'], gt_file=split['gt_file'], gt_dist_file=split['gt_dist_file'],
        model=model['model_path'], queries=split['query_count'], k=config['k'], ef=config['ef'],
        threads=config['threads'], normalize=int(config['space'] == 'cosine'),
        initial_interval=model['initial_interval'], min_interval=model['min_interval'],
        warmup_runs=config['warmups'], measured_runs=config['measured_runs'], output=output / STAGE_DIRS[stage])
    if validation:
        values['target_recall'] = config['target_recall']
    else:
        values['targets'] = plan['sources']['targets']['path']
    command = [sys.executable, str(ROOT / 'run_darth.py')]
    for name, value in values.items():
        command.extend(('--' + name.replace('_', '-'), str(value)))
    return command


def run_command(command, log_path, threads):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('OMP_PLACES', None)
    with log_path.open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            result = process.wait()
            require(result == 0, f'Stage exited with status {result}; see {log_path}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def stage_artifacts(output, stage):
    directory = output / STAGE_DIRS[stage]
    cached = read(directory / 'prepared.json')['artifacts'] if stage == 'prepare' else {}
    artifacts = []
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            record = identity(path)
            previous = cached.get(str(path))
            digest = previous['sha256'] if previous and all(previous[k] == v for k, v in record.items()) else sha(path)
            artifacts.append(dict(**record, sha256=digest))
    require(artifacts, f'No artifacts from completed stage {stage}')
    return artifacts


def execute(plan, args):
    output = Path(plan['output'])
    lock_path = SAGE_ROOT / 'runs/timing.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another experiment holds {lock_path}') from None
        if args.resume:
            require((output / 'pipeline.json').is_file(), '--resume requires an existing pipeline.json.')
            require(read(output / 'pipeline.json') == plan,
                    'Resume settings, inputs, source, Python environment or CPU affinity changed; use a new output.')
        else:
            output.mkdir(parents=True, exist_ok=False)
            (output / 'stages').mkdir()
            save(output / 'pipeline.json', plan)
        for stage in STAGES[:STAGES.index(args.stop_after) + 1]:
            check_inputs(plan)
            receipt = output / 'stages' / f'{stage}.json'
            if receipt.exists():
                completed = read(receipt)
                require(completed.get('complete') is True, f'Incomplete receipt: {receipt}')
                for record in completed['artifacts']:
                    require(fingerprint(record['path']) == record, f'Completed artifact changed: {record["path"]}')
                print(f'{stage}: verified completed artifacts', flush=True)
                continue
            require(not (output / STAGE_DIRS[stage]).exists(),
                    f'Partial {stage} output exists without a completion receipt; use a new output directory.')
            log = output / 'stages' / f'{stage}.log'
            require(not log.exists(), f'Prior {stage} attempt exists at {log}; use a new output directory.')
            command = ([sys.executable, str(Path(__file__).resolve()), '_worker', str(output / 'pipeline.json'), stage]
                       if stage in ('prepare', 'train') else evaluation_command(plan, stage))
            print(f'{stage}: running (log: {log})', flush=True)
            started = time.monotonic()
            try:
                run_command(command, log, plan['protocol']['threads'])
                check_inputs(plan)
                artifacts = stage_artifacts(output, stage)
                save(receipt, dict(complete=True, stage=stage, seconds=time.monotonic() - started,
                                   command=command, artifacts=artifacts))
            except BaseException as error:
                save(output / 'failure.json', dict(stage=stage, type=type(error).__name__, message=str(error)))
                raise
        if args.stop_after == 'test':
            source_summary = output / 'evaluation/test/summary.csv'
            summary = output / 'summary.csv'
            if summary.exists():
                require(sha(summary) == sha(source_summary), 'Final summary differs from completed test output.')
            else:
                shutil.copyfile(source_summary, summary)
            if not (output / 'complete.json').exists():
                save(output / 'complete.json', dict(complete=True, dataset=plan['dataset'],
                    paper_holdout_counts=plan['paper_holdout_counts'],
                    validation=read(output / 'evaluation/validation/complete.json'),
                    test=read(output / 'evaluation/test/complete.json'), summary=fingerprint(summary),
                    model_sha256=sha(output / 'training/model.txt')))
            print(f'Completed: {summary}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--targets', type=Path, required=True, help='Completed k10 FAISS targets.json')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'config/main6_online24/datasets.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--binary', type=Path, default=SAGE_ROOT / 'baselines/darth_faiss1141/build/benchmark_new')
    parser.add_argument('--trace-binary', type=Path, default=SAGE_ROOT / 'baselines/darth_faiss1141/build/train_darth_corrected')
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--learn-queries', type=int, default=10000)
    parser.add_argument('--validation-queries', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--batch-rows', type=int, default=25000)
    parser.add_argument('--trace-timeout-seconds', type=float, default=86400)
    parser.add_argument('--stop-after', choices=STAGES, default='test')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='Check inputs and print protocol; no output or ANN run')
    args = parser.parse_args()
    plan = make_plan(args)
    if args.preflight:
        print(json.dumps(dict(plan, no_ann_run=True), indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt('DARTH pipeline interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    execute(plan, args)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '_worker':
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
