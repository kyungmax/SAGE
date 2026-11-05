#!/usr/bin/env python3
"""Run the retained author-policy Ada-EF baseline on the catalog's HNSW graph."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

from main6_common import SAGE_ROOT, resolve_datasets

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def target_from_manifest(args, cfg):
    import h5py
    import numpy as np

    document = read(args.targets)
    if document.get('schema_version') != 1 or document.get('complete') is not True:
        raise ValueError('Expected a completed version-1 FAISS targets.json')
    if document.get('datasets_config_sha256') != sha(args.catalog):
        raise ValueError('Target manifest and dataset catalog differ')
    rows = [r for r in document['datasets'] if r['dataset'] == args.dataset]
    if len(rows) != 1:
        raise ValueError('Target manifest must contain exactly one record for this dataset')
    target = rows[0]
    if (target['query_count'], target['k']) != (cfg['query_count'], args.k):
        raise ValueError('Target query count/k differs from this run')
    if target['faiss_index'] != identity(cfg['faiss_index']['path']):
        raise ValueError('The FAISS graph used to produce the target has changed')
    label_path = (args.targets.parent / target['labels_file']).resolve()
    if not label_path.is_relative_to(args.targets.parent.resolve()):
        raise ValueError('Target label path must be inside the FAISS output directory')
    with h5py.File(cfg['dataset_path'], 'r') as handle:
        queries = np.asarray(handle['test'], dtype=np.float32, order='C')
        truth = np.asarray(handle['neighbors'][:, :args.k], dtype=np.int64, order='C')
    with np.load(label_path, allow_pickle=False) as handle:
        labels = np.ascontiguousarray(handle['labels'], dtype=np.int64)
    if labels.shape != truth.shape or truth.shape != (cfg['query_count'], args.k):
        raise ValueError('Target label or truth shape differs')
    for field, array in (('query_sha256', queries), ('truth_sha256', truth), ('labels_sha256', labels)):
        if hashlib.sha256(array.tobytes()).hexdigest() != target[field]:
            raise ValueError(f'Target hash mismatch: {field}')
    if (labels < 0).any() or (np.diff(np.sort(labels, axis=1), axis=1) <= 0).any():
        raise ValueError('Invalid target labels')
    count = int((labels[:, :, None] == truth[:, None, :]).any(2).sum())
    if type(target['target_hits']) is not int or count != target['target_hits']:
        raise ValueError('Target labels do not reproduce target_hits')
    return count / truth.size, count, dict(path=str(args.targets.resolve()), sha256=sha(args.targets), record=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--catalog', type=Path, default=HERE / 'config/main6_online24/datasets.json')
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target', type=float, help='Explicit requested recall, in [0,1]')
    target_group.add_argument('--targets', type=Path, help='Fresh completed FAISS targets.json')
    parser.add_argument('--k', type=int, choices=(10, 100), default=10)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measured-runs', type=int, default=5)
    parser.add_argument('--phase', choices=('offline', 'online', 'all'), default='all')
    parser.add_argument('--binary', type=Path, default=SAGE_ROOT / 'baselines/ada-ef/build/backend_runner')
    parser.add_argument('--artifacts-root', '--artifacts', type=Path,
                        help='Dedicated artifact directory for this dataset/k/target; defaults under output')
    parser.add_argument('--output-root', type=Path, default=SAGE_ROOT / 'runs/01_main_results/adaef')
    parser.add_argument('--preflight', action='store_true', help='Validate inputs and print commands without writes or ANN searches')
    args = parser.parse_args()
    if args.threads < 1 or args.warmup_runs < 0 or args.measured_runs < 1:
        parser.error('Invalid thread or repeat count')
    cfg = resolve_datasets(args.catalog, {'datasets': [args.dataset]})[0]
    if cfg['space'] not in ('cosine', 'ip'):
        parser.error('The retained Ada-EF implementation supports cosine/IP only; L2 is unsupported')
    if 'hnswlib_index' not in cfg:
        parser.error('Catalog must supply the shared hnswlib_index')
    for path in (args.binary, cfg['dataset_path'], cfg['hnswlib_index']['path']):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing input: {path}; build Ada-EF and prepare main data/indexes first')
    if not os.access(args.binary, os.X_OK):
        raise ValueError('Ada-EF backend_runner is not executable')
    if args.targets is not None:
        target, target_hits, source = target_from_manifest(args, cfg)
    else:
        target = args.target
        if not math.isfinite(target) or not 0 <= target <= 1:
            parser.error('target must be a finite recall in [0,1]')
        target_hits = math.ceil(target * cfg['query_count'] * args.k - 1e-12)
        source = dict(explicit_target=target)
    output = (args.output_root / args.dataset / f'k{args.k}' / f'target_hits_{target_hits}').resolve()
    artifacts = (args.artifacts_root or output / 'artifacts').resolve()
    m, efc = int(cfg.get('M', 32)), int(cfg.get('efConstruction', 500))
    graph = identity(cfg['hnswlib_index']['path'])
    receipt = dict(dataset=args.dataset, data=identity(cfg['dataset_path']), graph=graph,
        binary_sha256=sha(args.binary), metric='cd' if cfg['space'] == 'cosine' else 'ipd',
        k=args.k, M=m, efConstruction=efc, requested_recall=target, target_hits=target_hits,
        artifact_revision='author-v1-zero-recall-v1', statistics_length=1025,
        sampling_policy='upstream_sample_data', target_source=source)
    receipt_path = artifacts / 'run_inputs.json'
    if receipt_path.exists():
        if read(receipt_path) != receipt:
            raise ValueError('Existing Ada-EF artifacts use different inputs/target; choose a new --artifacts-root')
    elif artifacts.exists() and any(artifacts.iterdir()):
        raise ValueError('Nonempty artifact directory has no matching run_inputs.json; choose a new --artifacts-root')
    elif args.phase == 'online':
        raise FileNotFoundError('Run --phase offline (or all) first with these same inputs/target')
    graph_link = artifacts / 'index' / f'{args.dataset}-M{m}-efc-{efc}-parallel.hnsw'
    if graph_link.exists() or graph_link.is_symlink():
        if graph_link.resolve() != Path(graph['path']):
            raise ValueError('Ada-EF artifact graph link points to a different index')
    base = [str(args.binary.resolve()), '--dataset', args.dataset, '--data-path', cfg['dataset_path'],
        '--experiments-root', str(artifacts), '--m', str(m), '--ef-construction', str(efc),
        '--k', str(args.k), '--num-threads', str(args.threads), '--metric', receipt['metric'],
        '--expected-recall', format(target, '.17g')]
    commands = {}
    if args.phase in ('offline', 'all'):
        commands['offline'] = base + ['--phase', 'offline', '--output-json', str(output / 'offline.json')]
    if args.phase in ('online', 'all'):
        commands['online'] = base + ['--phase', 'online', '--parallel-queries', '--e2e-timing',
            '--warmup-runs', str(args.warmup_runs), '--measured-runs', str(args.measured_runs),
            '--output-json', str(output / 'online.json'), '--labels-output', str(output / 'labels.u64'),
            '--per-query-csv', str(output / 'per_query.csv')]
    phase_outputs = dict(offline=['offline.log', 'offline.json'],
        online=['online.log', 'online.json', 'labels.u64', 'labels.npz', 'per_query.csv',
                'summary.json', 'summary.csv'])
    for phase in commands:
        for name in phase_outputs[phase]:
            if (output / name).exists():
                raise FileExistsError(f'Refusing to overwrite {output / name}; choose a new --output-root')
    plan = dict(inputs=receipt, graph_link=str(graph_link), output=str(output), commands=commands)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return
    graph_link.parent.mkdir(parents=True, exist_ok=True)
    if not graph_link.is_symlink() and not graph_link.exists():
        graph_link.symlink_to(graph['path'])
    save(receipt_path, receipt)
    save(output / 'run_plan.json', plan)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for phase, command in commands.items():
        with (output / f'{phase}.log').open('x') as log:
            subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = read(output / f'{phase}.json')
        if result.get('status') != 'ok':
            raise RuntimeError(f'Ada-EF {phase} failed; inspect {output / (phase + ".json")}')
    if 'online' in commands:
        import h5py
        import numpy as np
        result = read(output / 'online.json')
        metrics = result['metrics']
        if metrics['artifact_revision'] != receipt['artifact_revision']:
            raise RuntimeError('Ada-EF binary used a different calibration revision')
        if len(metrics['measured_batch_ms']) != args.measured_runs or len(metrics['warmup_batch_ms']) != args.warmup_runs:
            raise RuntimeError('Ada-EF timing repeat counts differ')
        labels = np.fromfile(output / 'labels.u64', dtype='<u8').astype(np.int64).reshape(cfg['query_count'], args.k)
        with h5py.File(cfg['dataset_path'], 'r') as handle:
            truth = np.asarray(handle['neighbors'][:, :args.k], dtype=np.int64)
        if (labels < 0).any() or (labels >= cfg['train_count']).any() or (np.diff(np.sort(labels, axis=1), axis=1) <= 0).any():
            raise RuntimeError('Ada-EF returned invalid result labels')
        hits = (labels[:, :, None] == truth[:, None, :]).any(2).sum(1)
        if int(hits.sum()) != metrics['exact_hits']:
            raise RuntimeError('Ada-EF reported hits differ from exported labels')
        np.savez_compressed(output / 'labels.npz', labels=labels, hits=hits)
        save(output / 'summary.json', dict(dataset=args.dataset, method='adaef', k=args.k,
            requested_recall=target, target_hits=target_hits, achieved_hits=int(hits.sum()),
            achieved_recall=float(hits.sum() / truth.size), matched_target=bool(hits.sum() >= target_hits),
            mean_batch_seconds=metrics['mean_batch_latency_ms'] / 1000,
            measured_runs=args.measured_runs, warmup_runs=args.warmup_runs,
            artifact_revision=receipt['artifact_revision'], target_source=source))
        samples = [value / 1000 for value in metrics['measured_batch_ms']]
        row = dict(dataset=args.dataset, method='adaef', backend='hnswlib',
            reference_backend='faiss' if args.targets is not None else 'hnswlib',
            k=args.k, query_count=cfg['query_count'], hits=int(hits.sum()),
            recall=float(hits.sum() / truth.size), target_hits=target_hits,
            matched_target=bool(hits.sum() >= target_hits),
            batch_mean_seconds=statistics.mean(samples),
            batch_stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0,
            repeats=args.measured_runs, threads=args.threads)
        with (output / 'summary.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    print(json.dumps(dict(output=str(output), phases=list(commands))), flush=True)


if __name__ == '__main__':
    main()
