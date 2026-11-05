"""Input and measurement helpers for the calibration and sensitivity experiments."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CATALOG = ROOT / 'experiments_scripts/01_main_results/config/main6_online24/datasets.json'


def add_inputs(parser, *, output_name):
    parser.add_argument('--dataset', required=True, help='Dataset identifier from the main dataset catalog')
    parser.add_argument('--catalog', type=Path, default=CATALOG)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--measured-runs', type=int, default=5)
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs' / output_name)


def validate_args(args):
    if args.threads < 1 or args.measured_runs < 2 or args.warmup_runs < 0:
        raise ValueError('Use positive threads, at least two measured runs, and nonnegative warmup runs')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(args):
    import faiss
    import h5py
    import numpy as np
    from experiments_scripts.faiss.faiss_sage_index import Index

    common = importlib.import_module('experiments_scripts.01_main_results.main6_common')
    cfg = common.resolve_datasets(args.catalog, {'datasets': [args.dataset]})[0]
    for path in (cfg['dataset_path'], cfg['faiss_index']['path']):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing input: {path}; see 01_main_results/README.md for data and index preparation')
    faiss.omp_set_num_threads(args.threads)
    index = Index(space=cfg['space'], dim=cfg['dimension'])
    index.load_index(cfg['faiss_index']['path'])
    index.set_num_threads(args.threads)
    expected_metric = faiss.METRIC_L2 if cfg['space'] == 'l2' else faiss.METRIC_INNER_PRODUCT
    if index._require_index().metric_type != expected_metric:
        raise ValueError('Index metric differs from the catalog')
    if ('M' in cfg and index.M != cfg['M']) or ('efConstruction' in cfg and index.ef_construction != cfg['efConstruction']):
        raise ValueError('Index construction settings differ from the catalog')
    with h5py.File(cfg['dataset_path'], 'r') as handle:
        if handle['train'].shape != (index.get_current_count(), cfg['dimension']):
            raise ValueError('Dataset and index dimensions/counts differ')
        if index.get_current_count() != cfg['train_count']:
            raise ValueError('Dataset and catalog counts differ')
        ids = np.unique(np.linspace(0, cfg['train_count'] - 1, 33, dtype=np.int64))
        expected = index._prepare_vectors(np.asarray(handle['train'][ids], dtype=np.float32))
        np.testing.assert_allclose(index.get_items(ids), expected, rtol=1e-4, atol=1e-5)
    return cfg, index


def load_online(cfg):
    import h5py
    import numpy as np
    with h5py.File(cfg['dataset_path'], 'r') as handle:
        vectors = np.ascontiguousarray(handle['test'][:], dtype=np.float32)
        truth = np.ascontiguousarray(handle['neighbors'][:, :10], dtype=np.int64)
    if vectors.shape != (cfg['query_count'], cfg['dimension']) or truth.shape != (cfg['query_count'], 10):
        raise ValueError('Online query/truth shape differs from the catalog')
    return vectors, truth


def hits(labels, truth):
    import numpy as np
    labels, truth = np.asarray(labels), np.asarray(truth)
    if labels.shape != truth.shape or np.any(labels < 0):
        raise ValueError('Invalid result labels')
    if np.any(np.diff(np.sort(labels, axis=1), axis=1) <= 0):
        raise ValueError('Duplicate result labels')
    return (labels[:, :, None] == truth[:, None, :]).any(axis=2).sum(axis=1)


def hidden(index, vectors, ids, ef, threads):
    import numpy as np
    previous = index.ef
    try:
        index.set_ef(ef)
        labels, _ = index.knn_query_hide_node(vectors, k=10, hide_labels=ids, num_threads=threads)
    finally:
        index.set_ef(previous)
    if np.any(labels == ids[:, None]):
        raise RuntimeError('Calibration search returned the hidden node')
    return labels


def measure_jobs(jobs, invoke, truth, args, folder):
    """Randomize complete batches; verify labels after each measured call."""
    import numpy as np
    records, reference = [], {}
    for repeat in range(args.warmup_runs + args.measured_runs):
        phase = 'warmup' if repeat < args.warmup_runs else 'measured'
        for position, name in enumerate(np.random.default_rng(20261102 + repeat).permutation(jobs)):
            name = str(name)
            start = time.perf_counter()
            labels, distances = invoke(name)
            seconds = time.perf_counter() - start
            if not np.isfinite(distances).all():
                raise RuntimeError('Search returned nonfinite distances')
            if name in reference:
                np.testing.assert_array_equal(labels, reference[name])
            else:
                reference[name] = labels.copy()
                np.savez_compressed(folder / f'{name}.npz', labels=labels, distances=distances,
                                    hits=hits(labels, truth))
            records.append(dict(setting=name, repeat=repeat, phase=phase, position=position,
                                seconds=seconds))
            write_csv(folder / 'timings.csv', records)
            print(json.dumps(records[-1]), flush=True)
    result = {}
    for name in jobs:
        values = [r['seconds'] for r in records if r['setting'] == name and r['phase'] == 'measured']
        result[name] = dict(recall=float(hits(reference[name], truth).sum() / truth.size),
                            batch_seconds_mean=statistics.mean(values),
                            batch_seconds_sd=statistics.stdev(values),
                            latency_ms=1000 * statistics.mean(values) / len(truth),
                            measured_runs=args.measured_runs, warmup_runs=args.warmup_runs)
    return result
