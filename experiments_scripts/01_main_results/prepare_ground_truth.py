#!/usr/bin/env python3
"""Prepare exact top-k truth from the flat storage of an existing FAISS graph.

Ported from current_sage_k100_20261029/prepare_youtube_truth.py. No index build,
historical binding, or approximate HNSW search is used.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from main6_common import ROOT, SAGE_ROOT, resolve_datasets


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def require(value, message):
    if not value:
        raise ValueError(message)


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def inspect(args):
    import h5py
    document = json.loads(args.catalog.read_text())
    names = [row['dataset'] for row in document['datasets']]
    require(args.dataset in names, f'Dataset is absent from catalog: {args.dataset}')
    rows = resolve_datasets(args.catalog, {'datasets': names})
    for row in rows:
        row.pop('dataset_file', None)
        row['dataset_path'] = str(Path(row['dataset_path']).resolve())
        for backend in ('faiss', 'hnswlib'):
            if backend + '_index' in row:
                graph = row[backend + '_index']
                graph['path'] = str(Path(graph['path']).resolve())
    selected = next(row for row in rows if row['dataset'] == args.dataset)
    require(selected['space'] in ('cosine', 'ip', 'l2'), 'Unsupported vector space')
    data = identity(selected['dataset_path'])
    graph = identity(selected['faiss_index']['path'])
    with h5py.File(data['path'], 'r') as handle:
        require(handle['train'].shape == (selected['train_count'], selected['dimension']),
                'Training shape differs from catalog')
        require(handle['test'].shape == (selected['query_count'], selected['dimension']),
                'Query shape differs from catalog')
        previous_k = handle['neighbors'].shape[1] if 'neighbors' in handle else 0
    require(args.k <= selected['train_count'], 'k exceeds the training corpus size')
    require(not args.output.exists(), f'Use a new output directory: {args.output}')
    import faiss
    expected = SAGE_ROOT / 'faiss/build_sage_avx512/faiss/python/faiss'
    require(Path(faiss.__file__).resolve().parent == expected, 'Use the local AVX512 build through scripts/python.sh')
    return document, rows, selected, dict(dataset=args.dataset, original_data=data, graph=graph,
        k=args.k, previous_ground_truth_width=previous_k, threads=args.threads,
        queries=selected['query_count'], training_rows=selected['train_count'],
        dimension=selected['dimension'], runtime=faiss.__file__,
        method='exhaustive IndexFlat search over the same full index storage',
        output_dataset=str(args.output / f'{args.dataset}.top{args.k}.hdf5'),
        output_catalog=str(args.output / 'datasets.json'),
        graph_bytes_to_load=graph['bytes'], no_ann_run=True)


def prepare(args, document, rows, selected, plan):
    import faiss
    import h5py
    import numpy as np
    source = plan['original_data']['path']
    with h5py.File(source, 'r') as handle:
        queries = np.asarray(handle['test'], dtype=np.float32, order='C')
        attrs = dict(handle.attrs)
        old_truth = np.asarray(handle['neighbors'], dtype=np.int64) if 'neighbors' in handle else None
    require(np.isfinite(queries).all(), 'Nonfinite query vectors')
    prepared = queries.copy()
    if selected['space'] == 'cosine':
        faiss.normalize_L2(prepared)
    faiss.omp_set_num_threads(args.threads)
    faiss.InterruptCallback.clear_instance()
    lock_path = SAGE_ROOT / 'runs/timing.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f'Loading existing FAISS graph: {plan["graph"]["path"]}', flush=True)
        load_started = time.perf_counter()
        index = faiss.read_index(plan['graph']['path'])
        flat = faiss.downcast_index(index.storage)
        expected_metric = faiss.METRIC_L2 if selected['space'] == 'l2' else faiss.METRIC_INNER_PRODUCT
        require(isinstance(flat, faiss.IndexFlat), 'Graph storage must be full-precision IndexFlat')
        require((flat.ntotal, flat.d, flat.metric_type) ==
                (selected['train_count'], selected['dimension'], expected_metric),
                'Stored vectors or metric differ from catalog')
        require((index.hnsw.nb_neighbors(1), index.hnsw.efConstruction) ==
                (selected['M'], selected['efConstruction']), 'Graph construction settings differ')
        sample_ids = np.unique(np.linspace(0, flat.ntotal - 1, min(33, flat.ntotal), dtype=np.int64))
        with h5py.File(source, 'r') as handle:
            expected = np.asarray(handle['train'][sample_ids], dtype=np.float32, order='C')
        if selected['space'] == 'cosine':
            faiss.normalize_L2(expected)
        actual = np.vstack([flat.reconstruct(int(label)) for label in sample_ids])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5,
                                   err_msg='Graph vectors do not match HDF5 train rows')
        load_seconds = time.perf_counter() - load_started
        print(f'Exhaustive search: {flat.ntotal} vectors, {len(prepared)} queries, k={args.k}', flush=True)
        started = time.perf_counter()
        scores, labels = flat.search(prepared, args.k)
        search_seconds = time.perf_counter() - started
    require(labels.shape == (selected['query_count'], args.k), 'Unexpected label shape')
    require(((labels >= 0) & (labels < flat.ntotal)).all() and np.isfinite(scores).all(),
            'Invalid exhaustive-search results')
    require((np.diff(np.sort(labels, axis=1), axis=1) > 0).all(), 'Duplicate result labels')
    if selected['space'] == 'cosine':
        distances, representation = np.maximum(0, 1 - scores), '1 - normalized inner product'
    elif selected['space'] == 'ip':
        distances, representation = -scores, 'negative inner product'
    else:
        distances, representation = np.sqrt(np.maximum(0, scores)), 'Euclidean distance'
    require(identity(source) == plan['original_data'] and
            identity(plan['graph']['path']) == plan['graph'], 'Input changed during exact search')
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path(plan['output_dataset'])
    with h5py.File(path, 'x') as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle['train'] = h5py.ExternalLink(source, '/train')
        handle['test'] = h5py.ExternalLink(source, '/test')
        handle.create_dataset('neighbors', data=labels)
        handle.create_dataset('distances', data=distances)
        handle.attrs.update(ground_truth_method=plan['method'], ground_truth_complete=True,
            groundtruth_complete=True, groundtruth_k=args.k, groundtruth_threads=args.threads,
            groundtruth_kind='exact_faiss_flat', distance_representation=representation,
            source_hdf5=source, raw_query_sha256=array_hash(queries),
            prepared_query_sha256=array_hash(prepared), ground_truth_sha256=array_hash(labels))
    selected['dataset_path'] = str(path)
    document['datasets'] = rows
    (args.output / 'datasets.json').write_text(json.dumps(document, indent=2) + '\n')
    previous_overlap = None
    if old_truth is not None:
        previous_overlap = float(np.mean([len(set(old) & set(new)) / len(old)
                                          for old, new in zip(old_truth, labels)]))
    manifest = dict(plan, no_ann_run=False, complete=True,
        approximate_graph_search_used=False, existing_vectors_reused=True,
        raw_query_sha256=array_hash(queries), prepared_query_sha256=array_hash(prepared),
        truth_sha256=array_hash(labels), distance_sha256=array_hash(distances),
        original_catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        graph_load_validation_seconds=load_seconds, exact_search_seconds=search_seconds,
        previous_truth_coverage=previous_overlap, distance_representation=representation,
        source_hdf5_required=True, output_identity=identity(path))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(complete=True, datasets_config=plan['output_catalog'],
                         exact_search_seconds=search_seconds), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='youtube-15M-angular')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'config/main6_online24/datasets.json')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--preflight', action='store_true', help='Check metadata/import only; no graph load/search')
    args = parser.parse_args()
    if min(args.k, args.threads) < 1:
        parser.error('k and threads must be positive')
    args.catalog, args.output = args.catalog.resolve(), args.output.resolve()
    document, rows, selected, plan = inspect(args)
    if args.preflight:
        print(json.dumps(plan, indent=2))
        return
    prepare(args, document, rows, selected, plan)


if __name__ == '__main__':
    main()
