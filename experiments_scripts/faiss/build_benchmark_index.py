#!/usr/bin/env python3
"""Build an HNSW graph from training rows; atomically publish after reload checks."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import faiss
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'index', 'state-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--space', choices=['cosine', 'l2'], required=True)
    p.add_argument('--m', type=int, default=32)
    p.add_argument('--ef-construction', type=int, default=500)
    p.add_argument('--threads', type=int, default=24)
    p.add_argument('--batch-rows', type=int, default=25000)
    p.add_argument('--seed', type=int, default=12345)
    a = p.parse_args()
    if min(a.m, a.ef_construction, a.threads, a.batch_rows) < 1:
        p.error('Positive build parameters required')
    if not Path(faiss.__file__).resolve().is_relative_to(ROOT / 'faiss'):
        p.error('Use this checkout\'s scripts/python.sh')
    a.index = a.index.resolve()
    partial = a.index.with_suffix(a.index.suffix + '.partial')
    if a.index.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite an index or partial build')
    a.index.parent.mkdir(parents=True, exist_ok=True)
    a.state_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    stat = a.dataset.stat()
    binary = next(Path(m.__file__) for name, m in __import__('sys').modules.items()
                  if name.startswith('faiss._swigfaiss') and getattr(m, '__file__', '').endswith('.so'))
    with h5py.File(a.dataset) as h:
        count, dimension = h['train'].shape
        metric = faiss.METRIC_INNER_PRODUCT if a.space == 'cosine' else faiss.METRIC_L2
        index = faiss.IndexHNSWFlat(dimension, a.m, metric)
        index.hnsw.efConstruction = a.ef_construction
        index.hnsw.rng = faiss.RandomGenerator(a.seed)
        faiss.omp_set_num_threads(a.threads)
        manifest = dict(dataset=str(a.dataset.resolve()), source_bytes=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns, index=str(a.index), count=count, dimension=dimension,
            space=a.space, M=a.m, efConstruction=a.ef_construction, seed=a.seed,
            threads=a.threads, batch_rows=a.batch_rows, graph_backend='faiss',
            insertion_order='ascending HDF5 train row IDs', multithreaded_build=a.threads > 1,
            binary_sha256=sha(binary), builder_sha256=sha(Path(__file__)),
            cpu_affinity=sorted(os.sched_getaffinity(0)))
        save(a.state_dir / 'build_manifest.json', manifest)
        for begin in range(0, count, a.batch_rows):
            vectors = np.asarray(h['train'][begin:begin + a.batch_rows], dtype=np.float32, order='C')
            if not np.isfinite(vectors).all():
                raise ValueError('Non-finite training vectors')
            if a.space == 'cosine':
                faiss.normalize_L2(vectors)
            index.add(vectors)
            state = dict(stage='building', count=index.ntotal, target=count,
                         elapsed_seconds=time.perf_counter() - start, pid=os.getpid())
            save(a.state_dir / 'status.json', state)
            print(json.dumps(state), flush=True)
        ids = np.unique(np.linspace(0, count - 1, min(count, 32), dtype=np.int64))
        queries = np.asarray(h['train'][ids], dtype=np.float32, order='C')
    if a.space == 'cosine':
        faiss.normalize_L2(queries)
    index.hnsw.efSearch = 128
    expected = index.search(queries, min(10, count))
    faiss.write_index(index, str(partial))
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    del index
    gc.collect()
    index = faiss.read_index(str(partial))
    actual = index.search(queries, min(10, count))
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)
    after = a.dataset.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError('Dataset changed while building')
    if a.index.exists():
        raise FileExistsError(a.index)
    partial.replace(a.index)
    manifest.update(complete=True, actual_index_bytes=a.index.stat().st_size,
        index_sha256=sha(a.index), reload_check_queries=len(queries),
        reload_labels_distances_equal=True, elapsed_seconds=time.perf_counter() - start)
    save(a.state_dir / 'build_manifest.json', manifest)
    save(a.index.with_suffix(a.index.suffix + '.manifest.json'), manifest)
    save(a.state_dir / 'status.json', dict(stage='complete', index=str(a.index)))


if __name__ == '__main__':
    main()
