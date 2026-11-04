#!/usr/bin/env python3
"""Build a native hnswlib graph from HDF5 train rows, then verify save/reload."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import h5py
import hnswlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_metadata(dataset):
    source = dataset.resolve()
    stat = source.stat()
    external = []
    with h5py.File(dataset, 'r') as data:
        for name, offset, size in data['train'].external or []:
            path = Path(os.fsdecode(name)).resolve()
            info = path.stat()
            external.append(dict(path=str(path), bytes=info.st_size,
                                 mtime_ns=info.st_mtime_ns, offset=offset, size=size))
    return dict(dataset=str(source), source_bytes=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns, external_sources=external)


def save_native(index, path):
    """Publish only a fully flushed native file; retain the previous checkpoint."""
    expected = index.index_file_size()
    if shutil.disk_usage(path.parent).free < expected + (1 << 30):
        raise OSError('Insufficient space for native save plus 1 GiB reserve.')
    partial = path.with_name(path.name + '.partial')
    index.save_index(str(partial))
    if partial.stat().st_size != expected:
        raise OSError('Saved index size does not match native serialization size.')
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    partial.replace(path)
    sync_directory(path.parent)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--space', choices=('l2', 'ip', 'cosine'), required=True)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--batch-rows', type=int, default=25000)
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--ef-construction', type=int, default=500)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Save a resumable graph every N inserted rows (0 disables).')
    parser.add_argument('--resume-from', type=Path,
                        help='Resume from a checkpoint.json manifest; input/settings must match.')
    args = parser.parse_args()
    if min(args.threads, args.batch_rows, args.m, args.ef_construction) < 1 or args.m < 2:
        parser.error('Counts must be positive; M must be at least 2.')
    if args.checkpoint_every < 0:
        parser.error('--checkpoint-every must be nonnegative.')
    extension = Path(hnswlib.__file__).resolve()
    if not extension.is_relative_to(ROOT / 'hnswlib'):
        raise RuntimeError('Use this checkout\'s scripts/python.sh.')
    partial = args.index.with_name(args.index.name + '.partial')
    if args.index.exists() or (partial.exists() and not args.resume_from):
        raise FileExistsError('Refusing to overwrite an index or partial build.')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, 'r') as data:
        n, dim = data['train'].shape
        data['train'][0]  # Also check external HDF5 storage before allocating.
    if n < 10:
        raise ValueError('At least 10 train rows are required for reload checks.')
    # The current native layout reserves 100 extra level-0 link slots per node.
    base_stride = (2 * args.m + 100) * 4 + 4 + dim * 4 + 8
    estimated_bytes = int(96 + n * (base_stride + 4 + (args.m * 4 + 4) / (args.m - 1)) * 1.005)
    free = shutil.disk_usage(args.index.parent).free
    required_bytes = estimated_bytes * (2 if args.checkpoint_every else 1) + (1 << 30)
    if free < required_bytes:
        raise OSError(f'Insufficient space: estimated index {estimated_bytes} bytes, free {free}.')
    source_info = source_metadata(args.dataset)
    manifest = dict(**source_info, index=str(args.index.resolve()),
                    count=n, dimension=dim, M=args.m, efConstruction=args.ef_construction,
                    space=args.space, seed=args.seed, threads=args.threads,
                    insertion_order='ascending HDF5 train row IDs', batch_rows=args.batch_rows,
                    estimated_index_bytes=estimated_bytes, disk_free_before=free,
                    cpu_affinity=sorted(os.sched_getaffinity(0)),
                    extension_sha256=sha256(extension), builder_sha256=sha256(Path(__file__)),
                    multithreaded_build=args.threads > 1, graph_backend='hnswlib',
                    checkpoint_every=args.checkpoint_every, resumed_rows=0)
    fingerprint = {key: manifest[key] for key in [*source_info, 'count', 'dimension',
                   'M', 'efConstruction', 'space', 'seed', 'extension_sha256']}
    resume = None
    if args.resume_from:
        resume = json.loads(args.resume_from.read_text())
        if resume['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint input/settings/binary fingerprint mismatch.')
        checkpoint = Path(resume['checkpoint_path'])
        if (checkpoint.stat().st_size != resume['index_bytes'] or
                sha256(checkpoint) != resume['index_sha256']):
            raise ValueError('Checkpoint size/checksum mismatch.')
        manifest.update(resumed_rows=resume['inserted_rows'], resume_from=str(args.resume_from.resolve()),
                        resumed_checkpoint_sha256=resume['index_sha256'],
                        random_state_restored=False,
                        resume_note='Native serialization does not preserve insertion RNG state; '
                                    'the resumed graph need not match an uninterrupted build.')
    write_json(args.state_dir / 'build_manifest.json', manifest)
    started = time.perf_counter()
    index = hnswlib.Index(space=args.space, dim=dim)
    if resume:
        index.load_index(resume['checkpoint_path'], max_elements=n)
        if index.get_current_count() != resume['inserted_rows']:
            raise ValueError('Checkpoint count mismatch.')
    else:
        index.init_index(max_elements=n, M=args.m, ef_construction=args.ef_construction,
                         random_seed=args.seed)
    index.set_num_threads(args.threads)
    stop = threading.Event()
    state_lock = threading.Lock()
    current_stage = 'building'
    last_checkpoint = resume
    start_row = index.get_current_count()

    def progress(stage=None, **extra):
        nonlocal current_stage
        with state_lock:
            if stage is not None:
                current_stage = stage
            elapsed = time.perf_counter() - started
            count = index.get_current_count()
            value = dict(stage=current_stage, count=count, target=n, elapsed_seconds=elapsed,
                         resumed_rows=start_row, pid=os.getpid(),
                         average_rows_per_second=(count-start_row) / max(elapsed, 1e-9),
                         checkpoint_rows=0 if last_checkpoint is None else last_checkpoint['inserted_rows'],
                         **extra)
            write_json(args.state_dir / 'status.json', value)
            print(json.dumps(value), flush=True)

    def heartbeat():
        while not stop.wait(30):
            progress()

    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    def interrupted(signum, _frame):
        raise InterruptedError(f'Builder received signal {signum}; use the last committed checkpoint.')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        progress('building')
        checkpoint_at = ((start_row // args.checkpoint_every + 1) * args.checkpoint_every
                         if args.checkpoint_every else n + 1)
        with h5py.File(args.dataset, 'r') as data:
            for begin in range(start_row, n, args.batch_rows):
                end = min(n, begin + args.batch_rows)
                vectors = np.ascontiguousarray(data['train'][begin:end], dtype=np.float32)
                if not np.isfinite(vectors).all():
                    raise ValueError(f'Non-finite train vectors in rows {begin}:{end}.')
                index.add_items(vectors, ids=np.arange(begin, end), num_threads=args.threads)
                if end >= checkpoint_at and end < n:
                    path = args.index.with_name(args.index.name + f'.checkpoint.{end}.index')
                    progress('saving_checkpoint')
                    size = save_native(index, path)
                    saved = dict(checkpoint_path=str(path.resolve()), inserted_rows=end,
                                 index_bytes=size, index_sha256=sha256(path), fingerprint=fingerprint)
                    write_json(path.with_suffix(path.suffix + '.json'), saved)
                    write_json(args.state_dir / 'checkpoint.json', saved)
                    previous = last_checkpoint
                    last_checkpoint = saved
                    # Delete only this output's older generated checkpoint after committing the new one.
                    if previous:
                        old = Path(previous['checkpoint_path'])
                        if (old.parent == args.index.resolve().parent and old != path.resolve()
                                and old.name.startswith(args.index.name + '.checkpoint.')):
                            old.unlink(missing_ok=True)
                            old.with_suffix(old.suffix + '.json').unlink(missing_ok=True)
                    checkpoint_at = (end // args.checkpoint_every + 1) * args.checkpoint_every
                    progress('building')
            ids = np.unique(np.linspace(0, n - 1, 32, dtype=np.int64))
            checks = np.ascontiguousarray(data['train'][ids], dtype=np.float32)
        stop.set()
        reporter.join()
        assert index.element_count == n
        progress('checking_before_save')
        index.set_ef(128)
        expected = index.knn_query(checks, k=10, num_threads=1)
        expected_size = index.index_file_size()
        if shutil.disk_usage(args.index.parent).free < expected_size + (1 << 30):
            raise OSError('Free space changed during build; cannot safely persist the index.')
        progress('saving', expected_index_bytes=expected_size)
        save_native(index, partial)
        del index
        gc.collect()
        index = hnswlib.Index(space=args.space, dim=dim)
        index.load_index(str(partial))
        index.set_ef(128)
        actual = index.knn_query(checks, k=10, num_threads=1)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(before, after)
        if source_metadata(args.dataset) != source_info:
            raise ValueError('Input metadata changed during build.')
        if args.index.exists():
            raise FileExistsError('Final index appeared during the build.')
        partial.replace(args.index)
        sync_directory(args.index.parent)
        manifest.update(complete=True, actual_index_bytes=expected_size,
                        index_sha256=sha256(args.index), reload_check_queries=len(checks),
                        reload_labels_distances_equal=True, elapsed_seconds=time.perf_counter()-started)
        write_json(args.state_dir / 'build_manifest.json', manifest)
        write_json(args.index.with_suffix(args.index.suffix + '.manifest.json'), manifest)
        progress('complete', index=str(args.index))
    except BaseException as exc:
        stop.set()
        reporter.join()
        write_json(args.state_dir / 'status.json', dict(stage='failed', error=repr(exc),
                   elapsed_seconds=time.perf_counter()-started,
                   checkpoint_rows=0 if last_checkpoint is None else last_checkpoint['inserted_rows']))
        raise


if __name__ == '__main__':
    main()
