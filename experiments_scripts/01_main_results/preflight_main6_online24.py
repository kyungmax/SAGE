#!/usr/bin/env python3
"""Check main-experiment inputs and builds without loading or searching a graph."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

from main6_common import (ROOT, RUN_ROOT, SAGE_ROOT, effective_config, resolve_datasets,
                         validate_policy_config, validate_timing_config)

CONFIG = ROOT / 'config/main6_online24'


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    if stat.st_size == 0:
        raise ValueError(f'Empty input: {path}')
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def inspect(ds, required_k, backends):
    import h5py
    data = identity(ds['dataset_path'])
    indexes = {}
    with h5py.File(data['path'], 'r') as f:
        shapes = {name: list(f[name].shape) for name in ['train', 'test', 'neighbors']}
        if shapes['train'] != [ds['train_count'], ds['dimension']]:
            raise ValueError('Training shape differs from catalog')
        if shapes['test'] != [ds['query_count'], ds['dimension']]:
            raise ValueError('Query shape differs from catalog')
        if shapes['neighbors'][0] != ds['query_count'] or shapes['neighbors'][1] < required_k:
            raise ValueError(f'Ground truth must have at least {required_k} columns; prepare exact GT first')
        external = []
        for name, offset, size in f['train'].external or []:
            path = Path(name.decode() if isinstance(name, bytes) else name)
            if not path.is_absolute():
                path = Path(data['path']).parent / path
            external.append(identity(path) | dict(offset=int(offset), declared_bytes=int(size)))
    for backend in backends:
        expected = ds[backend + '_index']
        actual = identity(expected['path'])
        for field in ('bytes', 'mtime_ns'):
            if field in expected and actual[field] != expected[field]:
                raise ValueError(f'{backend} index {field} changed')
        indexes[backend] = actual
    if 'hnswlib' in backends:
        with Path(indexes['hnswlib']['path']).open('rb') as f:
            header = f.read(struct.calcsize('<6QiI3QdQ'))
        _, capacity, count, _, label_offset, data_offset, _, _, _, _, m, _, efc = struct.unpack('<6QiI3QdQ', header)
        if not (count == ds['train_count'] and count <= capacity and
                (label_offset - data_offset) // 4 == ds['dimension'] and
                m == ds['M'] and efc == ds['efConstruction']):
            raise ValueError('hnswlib graph header differs from catalog')
        if 'hnswlib_header' in ds and hashlib.sha256(header).hexdigest() != ds['hnswlib_header']['sha256']:
            raise ValueError('hnswlib header fingerprint changed')
    return dict(dataset=ds['dataset'], dataset_file=data, shapes=shapes, indexes=indexes,
                external_train_sources=external)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG / 'experiment.json')
    parser.add_argument('--datasets-config', type=Path, default=CONFIG / 'datasets.json')
    parser.add_argument('--output', type=Path, default=RUN_ROOT / 'preparation/main6_online24/preflight.json')
    parser.add_argument('--k', choices=['10', '100', 'all'], default='all')
    parser.add_argument('--cells', default='faiss,hnswlib,pip,abs')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_policy_config(config)
    cells = args.cells.split(',')
    if not set(cells) <= {'faiss', 'hnswlib', 'pip', 'abs'}:
        parser.error('Unknown cell')
    ks = config['k_values'] if args.k == 'all' else [int(args.k)]
    backends = sorted({'hnswlib' if c == 'hnswlib' else 'faiss' for c in cells})
    errors, checked, binaries = [], [], []
    try:
        validate_timing_config(config)
        cpus = config['timing'].get('cpus')
        if cpus and not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError('Configured CPU affinity is unavailable on this host')
        if config['timing'].get('numa_node') is not None and not shutil.which('numactl'):
            raise ValueError('Explicit NUMA binding requires numactl')
    except ValueError as exc:
        errors.append(dict(error=str(exc)))
    if config['timed_repeats'] < 2 or config['warmups_per_setting'] < 0:
        errors.append(dict(error='Need at least two measured runs and nonnegative warmups'))
    for k in ks:
        scoped = effective_config(args.config, k)
        try:
            datasets = resolve_datasets(args.datasets_config, scoped)
        except (ValueError, KeyError) as exc:
            errors.append(dict(k=k, error=str(exc)))
            continue
        for ds in datasets:
            try:
                checked.append(dict(k=k, **inspect(ds, k, backends)))
            except (OSError, ValueError, KeyError, struct.error) as exc:
                errors.append(dict(dataset=ds['dataset'], k=k, error=str(exc)))
    names = []
    if 'faiss' in backends:
        names.append('faiss/build_sage_avx512/faiss/python/faiss/_swigfaiss_avx512.so')
    if 'pip' in cells:
        names.append('baselines/pip/build/paper095/faiss/python/faiss/_swigfaiss_avx512.so')
    if 'abs' in cells:
        names += ['baselines/abs/build/libpaper_abs.so', 'baselines/abs/build/faiss/avx512/faiss/python/faiss/_swigfaiss_avx512.so']
    if 'hnswlib' in backends:
        extensions = list((SAGE_ROOT / 'hnswlib').glob('hnswlib*.so'))
        if len(extensions) != 1:
            errors.append(dict(error='Expected one local hnswlib extension'))
        else:
            names.append(str(extensions[0].relative_to(SAGE_ROOT)))
    for name in names:
        try:
            binaries.append(identity(SAGE_ROOT / name))
        except OSError as exc:
            errors.append(dict(binary=name, error=str(exc)))
    result = dict(passed=not errors, ready_to_run=not errors, no_ann_run=True,
        checked_datasets=checked, binaries=binaries, errors=errors,
        warmups=config['warmups_per_setting'], measured_runs=config['timed_repeats'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
