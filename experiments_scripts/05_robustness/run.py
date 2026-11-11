#!/usr/bin/env python3
"""Run aligned embedding-quality or three-dataset RaBitQ robustness curves."""
import argparse
import csv
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = ROOT / 'experiments_scripts/01_main_results'
sys.path[:0] = [str(ROOT), str(MAIN)]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def dependencies(path):
    """Validate HDF5 virtual/external payloads without reading vector matrices."""
    import h5py
    seen, pending = set(), [Path(path).resolve(strict=True)]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        require(current.is_file(), f'Missing HDF5 payload: {current}')
        seen.add(current)
        if current.suffix not in ('.hdf5', '.h5'):
            continue
        with h5py.File(current, 'r') as handle:
            for key in ('train', 'test'):
                if key not in handle:
                    continue
                data = handle[key]
                filenames = {item[0] for item in (data.external or [])}
                if data.is_virtual:
                    filenames.update(item.file_name for item in data.virtual_sources())
                for filename in filenames:
                    if isinstance(filename, bytes):
                        filename = filename.decode()
                    if filename == '.':
                        continue
                    target = Path(filename)
                    if not target.is_absolute():
                        target = current.parent / target
                    pending.append(target.resolve())
    return [identity(item) for item in sorted(seen)]


def selected_inputs(args, config):
    import h5py
    catalog = read(args.catalog)
    rows = catalog['datasets']
    names = args.datasets.split(',') if args.datasets else [row['dataset'] for row in rows]
    require(len(names) == len(set(names)) and set(names) <= {row['dataset'] for row in rows},
            'Choose distinct names from the supplied catalog')
    data_root = Path(os.environ.get('SAGE_DATA_DIR', ROOT / 'datasets'))
    index_root = Path(os.environ.get('SAGE_INDEX_DIR', ROOT / 'index'))
    entries = []
    for name in names:
        row = dict(next(item for item in rows if item['dataset'] == name))
        data = Path(row['dataset_path']) if 'dataset_path' in row else data_root / row['dataset_file']
        backend = 'faiss' if args.experiment == 'embedding' else 'rabitq'
        graph = row[backend + '_index']
        index = Path(graph['path']) if 'path' in graph else index_root / graph['file']
        row['dataset_path'] = str(data.resolve(strict=True))
        row[backend + '_index'] = identity(index)
        row['dependencies'] = dependencies(data)
        with h5py.File(data, 'r') as handle:
            require(handle['train'].shape == (row['train_count'], row['dimension']),
                    f'{name}: train shape differs from catalog')
            require(handle['test'].shape == (row['query_count'], row['dimension']),
                    f'{name}: test shape differs from catalog')
            require(handle['neighbors'].shape[0] == row['query_count'] and
                    handle['neighbors'].shape[1] >= config['k'], f'{name}: insufficient ground truth')
            require(row['train_count'] >= 10000, 'Calibration needs at least 10000 training vectors')
            if args.experiment == 'embedding':
                method = str(handle.attrs.get('ground_truth_method', '')).lower()
                require('exhaustive' in method or 'exact' in method,
                        f'{name}: HDF5 must record exact ground_truth_method for its embedding')
        if args.experiment == 'rabitq':
            row.update(data_path=row['dataset_path'], index_path=str(index.resolve()), raw=None,
                       shape=[row['train_count'], row['dimension']], degree=row['M'],
                       ef_construction=row['efConstruction'], total_bits=8)
        entries.append(row)
    if args.experiment == 'embedding':
        require(catalog.get('shared_corpus') and catalog.get('shared_split'),
                'Embedding catalog must identify the common corpus and ordered train/query split')
        require(len({(row['train_count'], row['query_count']) for row in entries}) == 1,
                'Embedding models must use the same corpus/query row counts')
    return entries


def protocol(args):
    from main6_common import validate_policy_config
    config = read(args.config)
    if args.cpus:
        config['timing']['cpus'] = [int(cpu) for cpu in args.cpus.split(',')]
        config['online_threads'] = config['offline_threads'] = len(config['timing']['cpus'])
    if args.numa_node is not None:
        config['timing']['numa_node'] = args.numa_node
    cpus = config['timing']['cpus']
    require(cpus and len(cpus) == len(set(cpus)), 'Set a nonempty, distinct CPU list')
    require(set(cpus) <= os.sched_getaffinity(0), 'Configured CPUs are unavailable; pass --cpus')
    cores = []
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        cores.append(((topology / 'physical_package_id').read_text().strip(),
                      (topology / 'core_id').read_text().strip()))
    require(len(set(cores)) == len(cpus), 'Choose one logical CPU per physical core')
    config['timing']['shared_physical_cores'] = len(cpus)
    require(len(cpus) == config['offline_threads'] == config['online_threads'],
            'CPU count must equal offline_threads and online_threads')
    require(config['k'] == 10 and config['timed_repeats'] == 5 and
            config['warmups_per_setting'] >= 1, 'Use k10, five measured runs and separate warmup')
    validate_policy_config(config)
    if config['timing'].get('numa_node') is not None:
        require(shutil.which('numactl'), 'Configured NUMA placement requires numactl')
    return config


def native_preflight(experiment):
    affinity = os.sched_getaffinity(0)
    try:
        if experiment == 'rabitq':
            from experiments_scripts.rabitq.final_policy import load_library
            library = load_library()
            return dict(module=library.__file__, policy_api=library.SAGE_POLICY_API)
        import faiss
        expected = ROOT / 'faiss/build_sage_avx512/faiss/python/faiss'
        require(Path(faiss.__file__).resolve().parent == expected,
                'Use scripts/python.sh with the local AVX512 FAISS build')
        from experiments_scripts.faiss.faiss_sage_index import Index
        require(callable(Index) and hasattr(faiss.IndexHNSWFlat, 'knn_query_sage'),
                'Missing local SAGE API; rebuild FAISS')
        return dict(module=faiss.__file__, version=faiss.__version__)
    finally:
        os.sched_setaffinity(0, affinity)


def source_signature(experiment):
    sources = list(HERE.glob('*.py')) + list(MAIN.glob('*.py'))
    for directory in ('common', 'faiss', 'rabitq'):
        sources.extend((ROOT / 'experiments_scripts' / directory).glob('*.py'))
    native = (ROOT / 'faiss/build_sage_avx512/faiss/python/faiss' if experiment == 'embedding'
              else ROOT / 'rabitq/build_sage/python/rabitqlib')
    binaries = list(native.glob('*.so'))
    require(binaries, f'No local native build at {native}')
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(sources + binaries)}


def verify_inputs(entries):
    for entry in entries:
        graph = entry.get('rabitq_index', entry.get('faiss_index'))
        for item in [graph, *entry['dependencies']]:
            require(identity(item['path']) == item, f'Input changed: {item["path"]}')


def worker(args):
    manifest = read(args.output / 'manifest.json')
    config, entries = manifest['config'], manifest['entries']
    require(source_signature(args.experiment) == manifest['sources'], 'Sources changed since launch')
    verify_inputs(entries)
    entry = next(row for row in entries if row['dataset'] == args.worker)
    os.sched_setaffinity(0, config['timing']['cpus'])
    if args.experiment == 'rabitq':
        benchmark = importlib.import_module('experiments_scripts.rabitq.benchmark')
        benchmark.cell_run(config, entry, 10, args.output / 'results' / args.worker)
    else:
        from types import SimpleNamespace
        from run_full_search import run
        worker_args = SimpleNamespace(backend='faiss', stage=args.phase,
            config=args.output / 'config.json', dataset_config=args.output / 'catalog.json',
            output=args.output / ('calibration' if args.phase == 'calibrate' else 'results') / args.worker,
            policy=args.output / 'calibration' / args.worker / 'policy.json' if args.phase == 'online' else None)
        run(worker_args, config, entry)
    verify_inputs(entries)
    require(source_signature(args.experiment) == manifest['sources'], 'Sources changed during execution')


def execute(args, config, entries, report):
    from run_baselines_main6 import environment
    require(not args.output.exists(), f'Use a new output directory: {args.output}')
    args.output.mkdir(parents=True)
    for folder in ('logs', 'calibration', 'results'):
        (args.output / folder).mkdir()
    save(args.output / 'config.json', config)
    save(args.output / 'catalog.json', dict(datasets=entries))
    save(args.output / 'manifest.json', dict(experiment=args.experiment, config=config,
        entries=entries, sources=source_signature(args.experiment), preflight=report,
        config_sha256=sha(args.config), catalog_sha256=sha(args.catalog), result_reuse=False))
    env = environment(config)
    if args.experiment == 'rabitq':
        env['OMP_PROC_BIND'] = 'FALSE'
        env.pop('OMP_PLACES', None)
    lock_path = ROOT / 'runs/timing.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in entries:
            phases = ('calibrate', 'online') if args.experiment == 'embedding' else ('online',)
            for phase in phases:
                command = ['bash', str(ROOT / 'scripts/python.sh'), str(Path(__file__).resolve()),
                           args.experiment, '--output', str(args.output), '--worker', entry['dataset'],
                           '--phase', phase]
                node = config['timing'].get('numa_node')
                if node is not None:
                    command = ['numactl', '--physcpubind=' + ','.join(map(str, config['timing']['cpus'])),
                               '--membind=' + str(node), *command]
                print(f'{entry["dataset"]}: {phase}', flush=True)
                with (args.output / 'logs' / f'{entry["dataset"]}.{phase}.log').open('x') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   pass_fds=(lock.fileno(),))
        for filename in ('summary.csv', 'offline_times.csv', 'raw_timings.csv'):
            rows = []
            for entry in entries:
                with (args.output / 'results' / entry['dataset'] / filename).open() as stream:
                    rows.extend(csv.DictReader(stream))
            with (args.output / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        save(args.output / 'complete.json', dict(passed=True, datasets=[row['dataset'] for row in entries]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('embedding', 'rabitq'))
    parser.add_argument('--config', type=Path, default=HERE / 'config/experiment.json')
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--datasets', help='Comma-separated catalog names; default: all')
    parser.add_argument('--cpus', help='Comma-separated physical CPU IDs; overrides thread counts')
    parser.add_argument('--numa-node', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true', help='Read metadata and import native API only')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('calibrate', 'online'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.catalog = (args.catalog or HERE / 'config' / f'{args.experiment}_datasets.json').resolve()
    args.output = (args.output or ROOT / 'runs/05_robustness' / args.experiment).resolve()
    if args.worker:
        worker(args)
        return
    config = protocol(args)
    entries = selected_inputs(args, config)
    report = dict(passed=True, experiment=args.experiment, datasets=[row['dataset'] for row in entries],
                  runtime=native_preflight(args.experiment), native_search_executed=False,
                  measured_runs=config['timed_repeats'], warmups=config['warmups_per_setting'])
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    execute(args, config, entries, report)


if __name__ == '__main__':
    main()
