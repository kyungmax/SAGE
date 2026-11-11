"""Portable input resolution and k-specific scope for the final experiment."""
from copy import deepcopy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAGE_ROOT = ROOT.parents[1]
RUN_ROOT = SAGE_ROOT / 'runs/01_main_results'


def effective_config(path, k=None):
    config = json.loads(Path(path).read_text())
    k = int(config['k'] if k is None else k)
    if k not in config.get('k_values', [config['k']]):
        raise ValueError(f'Unsupported k={k}')
    config['k'] = k
    excluded = config.get('exclude_datasets_by_k', {}).get(str(k), [])
    config['datasets'] = [d for d in config['datasets'] if d not in excluded]
    config['calibration']['shadow_enabled'] = k != 100
    config['calibration'].setdefault('lid_pool_size', 10000)
    config['calibration'].setdefault('internal_lid_k', 15)
    config['calibration'].setdefault('lid_search_ef', 512)
    config['calibration'].setdefault('lid_exclude_self', True)
    if k == 100:
        config['methods'] = {backend: ['sage_main4' if method == 'sage_shadow128' else method
                                      for method in methods]
                             for backend, methods in config['methods'].items()}
    return config


def validate_timing_config(config):
    """Thread counts are independent of placement unless CPUs are explicitly bound."""
    offline, online = config['offline_threads'], config['online_threads']
    if any(type(value) is not int or value < 1 for value in (offline, online)):
        raise ValueError('Offline/online thread counts must be positive integers')
    if offline != online:
        raise ValueError('Use the same offline and online thread counts')
    cpus = config['timing'].get('cpus')
    if cpus is not None:
        if (not isinstance(cpus, list) or
                any(type(cpu) is not int or cpu < 0 for cpu in cpus) or
                len(cpus) != len(set(cpus))):
            raise ValueError('CPU binding must be null or a list of distinct nonnegative CPU IDs')
        if cpus and len(cpus) != online:
            raise ValueError('Explicit CPU binding must match offline/online thread counts')
    node = config['timing'].get('numa_node')
    if node is not None and (type(node) is not int or node < 0):
        raise ValueError('NUMA binding must be null or a nonnegative node ID')


def with_numa_binding(command, config):
    """Add numactl only when a NUMA node was explicitly requested."""
    node = config['timing'].get('numa_node')
    if node is None:
        return command
    prefix = ['numactl']
    cpus = config['timing'].get('cpus')
    if cpus:
        prefix.append('--physcpubind=' + ','.join(map(str, cpus)))
    return [*prefix, f'--membind={node}', *command]


def validate_policy_config(config):
    # These are fixed policy choices in common.final_policy, not free knobs.
    if config['calibration'].get('lid_pool_size', 10000) != 10000:
        raise ValueError('Config does not match the final policy: lid_pool_size must be 10000')
    for key, value in dict(internal_lid_k=15, lid_search_ef=512, lid_exclude_self=True).items():
        if config['calibration'].get(key, value) != value:
            raise ValueError(f'Config does not match the final policy: {key} must be {value!r}')
    expected = dict(probe_count=500, self_hidden=True, pseudo_gt_ef=4096,
        offline_min_ef=64, steps_per_doubling=4, initial_cap=1024,
        cap_increment=1024, high_ef_grid_step=128, offline_max_ef=4096,
        extend_when_recommendation_hits_cap=True, epsilon=.001,
        cfr_signal='raw arithmetic mean', classify_start=1, classify_end=16,
        main_buckets=4, main_margin=2, shadow_ef=128, shadow_margin=4,
        shadow_enabled_from_ef=256, shadow_routes=[32, 64, 96])
    for key, value in expected.items():
        if config['calibration'].get(key) != value:
            raise ValueError(f'Config does not match the final policy: {key} must be {value!r}')


def resolve_datasets(path, config):
    rows = deepcopy(json.loads(Path(path).read_text())['datasets'])
    data_root = Path(os.environ.get('SAGE_DATA_DIR', SAGE_ROOT / 'datasets'))
    index_root = Path(os.environ.get('SAGE_INDEX_DIR', SAGE_ROOT / 'index'))
    selected = []
    for row in rows:
        if row['dataset'] not in config['datasets']:
            continue
        if 'dataset_file' in row:
            row['dataset_path'] = str(data_root / row['dataset_file'])
        for backend in ('faiss', 'hnswlib'):
            if backend + '_index' not in row:
                continue
            item = row[backend + '_index']
            if 'file' in item:
                item['path'] = str(index_root / item.pop('file'))
            graph = Path(item['path'])
            if graph.is_file():
                stat = graph.stat()
                for field, actual in [('bytes', stat.st_size), ('mtime_ns', stat.st_mtime_ns)]:
                    if field in item and item[field] != actual:
                        raise ValueError(f'{row["dataset"]}: {backend} index {field} changed')
                    item[field] = actual
        selected.append(row)
    if [r['dataset'] for r in selected] != config['datasets']:
        raise ValueError('Dataset catalog does not cover the selected scope in order')
    return selected


def hits_per_query(labels, truth):
    import numpy as np
    if labels.shape != truth.shape:
        raise ValueError('Result and truth shapes differ')
    return np.asarray([np.intersect1d(a, b).size for a, b in zip(labels, truth)], dtype=np.int64)
