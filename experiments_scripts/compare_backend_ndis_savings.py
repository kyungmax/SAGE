#!/usr/bin/env python3
"""Compare Vanilla vs SAGE/Ours HNSW distance computations by backend.

The runner measures Vanilla ndis with the native layer-0 CFR summary API and
reconstructs Ours ndis by gathering each query's Vanilla ndis at its routed
efSearch. This is the same accounting used by the saved Qwen distance-count
analysis: for paper-bucket SAGE, once a query is classified into a route,
logical distance work matches the vanilla trace at that routed efSearch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OURS_ROOT = HERE.parent
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HNSWLIB_ROOT = Path(os.environ.get('SAGE_HNSWLIB_EXTENSION_ROOT', str(REPO_ROOT / 'hnswlib'))).expanduser()
DEFAULT_FAISS_PYTHON = Path(os.environ.get('FAISS_PYTHON_PATH', str(REPO_ROOT / 'faiss/build_sage_avx512/faiss/python'))).expanduser()
DEFAULT_FAISS_WRAPPER = HERE / 'faiss'
DEFAULT_OUT = OURS_ROOT / 'final_analysis' / 'backend_ndis_savings'
DEFAULT_HNSWLIB_SWEEP = (
    OURS_ROOT
    / 'final_experiments/HNSWLib/hnswlib_vanilla_ours_final6_m32_efc500_ncal100_20260617/final/main_qps_latency_sweep.csv'
)
DEFAULT_FAISS_SWEEP = (
    OURS_ROOT
    / 'final_experiments/FAISS/faiss_vanilla_ours_final6_m32_efc500_ncal100_20260617/final/main_qps_latency_sweep.csv'
)


@dataclass(frozen=True)
class BackendSpec:
    name: str
    index_path: Path
    sweep_csv: Path


def parse_signature(value: Any, *, kind: str) -> tuple[int | float, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    text = text.replace(',', '/').replace(';', '/')
    parts = [part.strip() for part in text.split('/') if part.strip()]
    if kind == 'int':
        return tuple(int(float(part)) for part in parts)
    if kind == 'float':
        return tuple(float(part) for part in parts)
    raise ValueError(f'unsupported signature kind: {kind!r}')


def route_ef_for_cfr_ratio(
    *,
    selection_ef: int,
    k: int,
    route_efs: tuple[int, ...],
    bucket_gamma_ratios: tuple[float, ...],
    cfr_ratio: float,
) -> int:
    if not np.isfinite(cfr_ratio):
        return int(selection_ef)
    for route_ef, gamma in zip(route_efs, bucket_gamma_ratios):
        if float(cfr_ratio) <= float(gamma) + 1e-12:
            return max(int(k), int(route_ef))
    return int(selection_ef)


def route_count_signature(routed_efs: np.ndarray) -> str:
    values, counts = np.unique(np.asarray(routed_efs, dtype=np.int64), return_counts=True)
    return ';'.join(f'{int(value)}:{int(count)}' for value, count in zip(values, counts))


def recall_at_k(labels: np.ndarray, gt: np.ndarray, k: int) -> float:
    if gt.size == 0:
        return float('nan')
    recalls: list[float] = []
    for row, truth in zip(labels, gt):
        recalls.append(len(set(map(int, row[:k])).intersection(map(int, truth[:k]))) / float(k))
    return float(np.mean(recalls)) if recalls else float('nan')


def load_queries(dataset_hdf5: Path, query_key: str, gt_key: str, n_queries: int, k: int):
    with h5py.File(dataset_hdf5, 'r') as handle:
        queries = np.asarray(handle[query_key][:n_queries], dtype=np.float32)
        gt = None
        if gt_key in handle:
            gt = np.asarray(handle[gt_key][:n_queries, :k], dtype=np.int64)
    return queries, gt


def import_hnswlib(root: Path):
    sys.path.insert(0, str(root))
    import hnswlib  # type: ignore

    return hnswlib


def import_faiss_index(faiss_python: Path, wrapper_dir: Path):
    sys.path.insert(0, str(faiss_python))
    sys.path.insert(0, str(wrapper_dir))
    import faiss  # type: ignore  # noqa: F401
    from faiss_sage_index import Index as FaissIndex  # type: ignore

    return FaissIndex


def load_backend_index(
    spec: BackendSpec,
    *,
    space: str,
    dim: int,
    num_threads: int,
    hnswlib_root: Path,
    faiss_python: Path,
    faiss_wrapper: Path,
):
    if not spec.index_path.exists():
        raise FileNotFoundError(spec.index_path)
    if spec.name == 'hnswlib':
        hnswlib = import_hnswlib(hnswlib_root)
        index = hnswlib.Index(space=space, dim=int(dim))
        index.set_num_threads(int(num_threads))
        index.load_index(str(spec.index_path), max_elements=0)
        return index
    if spec.name == 'faiss':
        FaissIndex = import_faiss_index(faiss_python, faiss_wrapper)
        index = FaissIndex(space=space, dim=int(dim))
        index.set_num_threads(int(num_threads))
        index.load_index(str(spec.index_path))
        return index
    raise ValueError(f'unknown backend: {spec.name!r}')


def call_cfr_summary(index, queries: np.ndarray, *, k: int, ef: int, num_threads: int) -> dict[str, np.ndarray]:
    method = getattr(index, 'search_layer0_cfr_summary_batch', None)
    if method is not None:
        return method(queries, k=int(k), ef=int(ef), num_threads=int(num_threads))
    method = getattr(index, 'search_layer0_cfr_summary', None)
    if method is None:
        raise RuntimeError('index does not expose search_layer0_cfr_summary[_batch]')
    try:
        return method(queries, k=int(k), ef=int(ef), num_threads=int(num_threads))
    except TypeError:
        return method(queries, int(k), int(ef), num_threads=int(num_threads))


def load_sweep_policy(sweep_csv: Path, dataset_name: str, efs: list[int]) -> dict[int, dict[str, Any]]:
    if not sweep_csv.exists():
        raise FileNotFoundError(sweep_csv)
    df = pd.read_csv(sweep_csv)
    if 'method' in df.columns:
        ours = df[df['method'].astype(str) == 'Ours'].copy()
    else:
        ours = df.copy()
    if dataset_name:
        ours = ours[ours['dataset'].astype(str) == str(dataset_name)].copy()
    policies: dict[int, dict[str, Any]] = {}
    for ef in efs:
        part = ours[ours['ef'].astype(int) == int(ef)]
        if part.empty:
            raise ValueError(f'No Ours row for dataset={dataset_name!r}, ef={ef} in {sweep_csv}')
        row = part.iloc[0]
        route_signature = parse_signature(row.get('route_signature', ''), kind='int')
        if route_signature and int(route_signature[-1]) == int(ef):
            route_efs = tuple(int(value) for value in route_signature[:-1])
        else:
            route_efs = tuple(int(value) for value in route_signature if int(value) < int(ef))
        bucket_gammas = tuple(float(value) for value in parse_signature(row.get('bucket_gamma_signature', ''), kind='float'))
        policies[int(ef)] = {
            'tau': float(row.get('early_stop_ratio', np.nan)),
            'route_efs': route_efs,
            'bucket_gammas': bucket_gammas,
            'route_signature': str(row.get('route_signature', '')),
            'bucket_gamma_signature': str(row.get('bucket_gamma_signature', '')),
            'sweep_recall': float(row.get('recall', np.nan)),
            'sweep_qps_gain_vs_vanilla_pct': float(row.get('qps_gain_vs_vanilla_pct', np.nan)),
            'sweep_recall_loss_vs_vanilla_pp': float(row.get('recall_loss_vs_vanilla_pp', np.nan)),
        }
    return policies


def measure_backend(
    spec: BackendSpec,
    *,
    dataset_name: str,
    queries: np.ndarray,
    gt: np.ndarray | None,
    efs: list[int],
    k: int,
    space: str,
    dim: int,
    num_threads: int,
    hnswlib_root: Path,
    faiss_python: Path,
    faiss_wrapper: Path,
):
    policies = load_sweep_policy(spec.sweep_csv, dataset_name, efs)
    needed_efs = set(int(ef) for ef in efs)
    for policy in policies.values():
        needed_efs.update(int(value) for value in policy['route_efs'])

    print(f'[{spec.name}] loading index {spec.index_path}', flush=True)
    index = load_backend_index(
        spec,
        space=space,
        dim=dim,
        num_threads=num_threads,
        hnswlib_root=hnswlib_root,
        faiss_python=faiss_python,
        faiss_wrapper=faiss_wrapper,
    )

    summaries: dict[int, dict[str, np.ndarray]] = {}
    for ef in sorted(needed_efs):
        print(f'[{spec.name}] cfr_summary ef={ef}', flush=True)
        summaries[int(ef)] = call_cfr_summary(index, queries, k=k, ef=int(ef), num_threads=num_threads)

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    qids = np.arange(queries.shape[0], dtype=np.int64)

    for ef in efs:
        policy = policies[int(ef)]
        tau = float(policy['tau'])
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError(f'Invalid early_stop_ratio for {spec.name} ef={ef}: {tau}')
        route_efs = tuple(int(value) for value in policy['route_efs'])
        bucket_gammas = tuple(float(value) for value in policy['bucket_gammas'])
        if route_efs and len(route_efs) != len(bucket_gammas):
            raise ValueError(
                f'route/gamma length mismatch for {spec.name} ef={ef}: '
                f'{route_efs} vs {bucket_gammas}'
            )

        anchor = summaries[int(ef)]
        vanilla_counts = np.asarray(anchor['distance_counts'], dtype=np.float64)
        cfr_values = np.asarray(anchor['mean_smoothed_cfrs'], dtype=np.float64)
        usable = np.asarray(anchor.get('usable_flags', np.isfinite(cfr_values)), dtype=bool)
        routed_efs = np.full(queries.shape[0], int(ef), dtype=np.int64)
        cfr_ratios = cfr_values / max(tau, 1e-6)

        if route_efs and bucket_gammas:
            for i, cfr_ratio in enumerate(cfr_ratios):
                if not bool(usable[i]) or not np.isfinite(cfr_ratio):
                    continue
                routed_efs[i] = route_ef_for_cfr_ratio(
                    selection_ef=int(ef),
                    k=int(k),
                    route_efs=route_efs,
                    bucket_gamma_ratios=bucket_gammas,
                    cfr_ratio=float(cfr_ratio),
                )

        ours_counts = np.empty_like(vanilla_counts)
        for route_ef in np.unique(routed_efs):
            mask = routed_efs == int(route_ef)
            ours_counts[mask] = np.asarray(summaries[int(route_ef)]['distance_counts'], dtype=np.float64)[mask]

        saved = vanilla_counts - ours_counts
        saved_pct = np.where(vanilla_counts > 0, saved / vanilla_counts * 100.0, np.nan)
        vanilla_recall = float('nan')
        if gt is not None:
            try:
                set_ef = getattr(index, 'set_ef', None)
                if set_ef is not None:
                    set_ef(int(ef))
                labels, _ = index.knn_query(queries, k=int(k), num_threads=int(num_threads))
                vanilla_recall = recall_at_k(np.asarray(labels), gt, int(k))
            except Exception as exc:  # keep ndis measurement independent from recall helpers
                print(f'[{spec.name}] recall measurement failed for ef={ef}: {exc}', flush=True)

        for i in range(queries.shape[0]):
            long_rows.append(
                {
                    'backend': spec.name,
                    'dataset': dataset_name,
                    'qid': int(qids[i]),
                    'ef': int(ef),
                    'vanilla_ndis': int(vanilla_counts[i]),
                    'ours_ndis': int(ours_counts[i]),
                    'saved_ndis': int(saved[i]),
                    'saved_ndis_pct': float(saved_pct[i]),
                    'routed_ef': int(routed_efs[i]),
                    'usable_cfr': bool(usable[i]),
                    'mean_smoothed_cfr': float(cfr_values[i]),
                    'cfr_ratio': float(cfr_ratios[i]),
                    'early_stop_ratio': tau,
                }
            )

        summary_rows.append(
            {
                'backend': spec.name,
                'dataset': dataset_name,
                'ef': int(ef),
                'query_count': int(queries.shape[0]),
                'vanilla_mean_ndis': float(np.mean(vanilla_counts)),
                'ours_mean_ndis': float(np.mean(ours_counts)),
                'saved_mean_ndis': float(np.mean(saved)),
                'saved_ndis_pct': float(np.mean(saved) / np.mean(vanilla_counts) * 100.0),
                'ndis_speedup': float(np.mean(vanilla_counts) / np.mean(ours_counts)),
                'vanilla_p50_ndis': float(np.percentile(vanilla_counts, 50)),
                'ours_p50_ndis': float(np.percentile(ours_counts, 50)),
                'saved_p50_ndis': float(np.percentile(saved, 50)),
                'vanilla_p90_ndis': float(np.percentile(vanilla_counts, 90)),
                'ours_p90_ndis': float(np.percentile(ours_counts, 90)),
                'saved_p90_ndis': float(np.percentile(saved, 90)),
                'vanilla_p95_ndis': float(np.percentile(vanilla_counts, 95)),
                'ours_p95_ndis': float(np.percentile(ours_counts, 95)),
                'saved_p95_ndis': float(np.percentile(saved, 95)),
                'usable_cfr_query_count': int(np.sum(usable)),
                'routed_ef_count_signature': route_count_signature(routed_efs),
                'route_signature': policy['route_signature'],
                'bucket_gamma_signature': policy['bucket_gamma_signature'],
                'early_stop_ratio': tau,
                'measured_vanilla_recall': vanilla_recall,
                'sweep_ours_recall': policy['sweep_recall'],
                'sweep_qps_gain_vs_vanilla_pct': policy['sweep_qps_gain_vs_vanilla_pct'],
                'sweep_recall_loss_vs_vanilla_pp': policy['sweep_recall_loss_vs_vanilla_pp'],
            }
        )

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def write_backend_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty or summary['backend'].nunique() < 2:
        return
    pivot = summary.pivot_table(
        index=['dataset', 'ef'],
        columns='backend',
        values=['vanilla_mean_ndis', 'ours_mean_ndis', 'saved_ndis_pct', 'ndis_speedup'],
        aggfunc='first',
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()
    if {'saved_ndis_pct_faiss', 'saved_ndis_pct_hnswlib'}.issubset(pivot.columns):
        pivot['faiss_minus_hnswlib_saved_pct_pp'] = (
            pivot['saved_ndis_pct_faiss'] - pivot['saved_ndis_pct_hnswlib']
        )
    if {'vanilla_mean_ndis_faiss', 'vanilla_mean_ndis_hnswlib'}.issubset(pivot.columns):
        pivot['faiss_vanilla_ndis_over_hnswlib'] = (
            pivot['vanilla_mean_ndis_faiss'] / pivot['vanilla_mean_ndis_hnswlib']
        )
    if {'ours_mean_ndis_faiss', 'ours_mean_ndis_hnswlib'}.issubset(pivot.columns):
        pivot['faiss_ours_ndis_over_hnswlib'] = (
            pivot['ours_mean_ndis_faiss'] / pivot['ours_mean_ndis_hnswlib']
        )
    pivot.to_csv(out_dir / 'backend_ndis_comparison.csv', index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-hdf5', type=Path, required=True)
    parser.add_argument('--dataset-name', required=True, help='Dataset label used in the final sweep CSV.')
    parser.add_argument('--query-key', default='test')
    parser.add_argument('--gt-key', default='neighbors')
    parser.add_argument('--num-queries', type=int, default=10000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--efs', type=int, nargs='+', default=[64, 128, 256, 512, 1024])
    parser.add_argument('--space', required=True, choices=['cosine', 'ip', 'l2'])
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--num-threads', type=int, default=24)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--hnswlib-index', type=Path)
    parser.add_argument('--faiss-index', type=Path)
    parser.add_argument('--hnswlib-sweep-csv', type=Path, default=DEFAULT_HNSWLIB_SWEEP)
    parser.add_argument('--faiss-sweep-csv', type=Path, default=DEFAULT_FAISS_SWEEP)
    parser.add_argument('--hnswlib-root', type=Path, default=DEFAULT_HNSWLIB_ROOT)
    parser.add_argument('--faiss-python', type=Path, default=DEFAULT_FAISS_PYTHON)
    parser.add_argument('--faiss-wrapper', type=Path, default=DEFAULT_FAISS_WRAPPER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hnswlib_index is None and args.faiss_index is None:
        raise SystemExit('Provide at least one of --hnswlib-index or --faiss-index.')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)

    queries, gt = load_queries(args.dataset_hdf5, args.query_key, args.gt_key, args.num_queries, args.k)
    specs: list[BackendSpec] = []
    if args.hnswlib_index is not None:
        specs.append(BackendSpec('hnswlib', args.hnswlib_index, args.hnswlib_sweep_csv))
    if args.faiss_index is not None:
        specs.append(BackendSpec('faiss', args.faiss_index, args.faiss_sweep_csv))

    all_per_query = []
    all_summary = []
    for spec in specs:
        per_query, summary = measure_backend(
            spec,
            dataset_name=args.dataset_name,
            queries=queries,
            gt=gt,
            efs=[int(ef) for ef in args.efs],
            k=int(args.k),
            space=str(args.space),
            dim=int(args.dim),
            num_threads=int(args.num_threads),
            hnswlib_root=args.hnswlib_root,
            faiss_python=args.faiss_python,
            faiss_wrapper=args.faiss_wrapper,
        )
        per_query.to_csv(args.out_dir / f'{spec.name}__per_query_ndis.csv', index=False)
        summary.to_csv(args.out_dir / f'{spec.name}__ndis_summary.csv', index=False)
        all_per_query.append(per_query)
        all_summary.append(summary)

    combined_per_query = pd.concat(all_per_query, ignore_index=True) if all_per_query else pd.DataFrame()
    combined_summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    combined_per_query.to_csv(args.out_dir / 'per_query_ndis.csv', index=False)
    combined_summary.to_csv(args.out_dir / 'ndis_summary.csv', index=False)
    write_backend_comparison(combined_summary, args.out_dir)

    metadata = {
        'dataset_hdf5': str(args.dataset_hdf5),
        'dataset_name': str(args.dataset_name),
        'query_key': str(args.query_key),
        'gt_key': str(args.gt_key),
        'num_queries_requested': int(args.num_queries),
        'num_queries_used': int(queries.shape[0]),
        'k': int(args.k),
        'efs': [int(ef) for ef in args.efs],
        'space': str(args.space),
        'dim': int(args.dim),
        'num_threads': int(args.num_threads),
        'backends': [spec.__dict__ | {'index_path': str(spec.index_path), 'sweep_csv': str(spec.sweep_csv)} for spec in specs],
    }
    (args.out_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    print(f'[DONE] wrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
