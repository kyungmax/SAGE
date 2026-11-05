#!/usr/bin/env python3
"""Reproduce recommendation quality/cost, seed stability, and pseudo-GT accuracy."""
from __future__ import annotations

import argparse
import json
import statistics
import time

from calibration_common import (
    add_inputs, hidden, hits, load_inputs, load_online, measure_jobs, save,
    validate_args, write_csv,
)


def default_calibration(index, args, folder):
    from experiments_scripts.common.final_policy import calibrate_policy

    costs, reference = [], None
    for repeat in range(args.warmup_runs + args.measured_runs):
        policy = calibrate_policy(index, k=10, backend='faiss', num_threads=args.threads,
                                  min_ef=64, shadow_enabled=True)
        if reference is not None:
            for field in ('ef_init', 'probe_ids', 'main', 'shadow', 'cap_history'):
                if policy[field] != reference[field]:
                    raise RuntimeError(f'Calibration changed across repeats: {field}')
        reference = policy
        costs.append(dict(repeat=repeat, phase='warmup' if repeat < args.warmup_runs else 'measured',
                          **policy['offline_timings']))
        save(folder / 'calibration' / f'repeat{repeat}.json', policy)
        write_csv(folder / 'offline_timings.csv', costs)
        print(json.dumps(dict(stage='calibration', repeat=repeat, ef=policy['ef_init'])), flush=True)
    save(folder / 'policy.json', reference)
    measured = [r for r in costs if r['phase'] == 'measured']
    return reference, {key: statistics.mean(r[key] for r in measured)
                       for key in ('Samp.', 'Select.', 'Eval.', 'total_seconds')}


def stability_choices(index, args, folder):
    import numpy as np
    from experiments_scripts.common.final_policy import _recommend_ef
    from experiments_scripts.common.offline_calibration import compute_fixed_calibration_lid_pool
    from experiments_scripts.common.projected_local_acceptable_runtime import _select_dummy_queries

    choices = []
    for seed in range(args.seed_count):
        pool = compute_fixed_calibration_lid_pool(index, internal_lid_k=15,
            num_nodes=index.get_current_count(), lid_sample_seed=seed, num_threads=args.threads,
            dataset_name=args.dataset, lid_search_ef=512)
        valid = pool[np.isfinite(pool.lid) & (pool.lid > 0)].copy()
        valid['query_source'] = 'train'
        selected = _select_dummy_queries(lid_df=valid, num_calibration_queries=500,
            trim_low_percentile=1., trim_high_percentile=99.)
        lid_ids = selected.query_id.to_numpy(np.int64)
        # Preserve the independent random stream from the original stability study.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31007]))
        random_ids = rng.choice(index.get_current_count(), size=1600, replace=False).astype(np.int64)[:500]
        dest = folder / 'stability' / f'seed{seed:02d}'
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'selection.npz', pool_ids=pool.query_id.to_numpy(np.int64),
                            pool_lids=pool.lid.to_numpy(np.float32), lid_ids=lid_ids, random_ids=random_ids)
        for method in ('lid', 'random') if seed % 2 == 0 else ('random', 'lid'):
            ids = lid_ids if method == 'lid' else random_ids
            vectors = np.ascontiguousarray(index.get_items(ids), dtype=np.float32)
            truth = hidden(index, vectors, ids, 4096, args.threads)
            counted = {}

            def measure(ef):
                if ef not in counted:
                    counted[ef] = hits(hidden(index, vectors, ids, ef, args.threads), truth)
                return int(counted[ef].sum())

            ef, history = _recommend_ef(measure, total_hits=5000, min_ef=64, k=10)
            choice = dict(dataset=args.dataset, seed=seed, method=method, ef=ef,
                          cap=history[-1]['cap'], cap_history=history)
            choices.append(choice)
            save(dest / f'{method}.json', choice)
            np.savez_compressed(dest / f'{method}.npz', probe_ids=ids, pseudo_labels=truth,
                                **{f'hits_ef{e}': values for e, values in counted.items()})
            print(json.dumps(dict(stage='stability', seed=seed, method=method, ef=ef)), flush=True)
    save(folder / 'offline_choices.json', choices)
    return choices


def exact_topk(index, vectors, ids, *, chunk_size, k=10):
    """Brute force over stored vectors in bounded chunks, with each self ID hidden."""
    import faiss
    import numpy as np

    vectors = index._prepare_vectors(vectors)
    metric = faiss.METRIC_L2 if index.space == 'l2' else faiss.METRIC_INNER_PRODUCT
    best_values = np.full((len(ids), k), np.inf, dtype=np.float32)
    best_ids = np.full((len(ids), k), -1, dtype=np.int64)
    for start in range(0, index.get_current_count(), chunk_size):
        stop = min(start + chunk_size, index.get_current_count())
        block = np.ascontiguousarray(index.get_items(np.arange(start, stop, dtype=np.int64)))
        values, labels = faiss.knn(vectors, block, min(k + 1, len(block)), metric=metric)
        labels += start
        if metric == faiss.METRIC_INNER_PRODUCT:
            values = -values
        values[labels == ids[:, None]] = np.inf
        merged_values = np.concatenate((best_values, values), axis=1)
        merged_ids = np.concatenate((best_ids, labels), axis=1)
        keep = np.argpartition(merged_values, k - 1, axis=1)[:, :k]
        best_values = np.take_along_axis(merged_values, keep, axis=1)
        best_ids = np.take_along_axis(merged_ids, keep, axis=1)
    order = np.argsort(best_values, axis=1, kind='stable')
    result = np.take_along_axis(best_ids, order, axis=1)
    if np.any(result < 0) or np.any(result == ids[:, None]):
        raise RuntimeError('Exact search did not return k valid nonself labels')
    return result


def pseudo_validation(index, policy, args, folder):
    import numpy as np

    ids = np.asarray(policy['probe_ids'], dtype=np.int64)
    vectors = np.ascontiguousarray(index.get_items(ids), dtype=np.float32)
    outputs, timings = {}, []
    for repeat in range(args.warmup_runs + args.measured_runs):
        for kind in ('pseudo', 'exact') if repeat % 2 == 0 else ('exact', 'pseudo'):
            start = time.perf_counter()
            labels = (hidden(index, vectors, ids, 4096, args.threads) if kind == 'pseudo'
                      else exact_topk(index, vectors, ids, chunk_size=args.exact_chunk_size))
            seconds = time.perf_counter() - start
            if kind in outputs:
                np.testing.assert_array_equal(labels, outputs[kind])
            outputs[kind] = labels
            timings.append(dict(kind=kind, repeat=repeat,
                phase='warmup' if repeat < args.warmup_runs else 'measured', seconds=seconds))
            write_csv(folder / 'pseudo_exact_timings.csv', timings)
            print(json.dumps(dict(stage='pseudo_gt', **timings[-1])), flush=True)
    np.savez_compressed(folder / 'groundtruth.npz', probe_ids=ids, query_vectors=vectors,
                        pseudo_labels=outputs['pseudo'], exact_labels=outputs['exact'])
    summary = dict(recall=float(hits(outputs['pseudo'], outputs['exact']).sum() / (len(ids) * 10)),
                   exact_source='full stored index vectors; brute force; self hidden',
                   measured_runs=args.measured_runs, warmup_runs=args.warmup_runs)
    for kind in ('pseudo', 'exact'):
        samples = [r['seconds'] for r in timings if r['kind'] == kind and r['phase'] == 'measured']
        summary[kind + '_seconds'] = statistics.mean(samples)
    save(folder / 'pseudo_exact.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('quality', 'stability', 'pseudo-gt', 'all'), default='all')
    add_inputs(parser, output_name='03_calibration')
    parser.add_argument('--seed-count', type=int, default=10, help='Independent sampling seeds, separate from timing repeats')
    parser.add_argument('--exact-chunk-size', type=int, default=50000)
    args = parser.parse_args()
    validate_args(args)
    if args.seed_count < 2 or args.exact_chunk_size < 1:
        parser.error('seed-count must be >=2 and exact-chunk-size must be positive')
    cfg, index = load_inputs(args)
    folder = args.output_root / args.dataset / args.experiment
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'inputs.json', dict(dataset=cfg, threads=args.threads,
        measured_runs=args.measured_runs, warmup_runs=args.warmup_runs, seed_count=args.seed_count))
    summary, policy, choices = {}, None, []
    if args.experiment in ('quality', 'pseudo-gt', 'all'):
        policy, costs = default_calibration(index, args, folder)
        summary.update(recommended_ef=policy['ef_init'], offline_cap=policy['cap_history'][-1]['cap'],
                       calibration_seconds=costs)
    if args.experiment in ('stability', 'all'):
        choices = stability_choices(index, args, folder)
    if args.experiment in ('pseudo-gt', 'all'):
        summary['pseudo_gt'] = pseudo_validation(index, policy, args, folder)
    if args.experiment in ('quality', 'stability', 'all'):
        # Online queries enter only after calibration decisions are frozen.
        queries, truth = load_online(cfg)
        efs = {c['ef'] for c in choices}
        if policy is not None:
            efs.update((policy['ef_init'], policy['cap_history'][-1]['cap']))
        jobs = [f'ef{e}' for e in sorted(efs)]

        def invoke(name):
            index.set_ef(int(name[2:]))
            return index.knn_query(queries, k=10, num_threads=args.threads)

        online = measure_jobs(jobs, invoke, truth, args, folder)
        summary['online'] = online
        if policy is not None:
            recommended = online[f"ef{policy['ef_init']}"]['recall']
            at_cap = online[f"ef{policy['cap_history'][-1]['cap']}"]['recall']
            summary.update(recall_recommended=recommended, recall_cap=at_cap, recall_gap=at_cap - recommended)
        if choices:
            rows = [dict(dataset=args.dataset, method=c['method'], seed=c['seed'], ef=c['ef'],
                         **online[f"ef{c['ef']}"]) for c in choices]
            write_csv(folder / 'stability_per_seed.csv', rows)
            summary['stability'] = {method: dict(
                seed_count=args.seed_count,
                ef_sample_sd=statistics.stdev(r['ef'] for r in rows if r['method'] == method),
                recall_pp_sample_sd=100 * statistics.stdev(r['recall'] for r in rows if r['method'] == method),
                latency_ms_sample_sd=statistics.stdev(r['latency_ms'] for r in rows if r['method'] == method))
                for method in ('random', 'lid')}
    save(folder / 'summary.json', summary)
    print(json.dumps(dict(output=str(folder / 'summary.json'))), flush=True)


if __name__ == '__main__':
    main()
