#!/usr/bin/env python3
"""Replay all assignable tiers; report actual false-easy loss and tier-oracle cost."""
import argparse

import numpy as np

from detail_common import (add_inputs, candidate_tiers, hits, initial_budgets, load,
                           sage_work, save_csv, save_json, upper_hops, vanilla_work)


def classify(tiers, tier_hits, tier_steps, actual_hits, initial, final, sage_steps, k):
    tiers = np.asarray(tiers, np.int64)
    baseline = tier_hits[-1]
    enough = tier_hits >= baseline[None, :]
    if not enough[-1].all() or not np.isin(final, tiers).all():
        raise ValueError('Invalid complete tier matrix or final budgets')
    position = np.argmax(enough, axis=0)
    oracle = tiers[position]
    oracle_steps = tier_steps[position, np.arange(len(final))]
    ever_reduced = initial < tiers[-1]
    final_reduced = final < tiers[-1]
    lost = baseline - actual_hits
    ever_fe = ever_reduced & (lost > 0)
    actual_fe = final_reduced & (lost > 0)
    promoted_fe = ever_fe & ~final_reduced
    tier_fe, tier_fh = final < oracle, final > oracle
    binary_fh = (final == tiers[-1]) & (oracle < tiers[-1])
    extra = np.maximum(sage_steps - oracle_steps, 0)
    def pct(mask, denominator):
        return 100 * int(mask.sum()) / denominator if denominator else None
    summary = dict(all_queries=len(final), reduced_queries=int(final_reduced.sum()),
        false_easy_queries=int(actual_fe.sum()), false_easy_percent_reduced=pct(actual_fe, int(final_reduced.sum())),
        recall_drop_01_queries=int((actual_fe & (lost == 1)).sum()) if k == 10 else None,
        recall_drop_02_queries=int((actual_fe & (lost == 2)).sum()) if k == 10 else None,
        recall_drop_le_02_percent=pct(actual_fe & (lost / k <= .2), int(actual_fe.sum())),
        ever_reduced_queries=int(ever_reduced.sum()),
        ever_reduced_false_easy_queries=int(ever_fe.sum()),
        promoted_full_budget_false_easy_queries=int(promoted_fe.sum()),
        tier_false_easy_queries=int(tier_fe.sum()), tier_false_hard_queries=int(tier_fh.sum()),
        full_budget_false_hard_queries=int(binary_fh.sum()),
        full_budget_false_hard_percent_all=pct(binary_fh, len(final)),
        full_budget_false_hard_extra_steps=int(extra[binary_fh].sum()),
        full_budget_false_hard_extra_share_sage_steps_pct=(100 * int(extra[binary_fh].sum()) / int(sage_steps.sum())) if sage_steps.sum() else None,
        nonmonotone_tier_recall_queries=int((np.diff(tier_hits, axis=0) < 0).any(axis=0).sum()))
    rows = [dict(query_id=q, initial_ef=int(initial[q]), final_ef=int(final[q]),
        oracle_ef=int(oracle[q]), baseline_hits=int(baseline[q]), sage_hits=int(actual_hits[q]),
        lost_hits=int(lost[q]), ever_reduced=bool(ever_reduced[q]), final_reduced=bool(final_reduced[q]),
        false_easy=bool(actual_fe[q]), ever_reduced_false_easy=bool(ever_fe[q]),
        promoted_full_budget_false_easy=bool(promoted_fe[q]),
        tier_false_easy=bool(tier_fe[q]), tier_false_hard=bool(tier_fh[q]),
        full_budget_false_hard=bool(binary_fh[q]), sage_steps=int(sage_steps[q]),
        oracle_steps=int(oracle_steps[q]), excess_steps=int(extra[q])) for q in range(len(final))]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser, 'misclassification')
    args = parser.parse_args()
    data = load(args)
    if data is None:
        return
    index, queries, truth, policy = data
    k = policy['k']
    tiers = candidate_tiers(policy)
    upper = upper_hops(index, queries)
    sage = sage_work(index, queries, policy, args.threads, upper)
    initial = initial_budgets(policy, sage)
    tier_hits, tier_steps = [], []
    for ef in tiers:
        values = vanilla_work(index, queries, k, ef, args.threads, upper)
        values['hits'] = hits(values['labels'], truth)
        tier_hits.append(values['hits'])
        tier_steps.append(values['steps'])
        np.savez_compressed(args.output / f'vanilla_ef{ef}.npz', **values)
    sage['hits'] = hits(sage['labels'], truth)
    np.savez_compressed(args.output / 'sage.npz', **sage, initial_efs=initial)
    summary, rows = classify(tiers, np.array(tier_hits), np.array(tier_steps), sage['hits'],
        initial, sage['final_efs'], sage['steps'], k)
    save_csv(args.output / 'per_query.csv', rows)
    save_json(args.output / 'summary.json', dict(**summary, candidate_tiers=tiers,
        k=k, recommended_ef=policy['ef_init'], measured_latency=False,
        oracle='Minimum measured assignable Vanilla tier reaching recommended-ef Vanilla hits; no monotonicity assumption',
        false_easy='Final budget is below recommended ef and actual SAGE hits are below Vanilla at recommended ef',
        reduced_queries_definition='Final budget below recommended ef; ever-reduced and promoted-full losses reported separately',
        binary_false_hard='Final budget equals recommended ef, but a smaller Vanilla tier preserves recall',
        initial_budget_source='Native CFRs and shadow-route flag; final budget checked against one-tier promotion'))


if __name__ == '__main__':
    main()
