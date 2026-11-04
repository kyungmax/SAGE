"""Recall matching for the main experiment; no dataset paths or native imports.

ABS follows R12 worker.tune; PiP uses paper set overlap and positive Delta.
Selection uses evaluation ground truth retrospectively, not held-out calibration.
"""
import json
import random
import statistics


def ef_grid(config, target):
    cap = max(config['online']['initial_curve_cap'], target['offline_cap'],
              target['recommended_ef'])
    values = {config['k'], cap, target['recommended_ef'],
              *config['online']['low_ef_extra']}
    low = config['calibration']['offline_min_ef']
    steps = config['calibration']['steps_per_doubling']
    while low <= cap:
        step = max(1, low // steps)
        if 'high_ef_grid_step' in config['calibration']:
            step = min(step, config['calibration']['high_ef_grid_step'])
        values.update(range(low, min(2 * low, cap) + 1, step))
        low *= 2
    return sorted(v for v in values if v >= config['k'])


def same_result(actual, expected):
    for name in ('hits', 'labels_sha256', 'distances_sha256'):
        if actual[name] != expected[name]:
            raise ValueError(f'Non-reproducible search output: {name}')


class Evaluator:
    def __init__(self, call, record):
        self.call, self.record = call, record
        self.cache = {}

    def evaluate(self, method, **params):
        key = (method, json.dumps(params, sort_keys=True))
        if key not in self.cache:
            result = self.call(method, params)
            self.cache[key] = result
            self.record(dict(phase='tuning', **result))
        return self.cache[key]

    @property
    def rows(self):
        return list(self.cache.values())


def inversions(rows, parameter):
    ordered = sorted(rows, key=lambda r: r['config'][parameter])
    return [dict(lower=a['config'], lower_hits=a['hits'],
                 upper=b['config'], upper_hits=b['hits'])
            for a, b in zip(ordered, ordered[1:]) if a['hits'] > b['hits']]


def tune_abs(call, target_hits, settings, record):
    evaluator = Evaluator(call, record)
    scale = settings['gamma_resolution_denominator']
    lo, hi, cap = 0, settings['initial_gamma_tick'], settings['max_gamma_tick']
    if not (scale > 0 and 0 < hi <= cap):
        raise ValueError('Invalid ABS gamma bracket')

    def evaluate(tick):
        return evaluator.evaluate('abs', gamma=tick / scale)

    low = evaluate(lo)
    high = evaluate(hi)
    while high['hits'] < target_hits and hi < cap:
        lo, hi = hi, min(hi * 2, cap)
        high = evaluate(hi)
    if low['hits'] >= target_hits:
        hi = 0
    elif high['hits'] >= target_hits:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if evaluate(mid)['hits'] >= target_hits:
                hi = mid
            else:
                lo = mid
    radius = settings['local_radius_ticks']
    for tick in range(max(0, hi - radius), min(cap, hi + radius) + 1):
        evaluate(tick)
    passing = [r for r in evaluator.rows if r['hits'] >= target_hits]
    if not passing:
        raise ValueError(f'ABS did not reach target {target_hits} within gamma <= {cap / scale}')
    return dict(selected=min(passing, key=lambda r: r['config']['gamma']),
                candidates=passing, observed_recall_inversions=inversions(evaluator.rows, 'gamma'),
                rule='smallest tested passing gamma; bracket refinement and local checks')


def tune_pip(call, target_hits, grid, ntotal, settings, record, warmups, repeats):
    evaluator = Evaluator(call, record)
    candidates, observed_inversions = [], []
    cap = min(2**31 - 1, 2 * ntotal)
    threshold = settings['threshold']
    if threshold != 0.95:
        raise ValueError('The retained PiP protocol fixes saturation threshold at 0.95')
    for ef in grid:
        ceiling = evaluator.evaluate('vanilla', ef_search=ef)
        if ceiling['hits'] < target_hits:
            continue

        def evaluate(patience):
            return evaluator.evaluate('pip', ef_search=ef, patience=patience,
                                      threshold=threshold)

        if evaluate(1)['hits'] >= target_hits:
            hi = 1
        else:
            lo, hi = 1, min(max(8, ef), cap)
            high = evaluate(hi)
            while high['hits'] < target_hits and hi < cap:
                lo, hi = hi, min(hi * 2, cap)
                high = evaluate(hi)
            if high['hits'] >= target_hits:
                while hi - lo > 1:
                    mid = (hi + lo) // 2
                    if evaluate(mid)['hits'] >= target_hits:
                        hi = mid
                    else:
                        lo = mid
        radius = settings['local_radius_patience']
        for patience in range(max(1, hi - radius), min(cap, hi + radius) + 1):
            evaluate(patience)
        rows = [r for r in evaluator.rows if r['method'] == 'pip'
                and r['config']['ef_search'] == ef]
        observed_inversions.extend(inversions(rows, 'patience'))
        passing = [r for r in rows if r['hits'] >= target_hits]
        if passing:
            candidates.append(dict(min(passing, key=lambda r: r['config']['patience'])))
    if not candidates:
        raise ValueError('PiP did not reach the target within the efSearch/patience search range')

    # These repetitions select the setting. Final timing runs in another process.
    rng = random.Random(settings['selection_seed'])
    samples = [[] for _ in candidates]
    for repeat in range(warmups + repeats):
        order = list(range(len(candidates)))
        rng.shuffle(order)
        for position, number in enumerate(order):
            candidate = candidates[number]
            result = call('pip', candidate['config'])
            same_result(result, candidate)
            record(dict(phase='selection_warmup' if repeat < warmups else 'selection_timed',
                        repeat=repeat, order=position, **result))
            if repeat >= warmups:
                samples[number].append(result['seconds'])
    for candidate, times in zip(candidates, samples):
        candidate['selection_seconds'] = times
        candidate['selection_mean_seconds'] = statistics.mean(times)
    return dict(selected=min(candidates, key=lambda r: r['selection_mean_seconds']),
                candidates=candidates, observed_recall_inversions=observed_inversions,
                rule=f'fastest mean of {repeats} selection repeats among tested passing patience boundaries')
