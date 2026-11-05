#!/usr/bin/env python3
"""Validate five-run measurements and export robustness curves and matched recall."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, values):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def pareto(points):
    return [point for point in points if not any(
        float(other['recall']) >= float(point['recall']) and
        float(other['batch_mean_seconds']) <= float(point['batch_mean_seconds']) and
        (float(other['recall']) > float(point['recall']) or
         float(other['batch_mean_seconds']) < float(point['batch_mean_seconds']))
        for other in points)]


def matched_time(points, target):
    frontier = sorted(pareto(points), key=lambda point: float(point['recall']))
    for point in frontier:
        if math.isclose(float(point['recall']), target, abs_tol=1e-12, rel_tol=0):
            return float(point['batch_mean_seconds']), 'measured', point['ef_init'], point['ef_init']
    for left, right in zip(frontier, frontier[1:]):
        low, high = float(left['recall']), float(right['recall'])
        if low < target < high:
            fraction = (target - low) / (high - low)
            seconds = float(left['batch_mean_seconds']) + fraction * (
                float(right['batch_mean_seconds']) - float(left['batch_mean_seconds']))
            return seconds, 'linear_interpolation', left['ef_init'], right['ef_init']
    return None, 'unbracketed', '', ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    completion = json.loads((args.run / 'complete.json').read_text())
    if completion.get('passed') is not True:
        raise ValueError('Run is not complete')
    summary, raw = rows(args.run / 'summary.csv'), rows(args.run / 'raw_timings.csv')
    if not summary:
        raise ValueError('No measured curves')
    matched, curves = [], []
    for dataset in completion['datasets']:
        points = [row for row in summary if row['dataset'] == dataset]
        for point in points:
            calls = [row for row in raw if (row['dataset'], row['method'], row['ef_init']) ==
                     (dataset, point['method'], point['ef_init'])]
            if len(calls) != 5 or {row['repeat'] for row in calls} != {'1', '2', '3', '4', '5'}:
                raise ValueError(f'{dataset}: expected five distinct measured repetitions')
            seconds = [float(row['seconds']) for row in calls]
            if not math.isclose(statistics.mean(seconds), float(point['batch_mean_seconds']), rel_tol=1e-10):
                raise ValueError(f'{dataset}: timing mean does not match raw measurements')
            if not math.isclose(statistics.stdev(seconds), float(point['batch_stdev_seconds']), rel_tol=1e-10):
                raise ValueError(f'{dataset}: timing deviation does not match raw measurements')
            if int(point['repeats']) != 5:
                raise ValueError(f'{dataset}: invalid repeat count')
            if not math.isclose(int(point['hits']) / (int(point['query_count']) * int(point['k'])),
                                float(point['recall']), rel_tol=1e-10):
                raise ValueError(f'{dataset}: recall does not match measured hits')
        for method in sorted({point['method'] for point in points}):
            group = [point for point in points if point['method'] == method]
            frontier = pareto(group)
            for point in sorted(group, key=lambda row: int(row['ef_init'])):
                curves.append(dict(point, pareto=int(point in frontier)))
        stars = [point for point in points if point['method'] != 'vanilla' and
                 point['recommended'].lower() == 'true']
        if len(stars) != 1:
            raise ValueError(f'{dataset}: expected one recommended SAGE point')
        star = stars[0]
        vanilla = [point for point in points if point['method'] == 'vanilla']
        seconds, kind, low, high = matched_time(vanilla, float(star['recall']))
        matched.append(dict(dataset=dataset, recommended_ef=star['ef_init'], recall=star['recall'],
            sage_batch_seconds=star['batch_mean_seconds'], vanilla_matched_seconds=seconds,
            speedup=seconds / float(star['batch_mean_seconds']) if seconds is not None else None,
            vanilla_comparison=kind, lower_ef=low, upper_ef=high))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'curves.csv', curves)
    write(args.output / 'matched_recall.csv', matched)
    (args.output / 'validation.json').write_text(json.dumps(dict(
        passed=True, datasets=completion['datasets'], measured_runs=5,
        measured_points_preserved=True, matched_recall_interpolation_labeled=True,
        extrapolation=False), indent=2) + '\n')
    print(f'Wrote {len(curves)} measured points and {len(matched)} matched-recall comparisons')


if __name__ == '__main__':
    main()
