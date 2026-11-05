#!/usr/bin/env python3
"""Compare SAGE with the vanilla curve and retain measured baseline recall."""
import argparse
import csv
import math
from pathlib import Path


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def vanilla_match(points, target):
    by_hits = {}
    for point in sorted(points, key=lambda r: (float(r['batch_mean_seconds']),
                                              str(r.get('ef_init', r.get('ef', ''))))):
        by_hits.setdefault(int(point['hits']), point)
    frontier = sorted((point for hits, point in by_hits.items() if not any(
        other_hits > hits and float(other['batch_mean_seconds']) <= float(point['batch_mean_seconds'])
        for other_hits, other in by_hits.items())), key=lambda r: int(r['hits']))
    for point in frontier:
        if int(point['hits']) == target:
            return float(point['batch_mean_seconds']), 'measured', point, point
    for left, right in zip(frontier, frontier[1:]):
        low, high = int(left['hits']), int(right['hits'])
        if low < target < high:
            fraction = (target - low) / (high - low)
            seconds = float(left['batch_mean_seconds']) + fraction * (
                float(right['batch_mean_seconds']) - float(left['batch_mean_seconds']))
            return seconds, 'linear_interpolation', left, right
    return None, 'unbracketed', None, None


def matched_rows(rows):
    groups = {}
    for row in rows:
        backend = row.get('reference_backend') or row.get('backend') or 'faiss'
        key = (row['dataset'], backend, int(row['k']))
        groups.setdefault(key, []).append(row)
    output = []
    for (dataset, backend, k), points in sorted(groups.items()):
        sages = [r for r in points if r['method'].startswith('sage') and
                 str(r.get('recommended', '')).lower() in ('true', '1')]
        if len(sages) != 1:
            raise ValueError(f'{dataset}/{backend}/k{k}: expected exactly one recommended SAGE point')
        sage = sages[0]
        target = int(sage['hits'])
        sage_seconds = float(sage['batch_mean_seconds'])
        query_count = int(sage['query_count'])
        if query_count <= 0 or k <= 0:
            raise ValueError(f'{dataset}/{backend}: invalid query count/k')
        target_recall = target / (query_count * k)
        for point in points:
            hits, seconds = int(point['hits']), float(point['batch_mean_seconds'])
            if int(point['query_count']) != query_count or not 0 <= hits <= query_count * k:
                raise ValueError(f'{dataset}/{backend}: inconsistent query count or invalid hits')
            if not math.isfinite(seconds) or seconds <= 0 or not math.isclose(
                    float(point['recall']), hits / (query_count * k), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'{dataset}/{backend}: invalid timing or recall/hits mismatch')
            if point.get('target_hits') and int(point['target_hits']) != target:
                raise ValueError(f'{dataset}/{backend}: baseline target differs from recommended SAGE hits')
        for method in sorted({r['method'] for r in points}):
            candidates = [r for r in points if r['method'] == method]
            lower = upper = None
            if method == 'vanilla':
                seconds, comparison, lower, upper = vanilla_match(candidates, target)
                selected = lower if comparison == 'measured' else None
                matched = seconds is not None
            else:
                eligible = [r for r in candidates if int(r['hits']) >= target]
                if method.startswith('sage'):
                    selected = sage if method == sage['method'] else None
                elif eligible:
                    selected = min(eligible, key=lambda r: float(r['batch_mean_seconds']))
                else:
                    selected = min(candidates, key=lambda r: (-int(r['hits']), float(r['batch_mean_seconds'])))
                seconds = float(selected['batch_mean_seconds']) if selected else None
                matched = selected is not None and int(selected['hits']) >= target
                comparison = 'measured_target_met' if matched else 'measured_below_target'
                if method.startswith('sage'):
                    comparison = 'recommended' if selected else 'not_recommended'
            output.append(dict(dataset=dataset, backend=backend, k=k, method=method,
                target_hits=target, target_recall=target_recall, matched=matched,
                comparison=comparison,
                target_met='' if selected is None else int(selected['hits']) >= target,
                selected_ef='' if selected is None else selected.get('ef_init', selected.get('ef_search', selected.get('ef', ''))),
                hits='' if selected is None else selected['hits'],
                recall=(target_recall if comparison == 'linear_interpolation' else
                        '' if selected is None else selected['recall']),
                batch_mean_seconds='' if seconds is None else seconds,
                method_seconds_over_sage='' if seconds is None else seconds / sage_seconds,
                lower_ef='' if lower is None else lower.get('ef_init', lower.get('ef', '')),
                upper_ef='' if upper is None else upper.get('ef_init', upper.get('ef', '')),
                lower_recall='' if lower is None else lower['recall'],
                upper_recall='' if upper is None else upper['recall']))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One k directory with backend/method subdirectories.')
    parser.add_argument('--summary', type=Path, action='append', default=[], help='Additional Ada-EF/DARTH summary CSV.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for cell in sorted(args.input.iterdir()):
        if not cell.is_dir():
            continue
        files = [cell / 'summary.csv'] if (cell / 'summary.csv').is_file() else sorted(cell.glob('datasets/*/summary.csv'))
        for path in files:
            rows.extend(read_rows(path))
    for path in args.summary:
        rows.extend(read_rows(path))
    if not rows:
        parser.error('No measured summaries found.')
    result = matched_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


if __name__ == '__main__':
    main()
