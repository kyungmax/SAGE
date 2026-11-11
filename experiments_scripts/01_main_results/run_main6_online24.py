#!/usr/bin/env python3
"""Run the paper's main curves and matched-recall PiP/ABS measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from main6_common import (ROOT, RUN_ROOT, effective_config, resolve_datasets,
                         validate_policy_config, validate_timing_config)

CONFIG = ROOT / 'config/main6_online24'
CELLS = ('faiss', 'hnswlib', 'pip', 'abs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run-all', 'run-cell'])
    parser.add_argument('--cell', choices=CELLS)
    parser.add_argument('--config', type=Path, default=CONFIG / 'experiment.json')
    parser.add_argument('--datasets-config', type=Path, default=CONFIG / 'datasets.json')
    parser.add_argument('--output', type=Path, default=RUN_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print the plan without creating outputs.')
    mode.add_argument('--preflight', action='store_true', help='Inspect inputs without running ANN.')
    parser.add_argument('--k', choices=['10', '100', 'all'], default='10')
    args = parser.parse_args()
    if (args.command == 'run-cell') != bool(args.cell):
        parser.error('Use --cell with run-cell only.')
    for key in ('config', 'datasets_config', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    config = json.loads(args.config.read_text())
    validate_policy_config(config)
    validate_timing_config(config)
    cells = config['run_order'] if args.command == 'run-all' else [args.cell]
    ks = config['k_values'] if args.k == 'all' else [int(args.k)]
    if args.preflight:
        return subprocess.run([sys.executable, str(ROOT / 'preflight_main6_online24.py'),
            '--config', str(args.config), '--datasets-config', str(args.datasets_config),
            '--k', args.k, '--cells', ','.join(cells),
            '--output', str(args.output / 'preparation/main6_online24/preflight.json')]).returncode
    jobs = []
    for k in ks:
        scoped = effective_config(args.config, k)
        inputs = resolve_datasets(args.datasets_config, scoped)
        for cell in cells:
            worker = ROOT / config['cell_workers'][cell]
            target = args.output / f'k{k}/faiss/targets.json' if cell in ('pip', 'abs') else None
            backend = 'hnswlib' if cell == 'hnswlib' else 'faiss'
            jobs.append(dict(cell=cell, k=k, worker=str(worker), connected=worker.is_file(),
                output=str(args.output / f'k{k}' / cell), datasets=scoped['datasets'],
                inputs_ready=all(Path(d['dataset_path']).is_file() and
                    Path(d[backend + '_index']['path']).is_file() for d in inputs),
                targets=str(target) if target else None,
                targets_ready=target is None or target.is_file(),
                target_produced_by_this_plan=target is not None and 'faiss' in cells))
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs, threads=config['online_threads'],
            warmups=config['warmups_per_setting'], measured_runs=config['timed_repeats'],
            comparison='matched_recall', no_ann_run=True,
            ready_to_run=all(j['connected'] and j['inputs_ready'] and
                (j['targets_ready'] or j['target_produced_by_this_plan']) for j in jobs)), indent=2))
        return 0
    for job in jobs:
        if not job['connected'] or not job['inputs_ready']:
            parser.error(f"Missing worker or data/index inputs for {job['cell']}; inspect --dry-run.")
        if Path(job['output']).exists():
            parser.error(f"Use an unused output directory: {job['output']}")
    for job in jobs:
        argv = [sys.executable, job['worker'], '--config', str(args.config),
            '--datasets-config', str(args.datasets_config), '--output', job['output'], '--k', str(job['k'])]
        if job['targets']:
            if not Path(job['targets']).is_file():
                parser.error('Run the FAISS cell first to produce matched-recall targets.')
            argv += ['--targets', job['targets'], '--method', job['cell']]
        else:
            argv += ['--backend', job['cell']]
        result = subprocess.run(argv, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
