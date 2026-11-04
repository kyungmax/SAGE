#!/usr/bin/env python3
"""Main-six launch interface; worker integration is recorded explicitly in config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from main6_common import RUN_ROOT, effective_config, resolve_datasets

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'config/main6_online24'
OUTPUT=RUN_ROOT
CELLS=('faiss','hnswlib','pip','abs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run-all','run-cell'])
    parser.add_argument('--cell',choices=CELLS)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='Show the plan; create no results and run no search.')
    mode.add_argument('--preflight',action='store_true',help='Check metadata/binaries only; run no search.')
    parser.add_argument('--k', choices=['10', '100', 'all'], default='all')
    args=parser.parse_args()
    if (args.command=='run-cell') != bool(args.cell):
        parser.error('Use --cell with run-cell only.')
    config=json.loads((CONFIG/'experiment.json').read_text())
    cells=list(config['run_order']) if args.command=='run-all' else [args.cell]
    workers=config.get('cell_workers',{})
    cell_pending=config.get('cell_work_remaining',{})
    jobs=[]
    ks=config.get('k_values',[config['k']]) if args.k=='all' else [int(args.k)]
    for k in ks:
        scoped=effective_config(CONFIG/'experiment.json',k)
        inputs=resolve_datasets(CONFIG/'datasets.json',scoped)
        for cell in cells:
            worker=workers.get(cell)
            worker_path=None if worker is None else ROOT/worker
            target=OUTPUT/f'k{k}/faiss/targets.json' if cell in ('pip','abs') else None
            jobs.append(dict(cell=cell,k=k,output=str(OUTPUT/f'k{k}'/cell),datasets=scoped['datasets'],
                methods=[cell] if cell in ('pip','abs') else
                    [m for m in scoped['methods'][cell] if m not in ('pip','abs')],
                shadow_enabled=k!=100,
                inputs_ready=all(Path(d['dataset_path']).is_file() and
                    Path(d[('hnswlib' if cell=='hnswlib' else 'faiss')+'_index']['path']).is_file()
                    for d in inputs),
                worker=None if worker_path is None else str(worker_path),
                connected=worker_path is not None and worker_path.is_file(),
                targets=None if target is None else str(target),
                targets_ready=target is None or target.is_file(),
                target_produced_by_this_plan=target is not None and 'faiss' in cells,
                implementation_work_remaining=cell_pending.get(cell,[]),
                depends_on='new FAISS recommended-ef SAGE result for this k' if target else None))
    if args.preflight:
        return subprocess.run([sys.executable,str(ROOT/'preflight_main6_online24.py')]).returncode
    pending=[item for job in jobs for item in job['implementation_work_remaining']]
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs,k_values=ks,threads=config['online_threads'],
            repeats=config['timed_repeats'],reuse_old_results=False,
            workers_connected=all(j['connected'] for j in jobs),
            execution_authorized=config.get('execution_authorized',False),
            pre_run_builds=config.get('pre_run_builds',[]),
            ready_to_run=config.get('execution_authorized',False) and all(j['connected'] and j['inputs_ready'] and (j['targets_ready'] or j['target_produced_by_this_plan']) for j in jobs) and not pending,
            implementation_work_remaining=pending,no_ann_run=True),indent=2))
        return 0
    if not config.get('execution_authorized',False):
        parser.error('Final user approval is required before builds or measurement. No experiment started.')
    missing=[j['cell'] for j in jobs if not j['connected']]
    if missing or pending:
        print('The requested main6 cells still need measurement-worker integration.',file=sys.stderr)
        print('No calibration, tuning or search was started. Use --dry-run or --preflight to inspect preparation.',file=sys.stderr)
        if missing:print('Workers awaiting connection: '+', '.join(missing),file=sys.stderr)
        for item in pending:print('- '+item,file=sys.stderr)
        return 2
    # Once connected, each cell must produce new outputs. Earlier reference
    # data and existing result directories are never used as a cache.
    for j in jobs:
        if Path(j['output']).exists():
            parser.error(f"New measurement requires an unused output directory: {j['output']}")
    for j in jobs:
        argv=[sys.executable,j['worker'],'--config',str(CONFIG/'experiment.json'),
              '--datasets-config',str(CONFIG/'datasets.json'),'--output',j['output'],'--k',str(j['k'])]
        if j['cell'] in ('pip','abs'):
            target=Path(j['targets'])
            if not target.is_file():
                parser.error('PiP/ABS require the new FAISS target file from this main6 run. Run run_faiss.sh first.')
            argv+=['--targets',str(target),'--method',j['cell']]
        else:
            argv+=['--backend',j['cell']]
        result=subprocess.run(argv,cwd=ROOT)
        if result.returncode:return result.returncode
    return 0


if __name__=='__main__':raise SystemExit(main())
