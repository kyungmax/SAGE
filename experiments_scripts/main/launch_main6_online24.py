#!/usr/bin/env python3
"""Detach the approved experiment and persist queue, progress and exit state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from main6_common import ROOT, RUN_ROOT, SAGE_ROOT

PREPARATION = RUN_ROOT / 'preparation/main6_online24'
CONTROL = PREPARATION / 'active_run'
CONFIG = ROOT / 'config/main6_online24/experiment.json'
OUTPUT = RUN_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def siblings(cpus):
    found = set()
    for cpu in cpus:
        tokens = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(',')
        for token in tokens:
            ends = list(map(int, token.split('-')))
            found.update(range(ends[0], ends[-1] + 1))
    return sorted(found)


def cpu_ticks():
    return {int(parts[0][3:]): list(map(int, parts[1:9]))
            for line in Path('/proc/stat').read_text().splitlines()
            if (parts := line.split()) and parts[0].startswith('cpu') and parts[0][3:].isdigit()}


def utilization(before, after, cpus, physical_count):
    busy = []
    for cpu in cpus:
        delta = [b - a for a, b in zip(before[cpu], after[cpu])]
        busy.append(100 * (1 - (delta[3] + delta[4]) / max(1, sum(delta))))
    return dict(physical_capacity_busy_percent=sum(busy) / physical_count,
                busiest_logical_cpu_percent=max(busy), watched_cpus=cpus)


def progress():
    completed, active = [], []
    for k in (10, 100):
        for method in ('faiss', 'hnswlib', 'pip', 'abs'):
            folder = OUTPUT / f'k{k}' / method
            if (folder / 'complete.json').exists():
                completed.append(f'k{k}/{method}')
            elif folder.exists():
                statuses = list(folder.glob('datasets/*/status.json'))
                newest = max(statuses, key=lambda p: p.stat().st_mtime_ns) if statuses else None
                item = dict(cell=f'k{k}/{method}')
                if newest is not None:
                    item.update(dataset=newest.parent.name, detail=read(newest))
                else:
                    logs = list((folder / 'logs').glob('*.log'))
                    if logs:
                        item['latest_log'] = str(max(logs, key=lambda p: p.stat().st_mtime_ns))
                active.append(item)
    return dict(completed_cells=completed, active_cells=active)


def supervise():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    launch = read(CONTROL / 'launch.json')
    config = read(CONFIG)
    if not config.get('execution_authorized') or sha(CONFIG) != launch['config_sha256']:
        raise RuntimeError('Approved configuration changed before supervisor started')
    run = dict(supervisor_pid=os.getpid(), session_id=os.getsid(0),
               started_at=time.time(), log=str(CONTROL / 'run.log'))

    def state(stage, **details):
        run.update(stage=stage, updated_at=time.time(), **details)
        save(CONTROL / 'status.json', run)

    try:
        watched = siblings(config['timing']['cpus'])
        quiet = 0
        state('waiting_for_quiet_cpus', numa_node=config['timing']['numa_node'],
              cpus=config['timing']['cpus'], benchmark_started=False, stable_seconds=0)
        print('Waiting for 30 stable seconds on the selected physical cores and SMT siblings.', flush=True)
        while quiet < 30:
            before = cpu_ticks()
            time.sleep(10)
            load = utilization(before, cpu_ticks(), watched, len(config['timing']['cpus']))
            idle = load['physical_capacity_busy_percent'] < 5 and load['busiest_logical_cpu_percent'] < 80
            quiet = quiet + 10 if idle else 0
            state('waiting_for_quiet_cpus', stable_seconds=quiet, cpu_load=load)
        if sha(CONFIG) != launch['config_sha256']:
            raise RuntimeError('Approved configuration changed while queued')
        # The protocol remains frozen throughout all k/backend cells. Mutable
        # execution state belongs here, never in the hashed protocol JSON.
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        argv = ['bash', str(ROOT / 'main6_online24/run_all.sh')]
        child = subprocess.Popen(argv, cwd=SAGE_ROOT, stdin=subprocess.DEVNULL, env=env)
        state('running', benchmark_started=True, controller_pid=child.pid, **progress())
        print(f'Experiment controller started: PID {child.pid}', flush=True)
        while True:
            try:
                code = child.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                state('running', **progress())
        latest = progress()
        if code != 0:
            state('failed', returncode=code, **latest)
            raise RuntimeError(f'Experiment controller exited {code}; inspect the active cell logs')
        if len(latest['completed_cells']) != 8:
            raise RuntimeError('Controller exited without completing all eight k/backend cells')
        state('complete', returncode=0, finished_at=time.time(), **latest)
    except BaseException as error:
        state('failed', error=repr(error), **progress())
        raise


def launch():
    config = read(CONFIG)
    if not config.get('execution_authorized'):
        raise RuntimeError('Final user approval is required')
    if CONTROL.exists():
        raise FileExistsError(f'A launch record already exists: {CONTROL}')
    subprocess.run([sys.executable, str(ROOT / 'preflight_main6_online24.py')], check=True)
    if not read(PREPARATION / 'preflight.json')['ready_to_run']:
        raise RuntimeError('Preflight is not ready')
    CONTROL.mkdir()
    for source in (CONFIG, ROOT / 'config/main6_online24/datasets.json', PREPARATION / 'preflight.json'):
        shutil.copy2(source, CONTROL / source.name)
    info = dict(created_at=time.time(), config_sha256=sha(CONFIG),
        datasets_config_sha256=sha(ROOT / 'config/main6_online24/datasets.json'),
        k_values=config['k_values'], datasets=config['datasets'], timing=config['timing'],
        session_independent=True, automatic_restart=False, log=str(CONTROL / 'run.log'))
    save(CONTROL / 'launch.json', info)
    with (CONTROL / 'run.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervise'],
            cwd=SAGE_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    info['supervisor_pid'] = process.pid
    save(CONTROL / 'launch.json', info)
    print(json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        supervise()
    else:
        launch()


if __name__ == '__main__':
    main()
