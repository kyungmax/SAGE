"""Fit the corrected DARTH trace using the October 29 holdout-training recipe."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

FEATURES = ['step', 'dists', 'inserts', 'first_nn_dist', 'nn_dist',
            'furthest_dist', 'avg_dist', 'variance', 'percentile_25',
            'percentile_50', 'percentile_75']


def _save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class _HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block


def _kill_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def train(*, dataset, trace_binary, index, data_dir, output, ef, target,
          queries=10000, threads=24, logging_interval=2, trace_timeout_seconds=86400):
    """Stream every training observation, fit the model and derive its intervals.

    The caller supplies affinity and a fresh output directory. The trace timeout
    covers the native process and pipe consumption; it does not limit model fit.
    Validation queries are not used for fitting or interval selection.
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    if min(queries, threads, ef, logging_interval) <= 0:
        raise ValueError('Query/thread counts, ef and logging interval must be positive')
    if not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError('Target recall must be in (0, 1]')
    if not math.isfinite(trace_timeout_seconds) or trace_timeout_seconds <= 0:
        raise ValueError('Trace timeout must be positive and finite')
    binary, index, data = (Path(path).resolve(strict=True) for path in
                           (trace_binary, index, data_dir))
    inputs = {'--index': index, '--query-file': data / 'learn.fvecs',
              '--gt-file': data / 'learn.groundtruth.ivecs',
              '--gt-dist-file': data / 'learn.groundtruth.fvecs'}
    sources = {str(path): _sha(path) for path in (binary, Path(__file__), *inputs.values())}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
               OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    env.pop('OMP_PLACES', None)
    command = [str(binary)]
    for name, path in inputs.items():
        command.extend((name, str(path)))
    command.extend(('--queries', str(queries), '--k', '10', '--ef', str(ef),
                    '--threads', str(threads), '--logging-interval', str(logging_interval)))
    _save(output / 'started.json', dict(dataset=dataset, command=command,
          trace_output='inherited pipe; raw CSV is hashed but not retained',
          source_input_sha256=sources, features=FEATURES, target=target, queries=queries,
          trace_timeout_seconds=trace_timeout_seconds,
          versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, pandas=pd.__version__),
          lightgbm_library_sha256=_sha(lgb.basic._LIB._name)))

    first_reaching = np.full(queries, np.inf)
    query_rows = np.zeros(queries, dtype=np.int64)
    feature_min, feature_max = np.full(len(FEATURES), np.inf), np.full(len(FEATURES), -np.inf)
    pieces_x, pieces_y = [], []
    trace_rows = selected_rows = 0
    read_fd = write_fd = process = timer = None
    timed_out = threading.Event()
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        def interrupted(signum, frame):
            raise KeyboardInterrupt('DARTH trace interrupted')
        previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        read_fd, write_fd = os.pipe()
        command.extend(('--output', f'/proc/self/fd/{write_fd}'))
        with (output / 'trace.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, pass_fds=(write_fd,), start_new_session=True)
            os.close(write_fd)
            write_fd = None

            def expire():
                timed_out.set()
                _kill_group(process, signal.SIGKILL)

            timer = threading.Timer(trace_timeout_seconds, expire)
            timer.daemon = True
            timer.start()
            with os.fdopen(read_fd, 'rb') as stream:
                read_fd = None
                reader = _HashedReader(stream)
                for chunk in pd.read_csv(reader, usecols=['qid', *FEATURES, 'r'],
                        chunksize=500000, dtype={'qid': np.int64,
                        **{name: np.float64 for name in [*FEATURES, 'r']}}):
                    trace_rows += len(chunk)
                    qid = chunk.qid.to_numpy()
                    distances, labels = chunk.dists.to_numpy(), chunk.r.to_numpy()
                    if ((qid < 0) | (qid >= queries)).any():
                        raise ValueError('Unexpected trace query ID')
                    if not np.isfinite(labels).all() or ((labels < 0) | (labels > 1)).any():
                        raise ValueError('Invalid trace recall label')
                    if not np.isfinite(distances).all() or (distances < 0).any():
                        raise ValueError('Invalid logical distance count')
                    reaching = labels >= target
                    np.minimum.at(first_reaching, qid[reaching], distances[reaching])
                    selected = chunk.loc[(distances % logging_interval) == 0]
                    if selected.empty:
                        continue
                    x = selected[FEATURES].to_numpy(dtype=np.float64, copy=True)
                    y = selected.r.to_numpy(dtype=np.float64, copy=True)
                    if not np.isfinite(x).all():
                        raise ValueError('Nonfinite training feature')
                    if not ((x[:, 4] <= x[:, 8] + 1e-6).all()
                            and (x[:, 8] <= x[:, 9] + 1e-6).all()
                            and (x[:, 9] <= x[:, 10] + 1e-6).all()
                            and (x[:, 10] <= x[:, 5] + 1e-6).all()
                            and (x[:, 6] >= x[:, 4] - 1e-6).all()
                            and (x[:, 6] <= x[:, 5] + 1e-6).all()):
                        raise ValueError('Invalid corrected distance feature order')
                    feature_min = np.minimum(feature_min, x.min(axis=0))
                    feature_max = np.maximum(feature_max, x.max(axis=0))
                    query_rows += np.bincount(selected.qid.to_numpy(), minlength=queries)
                    pieces_x.append(x)
                    pieces_y.append(y)
                    selected_rows += len(y)
                trace_sha, trace_bytes = reader.digest.hexdigest(), reader.bytes
            result = process.wait(timeout=60)
            if timed_out.is_set():
                raise TimeoutError('DARTH trace exceeded its timeout; see trace.log')
            if result:
                raise RuntimeError(f'DARTH trace exited with status {result}; see trace.log')
    except BaseException as error:
        if timed_out.is_set():
            error = TimeoutError('DARTH trace exceeded its timeout; see trace.log')
        elif process is not None and process.poll() not in (None, 0):
            error = RuntimeError(f'DARTH trace exited with status {process.returncode}; see trace.log')
        _save(output / 'failure.json', dict(stage='trace', type=type(error).__name__, message=str(error)))
        raise error
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            _kill_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_group(process, signal.SIGKILL)
                process.wait(timeout=5)
            # A failed native wrapper must not leave inherited pipe writers alive.
            _kill_group(process, signal.SIGKILL)
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    if not (query_rows > 0).all():
        raise ValueError('Not all learning queries contributed training rows')
    reached = np.isfinite(first_reaching)
    if not reached.any():
        raise ValueError('No training query reached target recall')
    metadata = [json.loads(line) for line in (output / 'trace.log').read_text().splitlines()
                if line.startswith('{')]
    if not metadata or any(metadata[-1].get(key) != value for key, value in
            dict(query_count=queries, k=10, threads=threads, ef=ef,
                 logging_interval=logging_interval).items()):
        raise ValueError('Native trace metadata does not match the training request')
    _save(output / 'trace.json', dict(native=metadata[-1], trace_rows=trace_rows,
          trace_sha256=trace_sha, trace_bytes=trace_bytes, trace_retained=False))
    mean_first = float(first_reaching[reached].mean())
    initial = max(1, round(mean_first / 2.0))
    minimum = min(initial, max(1, round(mean_first / 10.0)))
    intervals = dict(dataset=dataset, ef_search=ef, k=10, target_recall=target,
        initial_interval=initial, min_interval=minimum, avg_dists_rt=mean_first,
        queries_reaching_target=int(reached.sum()), queries_not_reaching_target=int((~reached).sum()),
        interval_rule='Mean first target-reaching dists over all raw observations; Python round(mean/2), round(mean/10), clamp >=1',
        trace_rows=trace_rows, training_rows=selected_rows, trace_sha256=trace_sha)
    _save(output / 'intervals.json', intervals)
    np.savez(output / 'query_training_stats.npz', fitting_rows=query_rows, first_target_dists=first_reaching)
    collected_seconds = time.monotonic() - started
    x, y = np.concatenate(pieces_x, axis=0), np.concatenate(pieces_y)
    del pieces_x, pieces_y, chunk, selected
    gc.collect()
    model = lgb.LGBMRegressor(objective='regression', random_state=42,
                             n_estimators=100, verbose=-1, n_jobs=threads)
    fit_started = time.monotonic()
    model.fit(pd.DataFrame(x, columns=FEATURES, copy=False), y)
    fit_seconds = time.monotonic() - fit_started
    model_path = output / 'model.txt'
    model.booster_.save_model(str(model_path))
    result = dict(intervals, features=FEATURES, query_num=queries, logging_interval=logging_interval,
        threads=threads, model_parameters=model.get_params(), model_path=str(model_path),
        model_sha256=_sha(model_path), source_input_sha256=sources, trace_bytes=trace_bytes,
        trace_retained=False, training_policy='Full filtered trace; no subsampling, query reweighting, validation fit or test tuning',
        feature_ranges={name: [float(low), float(high)] for name, low, high in zip(FEATURES, feature_min, feature_max)},
        collect_and_parse_seconds=collected_seconds, fit_seconds=fit_seconds,
        total_seconds=time.monotonic() - started)
    _save(output / 'training.json', result)
    return result
