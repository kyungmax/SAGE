"""Minimal benchmark utilities for SAGE experiment scripts."""

from __future__ import annotations

import time

import numpy as np


def benchmark_query_batch(run_once, query_count: int, warmup_runs: int = 1, measured_runs: int = 5):
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")

    last_output = None
    for _ in range(warmup_runs):
        last_output = run_once()

    durations = []
    for _ in range(measured_runs):
        t0 = time.perf_counter()
        last_output = run_once()
        durations.append(time.perf_counter() - t0)

    durations = np.asarray(durations, dtype=np.float64)
    total_duration = float(durations.sum())
    total_queries = int(query_count) * int(measured_runs)

    return {
        "last_output": last_output,
        "durations_s": durations,
        "total_duration_s": total_duration,
        "avg_duration_s": float(durations.mean()),
        "median_duration_s": float(np.median(durations)),
        "min_duration_s": float(durations.min()),
        "max_duration_s": float(durations.max()),
        "qps": (total_queries / total_duration) if total_duration > 0 else 0.0,
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
    }
