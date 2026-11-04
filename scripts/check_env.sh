#!/usr/bin/env bash
# Synthetic correctness checks only; no external datasets or benchmark sweeps.
set -euo pipefail
CHECK_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$CHECK_ROOT/scripts/python.sh" - <<'PY'
import os
from pathlib import Path
import numpy as np
import faiss
import hnswlib

root = Path(os.environ['SAGE_ROOT'])
for module in (faiss, hnswlib):
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(root), path
    print(f'{module.__name__}: {path}', flush=True)
assert hasattr(faiss, 'SearchParametersHNSWAdaptiveLight')
for field in ('shadow_ef', 'shadow_bucket_mode', 'shadow_bucket_gamma_ratio_2'):
    assert hasattr(faiss.SearchParametersHNSWAdaptiveLight, field), field
assert hasattr(hnswlib.Index, 'knn_query_adaptive_light')
x = np.random.default_rng(42).normal(size=(128, 16)).astype(np.float32)
faiss.omp_set_num_threads(1)
f = faiss.IndexHNSWFlat(16, 16)
f.add(x)
f.hnsw.efSearch = 128
_, ids = f.search(x[:8], 1)
np.testing.assert_array_equal(ids[:, 0], np.arange(8))
h = hnswlib.Index(space='l2', dim=16)
h.init_index(max_elements=128, M=16, ef_construction=100)
h.set_num_threads(1)
h.add_items(x)
h.set_ef(128)
ids, _ = h.knn_query(x[:8], k=1)
np.testing.assert_array_equal(ids[:, 0], np.arange(8))
print('SAGE backend imports and synthetic searches passed.', flush=True)
PY
bash "$CHECK_ROOT/scripts/python.sh" "$CHECK_ROOT/baselines/pip/tests/test_pip.py" -q
bash "$CHECK_ROOT/scripts/python.sh" "$CHECK_ROOT/baselines/abs/tests/test_abs.py" -q
