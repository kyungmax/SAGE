"""Explicit Euclidean lifting for normalized IP data that include zero vectors.

Database: P(x) = (x, sqrt(1-||x||^2), 0).
Queries:  Q(q) = (q, 0, sqrt(1-||q||^2)).
Both have unit norm, and ||Q(q)-P(x)||^2 = 2*(1-q.x), including zeros.
The separate added coordinates preserve zero-query scores against zero database
vectors too. The native kernel already computes half this squared distance.
No coordinates, graph edges, IP scores, or native search rules need changing.
Only unit or exactly zero input vectors are accepted, not arbitrary raw IP.
"""
import numpy as np

from abs_runtime import ABSSearch


class UnitOrZeroIPABSSearch(ABSSearch):
    """Opt-in lifting; the ordinary ABSSearch still rejects zero IP vectors."""

    def __init__(self, index, threads=1, *, unit_normalized=False, build_dir=None):
        self.zero_database_vectors = 0
        super().__init__(index, threads=threads, unit_normalized=unit_normalized,
                         build_dir=build_dir)

    def _check_unit(self, vectors, kind):
        squared_norms = np.einsum("ij,ij->i", vectors, vectors, dtype=np.float64)
        zero = squared_norms == 0
        unit = np.isclose(squared_norms, 1, rtol=0, atol=2e-5)
        if not np.all(np.isfinite(squared_norms) & (zero | unit)):
            raise ValueError(f"Lifted IP {kind} vectors must be unit or exactly zero")
        if kind == "database":
            self.zero_database_vectors += int(zero.sum())
