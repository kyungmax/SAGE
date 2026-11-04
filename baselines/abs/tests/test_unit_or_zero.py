"""Verify implicit zero-vector handling against explicit Euclidean lifting."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from abs_runtime import ABSSearch, load_faiss
from unit_or_zero_runtime import UnitOrZeroIPABSSearch
from test_abs import make_graph

faiss = load_faiss()


class UnitOrZeroTests(unittest.TestCase):
    def test_implicit_matches_explicit_lifting_with_zero_queries_and_database(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]], np.float32)
        queries = np.array([[0, 0], [1, 0], [0, 1], [-1, 0]], np.float32)
        edges = [[j for j in range(len(points)) if j != i] for i in range(len(points))]
        lifted_points = np.column_stack((points, (points == 0).all(axis=1), np.zeros(len(points)))).astype('float32')
        lifted_queries = np.column_stack((queries, np.zeros(len(queries)), (queries == 0).all(axis=1))).astype('float32')
        index = make_graph(points, edges, faiss.METRIC_INNER_PRODUCT)
        lifted = make_graph(lifted_points, edges, faiss.METRIC_INNER_PRODUCT)
        np.testing.assert_array_equal(queries @ points.T, lifted_queries @ lifted_points.T)
        with self.assertRaises(ValueError):
            ABSSearch(index, unit_normalized=True)
        for threads in (1, 4):
            with UnitOrZeroIPABSSearch(index, threads, unit_normalized=True) as actual, \
                    ABSSearch(lifted, threads, unit_normalized=True) as expected:
                self.assertEqual(actual.zero_database_vectors, 2)
                for gamma in (0, .001, .2, 1, 2):
                    a = actual.search(queries, 3, gamma=gamma, entry=0)
                    b = expected.search(lifted_queries, 3, gamma=gamma, entry=0)
                    np.testing.assert_array_equal(a[0], b[0])
                    np.testing.assert_array_equal(a[1], b[1])
                    for key in a[2]:
                        np.testing.assert_array_equal(a[2][key], b[2][key])

    def test_rejects_nonunit_nonzero_and_nonfinite_inputs(self):
        for value in (.5, 2, float('nan'), float('inf')):
            index = make_graph(np.array([[value, 0], [0, 1]], np.float32), [[1], [0]],
                               faiss.METRIC_INNER_PRODUCT)
            with self.assertRaises(ValueError):
                UnitOrZeroIPABSSearch(index, unit_normalized=True)
        index = make_graph(np.array([[1, 0], [0, 1]], np.float32), [[1], [0]],
                           faiss.METRIC_INNER_PRODUCT)
        with UnitOrZeroIPABSSearch(index, unit_normalized=True) as search:
            for value in (.5, 2, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    search.search(np.array([[value, 0]], np.float32), 1, gamma=.1)


if __name__ == "__main__":
    unittest.main()
