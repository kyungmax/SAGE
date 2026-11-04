"""PiP behavior on controlled graphs and FAISS result heaps."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pip_runtime import load_faiss, parameters, search_with_stats

PAPER_BUILD = ROOT / "build" / "paper095" / "faiss" / "python"
faiss = load_faiss(os.environ.get("PIP_PAPER_FAISS_PYTHON_PATH", PAPER_BUILD))


def graph(values, edges=None):
    """Explicit directed level-0 graph with entry ID 0 and scalar L2 data."""
    xb = np.asarray(values, dtype="float32").reshape(-1, 1)
    index = faiss.IndexHNSWFlat(1, 8)
    index.add(xb)
    n = len(xb)
    degree = index.hnsw.nb_neighbors(0)
    neighbors = np.full((n, degree), -1, dtype="int32")
    for node in range(n):
        links = edges.get(node, []) if edges is not None else (
            [node + 1] if node + 1 < n else [])
        assert len(links) <= degree
        neighbors[node, :len(links)] = links
    faiss.copy_array_to_vector(np.ones(n, dtype="int32"), index.hnsw.levels)
    faiss.copy_array_to_vector(
        np.arange(n + 1, dtype="uint64") * degree, index.hnsw.offsets)
    faiss.copy_array_to_vector(neighbors.ravel(), index.hnsw.neighbors)
    index.hnsw.max_level = 0
    index.hnsw.entry_point = 0
    return index


class PiPTest(unittest.TestCase):
    def setUp(self):
        faiss.omp_set_num_threads(1)

    def run_graph(self, index, *, k=1, threshold=0.95, patience=1,
                  selector=None):
        p = parameters(faiss, 1024, pip=True, threshold=threshold,
                       patience=patience)
        if selector is not None:
            p.sel = selector
        return search_with_stats(
            faiss, index, np.zeros((1, 1), dtype="float32"), k, p)

    def test_entry_point_snapshot_and_exact_positive_patience(self):
        for patience in (1, 2, 7):
            with self.subTest(patience=patience):
                _, ids, stats = self.run_graph(graph(range(100)),
                                               patience=patience)
                self.assertEqual(ids[0, 0], 0)
                self.assertEqual(stats["nexpan"], patience)
                self.assertEqual(stats["ndis"], patience)
                self.assertEqual(stats["npip_stops"], 1)

    def test_true_result_change_resets_streak(self):
        values = [10, 9, 9.5, 9.6, 8, 8.5, 8.6, 8.7, 8.8, 8.9]
        _, ids, stats = self.run_graph(graph(values), patience=3)
        self.assertEqual(ids[0, 0], 4)
        self.assertEqual(stats["nexpan"], 7)
        self.assertEqual(stats["npip_stops"], 1)

    def test_cumulative_accepted_count_cannot_cause_saturation(self):
        # Every expansion changes the true top-10. The former cumulative
        # acceptance formula eventually saturates despite ongoing changes.
        _, ids, stats = self.run_graph(graph(range(100, 0, -1)), k=10)
        np.testing.assert_array_equal(ids[0], np.arange(99, 89, -1))
        self.assertEqual(stats["nexpan"], 100)
        self.assertEqual(stats["npip_stops"], 1)

    def test_checks_after_entire_neighbor_expansion(self):
        index = graph([0] + list(range(10, 2, -1)),
                      {0: list(range(1, 9))})
        _, ids, stats = self.run_graph(index)
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(stats["nexpan"], 1)
        self.assertEqual(stats["ndis"], 8)

    def test_transient_replacements_preserve_exact_overlap(self):
        # At step 2, three accepted neighbors evict previous ID 1 and then
        # two transient IDs; {0,1} intersect {0,4} still contains one ID.
        index = graph([0, 10, 9, 8, 7], {0: [1], 1: [2, 3, 4]})
        _, ids, stats = self.run_graph(
            index, k=2, threshold=0.5, patience=2)
        np.testing.assert_array_equal(ids[0], [0, 4])
        self.assertEqual(stats["nexpan"], 2)
        self.assertEqual(stats["ndis"], 4)
        self.assertEqual(stats["npip_stops"], 1)

    def test_fallback_patience_depends_on_k_not_ef(self):
        for k, expected in ((1, 7), (10, 16), (100, 129)):
            with self.subTest(k=k):
                _, _, stats = self.run_graph(
                    graph(range(300)), k=k, threshold=1.0, patience=-1)
                self.assertEqual(stats["nexpan"], expected)

    def test_invalid_labels_do_not_inflate_overlap(self):
        index = graph(range(20))
        _, ids, stats = self.run_graph(
            index, k=10, selector=faiss.IDSelectorRange(9, 10))
        self.assertEqual(ids[0, 0], 9)
        self.assertTrue(np.all(ids[0, 1:] == -1))
        self.assertEqual(stats["nexpan"], 20)
        self.assertEqual(stats["npip_stops"], 0)

    def test_empty_result_set_cannot_trigger_stop(self):
        _, ids, stats = self.run_graph(
            graph(range(20)), selector=faiss.IDSelectorRange(20, 21))
        self.assertTrue(np.all(ids == -1))
        self.assertEqual(stats["nexpan"], 20)
        self.assertEqual(stats["npip_stops"], 0)

    def test_normal_hnsw_stopping_still_applies(self):
        p = parameters(faiss, 3, pip=True, threshold=0.95, patience=1000)
        _, _, stats = search_with_stats(
            faiss, graph(range(100)), np.zeros((1, 1), dtype="float32"), 1, p)
        self.assertLess(stats["nexpan"], 100)
        self.assertEqual(stats["npip_stops"], 0)

    def test_off_and_infinite_patience_match_vanilla_l2_and_ip(self):
        rng = np.random.default_rng(12)
        xb = rng.standard_normal((512, 16), dtype=np.float32)
        xq = rng.standard_normal((29, 16), dtype=np.float32)
        for metric in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
            index = faiss.IndexHNSWFlat(16, 12, metric)
            index.add(xb)
            for check_distance in (False, True):
                base = parameters(faiss, 64)
                base.check_relative_distance = check_distance
                expected = search_with_stats(faiss, index, xq, 10, base)
                for enabled, patience in ((False, 0), (True, 2**31 - 1)):
                    with self.subTest(metric=metric,
                                      check_distance=check_distance,
                                      enabled=enabled):
                        p = parameters(faiss, 64, pip=True, enabled=enabled,
                                       threshold=0.95, patience=patience)
                        p.check_relative_distance = check_distance
                        actual = search_with_stats(faiss, index, xq, 10, p)
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[1], expected[1])
                        self.assertEqual(actual[2], expected[2])

    def test_query_local_state_and_1_vs_24_thread_parity(self):
        rng = np.random.default_rng(123)
        index = faiss.IndexHNSWFlat(16, 12)
        index.add(rng.standard_normal((512, 16), dtype=np.float32))
        xq = rng.standard_normal((53, 16), dtype=np.float32)
        p = parameters(faiss, 128, pip=True, threshold=0.95, patience=2)
        expected = search_with_stats(faiss, index, xq, 10, p)
        self.assertGreater(expected[2]["npip_stops"], 0)
        faiss.omp_set_num_threads(24)
        actual = search_with_stats(faiss, index, xq, 10, p)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        faiss.omp_set_num_threads(1)
        summed = dict.fromkeys(expected[2], 0)
        for q in xq:
            _, _, stats = search_with_stats(faiss, index, q[None], 10, p)
            summed = {key: summed[key] + stats[key] for key in summed}
        self.assertEqual(summed, expected[2])


    def test_k10_threshold_095_and_0995_are_equivalent(self):
        rng = np.random.default_rng(321)
        index = faiss.IndexHNSWFlat(16, 12)
        index.add(rng.standard_normal((512, 16), dtype=np.float32))
        xq = rng.standard_normal((29, 16), dtype=np.float32)
        for patience in (1, 7, 23):
            with self.subTest(patience=patience):
                a = search_with_stats(
                    faiss, index, xq, 10,
                    parameters(faiss, 128, pip=True, threshold=0.95,
                               patience=patience))
                b = search_with_stats(
                    faiss, index, xq, 10,
                    parameters(faiss, 128, pip=True, threshold=0.995,
                               patience=patience))
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
                self.assertEqual(a[2], b[2])

    def test_serialized_index_needs_no_format_change(self):
        index = graph(range(100))
        data = faiss.serialize_index(index)
        expected = self.run_graph(index, k=10)
        actual = self.run_graph(faiss.deserialize_index(data), k=10)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2], expected[2])
        np.testing.assert_array_equal(faiss.serialize_index(index), data)

    def test_invalid_parameters_and_unsupported_paths_fail(self):
        index = graph(range(20))
        q = np.zeros((1, 1), dtype="float32")
        for key, value in (("saturation_threshold", 0),
                           ("saturation_threshold", 1.1),
                           ("saturation_threshold", float("nan")),
                           ("patience", -2), ("patience", 0),
                           ("bounded_queue", False)):
            with self.subTest(key=key, value=value):
                p = parameters(faiss, 128, pip=True, threshold=0.95,
                               patience=1)
                setattr(p, key, value)
                with self.assertRaises(RuntimeError):
                    index.search(q, 1, params=p)
        with self.assertRaises(RuntimeError):
            index.range_search(
                q, 10, params=parameters(faiss, 128, pip=True,
                                         threshold=0.95, patience=1))

    def test_k_larger_than_index_and_empty_index(self):
        _, ids, stats = self.run_graph(graph(range(4)), k=10)
        self.assertEqual(np.count_nonzero(ids >= 0), 4)
        self.assertEqual(stats["npip_stops"], 0)
        empty = faiss.IndexHNSWFlat(1, 8)
        _, ids, stats = self.run_graph(empty, k=10)
        self.assertTrue(np.all(ids == -1))
        self.assertEqual(stats["npip_stops"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
