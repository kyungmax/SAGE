"""Paper ABS checks, including an independent, unpruned Algorithm 1 oracle."""
from concurrent.futures import ThreadPoolExecutor
import heapq
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from abs_runtime import ABSSearch, ROOT, active_binary, load_faiss

faiss = load_faiss()
faiss.omp_set_num_threads(1)


def make_graph(points, edges, metric=faiss.METRIC_L2):
    """Install a layer-0 graph directly, without an HNSW construction heuristic."""
    points = np.ascontiguousarray(points, np.float32)
    degree = max(4, 2 * ((max(map(len, edges), default=0) + 1) // 2))
    index = faiss.IndexHNSWFlat(points.shape[1], degree // 2, metric)
    faiss.downcast_index(index.storage).add(points)
    index.ntotal = len(points)
    h = index.hnsw
    h.max_level, h.entry_point = 0, 0
    faiss.copy_array_to_vector(np.ones(len(points), np.int32), h.levels)
    faiss.copy_array_to_vector(np.arange(len(points) + 1, dtype=np.uint64) * degree, h.offsets)
    links = np.full((len(points), degree), -1, np.int32)
    for node, neighbors in enumerate(edges):
        links[node, :len(neighbors)] = neighbors
    faiss.copy_array_to_vector(links.ravel(), h.neighbors)
    return index


def adjacency(index):
    offsets = faiss.vector_to_array(index.hnsw.offsets)
    links = faiss.vector_to_array(index.hnsw.neighbors)
    degree = index.hnsw.nb_neighbors(0)
    result = []
    for start in offsets[:-1]:
        row = links[int(start):int(start) + degree]
        result.append(row[row >= 0].tolist())
    return result


def unpruned_reference(index, query, k, gamma, entry):
    # Algorithm 1 keeps ALL discovered nodes. It sorts D on every iteration,
    # without Algorithm 2's bounded result heap or candidate admission filter.
    query = np.ascontiguousarray(query, np.float32)
    dc = faiss.downcast_index(index.storage).get_distance_computer()
    dc.set_query(faiss.swig_ptr(query))
    ip = index.metric_type == faiss.METRIC_INNER_PRODUCT

    def distance(node):
        raw = float(dc(int(node)))
        return max(0.0, 1.0 - raw) if ip else raw

    graph = adjacency(index)
    discovered = {entry: distance(entry)}
    candidates = [(discovered[entry], entry)]
    expanded = 0
    while candidates:
        distance_to_next, node = heapq.heappop(candidates)
        ordered = sorted((dist, node_id) for node_id, dist in discovered.items())
        if len(ordered) >= k and distance_to_next >= (1 + gamma) ** 2 * ordered[k - 1][0]:
            break
        expanded += 1
        for neighbor in graph[node]:
            if neighbor not in discovered:
                discovered[neighbor] = distance(neighbor)
                heapq.heappush(candidates, (discovered[neighbor], neighbor))
    result = sorted((dist, node_id) for node_id, dist in discovered.items())[:k]
    return [node for _, node in result], [dist for dist, _ in result], len(discovered), expanded


class PaperABSTests(unittest.TestCase):
    def test_default_build_is_self_contained(self):
        self.assertTrue(active_binary().is_relative_to(ROOT))
        index = make_graph([[1], [.1]], [[1], []])
        with ABSSearch(index) as search:
            self.assertTrue(Path(search.lib._name).resolve().is_relative_to(ROOT))
            for source in search.build["source_sha256"]:
                self.assertTrue(Path(source).resolve().is_relative_to(ROOT), source)
            _, ids, _ = search.search([[0]], 1, gamma=.2, entry=0)
            np.testing.assert_array_equal(ids, [[1]])

    def assert_same(self, a, b):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual(a[2].keys(), b[2].keys())
        for name in a[2]:
            np.testing.assert_array_equal(a[2][name], b[2][name], err_msg=name)

    def test_strict_candidate_admission_at_equality(self):
        index = make_graph([[1], [2], [.25]], [[1], [2], []])
        with ABSSearch(index) as search:
            distances, ids, stats = search.search([[0]], 1, gamma=1, entry=0)
        np.testing.assert_array_equal(ids, [[0]])
        np.testing.assert_array_equal(distances, [[1]])
        self.assertEqual(stats["ndis"][0], 2)
        self.assertEqual(stats["nexpan"][0], 1)

    def test_stops_at_equality_after_threshold_shrinks(self):
        index = make_graph([[3], [2], [1], [.25]], [[1, 2], [3], [], []])
        with ABSSearch(index) as search:
            _, ids, stats = search.search([[0]], 1, gamma=1, entry=0)
        np.testing.assert_array_equal(ids, [[2]])
        self.assertEqual(stats["ndis"][0], 3)
        self.assertEqual(stats["nexpan"][0], 2)
        self.assertEqual(stats["npop"][0], 3)
        self.assertEqual(stats["stopped_by_bound"][0], 1)

    def test_squared_l2_and_bridge_outside_top_k(self):
        # 1.15^2 > 1.2, but 1.15^2 < 1.2^2. A linear squared-L2
        # multiplier, or truncating C to k=1, would miss node 2.
        index = make_graph([[1], [1.15], [.1]], [[1], [2], []])
        with ABSSearch(index) as search:
            _, ids, stats = search.search([[0]], 1, gamma=.2, entry=0)
        np.testing.assert_array_equal(ids, [[2]])
        self.assertEqual(stats["ndis"][0], 3)

    def test_literal_gamma_zero_and_zero_distance(self):
        index = make_graph([[1], [.1], [0]], [[1], [2], []])
        with ABSSearch(index) as search:
            for gamma, entry in ((0, 0), (.2, 2)):
                _, ids, stats = search.search([[0]], 1, gamma=gamma, entry=entry)
                np.testing.assert_array_equal(ids, [[entry]])
                self.assertEqual(stats["ndis"][0], 1)
                self.assertEqual(stats["nexpan"][0], 0)
                self.assertEqual(stats["stopped_by_bound"][0], 1)

    def test_no_ef_cap_or_k_clamping(self):
        # A binary tree produces a frontier >100,000 while k=1, efSearch=1.
        n = 200011
        edges = [[j for j in (2 * i + 1, 2 * i + 2) if j < n] for i in range(n)]
        index = make_graph(np.ones((n, 1), np.float32), edges)
        with ABSSearch(index) as search:
            index.hnsw.efSearch = 1
            tiny = search.search([[0]], 1, gamma=1, entry=0)
            self.assertGreater(tiny[2]["frontier_peak"][0], 100000)
            self.assertEqual(tiny[2]["ndis"][0], n)
            self.assertEqual(tiny[2]["nexpan"][0], n)
            index.hnsw.efSearch = 4096
            self.assert_same(tiny, search.search([[0]], 1, gamma=1, entry=0))
            index.hnsw.efSearch = 1
            _, ids, _ = search.search([[0]], 10, gamma=1, entry=0)
            np.testing.assert_array_equal(ids, [np.arange(10)])

    def test_unpruned_reference_l2_and_normalized_ip(self):
        rng = np.random.default_rng(20260919)
        cases = 0
        for dimension in (8, 13):
            for metric in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
                points = rng.normal(size=(91, dimension)).astype(np.float32)
                queries = rng.normal(size=(11, dimension)).astype(np.float32)
                if metric == faiss.METRIC_INNER_PRODUCT:
                    faiss.normalize_L2(points)
                    faiss.normalize_L2(queries)
                index = faiss.IndexHNSWFlat(dimension, 8, metric)
                index.add(points)
                with ABSSearch(index, threads=3, unit_normalized=True) as search:
                    for gamma in (0, .03, .15, .6, 2):
                        for k in (1, 7, 91):
                            distances, ids, stats = search.search(queries, k, gamma=gamma, entry=0)
                            for j, query in enumerate(queries):
                                expected_ids, expected_distances, calls, expansions = unpruned_reference(
                                    index, query, k, gamma, 0)
                                np.testing.assert_array_equal(ids[j], expected_ids)
                                actual = 1 - distances[j] if metric == faiss.METRIC_INNER_PRODUCT else distances[j]
                                np.testing.assert_allclose(actual, expected_distances, atol=2e-6, rtol=2e-6)
                                self.assertEqual(stats["ndis"][j], calls)
                                self.assertEqual(stats["nexpan"][j], expansions)
                                cases += 1
        self.assertEqual(cases, 660)

    def test_navigable_line_theorem_and_gamma_two(self):
        points = (np.arange(31) * 3).astype(np.float32)[:, None]
        edges = [[j for j in (i - 1, i + 1) if 0 <= j < len(points)] for i in range(len(points))]
        index = make_graph(points, edges)
        queries = np.array([[-1], [0], [16.25], [82.125], [92]], np.float32)
        with ABSSearch(index) as search:
            for gamma in (.03, .2, 1, 2):
                for k in (1, 3, 10, 31):
                    distances, ids, _ = search.search(queries, k, gamma=gamma, entry=0)
                    for j, query in enumerate(queries):
                        all_distances = np.abs(points[:, 0] - query[0])
                        outside = np.ones(len(points), bool)
                        outside[ids[j]] = False
                        self.assertTrue(np.all(all_distances[outside] + 1e-6 >=
                                               gamma / 2 * np.sqrt(distances[j].max())))
                        if gamma == 2:
                            np.testing.assert_array_equal(distances[j], np.sort(all_distances ** 2)[:k])

    def test_upper_descent_reuses_entry_and_preserves_index(self):
        rng = np.random.default_rng(99)
        points = rng.normal(size=(300, 13)).astype(np.float32)
        queries = rng.normal(size=(9, 13)).astype(np.float32)
        index = faiss.IndexHNSWFlat(13, 8)
        index.add(points)
        self.assertGreater(index.hnsw.max_level, 0)
        snapshot = faiss.serialize_index(index)
        with ABSSearch(index, threads=3) as search:
            distances, ids, stats = search.search(queries, 7, gamma=.2)
            for j, query in enumerate(queries):
                forced = search.search(query[None], 7, gamma=.2, entry=stats["entry_point"][j])
                np.testing.assert_array_equal(distances[j], forced[0][0])
                np.testing.assert_array_equal(ids[j], forced[1][0])
                self.assertGreater(stats["ndis_upper"][j], 0)
                self.assertEqual(stats["ndis_level0"][j], forced[2]["ndis"][0] - 1)
                self.assertEqual(stats["ndis"][j], stats["ndis_upper"][j] + stats["ndis_level0"][j])
        np.testing.assert_array_equal(snapshot, faiss.serialize_index(index))
        restored = faiss.deserialize_index(snapshot)
        with ABSSearch(restored) as search:
            actual = search.search(queries, 7, gamma=.2)
        self.assert_same((distances, ids, stats), actual)

    def test_workspaces_reset_and_gamma_is_query_local(self):
        rng = np.random.default_rng(12)
        points = rng.normal(size=(130, 8)).astype(np.float32)
        query = rng.normal(size=(1, 8)).astype(np.float32)
        index = faiss.IndexHNSWFlat(8, 8)
        index.add(points)
        with ABSSearch(index, threads=1) as single, ABSSearch(index, threads=4) as parallel:
            expected = {g: single.search(query, 7, gamma=g) for g in (.01, 2)}
            # More than 250 searches through the same workspace exercises the
            # uint8 visited-generation wraparound as well as reset between calls.
            many = np.repeat(query, 260, axis=0)
            repeated = single.search(many, 7, gamma=.01)
            for j in range(len(many)):
                self.assert_same(expected[.01], (repeated[0][j:j+1], repeated[1][j:j+1],
                    {name: values[j:j+1] for name, values in repeated[2].items()}))
            self.assert_same(repeated, parallel.search(many, 7, gamma=.01))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(3):
                    low = pool.submit(single.search, query, 7, gamma=.01)
                    high = pool.submit(parallel.search, query, 7, gamma=2)
                    self.assert_same(expected[.01], low.result())
                    self.assert_same(expected[2], high.result())

    def test_disconnected_graph_padding_and_duplicate_neighbors(self):
        index = make_graph([[0], [0], [1], [10]], [[0, 1, 1], [2, 2], [], []])
        with ABSSearch(index) as search:
            distances, ids, stats = search.search([[0]], 4, gamma=.1, entry=0)
        np.testing.assert_array_equal(ids, [[0, 1, 2, -1]])
        np.testing.assert_array_equal(distances, [[0, 0, 1, np.inf]])
        self.assertEqual(stats["ndis"][0], 3)
        ip = make_graph([[1, 0], [0, 1]], [[], []], faiss.METRIC_INNER_PRODUCT)
        with ABSSearch(ip, unit_normalized=True) as search:
            distances, ids, _ = search.search([[1, 0]], 2, gamma=.1, entry=0)
        np.testing.assert_array_equal(ids, [[0, -1]])
        np.testing.assert_array_equal(distances, [[1, -np.inf]])

    def test_unit_ip_validation_checks_database_and_queries(self):
        points = np.array([[1, 0], [0, 1]], np.float32)
        index = make_graph(points, [[1], [0]], faiss.METRIC_INNER_PRODUCT)
        with self.assertRaises(ValueError):
            ABSSearch(index)
        with ABSSearch(index, unit_normalized=True) as search:
            for query in ([[2, 0]], [[0, 0]], [[np.nan, 0]]):
                with self.assertRaises(ValueError):
                    search.search(query, 1, gamma=.1)
        # The invalid vector lies beyond the first validation chunk.
        points = np.zeros((16385, 2), np.float32)
        points[:, 0] = 1
        points[-1, 0] = 2
        index = make_graph(points, [[] for _ in points], faiss.METRIC_INNER_PRODUCT)
        with self.assertRaises(ValueError):
            ABSSearch(index, unit_normalized=True)

    def test_invalid_arguments_empty_queries_and_closed_context(self):
        with self.assertRaises(TypeError):
            ABSSearch(faiss.IndexFlatL2(1))
        with self.assertRaises(ValueError):
            ABSSearch(faiss.IndexHNSWFlat(1, 2))
        index = make_graph([[1], [.1]], [[1], [0]])
        with self.assertRaises(ValueError):
            ABSSearch(index, threads=0)
        with ABSSearch(index) as search:
            for gamma in (-1, np.nan, np.inf, 1e300):
                with self.assertRaises(ValueError):
                    search.search([[0]], 1, gamma=gamma)
            for k in (0, 3, 1.5, True):
                with self.assertRaises(ValueError):
                    search.search([[0]], k, gamma=.1)
            for entry in (-2, 2, .5):
                with self.assertRaises(ValueError):
                    search.search([[0]], 1, gamma=.1, entry=entry)
            for query in ([0], [[0, 0]], [[np.inf]]):
                with self.assertRaises(ValueError):
                    search.search(query, 1, gamma=.1)
            distances, ids, stats = search.search(np.empty((0, 1)), 1, gamma=.1)
            self.assertEqual(distances.shape, (0, 1))
            self.assertEqual(ids.shape, (0, 1))
            self.assertEqual(stats["ndis"].shape, (0,))
        search.close()
        with self.assertRaises(RuntimeError):
            search.search([[0]], 1, gamma=.1)

    def test_changed_index_is_rejected(self):
        index = faiss.IndexHNSWFlat(1, 2)
        index.add(np.array([[1], [2]], np.float32))
        with ABSSearch(index) as search:
            index.add(np.array([[3]], np.float32))
            with self.assertRaisesRegex(RuntimeError, "Index changed"):
                search.search([[0]], 1, gamma=.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
