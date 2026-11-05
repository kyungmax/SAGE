"""Prepare DARTH's disjoint learning/validation graph and original online inputs.

Ported from the October 29, 2026 main-six holdout campaign. Only training and
validation use the reduced graph; online queries keep the original full graph.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_vecs(path, values, floating):
    values = np.asarray(values, dtype='<f4' if floating else '<i4')
    packed = np.empty((len(values), values.shape[1] + 1), dtype=values.dtype)
    packed[:, 0].view('<i4')[:] = values.shape[1]
    packed[:, 1:] = values
    with Path(path).open('xb') as stream:
        packed.tofile(stream)


def _read_rows(dataset, ids):
    ids = np.asarray(ids, dtype=np.int64)
    unique, inverse = np.unique(ids, return_inverse=True)
    rows = np.empty((len(unique), dataset.shape[1]), dtype=np.float32)
    for begin in range(0, len(unique), 1024):
        end = min(begin + 1024, len(unique))
        rows[begin:end] = dataset[unique[begin:end]]
    return np.ascontiguousarray(rows[inverse])


def _normalize(rows, cosine, faiss):
    rows = np.array(rows, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(rows).all():
        raise ValueError('Non-finite source vectors')
    if cosine:
        faiss.normalize_L2(rows)
    rows[rows == 0] = 0  # Signed zero must not distinguish equal vectors.
    return rows


def _fingerprint(rows, coordinates):
    projected = np.ascontiguousarray(rows[:, coordinates])
    return projected.view(np.dtype((np.void, projected.shape[1] * 4))).reshape(-1)


def prepare_inputs(dataset_path, full_index_path, output_dir, *, k, ef, space, M,
                   ef_construction, learn_count=10000, validation_count=1000,
                   seed=12345, threads=24, batch_rows=25000):
    """Write a fresh input directory and return its ``prepared.json`` manifest.

    Exact GT uses retained rows and their auxiliary contiguous IDs. Learning and
    validation exclude all equal float32 vectors (after cosine normalization),
    while evaluation preserves the HDF5 test rows and official neighbor IDs.
    Counts can be reduced explicitly for synthetic checks; defaults are the
    paper's 10,000 learning and 1,000 validation queries.
    """
    import faiss
    import h5py

    started = time.perf_counter()
    integers = dict(k=k, ef=ef, M=M, ef_construction=ef_construction,
                    learn_count=learn_count, validation_count=validation_count,
                    threads=threads, batch_rows=batch_rows)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           or value <= 0 for value in integers.values()):
        raise ValueError('Counts, graph parameters and threads must be positive integers')
    if space not in ('cosine', 'l2'):
        raise ValueError('DARTH preparation supports only cosine and l2')
    if ef < k:
        raise ValueError('Training ef must be at least k')
    dataset_path, full_index_path = Path(dataset_path).resolve(strict=True), Path(full_index_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new preparation directory: {output}')
    inputs = dict(dataset=_identity(dataset_path), full_index=_identity(full_index_path))
    cosine = space == 'cosine'
    metric = faiss.METRIC_INNER_PRODUCT if cosine else faiss.METRIC_L2
    faiss.omp_set_num_threads(int(threads))

    with h5py.File(dataset_path, 'r') as data:
        base = data['train']
        count, dimension = base.shape
        query_count = len(data['test'])
        if (dimension < 1 or query_count < 1 or data['test'].shape[1] != dimension
                or data['neighbors'].shape[0] != query_count or data['neighbors'].shape[1] < k):
            raise ValueError('Dataset train/test/ground-truth shapes disagree')
        if count <= learn_count + validation_count + k:
            raise ValueError('Too few base rows for the requested holdout and k')
        if count > np.iinfo(np.int32).max:
            raise ValueError('Original IDs exceed the ivecs int32 format')
        external_inputs = []
        for name, offset, size in base.external or []:
            path = Path(name)
            if not path.is_absolute():
                path = dataset_path.parent / path
            external_inputs.append(dict(**_identity(path), offset=offset, external_bytes=size))
        inputs['external_train_storage'] = external_inputs

        full = faiss.read_index(str(full_index_path))
        if (not hasattr(full, 'hnsw') or full.ntotal != count or full.d != dimension
                or full.metric_type != metric or full.hnsw.nb_neighbors(1) != M
                or full.hnsw.efConstruction != ef_construction):
            raise ValueError('Full graph count/dimension/metric/M/efConstruction differs from the input protocol')
        sample_ids = np.unique(np.linspace(0, count - 1, min(33, count), dtype=np.int64))
        expected = _normalize(_read_rows(base, sample_ids), cosine, faiss)
        np.testing.assert_allclose(full.reconstruct_batch(sample_ids), expected,
                                   rtol=1e-5, atol=1e-6,
                                   err_msg='Full graph vectors differ from the source dataset')
        del full
        gc.collect()

        rng = np.random.default_rng(seed)
        selected = rng.choice(count, size=learn_count + validation_count, replace=False)
        used_ids, keys, replacements = set(map(int, selected)), set(), []
        raw_queries = _read_rows(base, selected)
        queries = _normalize(raw_queries, cosine, faiss)
        for position in range(len(selected)):
            key = queries[position].tobytes()
            while key in keys:
                if len(used_ids) == count:
                    raise ValueError('Too few distinct metric-equivalent vectors for disjoint holdouts')
                candidate = int(rng.integers(count))
                if candidate in used_ids:
                    continue
                used_ids.add(candidate)
                raw = np.asarray(base[candidate], dtype=np.float32).reshape(1, -1)
                normalized = _normalize(raw, cosine, faiss)[0]
                if normalized.tobytes() in keys:
                    continue
                replacements.append(dict(position=position, old_id=int(selected[position]),
                                         replacement_id=candidate))
                selected[position], raw_queries[position], queries[position] = candidate, raw[0], normalized
                key = normalized.tobytes()
            keys.add(key)

        output.mkdir(parents=True, exist_ok=False)
        splits = {}
        for name, subset in (('learn', slice(0, learn_count)), ('validation', slice(learn_count, None))):
            source_ids = output / f'{name}_source_ids.npy'
            np.save(source_ids, selected[subset])
            query_file = output / f'{name}.fvecs'
            _write_vecs(query_file, raw_queries[subset], True)
            splits[name] = dict(query_file=str(query_file),
                gt_file=str(output / f'{name}.groundtruth.ivecs'),
                gt_dist_file=str(output / f'{name}.groundtruth.fvecs'),
                source_ids=str(source_ids), original_gt_ids=str(output / f'{name}.groundtruth.original_ids.npy'),
                query_count=len(selected[subset]), id_space='auxiliary contiguous IDs')

        coordinates = np.unique(np.linspace(0, dimension - 1, min(dimension, 16), dtype=np.int64))
        selected_fingerprints = np.unique(_fingerprint(queries, coordinates))
        graph = faiss.IndexHNSWFlat(dimension, int(M), metric)
        graph.hnsw.efConstruction = int(ef_construction)
        graph.hnsw.rng = faiss.RandomGenerator(int(seed))
        graph.hnsw.efSearch = int(ef)
        flat = faiss.IndexFlatIP(dimension) if cosine else faiss.IndexFlatL2(dimension)
        kept = np.empty(count, np.int64)
        removed_chunks, kept_count = [], 0
        build_started = time.perf_counter()
        for begin in range(0, count, batch_rows):
            end = min(begin + batch_rows, count)
            rows = _normalize(base[begin:end], cosine, faiss)
            fingerprints = _fingerprint(rows, coordinates)
            positions = np.searchsorted(selected_fingerprints, fingerprints)
            candidates = np.flatnonzero(positions < len(selected_fingerprints))
            candidates = candidates[selected_fingerprints[positions[candidates]] == fingerprints[candidates]]
            # The coordinate fingerprint only filters candidates; full-row
            # equality decides exclusion, so collisions cannot remove a row.
            excluded = np.array([i for i in candidates if rows[i].tobytes() in keys], dtype=np.int64)
            keep = np.ones(len(rows), bool)
            keep[excluded] = False
            removed_chunks.append(begin + excluded)
            ids = np.arange(begin, end, dtype=np.int64)[keep]
            retained = np.ascontiguousarray(rows[keep])
            kept[kept_count:kept_count + len(ids)] = ids
            kept_count += len(ids)
            flat.add(retained)
            graph.add(retained)
        kept = kept[:kept_count]
        removed = np.concatenate(removed_chunks)
        if kept_count < k or not np.isin(selected, removed).all():
            raise ValueError('Heldout exclusion left an invalid auxiliary graph')
        graph_path = output / 'auxiliary.index'
        mapping_path = output / 'auxiliary_to_original.npy'
        np.save(mapping_path, kept)
        np.save(output / 'excluded_source_ids.npy', removed)
        check_queries = np.ascontiguousarray(queries[:min(32, len(queries))])
        before = graph.search(check_queries, k)
        faiss.write_index(graph, str(graph_path))
        del graph
        gc.collect()
        reloaded = faiss.read_index(str(graph_path))
        after = reloaded.search(check_queries, k)
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(actual, expected, err_msg='Auxiliary graph serialization changed search outputs')
        del reloaded
        gc.collect()
        build_seconds = time.perf_counter() - build_started

        exact_started = time.perf_counter()
        previous_blas_threshold = faiss.cvar.distance_compute_blas_threshold
        faiss.cvar.distance_compute_blas_threshold = np.iinfo(np.int32).max
        try:
            for name, subset in (('learn', slice(0, learn_count)), ('validation', slice(learn_count, None))):
                split_queries = queries[subset]
                all_ids = np.empty((len(split_queries), k), np.int64)
                all_scores = np.empty((len(split_queries), k), np.float32)
                for begin in range(0, len(split_queries), 128):
                    end = min(begin + 128, len(split_queries))
                    batch = split_queries[begin:end]
                    padded_count = max(128, threads)
                    if len(batch) < padded_count:
                        batch = np.concatenate((batch, np.repeat(batch[-1:], padded_count - len(batch), axis=0)))
                    scores, ids = flat.search(np.ascontiguousarray(batch), k)
                    all_ids[begin:end], all_scores[begin:end] = ids[:end-begin], scores[:end-begin]
                if (not np.isfinite(all_scores).all() or np.any(all_ids < 0)
                        or np.any(all_ids >= kept_count)):
                    raise ValueError('Invalid exact auxiliary ground truth')
                _write_vecs(splits[name]['gt_file'], all_ids, False)
                _write_vecs(splits[name]['gt_dist_file'], all_scores, True)
                np.save(splits[name]['original_gt_ids'], kept[all_ids])
        finally:
            faiss.cvar.distance_compute_blas_threshold = previous_blas_threshold
        del flat
        gc.collect()
        exact_seconds = time.perf_counter() - exact_started

        evaluation_dir = output / 'evaluation_inputs'
        evaluation_dir.mkdir()
        raw_test = np.asarray(data['test'], dtype=np.float32, order='C')
        truth = np.asarray(data['neighbors'][:, :k], dtype=np.int64, order='C')
        if ((truth < 0).any() or (truth >= count).any()
                or (np.diff(np.sort(truth, axis=1), axis=1) <= 0).any()):
            raise ValueError('Official online ground truth has invalid or duplicate IDs')
        normalized_test = _normalize(raw_test, cosine, faiss)
        truth_scores = np.empty(truth.shape, np.float32)
        for begin in range(0, query_count, 128):
            end = min(begin + 128, query_count)
            neighbors = _normalize(_read_rows(base, truth[begin:end].reshape(-1)), cosine, faiss)
            neighbors = neighbors.reshape(end - begin, k, dimension)
            if cosine:
                truth_scores[begin:end] = np.einsum('nd,nkd->nk', normalized_test[begin:end], neighbors)
            else:
                delta = normalized_test[begin:end, None, :] - neighbors
                truth_scores[begin:end] = np.einsum('nkd,nkd->nk', delta, delta)
        evaluation = dict(query_file=str(evaluation_dir / 'query.fvecs'),
            gt_file=str(evaluation_dir / 'query.groundtruth.ivecs'),
            gt_dist_file=str(evaluation_dir / 'query.groundtruth.fvecs'),
            query_count=query_count, id_space='original full-base IDs')
        _write_vecs(evaluation['query_file'], raw_test, True)
        _write_vecs(evaluation['gt_file'], truth, False)
        _write_vecs(evaluation['gt_dist_file'], truth_scores, True)

    if (_identity(dataset_path) != inputs['dataset']
            or _identity(full_index_path) != inputs['full_index']
            or any(_identity(record['path']) != {key: record[key] for key in ('path', 'bytes', 'mtime_ns')}
                   for record in external_inputs)):
        raise ValueError('Source dataset or full index changed during preparation')
    artifacts = {str(path): dict(**_identity(path), sha256=_sha(path))
                 for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = dict(schema_version=1, complete=True, dataset_path=str(dataset_path),
        full_index_path=str(full_index_path), auxiliary_index=str(graph_path),
        auxiliary_to_original=str(mapping_path), learn_count=learn_count,
        validation_count=validation_count, query_count=query_count, k=k, ef=ef,
        space=space, M=M, ef_construction=ef_construction, seed=seed, threads=threads,
        batch_rows=batch_rows, dimension=dimension, full_base_count=count,
        auxiliary_count=kept_count, learn=splits['learn'], validation=splits['validation'],
        evaluation=evaluation, inputs=inputs, artifacts=artifacts,
        query_sha256=hashlib.sha256(raw_test.tobytes()).hexdigest(),
        truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
        exclusion=dict(sampled_rows=len(selected), excluded_rows=len(removed),
            additional_equivalent_rows_excluded=len(removed)-len(selected),
            excluded_source_ids=str(output / 'excluded_source_ids.npy'), replacements=replacements,
            fingerprint_coordinates=coordinates.tolist(),
            equality='Exact normalized float32 rows for cosine, raw float32 rows for L2; signed zero canonicalized'),
        sampling='default_rng choice without replacement; duplicate vectors resampled across both holdout splits',
        insertion_order='Ascending original source IDs after heldout-equivalent rows excluded',
        exact_ground_truth='Exhaustive IndexFlat direct kernel; padded batches of at least 128 queries',
        source_inputs_unchanged=True, graph_reload_outputs_equal=True,
        full_graph_training_vectors_checked=len(sample_ids),
        timings=dict(auxiliary_build_seconds=build_seconds, exact_gt_seconds=exact_seconds,
                     total_seconds=time.perf_counter()-started))
    with (output / 'prepared.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest
