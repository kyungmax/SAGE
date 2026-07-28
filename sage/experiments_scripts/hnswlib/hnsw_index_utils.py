"""Backend-specific HNSW index helpers for the HNSW implementation."""

from __future__ import annotations

import os

import numpy as np

from common.dataset_utils import build_index_cache_dataset_name


def make_hnswlib_index_path(M, efConstruction, n_nodes, dim=None, dataset_name=None):
    parts = []
    if dataset_name:
        clean_name = str(dataset_name).replace(".hdf5", "").replace(".fvecs", "").replace(".bvecs", "")
        parts.append(clean_name)
    parts.append(f"M{M}")
    parts.append(f"efC{efConstruction}")
    parts.append(f"n{n_nodes}")
    if dim is not None:
        parts.append(f"dim{dim}")
    return "_".join(parts)


def build_or_load_hnsw_index(train, order, dim, space, M, efc, index_path, num_threads=-1):
    import hnswlib

    index = hnswlib.Index(space=space, dim=dim)
    index.set_num_threads(int(num_threads))

    if os.path.exists(index_path):
        print(f"[LOAD] Loading existing index: {index_path}")
        index.load_index(index_path, max_elements=len(train))
        print("[LOAD] Index loaded. Current count:", index.get_current_count())
    else:
        print(f"[BUILD] Building new index: {index_path}")
        index.init_index(max_elements=len(train), ef_construction=int(efc), M=int(M))
        batch_size = 20000
        total_added = 0
        for start in range(0, len(order), batch_size):
            chunk_idx = np.asarray(order[start:start + batch_size], dtype=np.int32)
            print(f"Adding batch: {start} to {start + len(chunk_idx)} (total so far: {total_added})")
            index.add_items(train[chunk_idx], ids=chunk_idx, num_threads=int(num_threads))
            total_added += len(chunk_idx)
        print("[BUILD] Finished. Total added:", total_added)
        index.save_index(index_path)
        print("[SAVE] Index saved:", index_path)

    return index


def setup_index(train, index_dir="../index", M=4, efConstruction=50, distance_method="cosine", seed=42, dataset_name=None, num_threads=-1):
    del seed
    os.makedirs(index_dir, exist_ok=True)
    dim = train.shape[1]
    naive_order = list(range(len(train)))
    index_path = os.path.join(
        index_dir,
        make_hnswlib_index_path(
            M=M,
            efConstruction=efConstruction,
            n_nodes=len(train),
            dim=dim,
            dataset_name=dataset_name,
        ),
    )
    print("Index path:", index_path)
    index = build_or_load_hnsw_index(
        train=train,
        order=naive_order,
        dim=dim,
        space=distance_method,
        M=M,
        efc=efConstruction,
        index_path=index_path,
        num_threads=num_threads,
    )
    return index, index_path, naive_order

