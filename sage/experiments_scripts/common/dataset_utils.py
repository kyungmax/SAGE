"""Dataset and exact-kNN helpers for SAGE experiment scripts."""

from __future__ import annotations

import os
import struct

import h5py
import numpy as np
from scipy import sparse


def is_text2image_dataset(dataset_name: str) -> bool:
    low_file = str(dataset_name).lower()
    return "text2image" in low_file or ("t2i" in low_file and "coco" not in low_file)


def resolve_space_type(dataset_name: str) -> str:
    low_file = str(dataset_name).lower()
    if any(token in low_file for token in ["msmarco", "msmacro", "-ip", "_ip", "dot"]):
        return "ip"
    if is_text2image_dataset(dataset_name):
        return "ip"
    if any(token in low_file for token in ["angular", "cosine", "coco", "nytimes", "deep", "glove", "openai"]):
        return "cosine"
    if any(token in low_file for token in ["euclidean", "l2", "sift", "gist", "mnist", "learn"]):
        return "l2"
    return "l2"


def build_index_cache_dataset_name(dataset_name: str, space_type: str) -> str:
    low_file = str(dataset_name).lower()
    if "coco" in low_file and "t2i" in low_file:
        return f"{dataset_name}__space_{space_type.lower()}"
    return str(dataset_name)


def read_fvecs(filename):
    with open(filename, "rb") as f:
        data = f.read()
    dim = struct.unpack("i", data[:4])[0]
    vecs = np.frombuffer(data, dtype=np.float32)
    return vecs.reshape(-1, dim + 1)[:, 1:]


def read_ivecs(filename):
    with open(filename, "rb") as f:
        data = f.read()
    dim = struct.unpack("i", data[:4])[0]
    vecs = np.frombuffer(data, dtype=np.int32)
    return vecs.reshape(-1, dim + 1)[:, 1:]


def exact_topk(train_subset, queries, K, distance_type="l2"):
    n_queries = len(queries)
    K = int(K)
    out = np.empty((n_queries, K), dtype=np.int32)

    if distance_type == "l2":
        batch_size = 100
        train_sq = np.sum(train_subset ** 2, axis=1)
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_queries = queries[start:end]
            q_sq = np.sum(batch_queries ** 2, axis=1, keepdims=True)
            d = q_sq + train_sq - 2 * (batch_queries @ train_subset.T)
            for i, di in enumerate(d):
                idx = np.argpartition(di, K)[:K]
                out[start + i] = idx[np.argsort(di[idx])]
    elif distance_type in ("angular", "cosine"):
        batch_size = 100
        train_norms = np.linalg.norm(train_subset, axis=1, keepdims=True) + 1e-12
        train_normalized = train_subset / train_norms
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch_queries = queries[start:end]
            q_norms = np.linalg.norm(batch_queries, axis=1, keepdims=True) + 1e-12
            q_normalized = batch_queries / q_norms
            d = 1.0 - (q_normalized @ train_normalized.T)
            for i, di in enumerate(d):
                idx = np.argpartition(di, K)[:K]
                out[start + i] = idx[np.argsort(di[idx])]
    elif distance_type == "ip":
        batch_size = 100
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            scores = queries[start:end] @ train_subset.T
            d = -scores
            for i, di in enumerate(d):
                idx = np.argpartition(di, K)[:K]
                out[start + i] = idx[np.argsort(di[idx])]
    else:
        raise ValueError("distance_type must be 'l2', 'angular', 'cosine', or 'ip'.")

    return out



def load_dataset(base_path="../datasets", file_name="glove-50-angular.hdf5"):
    file_path = os.path.join(base_path, file_name)
    with h5py.File(file_path, "r") as f:
        print(f"Keys in HDF5 file: {list(f.keys())}")
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
    return train, test, neighbors


def load_sparse_csr(base_path, stem, split):
    candidates = [
        os.path.join(base_path, f"{stem}_{split}.npz"),
        os.path.join(base_path, stem, f"{split}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return sparse.load_npz(path).astype(np.float32).tocsr()
    raise FileNotFoundError(f"Could not find sparse split '{split}' for {stem}: {candidates}")
