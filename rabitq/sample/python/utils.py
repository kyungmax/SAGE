import numpy as np
import faiss
from time import time


# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────

def read_ibin(filename: str) -> np.ndarray:
    print(f"Reading File - {filename}")
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Bad ibin header: {filename}")
        n, d = int(header[0]), int(header[1])
        data = np.fromfile(f, dtype=np.int32, count=n * d)
    if data.size != n * d:
        raise ValueError(f"Bad ibin payload: expected {n * d}, got {data.size}")
    print(f"	{filename} read, shape=({n}, {d})")
    return data.reshape(n, d)


def read_fbin(filename: str) -> np.ndarray:
    print(f"Reading File - {filename}")
    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Bad fbin header: {filename}")
        n, d = int(header[0]), int(header[1])
        data = np.fromfile(f, dtype=np.float32, count=n * d)
    if data.size != n * d:
        raise ValueError(f"Bad fbin payload: expected {n * d}, got {data.size}")
    print(f"	{filename} read, shape=({n}, {d})")
    return data.reshape(n, d)


def read_hdf5_dataset(filename: str, key: str) -> np.ndarray:
    import h5py

    print(f"Reading HDF5 - {filename}:{key}")
    with h5py.File(filename, "r") as f:
        if key not in f:
            raise KeyError(f"HDF5 key {key!r} not found in {filename}; keys={list(f.keys())}")
        data = np.asarray(f[key])
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    print(f"\t{filename}:{key} read, shape={data.shape}, dtype={data.dtype}")
    return data


def read_hdf5_neighbors(filename: str) -> np.ndarray:
    import h5py

    print(f"Reading HDF5 - {filename}:neighbors")
    with h5py.File(filename, "r") as f:
        if "neighbors" not in f:
            raise KeyError(f"HDF5 key 'neighbors' not found in {filename}; keys={list(f.keys())}")
        data = np.asarray(f["neighbors"])
    print(f"\t{filename}:neighbors read, shape={data.shape}, dtype={data.dtype}")
    return data.astype(np.int64, copy=False)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    x /= norms
    return x


def read_ivecs(filename: str, hdf5_key: str = "neighbors") -> np.ndarray:
    if filename.endswith(".hdf5") or filename.endswith(".h5"):
        return read_hdf5_neighbors(filename)
    if filename.endswith(".ibin"):
        return read_ibin(filename)
    print(f"Reading File - {filename}")
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"	{filename} read, dim={d}")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename: str, hdf5_key: str = "train") -> np.ndarray:
    if filename.endswith(".hdf5") or filename.endswith(".h5"):
        return read_hdf5_dataset(filename, hdf5_key)
    if filename.endswith(".fbin"):
        return read_fbin(filename)
    return read_ivecs(filename).view("float32")

# ──────────────────────────────────────────────
# Benchmarking utilities
# ──────────────────────────────────────────────

def compute_recall(ids: np.ndarray, gt: np.ndarray, topk: int) -> float:
    """Compute recall@topk: fraction of gt top-k found in returned top-k."""
    nq = ids.shape[0]
    total_correct = 0
    for i in range(nq):
        gt_set = set(gt[i, :topk].tolist())
        for j in range(topk):
            if ids[i, j] in gt_set:
                total_correct += 1
    return total_correct / (nq * topk)

# ──────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────

def cluster_data(
    X: np.ndarray,
    K: int,
    metric_str: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster X into K clusters using FAISS IVF.

    Returns
    -------
    centroids   : np.ndarray, shape (K, dim), float32
    cluster_ids : np.ndarray, shape (n,),     uint32
    """
    dim = X.shape[1]

    if metric_str == "ip":
        metric = faiss.METRIC_INNER_PRODUCT
        print("Clustering metric: InnerProduct")
    else:
        metric = faiss.METRIC_L2
        print("Clustering metric: L2")

    index = faiss.index_factory(dim, f"IVF{K},Flat", metric)
    index.verbose = True

    t0 = time()
    index.train(X)
    print(f"IVF training time: {time() - t0:.2f}s")

    centroids = index.quantizer.reconstruct_n(0, index.nlist)       # (K, dim) float32
    _, cluster_ids_2d = index.quantizer.search(X, 1)               # (n, 1)   int64
    cluster_ids = cluster_ids_2d.flatten().astype(np.uint32)        # (n,)     uint32

    return centroids, cluster_ids