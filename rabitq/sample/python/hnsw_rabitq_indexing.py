import argparse
from time import time
import numpy as np
from rabitqlib import HnswIndex
from utils import read_fvecs, cluster_data, l2_normalize_rows

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
NUM_CLUSTERS    = 256           # number of clusters
M               = 16            # degree bound for HNSW
EF_CONSTRUCTION = 200           # ef for indexing
TOTAL_BITS      = 8             # total number of bits for quantization
METRIC          = "l2"          # "l2" or "ip"
FASTER_QUANT    = False         # use faster quantization
NUM_THREADS     = 4             # number of threads for build
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file, hdf5_key=args.hdf5_key)
    print(f"Data shape: {data.shape}")
    if args.normalize:
        print("L2-normalizing vectors")
        data = l2_normalize_rows(data)
    if args.sanitize_nonfinite:
        nonfinite_mask = ~np.isfinite(data)
        nonfinite_count = int(nonfinite_mask.sum())
        if nonfinite_count:
            bad_rows = int(nonfinite_mask.any(axis=1).sum())
            print(
                f"Sanitizing non-finite values: {nonfinite_count} values "
                f"across {bad_rows} rows -> 0"
            )
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Cluster with FAISS
    centroids, cluster_ids = cluster_data(data, args.num_clusters, args.metric)
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")

    # 3. Build HNSW index
    n, dim = data.shape
    print(f"\nBuilding HNSW index: n={n}, dim={dim}, M={args.degree}, "
          f"ef={args.ef_construction}, bits={args.total_bits}, metric={args.metric}")

    idx = HnswIndex(
        dim=dim,
        max_elements=n,
        M=args.degree,
        ef_construction=args.ef_construction,
        nbits=args.total_bits,
        metric=args.metric,
    )

    t0 = time()
    idx.build(
        data,
        centroids,
        cluster_ids,
        num_threads=args.num_threads,
        fast_quantization=args.faster_quant,
    )
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(args.index_file)
    print(f"Index saved → {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ HNSW Manager")

    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument("--num-clusters", dest="num_clusters", type=int, metavar="INT", default=256, help="Number of clusters for quantization")
    parser.add_argument("--degree", dest="degree", type=int, metavar="INT", default=M, help="Degree bound for HNSW")
    parser.add_argument("--ef-construction", dest="ef_construction", type=int, metavar="INT", default=EF_CONSTRUCTION, help="EF parameter for index construction")
    parser.add_argument("--total-bits", dest="total_bits", type=int, metavar="INT", default=TOTAL_BITS, help="Total number of bits for quantization")
    parser.add_argument("--metric", dest="metric", type=str, default="l2", choices=["l2", "ip"], help="Distance metric (l2 or ip)")
    parser.add_argument("--faster-quant", dest="faster_quant", action="store_true", help="Use faster quantization method")
    parser.add_argument("--num-threads", dest="num_threads", type=int, metavar="INT", default=NUM_THREADS, help="Number of threads for building the index")
    parser.add_argument("--sanitize-nonfinite", dest="sanitize_nonfinite", action="store_true", help="Replace NaN/Inf values with 0 before clustering and indexing")
    parser.add_argument("--hdf5-key", dest="hdf5_key", type=str, default="train", help="HDF5 dataset key for vectors")
    parser.add_argument("--normalize", dest="normalize", action="store_true", help="L2-normalize vectors before clustering and indexing")

    args = parser.parse_args()

    main(args)