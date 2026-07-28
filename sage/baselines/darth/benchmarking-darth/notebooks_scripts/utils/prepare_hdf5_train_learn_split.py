import argparse
import json
import os
from pathlib import Path

import h5py
import hnswlib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare benchmarking-darth processed datasets from ANN-Benchmarks-style HDF5 files. "
            "Unlike the generic HDF5 helper, this script samples learn/validation queries from the "
            "raw train split and keeps the original test split as query."
        )
    )
    parser.add_argument("--hdf5", required=True, help="Source HDF5 dataset.")
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Output dataset directory name. Defaults to the HDF5 stem.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Processed dataset root. Defaults to $DARTH_ROOT/datasets/processed.",
    )
    parser.add_argument(
        "--learn-queries",
        type=int,
        default=10000,
        help="Number of learn queries sampled from the raw train split.",
    )
    parser.add_argument(
        "--validation-queries",
        type=int,
        default=1000,
        help="Number of validation queries sampled from the raw train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=987,
        help="Sampling seed. Matches notebooks_scripts/utils/organize_datasets.py.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 4),
        help="Threads used by hnswlib BFIndex.",
    )
    parser.add_argument(
        "--base-batch-size",
        type=int,
        default=65536,
        help="Rows per batch when streaming the train split into base.fvecs and BFIndex.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=2048,
        help="Rows per batch when querying BFIndex for ground truth.",
    )
    return parser.parse_args()


def infer_source_metric(name: str) -> str:
    lowered = name.lower()
    if "angular" in lowered:
        return "angular"
    if "-ip" in lowered or "_ip" in lowered or lowered.endswith("ip"):
        return "ip"
    if "euclidean" in lowered or "l2" in lowered:
        return "euclidean"
    raise ValueError(f"Could not infer metric from dataset name: {name}")


def resolve_output_root(path_value: str) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    darth_root = os.environ.get("DARTH_ROOT")
    if not darth_root:
        raise ValueError("--output-root is required when DARTH_ROOT is not set.")
    return Path(darth_root).expanduser().resolve() / "datasets" / "processed"


def write_fvecs_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D float matrix for {path}, got {matrix.shape}")
    d = matrix.shape[1]
    prefix = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    buffer = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    buffer[:, :4] = prefix.view(np.uint8).reshape(-1, 4)
    buffer[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    with path.open("wb") as fh:
        fh.write(buffer.tobytes())


def append_fvecs_rows(fh, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    d = matrix.shape[1]
    prefix = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    buffer = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    buffer[:, :4] = prefix.view(np.uint8).reshape(-1, 4)
    buffer[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    fh.write(buffer.tobytes())


def write_ivecs_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.int32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D int matrix for {path}, got {matrix.shape}")
    d = matrix.shape[1]
    prefix = np.full((matrix.shape[0], 1), d, dtype=np.int32)
    buffer = np.empty((matrix.shape[0], 4 + 4 * d), dtype=np.uint8)
    buffer[:, :4] = prefix.view(np.uint8).reshape(-1, 4)
    buffer[:, 4:] = matrix.view(np.uint8).reshape(matrix.shape[0], 4 * d)
    with path.open("wb") as fh:
        fh.write(buffer.tobytes())


def normalize_for_metric(matrix: np.ndarray, source_metric: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if source_metric != "angular":
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def score_from_hnsw_distance(distances: np.ndarray, source_metric: str) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    if source_metric in {"angular", "ip"}:
        return 1.0 - distances
    if source_metric == "euclidean":
        return distances
    raise ValueError(f"Unsupported metric: {source_metric}")


def load_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    rows = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return rows[inverse]


def write_base_and_build_index(
    train_dataset: h5py.Dataset,
    output_path: Path,
    source_metric: str,
    threads: int,
    batch_size: int,
) -> hnswlib.BFIndex:
    dim = int(train_dataset.shape[1])
    space = "l2" if source_metric == "euclidean" else "ip"
    bf = hnswlib.BFIndex(space=space, dim=dim)
    bf.init_index(max_elements=int(train_dataset.shape[0]))
    bf.set_num_threads(threads)

    with output_path.open("wb") as fh:
        for start in range(0, int(train_dataset.shape[0]), batch_size):
            end = min(start + batch_size, int(train_dataset.shape[0]))
            chunk = np.asarray(train_dataset[start:end], dtype=np.float32)
            append_fvecs_rows(fh, chunk)
            bf.add_items(
                normalize_for_metric(chunk, source_metric),
                ids=np.arange(start, end, dtype=np.int64),
            )
            print(f">> Indexed base rows [{start}:{end})")

    return bf


def search_groundtruth(
    bf: hnswlib.BFIndex,
    vectors: np.ndarray,
    source_metric: str,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32, order="C")
    all_ids = []
    all_scores = []
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        ids, distances = bf.knn_query(normalize_for_metric(vectors[start:end], source_metric), k=k)
        all_ids.append(np.asarray(ids, dtype=np.int32))
        all_scores.append(score_from_hnsw_distance(distances, source_metric))
        print(f">> Searched GT rows [{start}:{end})")
    return np.vstack(all_ids), np.vstack(all_scores)


def write_split_with_groundtruth(
    output_dir: Path,
    stem: str,
    vectors: np.ndarray,
    gt_ids: np.ndarray,
    gt_scores: np.ndarray,
) -> None:
    write_fvecs_matrix(output_dir / f"{stem}.fvecs", vectors)
    write_ivecs_matrix(output_dir / f"{stem}.groundtruth.ivecs", gt_ids)
    write_fvecs_matrix(output_dir / f"{stem}.groundtruth.fvecs", gt_scores)


def main() -> None:
    args = parse_args()

    hdf5_path = Path(args.hdf5).expanduser().resolve()
    dataset_name = args.dataset_name or hdf5_path.stem
    source_metric = infer_source_metric(dataset_name)
    output_root = resolve_output_root(args.output_root)
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5f:
        train_dataset = h5f["train"]
        query_vectors = np.asarray(h5f["test"], dtype=np.float32)
        if "neighbors" not in h5f:
            raise ValueError(f"{hdf5_path} is missing neighbors ground truth.")
        groundtruth_k = int(np.asarray(h5f["neighbors"]).shape[1])

        train_size = int(train_dataset.shape[0])
        dim = int(train_dataset.shape[1])
        learn_total = args.learn_queries + args.validation_queries
        if learn_total > train_size:
            raise ValueError(
                f"Requested learn+validation={learn_total} exceeds train size {train_size}"
            )

        np.random.seed(args.seed)
        sampled_train_indices = np.random.choice(train_size, size=learn_total, replace=False)
        learn_indices = sampled_train_indices[: args.learn_queries]
        validation_indices = sampled_train_indices[args.learn_queries :]
        learn_vectors = load_rows(train_dataset, learn_indices)
        validation_vectors = load_rows(train_dataset, validation_indices)

        print(f">> Source dataset: {hdf5_path}")
        print(f">> Output dataset: {output_dir}")
        print(f">> Metric: {source_metric}")
        print(f">> Train shape: {train_dataset.shape}")
        print(f">> Query shape: {query_vectors.shape}")
        print(
            f">> Train-sampled split sizes: learn={len(learn_vectors)}, "
            f"validation={len(validation_vectors)}, query={len(query_vectors)}"
        )
        print(f">> Ground-truth depth: {groundtruth_k}")
        print(f">> Threads: {args.threads}")

        bf = write_base_and_build_index(
            train_dataset=train_dataset,
            output_path=output_dir / "base.fvecs",
            source_metric=source_metric,
            threads=args.threads,
            batch_size=args.base_batch_size,
        )

    learn_gt_ids, learn_gt_scores = search_groundtruth(
        bf=bf,
        vectors=learn_vectors,
        source_metric=source_metric,
        k=groundtruth_k,
        batch_size=args.query_batch_size,
    )
    validation_gt_ids, validation_gt_scores = search_groundtruth(
        bf=bf,
        vectors=validation_vectors,
        source_metric=source_metric,
        k=groundtruth_k,
        batch_size=args.query_batch_size,
    )
    query_gt_ids, query_gt_scores = search_groundtruth(
        bf=bf,
        vectors=query_vectors,
        source_metric=source_metric,
        k=groundtruth_k,
        batch_size=args.query_batch_size,
    )

    write_split_with_groundtruth(output_dir, "learn", learn_vectors, learn_gt_ids, learn_gt_scores)
    write_split_with_groundtruth(
        output_dir,
        "validation",
        validation_vectors,
        validation_gt_ids,
        validation_gt_scores,
    )
    write_split_with_groundtruth(output_dir, "query", query_vectors, query_gt_ids, query_gt_scores)

    metadata = {
        "dataset_name": dataset_name,
        "dimension": dim,
        "groundtruth_k": groundtruth_k,
        "learn_queries": int(args.learn_queries),
        "validation_queries": int(args.validation_queries),
        "test_queries": int(query_vectors.shape[0]),
        "seed": int(args.seed),
        "source_hdf5": str(hdf5_path),
        "source_metric": source_metric,
        "train_sample_origin": "hdf5/train",
        "query_origin": "hdf5/test",
        "train_vectors": train_size,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f">> Wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
