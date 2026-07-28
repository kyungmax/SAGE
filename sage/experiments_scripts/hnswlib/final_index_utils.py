"""Original-index loading/building helpers for final HNSWLib experiments."""

from __future__ import annotations

import numpy as np

from common.dataset_utils import resolve_space_type
from hnsw_index_utils import build_index_cache_dataset_name, setup_index


def get_space_type(dataset_name: str) -> str:
    return resolve_space_type(dataset_name)


def resolve_index_dataset_name(dataset_name: str) -> str:
    return str(dataset_name)


def build_original_index(
    *,
    train: np.ndarray,
    dataset_name: str,
    index_dir: str,
    param_m: int,
    ef_construction: int,
    num_threads: int,
):
    space = get_space_type(dataset_name)
    index_dataset_name = resolve_index_dataset_name(dataset_name)
    cache_dataset_name = build_index_cache_dataset_name(index_dataset_name, space)
    index, _, _ = setup_index(
        train,
        index_dir,
        M=int(param_m),
        efConstruction=int(ef_construction),
        distance_method=space,
        dataset_name=cache_dataset_name + f"_M{int(param_m)}",
        num_threads=int(num_threads),
    )
    return index, space, index_dataset_name
