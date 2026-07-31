#!/usr/bin/env python3
"""Create 6,980-query MSMARCO embedding HDF5 subsets by query_id.

The BGE/Gemma full files contain all `queries.dev` queries.  The older
MSMARCO ada/GloVe experiments used the 6,980 query IDs stored next to the ada
full HDF5.  This script selects those query rows in the same order, writes
test/neighbors/distances, and external-links train to the full source file so
the corpus embeddings are not duplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path(os.environ.get("SAGE_DATA_DIR", str(REPO_ROOT / "datasets"))).expanduser()
MARCO_EMBEDDING_DIR = DATASET_ROOT / "marco_embeddings"
REFERENCE_QUERY_IDS = DATASET_ROOT / "msmarco-v1-openai-ada2-full-ip_query_ids.csv"

DATASETS = (
    (
        "msmarco-v1-bge-m3-fp32-full-ip.hdf5",
        "msmarco-v1-bge-m3-fp32-dev6980-ip.hdf5",
    ),
    (
        "msmarco-v1-embeddinggemma-300m-fp32-full-ip.hdf5",
        "msmarco-v1-embeddinggemma-300m-fp32-dev6980-ip.hdf5",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=MARCO_EMBEDDING_DIR)
    parser.add_argument("--reference-query-ids", type=Path, default=REFERENCE_QUERY_IDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_query_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        field_map = {field.lower(): field for field in reader.fieldnames}
        query_field = field_map.get("query_id") or field_map.get("id")
        if query_field is None:
            raise ValueError(f"{path} must contain query_id.")
        ids = [str(row[query_field]) for row in reader]
    if not ids:
        raise ValueError(f"No query IDs found in {path}.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate query IDs.")
    return ids


def write_query_ids(path: Path, query_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "query_id"])
        for row_index, query_id in enumerate(query_ids):
            writer.writerow([row_index, query_id])


def gather_rows(dataset: h5py.Dataset, positions: np.ndarray) -> np.ndarray:
    order = np.argsort(positions, kind="mergesort")
    sorted_positions = np.asarray(positions[order], dtype=np.int64)
    selected_sorted = np.asarray(dataset[sorted_positions], dtype=dataset.dtype)
    selected = np.empty((len(positions),) + dataset.shape[1:], dtype=selected_sorted.dtype)
    selected[order] = selected_sorted
    return selected


def jsonable_attrs(attrs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.generic):
            value = value.item()
        payload[str(key)] = value
    return payload


def make_subset(
    *,
    source_hdf5: Path,
    output_hdf5: Path,
    reference_query_ids_path: Path,
    reference_query_ids: list[str],
    overwrite: bool,
) -> None:
    if output_hdf5.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_hdf5}; pass --overwrite.")

    source_query_ids_path = source_hdf5.with_suffix("").with_name(source_hdf5.stem + "_query_ids.csv")
    source_meta_path = source_hdf5.with_suffix("").with_name(source_hdf5.stem + "_meta.json")
    output_query_ids_path = output_hdf5.with_suffix("").with_name(output_hdf5.stem + "_query_ids.csv")
    output_meta_path = output_hdf5.with_suffix("").with_name(output_hdf5.stem + "_meta.json")

    source_query_ids = read_query_ids(source_query_ids_path)
    source_positions = {query_id: row_index for row_index, query_id in enumerate(source_query_ids)}
    missing = [query_id for query_id in reference_query_ids if query_id not in source_positions]
    if missing:
        raise ValueError(
            f"{source_hdf5.name} is missing {len(missing)} reference query IDs; "
            f"examples={missing[:10]}"
        )

    positions = np.asarray([source_positions[query_id] for query_id in reference_query_ids], dtype=np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"Duplicate source positions found for {source_hdf5.name}.")

    tmp_hdf5 = output_hdf5.with_suffix(output_hdf5.suffix + ".part")
    for path in (tmp_hdf5, output_query_ids_path, output_meta_path):
        if path.exists():
            path.unlink()

    with h5py.File(source_hdf5, "r") as source:
        test = gather_rows(source["test"], positions)
        neighbors = gather_rows(source["neighbors"], positions)
        distances = gather_rows(source["distances"], positions) if "distances" in source else None
        train_shape = tuple(int(value) for value in source["train"].shape)
        source_attrs = jsonable_attrs(source.attrs)

    with h5py.File(tmp_hdf5, "w") as out:
        out.attrs.update(source_attrs)
        out.attrs["query_subset"] = "msmarco-v1-openai-ada2-full-ip_query_ids"
        out.attrs["source_hdf5"] = str(source_hdf5)
        out.attrs["source_query_rows"] = int(len(source_query_ids))
        out.attrs["query_rows"] = int(len(reference_query_ids))
        out.attrs["query_id_order"] = str(reference_query_ids_path)
        out["train"] = h5py.ExternalLink(str(source_hdf5), "/train")
        out.create_dataset("test", data=test, dtype=np.float32)
        out.create_dataset("neighbors", data=neighbors, dtype=np.int32)
        if distances is not None:
            out.create_dataset("distances", data=distances, dtype=np.float32)

    tmp_hdf5.replace(output_hdf5)
    write_query_ids(output_query_ids_path, reference_query_ids)

    source_meta: dict[str, Any] = {}
    if source_meta_path.exists():
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))

    output_meta = {
        "output_hdf5": str(output_hdf5),
        "source_hdf5": str(source_hdf5),
        "source_meta_json": str(source_meta_path) if source_meta_path.exists() else None,
        "query_subset_source": str(reference_query_ids_path),
        "query_ids_csv": str(output_query_ids_path),
        "query_rows": int(len(reference_query_ids)),
        "source_query_rows": int(len(source_query_ids)),
        "train_rows": int(train_shape[0]),
        "dim": int(train_shape[1]),
        "train_storage": {
            "type": "hdf5_external_link",
            "target_hdf5": str(source_hdf5),
            "target_path": "/train",
        },
        "selected_source_row_min": int(np.min(positions)),
        "selected_source_row_max": int(np.max(positions)),
        "selected_source_rows_are_monotonic": bool(np.all(positions[:-1] <= positions[1:])),
        "source_attrs": source_attrs,
        "source_meta": source_meta,
    }
    output_meta_path.write_text(json.dumps(output_meta, indent=2, sort_keys=True), encoding="utf-8")

    with h5py.File(output_hdf5, "r") as check:
        shapes = {
            "train": tuple(int(value) for value in check["train"].shape),
            "test": tuple(int(value) for value in check["test"].shape),
            "neighbors": tuple(int(value) for value in check["neighbors"].shape),
        }
        if "distances" in check:
            shapes["distances"] = tuple(int(value) for value in check["distances"].shape)
    print(f"[OK] {output_hdf5}")
    print(f"     source={source_hdf5.name}")
    print(f"     shapes={shapes}")
    print(f"     query_ids={output_query_ids_path}")
    print(f"     meta={output_meta_path}")


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir).expanduser().resolve()
    reference_query_ids_path = Path(args.reference_query_ids).expanduser().resolve()
    reference_query_ids = read_query_ids(reference_query_ids_path)

    print(f"[REFERENCE] {reference_query_ids_path} count={len(reference_query_ids):,}")
    for source_name, output_name in DATASETS:
        make_subset(
            source_hdf5=embedding_dir / source_name,
            output_hdf5=embedding_dir / output_name,
            reference_query_ids_path=reference_query_ids_path,
            reference_query_ids=reference_query_ids,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
