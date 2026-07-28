#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BENCHMARKING_DARTH_ROOT = THIS_DIR.parent
SNAPSHOT_ROOT = BENCHMARKING_DARTH_ROOT.parent


def _resolve_path(path_value: str | None, default: Path) -> Path:
    if path_value:
        return Path(path_value).expanduser().resolve()
    return default.expanduser().resolve()


def _default_darth_index_root() -> Path:
    if os.environ.get("DARTH_INDEX_ROOT"):
        return Path(os.environ["DARTH_INDEX_ROOT"])
    if os.environ.get("INDEX_ROOT"):
        return Path(os.environ["INDEX_ROOT"]) / "DARTH"
    return Path("/home/kyungmin/vectordb/hnsw-playground/index/DARTH")


def _default_darth_dataset_root() -> Path:
    if os.environ.get("DARTH_DATASET_ROOT"):
        return Path(os.environ["DARTH_DATASET_ROOT"])
    if os.environ.get("DATASETS_ROOT"):
        return Path(os.environ["DATASETS_ROOT"]) / "DARTH"
    return Path("/home/kyungmin/vectordb/hnsw-playground/datasets/processed/DARTH")


SHARED_INDEX_ROOT = _resolve_path(None, _default_darth_index_root())
SHARED_DATASET_ROOT = _resolve_path(None, _default_darth_dataset_root())
SHARED_INTERVAL_ROOT = SHARED_INDEX_ROOT / "intervals"
SHARED_TRAINING_ROOT = SHARED_INDEX_ROOT / "et_training_data"
LOCAL_PREDICTOR_ROOT = _resolve_path(
    os.environ.get("DARTH_PREDICTOR_ROOT"),
    BENCHMARKING_DARTH_ROOT / "predictor_models" / "darth",
)


@dataclass(frozen=True)
class SharedDatasetSpec:
    dataset: str
    eval_query_num: int
    training_query_num: int = 10000


DATASET_SPECS: tuple[SharedDatasetSpec, ...] = (
    SharedDatasetSpec("glove-100-angular", 1000),
    SharedDatasetSpec("glove-200-angular", 1000),
    SharedDatasetSpec("gist-960-euclidean", 800),
    SharedDatasetSpec("nytimes-256-angular", 1000),
    SharedDatasetSpec("deep-image-96-angular", 1000),
    SharedDatasetSpec("dbpedia-openai-1000k-angular", 1000),
    SharedDatasetSpec("agnews-mxbai-1024-euclidean", 800),
    SharedDatasetSpec("msmarco-v1-openai-ada2-1M-ip", 1000),
)


def get_dataset_specs(names: list[str] | tuple[str, ...] | None = None) -> list[SharedDatasetSpec]:
    if not names:
        return list(DATASET_SPECS)
    wanted = set(names)
    specs = [spec for spec in DATASET_SPECS if spec.dataset in wanted]
    missing = sorted(wanted - {spec.dataset for spec in specs})
    if missing:
        raise ValueError(f"Unsupported datasets: {missing}")
    return specs


def find_dataset_spec(name: str) -> SharedDatasetSpec:
    for spec in DATASET_SPECS:
        if spec.dataset == name:
            return spec
    raise ValueError(f"Unsupported dataset: {name}")


def shared_index_path(dataset: str, *, m: int, efc: int) -> Path:
    return SHARED_INDEX_ROOT / dataset / f"M{m}_efC{efc}.index"


def shared_training_csv_path(
    dataset: str,
    *,
    k: int,
    m: int,
    efc: int,
    efs: int,
    qs: int,
    li: int,
) -> Path:
    return SHARED_TRAINING_ROOT / dataset / f"k{k}" / f"M{m}_efC{efc}_efS{efs}_qs{qs}_li{li}.csv"


def shared_interval_json_path(
    dataset: str,
    *,
    k: int,
    efc: int,
    efs: int,
    target_recall: float,
    qs: int,
) -> Path:
    return SHARED_INTERVAL_ROOT / f"{dataset}_k{k}_efC{efc}_efS{efs}_rt{target_recall:.2f}_qs{qs}.json"


def predictor_model_path(
    dataset: str,
    *,
    k: int,
    m: int,
    efc: int,
    efs: int,
    qs: int,
    n_estimators: int,
    li: int,
) -> Path:
    return (
        LOCAL_PREDICTOR_ROOT
        / f"{dataset}_M{m}_efC{efc}_efS{efs}_s{qs}_k{k}_nestim{n_estimators}_li{li}_all_feats.txt"
    )
