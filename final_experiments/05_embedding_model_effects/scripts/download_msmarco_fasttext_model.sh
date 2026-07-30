#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${SAGE_DATA_DIR:-/home/kyungmin/vectordb/hnsw-playground/datasets}"
PYTHON="${SAGE_PYTHON:-python3}"
FASTTEXT_URL="${FASTTEXT_URL:-https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz}"
FASTTEXT_GZ="${FASTTEXT_GZ:-${DATASET_DIR}/msmarco_passage_fasttext_static/raw/cc.en.300.bin.gz}"
FASTTEXT_BIN="${FASTTEXT_BIN:-${DATASET_DIR}/msmarco_passage_fasttext_static/raw/cc.en.300.bin}"

cd "${ROOT_DIR}"
export FASTTEXT_URL FASTTEXT_GZ FASTTEXT_BIN
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

"${PYTHON}" - <<'PYMODEL'
import os
from pathlib import Path
from prepare_msmarco_glove_static_hdf5 import download_file
from prepare_msmarco_fasttext_static_hdf5 import extract_gzip_file

url = os.environ["FASTTEXT_URL"]
gz = Path(os.environ["FASTTEXT_GZ"]).expanduser().resolve()
bin_path = Path(os.environ["FASTTEXT_BIN"]).expanduser().resolve()
gz.parent.mkdir(parents=True, exist_ok=True)

download_file(url, gz)
extract_gzip_file(gz, bin_path)
print(f"[READY] {bin_path} size={bin_path.stat().st_size / 1e9:.2f} GB")
PYMODEL
